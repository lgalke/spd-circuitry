import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from spd.module_utils import init_param_


class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_heads=1):
        super().__init__()

        # Project input into hidden space first
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )

        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):

        x = self.input_proj(x)

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        x = self.norm2(x + self.ff(x))

        return self.out_proj(x)
    

class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    NOTE: In the paper, we use V and U for A and B, respectively.

    The weight matrix W is decomposed as W = B^T @ A^T, where A and B are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, K: int, bias: Tensor | None):
        super().__init__()
        self.C = C
        self.K = K

        self.A = nn.Parameter(torch.empty(d_in, C, K))
        self.B = nn.Parameter(torch.empty(C, K, d_out))
        self.bias = bias

        init_param_(self.A, fan_val=d_out, nonlinearity="linear")
        init_param_(self.B, fan_val=C * K, nonlinearity="linear")

        self.mask: Float[Tensor, "... C"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """B^T @ A^T"""
        return einops.einsum(self.A, self.B, "d_in C K, C K d_out -> d_out d_in")

    # @torch.compile
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.A, "... d_in, d_in C K -> ... C K")

        if self.mask is not None:
            component_acts *= self.mask.unsqueeze(-1)

        out = einops.einsum(component_acts, self.B, "... C K, C K d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        C: int,
        K: int,
    ):
        super().__init__()
        self.C = C
        self.K = K

        self.A = nn.Parameter(torch.empty(vocab_size, C, K))
        self.B = nn.Parameter(torch.empty(C, K, embedding_dim))

        init_param_(self.A, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.B, fan_val=C* K, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "batch pos C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """A @ B"""
        return einops.einsum(
            self.A, self.B, "vocab_size C K, ... C K embedding_dim -> vocab_size embedding_dim"
        )

    # @torch.compile
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.A[x]  # (batch pos C K)

        if self.mask is not None:
            #was component_acts *= self.mask, but k causes dimension mismatch?
            component_acts *= self.mask.unsqueeze(-1)

        out = einops.einsum(
            component_acts, self.B, "batch pos C K, ... C K embedding_dim -> batch pos embedding_dim"
        )
        return out
