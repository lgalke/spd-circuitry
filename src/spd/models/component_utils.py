from collections.abc import Mapping

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, LinearComponent, Transformer
import torch.nn as nn
from spd.utils import extract_batch_data



def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append(
            {layer: ci + (1 - ci) * torch.rand_like(ci) for layer, ci in causal_importances.items()}
        )
    return stochastic_masks


def calc_ci_l_zero(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    cutoff: float = 1e-2,
) -> dict[str, float]:
    """Calculate the L0 loss on the causal importances, summed over the C dimension."""
    ci_l_zero = {}
    for layer_name, ci in causal_importances.items():
        mean_dims = tuple(range(ci.ndim - 1))
        ci_l_zero[layer_name] = (ci > cutoff).float().mean(dim=mean_dims).sum().item()
    return ci_l_zero


def component_activation_statistics(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_steps: int,
    device: str,
    threshold: float = 0.1,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " C"]]]:
    """Get the number and strength of the masks over the full dataset."""
  
    # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.C, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: v.A for module_name, v in components.items()}

        causal_importances, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            As=As,
            gates = model.gates,
            detach_inputs=False,
        )
        for module_name, ci in causal_importances.items():
            # mask (batch, pos, C) or (batch, C)
            n_tokens[module_name] += ci.shape[:-1].numel()

            # Count the number of components that are active above the threshold
            active_components = ci > threshold
            total_n_active_components[module_name] += int(active_components.sum().item())

            sum_dims = tuple(range(ci.ndim - 1))
            component_activation_counts[module_name] += active_components.sum(dim=sum_dims)

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts


def lower_leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(x > 0, torch.clamp(x, max=1), alpha * x)


def upper_leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    # TODO: Make more memory efficient
    return torch.where(x > 1, 1 + alpha * (x - 1), F.relu(x))





def calc_causal_importances(
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
    As: Mapping[str, Float[Tensor, "d_in C"]],
    gates: nn.ModuleDict,
    detach_inputs: bool = False,
) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:

    causal_importances = {}
    causal_importances_upper_leaky = {}
    for param_name in pre_weight_acts:
        acts = pre_weight_acts[param_name]
        
        if not acts.dtype.is_floating_point:
            component_act_k = As[param_name][acts]
        else:
            A = As[param_name]
            component_act_k = einops.einsum(acts, A, "... d_in, d_in C k -> ... C k")

        gate_feats = component_act_k.detach() if detach_inputs else component_act_k # gate_feats has shape (batch, C, k)

        # Run this layer's attention only over its own components
        gate_key = param_name.replace(".", "-")
        gate_out = gates[gate_key].forward(gate_feats).squeeze(-1) #Transformer returns (batch, C)
      
        importance = torch.softmax(gate_out, dim=-1)
        causal_importances[param_name] = importance
        causal_importances_upper_leaky[param_name] = importance
 
    return causal_importances, causal_importances_upper_leaky


