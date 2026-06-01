import math
import torch
from torch import Tensor

class SimplexDataset:
    def __init__(self, dimensions: list[int], device: str = "cpu",
                 data_generation_type: str = "at_least_zero_active",
                 feature_probability: float = 0.75,
                 degeneracy_eps: float = 1e-6,
                 max_resample_attempts: int = 100):
        self.dimensions = dimensions
        self.device = device
        self.data_generation_type = data_generation_type
        self.feature_probability = feature_probability
        self.degeneracy_eps = degeneracy_eps
        self.max_resample_attempts = max_resample_attempts

        self.group_sizes = [k * (k + 1) for k in dimensions]
        self.n_features = sum(self.group_sizes)
        self.output_dim = len(dimensions)

        self.groups = []
        start = 0
        for size in self.group_sizes:
            self.groups.append(list(range(start, start + size)))
            start += size

    def __len__(self) -> int:
        return 2**31

    def _edge_dets(self, vertices: Tensor) -> Tensor:
        """vertices: (..., k+1, k). Returns (...,) signed determinants of edge matrix."""
        edges = vertices[..., 1:, :] - vertices[..., :1, :]
        return torch.det(edges)

    def _volume_from_vertices(self, vertices: Tensor) -> Tensor:
        """vertices: (..., k+1, k). Returns (...,) volumes via |det| / k!."""
        edges = vertices[..., 1:, :] - vertices[..., :1, :]
        k = edges.shape[-2]
        return torch.abs(torch.det(edges)) / math.factorial(k)

    def _sample_valid_simplices(self, num: int, k: int) -> Tensor:
        """Sample `num` non-degenerate (k+1, k) simplices in [0,1]^k via rejection."""
        vertices = torch.rand(num, k + 1, k, device=self.device)

        for _ in range(self.max_resample_attempts):
            dets = self._edge_dets(vertices)

            #a simplex is invalid if the determinant of the edge set v0-v1,v0-v2,... is linearly dependent, i.e if the determinant is zero
            bad = torch.abs(dets) < self.degeneracy_eps
            n_bad = int(bad.sum().item())
            if n_bad == 0:
                return vertices
            #check is not that efficient, would technically only have to recompute the bad ones,
            #but for the toy model its fine.
            vertices[bad] = torch.rand(n_bad, k + 1, k, device=self.device)

        raise RuntimeError(
            f"failed to sample non-degenerate simplices after "
            f"{self.max_resample_attempts} attempts (k={k}, eps={self.degeneracy_eps})"
        )

    def generate_batch(self, batch_size: int):
        batch = torch.zeros(batch_size, self.n_features, device=self.device)
        labels = torch.zeros(batch_size, self.output_dim, device=self.device)

        if self.data_generation_type == "at_least_zero_active":
            #produces a mask for every simplex in a batch, and see if it should be active.
            group_active = torch.rand(batch_size, len(self.dimensions), device=self.device) < self.feature_probability
        elif self.data_generation_type == "exactly_one_active":
            #select a random simplex for every batch, and set it to active - a 1D vector of length batch_size.
            chosen = torch.randint(len(self.dimensions), (batch_size,), device=self.device)
            #Create a matrix of shape batch x simplex groups
            group_active = torch.zeros(batch_size, len(self.dimensions), dtype=torch.bool, device=self.device)
            #Index into the batch row and set column chosen to true.
            group_active[torch.arange(batch_size), chosen] = True
        else:
            raise ValueError("not a valid data_generation_type for Simplex")

        for g_idx, (k, group) in enumerate(zip(self.dimensions, self.groups)):
            #select the active columns of the current simplex group, and get the batch indices where they are active.
            active_rows = group_active[:, g_idx].nonzero(as_tuple=True)[0]
            if active_rows.numel() == 0:
                continue
            
            num_active = active_rows.numel()
            #Num of active simplices in the batch, and the dimension of the simplex group.
            vertices = self._sample_valid_simplices(num_active, k) #returns batch x (k+1) x k tensor

            rows = active_rows.reshape(-1, 1)
            batch[rows, group] = vertices.reshape(num_active, -1)
            labels[active_rows, g_idx] = self._volume_from_vertices(vertices)

        return batch, labels


if __name__ == "__main__":
    torch.manual_seed(67)

    # sanity: known formulas still work
    ds = SimplexDataset([2])
    print(ds.generate_batch(2))
