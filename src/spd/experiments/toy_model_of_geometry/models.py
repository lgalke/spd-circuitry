from pathlib import Path
from typing import Any, Literal

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.wandb_utils import download_wandb_file, fetch_latest_wandb_checkpoint, fetch_wandb_run_dir


class GeometryModelPaths(BaseModel):
    """Paths to output files from a GeometryModel training run."""

    geometry_train_config: Path
    checkpoint: Path


class GeometryModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ranks: list[int]
    n_hidden: PositiveInt
    init_bias_to_zero: bool
    device: str
    output_activation: Literal["relu", "identity"] = "relu"


class GeometryModel(nn.Module):
    def __init__(self, config: GeometryModelConfig):
        super().__init__()
        self.config = config
        self.input_dim = sum(k * (k + 1) for k in config.ranks)
        self.output_dim = len(config.ranks)
        self.linear1 = nn.Linear(self.input_dim, config.n_hidden, bias=True)
        self.linear2 = nn.Linear(config.n_hidden, self.output_dim, bias=True)

        if config.init_bias_to_zero:
            self.linear2.bias.data.zero_()

    def forward(self, x: Float[Tensor, "... input_dim"], **_: Any) -> Float[Tensor, "... output_dim"]:
        x = self.linear1(x)
        x = F.relu(x)
        out = self.linear2(x)
        if self.config.output_activation == "relu":
            out = F.relu(out)
        return out

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> GeometryModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        geometry_model_config_path = download_wandb_file(run, run_dir, "geometry_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return GeometryModelPaths(geometry_train_config=geometry_model_config_path, checkpoint=checkpoint_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["GeometryModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `resid_mlp_train_config.yaml` and `label_coeffs.json` are in the same
                directory as the checkpoint.

        Returns:
            model: The pretrained GeometryModel
            geometry_model_config_dict: The config dict used to train the model (we don't
                instantiate a train config due to circular import issues)
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = GeometryModelPaths(
                geometry_train_config=Path(path).parent / "geometry_train_config.yaml",
                checkpoint=Path(path),
            )

        with open(paths.geometry_train_config) as f:
            geometry_train_config_dict = yaml.safe_load(f)

        geometry_config = GeometryModelConfig(**geometry_train_config_dict["geometry_model_config"])
        geometry_model = cls(config=geometry_config)
        params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        geometry_model.load_state_dict(params)

        return geometry_model, geometry_train_config_dict