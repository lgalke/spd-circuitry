"""Run SPD on a Geometry model."""

from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import torch
import wandb
import yaml

from spd.configs import Config, GeometryTaskConfig
from spd.data_utils import DatasetGeneratedDataLoader
from spd.experiments.toy_model_of_geometry.simplex_dataset import SimplexDataset
from spd.experiments.toy_model_of_geometry.models import GeometryModel, GeometryModelConfig
from spd.log import logger
from spd.plotting import create_toy_model_plot_results
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import get_device, load_config, set_seed
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(config: Config, geometry_model_config: GeometryModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        input_dim = sum(k * (k + 1) for k in geometry_model_config.ranks)
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"_input-dim{input_dim}_"
        run_suffix += f"hid{geometry_model_config.n_hidden}_"
        run_suffix += f"ranks{'-'.join(str(r) for r in geometry_model_config.ranks)}"
    return config.wandb_run_name_prefix + run_suffix


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    geometry_model: GeometryModel,
    geometry_model_train_config_dict: dict[str, Any],
) -> None:
    torch.save(geometry_model.state_dict(), out_dir / "geometry.pth")

    with open(out_dir / "geometry_train_config.yaml", "w") as f:
        yaml.dump(geometry_model_train_config_dict, f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "geometry.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "geometry_train_config.yaml"), base_path=out_dir, policy="now")


def main(config_path_or_obj: Path | str | Config) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project)

    task_config = config.task_config
    assert isinstance(task_config, GeometryTaskConfig)
    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict = GeometryModel.from_pretrained(
        config.pretrained_model_path,
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=config, geometry_model_config=target_model.config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        geometry_model=target_model,
        geometry_model_train_config_dict=target_model_train_config_dict,
    )

    dataset = SimplexDataset(
        dimensions=target_model.config.ranks,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # No tied weights in geometry model (input_dim != output_dim)
    tied_weights = None

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        plot_results_fn=create_toy_model_plot_results,
        tied_weights=tied_weights,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)