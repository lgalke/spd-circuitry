from datetime import datetime
from pathlib import Path
from typing import Literal
import numpy as np
import torch
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt
from tqdm import tqdm, trange

from spd.data_utils import DatasetGeneratedDataLoader
from spd.experiments.toy_model_of_geometry.models import GeometryModel, GeometryModelConfig
from spd.log import logger
from spd.utils import set_seed
from spd.experiments.toy_model_of_geometry.simplex_dataset import SimplexDataset
wandb.require("core")


class GeometryTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    geometry_model_config: GeometryModelConfig 
    feature_probability: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr: float
    data_generation_type: Literal["at_least_zero_active", "exactly_one_active"]
    lr_schedule: Literal["linear", "cosine", "constant"] = "linear"

def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))

def train(
    model: GeometryModel,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    log_wandb: bool,
    importance: float,
    steps: int,
    print_freq: int,
    lr: float,
    lr_schedule: Literal["linear", "cosine", "constant"],
    ) -> None:
    hooks = []

    if lr_schedule == "linear":
        lr_schedule_fn = linear_lr
    elif lr_schedule == "cosine":
        lr_schedule_fn = cosine_decay_lr
    elif lr_schedule == "constant":
        lr_schedule_fn = constant_lr
    else:
        raise ValueError(f"Invalid lr_schedule: {lr_schedule}")

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule_fn(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch, labels = next(data_iter)
            out = model(batch)
            error = importance * (labels - out) ** 2

            loss = error.mean()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(
                    model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item()}")
                t.set_postfix(
                    loss=loss.item(),
                    lr=step_lr,
                )
                if log_wandb:
                    wandb.log({"loss": loss.item(), "lr": step_lr}, step=step)



def get_model_and_dataloader(
    config: GeometryTrainConfig, device: str
    ) -> tuple[GeometryModel, DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    model = GeometryModel(config=config.geometry_model_config)
    model.to(device)

    dataset = SimplexDataset(
        dimensions=config.geometry_model_config.ranks,
        feature_probability=config.feature_probability,
        device=device,
        data_generation_type=config.data_generation_type,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)
    return model, dataloader


def run_train(config: GeometryTrainConfig, device: str) -> None:
    model, dataloader = get_model_and_dataloader(config, device)

    model_cfg = config.geometry_model_config
    input_dim = sum(k * (k + 1) for k in model_cfg.ranks)
    run_name = (
        f"geometry_input-dim{input_dim}_n-hidden{model_cfg.n_hidden}_"
        f"feat_prob{config.feature_probability}_seed{config.seed}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.wandb_project:
        wandb.init(project=config.wandb_project, name=run_name)

    # Save config
    config_path = out_dir / "geometry_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    train(
        model,
        dataloader=dataloader,
        log_wandb=config.wandb_project is not None,
        steps=config.steps,
        importance=1.0,
        print_freq=100,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
    )

    model_path = out_dir / "geometry.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GeometryTrainConfig(
    wandb_project="spd-train-tms",
    geometry_model_config=GeometryModelConfig(
    ranks=[2, 2, 2],
    n_hidden=150,
    device=device,
    init_bias_to_zero=False,
    output_activation="relu",
),
    feature_probability=0.40,
    batch_size=4096,               
    steps=15000,
    seed=0,
    lr=1e-3,
    lr_schedule="cosine",
    data_generation_type="at_least_zero_active",)
    set_seed(config.seed)

    run_train(config, device)
    