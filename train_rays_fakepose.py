import datetime
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from torch import autocast
from tqdm import tqdm

from utils.pose import make_fake_poses
from utils.training import get_batch, get_dataloaders, logger

LOG_INTERVAL = 100


def round_to_nearest_multiple(value, multiple=14):
    return multiple * round(value / multiple)


def backbone_feats(cfg, image_batch, backbone):
    _, _, height, width = image_batch.shape
    downscale_factor = 0.5 if cfg.ratio == "fixed" else np.random.uniform(0.25, 0.5)
    low_height = round_to_nearest_multiple(height * downscale_factor, backbone.patch_size)
    low_width = round_to_nearest_multiple(width * downscale_factor, backbone.patch_size)

    with torch.no_grad():
        hr_patch_tokens, _ = backbone(image_batch)
        low_res_batch = F.interpolate(image_batch, size=(low_height, low_width), mode="bilinear")
        lr_patch_tokens, _ = backbone(low_res_batch)

    return hr_patch_tokens, lr_patch_tokens


def setup_logging(cfg):
    log_dir = HydraConfig.get().runtime.output_dir
    writer, _, checkpoint_dir = logger(cfg, log_dir)
    terminal_console = Console()
    file_console = Console(file=open("train.log", "w", encoding="utf-8"))

    def log_print(*args, **kwargs):
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    return writer, checkpoint_dir, file_console, log_print


def save_checkpoint(path, raysup, optimizer_raysup, epoch, cfg, log_print):
    torch.save(
        {
            "optimizer_raysup": optimizer_raysup.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
            "raysup": raysup.state_dict(),
        },
        path,
    )
    log_print(f"Saved checkpoint: {path}")


@hydra.main(config_path="config", config_name="base")
def trainer(cfg: DictConfig):
    print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=True))
    torch.manual_seed(0)

    writer, checkpoint_dir, file_console, log_print = setup_logging(cfg)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'=' * 50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = instantiate(cfg.backbone).to(device)

    log_print("[bold yellow]Using fake camera poses.[/bold yellow]")
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")
    log_print(f"[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    raysup = instantiate(cfg.model).to(device)
    raysup.train()

    train_dataloader, _ = get_dataloaders(cfg, backbone, is_evaluation=False)
    log_print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    criterion = instantiate(cfg.loss, dim=backbone.embed_dim)
    optimizer_raysup = instantiate(cfg.optimizer, params=list(raysup.parameters()))
    log_print(f"[bold cyan]RaysUp parameters: {sum(p.numel() for p in raysup.parameters()):,}[/bold cyan]")

    checkpoint_interval = max(1, int(cfg.max_steps * 0.25))
    total_steps = max(1, cfg.epochs * cfg.max_steps)
    amp_enabled = device.type == "cuda" and cfg.bfloat16
    current_step = 0

    try:
        for epoch in range(cfg.epochs):
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                current_step = epoch * len(train_dataloader) + batch_idx
                progress = (current_step / total_steps) * 100

                batch = get_batch(batch, device)
                image_batch = batch["image"]

                with autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
                    hr_feats, lr_feats = backbone_feats(cfg, image_batch, backbone)
                    _, _, height, width = hr_feats.shape
                    image_batch = F.interpolate(image_batch, scale_factor=0.5, mode="bilinear")
                    poses = make_fake_poses(image_batch)
                    raysup_hr_feats = raysup(image_batch, lr_feats, poses, (height, width))
                    loss_raysup = criterion(raysup_hr_feats, hr_feats)["total"]

                optimizer_raysup.zero_grad()
                loss_raysup.backward()
                optimizer_raysup.step()

                if batch_idx % LOG_INTERVAL == 0:
                    writer.add_scalar("Loss/raysup_hr", loss_raysup.item(), current_step)
                    writer.add_scalar("Learning Rate RaysUp", optimizer_raysup.param_groups[0]["lr"], current_step)
                    log_print(
                        f"Epoch={epoch}/{cfg.epochs} | "
                        f"Batch={batch_idx}/{len(train_dataloader)} | "
                        f"Progress={progress:.1f}% | "
                        f"FakePose | "
                        f"raysup_hr={loss_raysup.item():.4f}"
                    )

                should_save = (batch_idx % checkpoint_interval == 0 and batch_idx != 0) or current_step >= cfg.max_steps
                if should_save:
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_{current_step}steps.pth")
                    save_checkpoint(checkpoint_path, raysup, optimizer_raysup, epoch, cfg, log_print)

                if current_step >= cfg.max_steps or cfg.sanity:
                    break

            writer.flush()
            if current_step >= cfg.max_steps or cfg.sanity:
                break
    finally:
        writer.close()
        file_console.file.close()


if __name__ == "__main__":
    trainer()
