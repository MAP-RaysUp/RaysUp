import numpy as np
import torch
from omegaconf import OmegaConf


DA3_SIZE_TO_NAME = {
    "small": "depth-anything/da3-small",
    "base": "depth-anything/da3-base",
    "large": "depth-anything/da3-large",
}


def resolve_da3_name(cfg):
    da3_name = OmegaConf.select(cfg, "da3.name")
    if da3_name:
        return da3_name

    da3_size = OmegaConf.select(cfg, "da3.size") or "large"
    if da3_size not in DA3_SIZE_TO_NAME:
        raise ValueError(
            f"Unknown da3.size='{da3_size}'. "
            f"Choose from {sorted(DA3_SIZE_TO_NAME)} or set da3.name explicitly."
        )
    return DA3_SIZE_TO_NAME[da3_size]


def load_da3_pose_estimator(cfg, device, log_print):
    from depth_anything_3.api import DepthAnything3

    da3_name = resolve_da3_name(cfg)
    log_print(f"[bold yellow]Loading DA3 pose estimator: {da3_name}[/bold yellow]")
    pose_estimator = DepthAnything3.from_pretrained(da3_name).to(device)
    pose_estimator.eval()
    log_print(f"[bold green]Loaded DA3 pose estimator: {da3_name}[/bold green]")
    return pose_estimator


def estimate_da3_poses(image_batch, pose_estimator, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_batch.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_batch.device).view(3, 1, 1)
    camera_poses = []
    intrinsics = []

    with torch.autocast(device_type=image_batch.device.type, enabled=False):
        with torch.no_grad():
            for image in image_batch:
                image_np = (image * std + mean).permute(1, 2, 0).detach().cpu().numpy()
                image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
                prediction = pose_estimator.inference([image_np])
                camera_poses.append(torch.as_tensor(prediction.extrinsics[0], dtype=torch.float32))
                intrinsics.append(torch.as_tensor(prediction.intrinsics[0], dtype=torch.float32))

    camera_poses = torch.stack(camera_poses).to(device)
    intrinsics = torch.stack(intrinsics).to(device)
    fxfycxcy = torch.stack(
        [
            intrinsics[:, 0, 0],
            intrinsics[:, 1, 1],
            intrinsics[:, 0, 2],
            intrinsics[:, 1, 2],
        ],
        dim=-1,
    )
    return camera_poses, fxfycxcy


def make_fake_poses(image_batch):
    batch_size, _, height, width = image_batch.shape
    device = image_batch.device
    camera_poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    fxfycxcy = torch.tensor(
        [float(width), float(height), width / 2.0, height / 2.0],
        device=device,
    ).unsqueeze(0).repeat(batch_size, 1)
    return camera_poses, fxfycxcy
