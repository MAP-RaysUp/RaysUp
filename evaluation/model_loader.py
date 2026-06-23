import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf


DA3_SIZE_TO_NAME = {
    "small": "depth-anything/da3-small",
    "base": "depth-anything/da3-base",
    "large": "depth-anything/da3-large",
}

EXTERNAL_UPSAMPLERS = {
    "jafarmain": {"entrypoint": "jafar", "path_key": "jafarmain_repo_path", "clear_src": True},
    "anyup": {"entrypoint": "anyup", "path_key": "anyup_repo_path", "use_natten": True},
    "anyup_multi_backbone": {
        "entrypoint": "anyup_multi_backbone",
        "path_key": "anyup_repo_path",
        "use_natten": True,
    },
    "featup": {"entrypoint": "dinov2", "path_key": "featup_repo_path"},
    "loftup": {"entrypoint": "loftup_dinov2s", "path_key": "loftup_repo_path"},
}


class BilinearUpsampler(nn.Module):
    def forward(self, image_batch, patch_tokens, target_size):
        del image_batch
        return F.interpolate(patch_tokens, size=target_size, mode="bilinear", align_corners=False)


class NoPoseUpsamplerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_batch, patch_tokens, target_size):
        try:
            return self.model(image_batch, patch_tokens, target_size)
        except TypeError:
            pass

        try:
            return self.model(image_batch, patch_tokens, output_size=target_size)
        except TypeError:
            pass

        return self.model(image_batch, patch_tokens)


def get_eval_option(cfg, key, default=None):
    value = OmegaConf.select(cfg, f"eval.{key}")
    return default if value is None else value


def get_upsampler_name(cfg):
    return get_eval_option(cfg, "upsampler_name", cfg.model.name)


def clear_src_modules():
    keys_to_remove = [key for key in sys.modules.keys() if key.startswith("src.") or key == "src"]
    for key in keys_to_remove:
        del sys.modules[key]


def resolve_config_path(cfg, path):
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        project_root = OmegaConf.select(cfg, "project_root")
        if project_root:
            path = os.path.join(str(project_root), path)
    return os.path.abspath(path)


def _require_repo_path(cfg, upsampler_name):
    spec = EXTERNAL_UPSAMPLERS[upsampler_name]
    repo_path = get_eval_option(cfg, spec["path_key"], None)
    repo_path = get_eval_option(cfg, "external_repo_path", repo_path)
    if not repo_path:
        raise ValueError(
            f"upsampler_name='{upsampler_name}' requires eval.external_repo_path "
            f"or eval.{spec['path_key']} to point to the local torch.hub repository."
        )
    return resolve_config_path(cfg, repo_path)


def load_local_hub_model(repo_path, entrypoint, device, log_print, clear_src=False, **kwargs):
    if clear_src:
        clear_src_modules()
    log_print(f"[green]Loading local torch.hub model: {entrypoint} from {repo_path}[/green]")
    return torch.hub.load(repo_path, entrypoint, source="local", **kwargs).to(device)


def load_checkpoint_state(model, checkpoint_path, device, log_print):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("raysup", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log_print(f"[green]Loaded model checkpoint: {checkpoint_path}[/green]")
    if missing:
        log_print(f"[yellow]Missing checkpoint keys: {len(missing)}[/yellow]")
    if unexpected:
        log_print(f"[yellow]Unexpected checkpoint keys: {len(unexpected)}[/yellow]")


def load_upsampler(cfg, device, log_print):
    upsampler_name = get_upsampler_name(cfg)

    if upsampler_name == "bilinear":
        model = BilinearUpsampler().to(device).eval()
        log_print("[bold green]Using bilinear interpolation upsampler.[/bold green]")
        return model

    if upsampler_name == "raysup":
        model = instantiate(cfg.model).to(device)
        checkpoint_path = get_eval_option(cfg, "model_ckpt", None)
        if checkpoint_path:
            checkpoint_path = resolve_config_path(cfg, checkpoint_path)
            load_checkpoint_state(model, checkpoint_path, device, log_print)
        model.eval()
        return model

    if upsampler_name in EXTERNAL_UPSAMPLERS:
        spec = EXTERNAL_UPSAMPLERS[upsampler_name]
        kwargs = {}
        if spec.get("use_natten"):
            kwargs["use_natten"] = get_eval_option(cfg, "use_natten", True)
        model = load_local_hub_model(
            _require_repo_path(cfg, upsampler_name),
            spec["entrypoint"],
            device,
            log_print,
            clear_src=spec.get("clear_src", False),
            **kwargs,
        )
        return NoPoseUpsamplerWrapper(model).to(device).eval()

    raise ValueError(f"Unsupported upsampler_name: {upsampler_name}")


def run_upsampler(model, upsampler_name, image_batch, patch_tokens, target_size, poses=None):
    if upsampler_name == "raysup":
        if poses is None:
            raise ValueError("RaysUp evaluation requires camera poses.")
        return model(image_batch, patch_tokens, poses, target_size)
    return model(image_batch, patch_tokens, target_size)


def make_fake_poses(image_batch):
    batch_size, _, height, width = image_batch.shape
    device = image_batch.device
    camera_poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    fxfycxcy = torch.tensor(
        [float(width), float(height), width / 2.0, height / 2.0],
        device=device,
    ).unsqueeze(0).repeat(batch_size, 1)
    return camera_poses, fxfycxcy


def estimate_poses(image_batch, pose_estimator, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_batch.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_batch.device).view(3, 1, 1)
    camera_poses_list = []
    intrinsics_list = []

    with torch.autocast(device_type="cuda", enabled=False):
        with torch.no_grad():
            for image in image_batch:
                image_np = (image * std + mean).permute(1, 2, 0).detach().cpu().numpy()
                image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
                prediction = pose_estimator.inference([image_np])

                extrinsic = prediction.extrinsics[0]
                intrinsic = prediction.intrinsics[0]
                camera_poses_list.append(torch.as_tensor(extrinsic, dtype=torch.float32))
                intrinsics_list.append(torch.as_tensor(intrinsic, dtype=torch.float32))

    camera_poses = torch.stack(camera_poses_list).to(device)
    intrinsics = torch.stack(intrinsics_list).to(device)
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


def load_da3_pose_estimator(cfg, device, log_print):
    da3_name = get_eval_option(cfg, "da3_name", None)
    if da3_name is None:
        da3_size = get_eval_option(cfg, "da3_size", "large")
        if da3_size not in DA3_SIZE_TO_NAME:
            raise ValueError(
                f"Unknown eval.da3_size='{da3_size}'. "
                f"Choose from {sorted(DA3_SIZE_TO_NAME)} or set eval.da3_name explicitly."
            )
        da3_name = DA3_SIZE_TO_NAME[da3_size]

    from depth_anything_3.api import DepthAnything3

    log_print(f"[bold yellow]Loading DA3 for pose estimation: {da3_name}[/bold yellow]")
    pose_estimator = DepthAnything3.from_pretrained(da3_name).to(device)
    pose_estimator.eval()
    log_print(f"[bold green]Loaded DA3 pose estimator: {da3_name}[/bold green]")
    return pose_estimator
