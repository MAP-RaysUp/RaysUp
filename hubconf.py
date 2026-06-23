dependencies = ["torch", "natten"]

import os

import torch

from src.upsampler.raysup import RaysUp


def raysup(pretrained: bool = True, checkpoint_path: str | None = None, device="cpu"):
    model = RaysUp().to(device)
    if pretrained:
        checkpoint_path = checkpoint_path or os.environ.get("RAYSUP_CKPT")
        if checkpoint_path is None:
            raise ValueError("Set checkpoint_path or RAYSUP_CKPT to load pretrained RaysUp weights.")

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = state_dict["raysup"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)
    return model
