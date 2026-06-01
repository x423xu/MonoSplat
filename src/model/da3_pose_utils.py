from __future__ import annotations

import torch
from torch import Tensor

from ..dataset.types import BatchedExample


def da3_pixel_intrinsics_to_normalized(
    intrinsics: Tensor,
    height: int,
    width: int,
) -> Tensor:
    normalized = intrinsics.clone()
    normalized[..., 0, 0] = normalized[..., 0, 0] / width
    normalized[..., 1, 1] = normalized[..., 1, 1] / height
    normalized[..., 0, 2] = normalized[..., 0, 2] / width
    normalized[..., 1, 2] = normalized[..., 1, 2] / height
    return normalized


def da3_w2c_to_c2w(
    extrinsics: Tensor,
    height: int | None = None,
    width: int | None = None,
) -> Tensor:
    if extrinsics.shape[-2:] == (3, 4):
        bottom = torch.zeros(
            *extrinsics.shape[:-2],
            1,
            4,
            dtype=extrinsics.dtype,
            device=extrinsics.device,
        )
        bottom[..., 0, 3] = 1
        extrinsics = torch.cat([extrinsics, bottom], dim=-2)
    elif extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            "Expected DA3 extrinsics with shape (..., 3, 4) or (..., 4, 4), "
            f"got {tuple(extrinsics.shape)}"
        )

    return torch.linalg.inv(extrinsics)


def replace_batch_poses_from_da3(
    batch: BatchedExample,
    da3_w2c: Tensor,
    da3_intrinsics: Tensor,
    height: int,
    width: int,
) -> BatchedExample:
    num_context = batch["context"]["image"].shape[1]
    num_target = batch["target"]["image"].shape[1]
    expected_views = num_context + num_target
    if da3_w2c.shape[1] != expected_views:
        raise ValueError(f"DA3 returned {da3_w2c.shape[1]} views, expected {expected_views}")
    if da3_intrinsics.shape[1] != expected_views:
        raise ValueError(
            f"DA3 returned {da3_intrinsics.shape[1]} intrinsics, expected {expected_views}"
        )

    c2w = da3_w2c_to_c2w(da3_w2c, height=height, width=width)
    intrinsics = da3_pixel_intrinsics_to_normalized(
        da3_intrinsics,
        height=height,
        width=width,
    )

    context = {
        **batch["context"],
        "extrinsics": c2w[:, :num_context].to(batch["context"]["extrinsics"].dtype),
        "intrinsics": intrinsics[:, :num_context].to(
            batch["context"]["intrinsics"].dtype
        ),
    }
    target = {
        **batch["target"],
        "extrinsics": c2w[:, num_context : num_context + num_target].to(
            batch["target"]["extrinsics"].dtype
        ),
        "intrinsics": intrinsics[:, num_context : num_context + num_target].to(
            batch["target"]["intrinsics"].dtype
        ),
    }
    return {**batch, "context": context, "target": target}