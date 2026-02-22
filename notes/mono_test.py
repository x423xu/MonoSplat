#!/usr/bin/env python3
"""Streaming-style MonoSplat test script for RE10K.

At each time step, the evaluator consumes 4 input frames (1 second) and renders
15 output frames, then reports PSNR and timing for that step.
"""

from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from PIL import Image

from src.config import load_typed_root_config
from src.dataset.shims.crop_shim import apply_crop_shim
from src.evaluation.metrics import compute_psnr
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming MonoSplat evaluator")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/data0/xxy/code/MonoSplat/outputs/2026-02-10/13-21-17/checkpoints/epoch_9-step_300000.ckpt"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/data0/xxy/data/re10k"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num-context", type=int, default=4)
    parser.add_argument("--num-target", type=int, default=15)
    parser.add_argument("--step-stride", type=int, default=4, help="Frame stride between time steps")
    parser.add_argument("--max-time-steps", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=Path, default=Path("notes/mono_test_results.json"))
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> ModelWrapper:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+experiment=re10k",
                "mode=test",
                "wandb.mode=disabled",
                f"dataset.roots=[{args.dataset_root}]",
                f"dataset.image_shape=[{args.image_height},{args.image_width}]",
                "test.compute_scores=false",
                "test.save_image=false",
                "test.save_video=false",
            ],
        )
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model = ModelWrapper(
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=get_decoder(cfg.model.decoder, cfg.dataset),
        losses=get_losses(cfg.loss),
        step_tracker=None,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def convert_poses(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _ = poses.shape
    intrinsics = repeat(torch.eye(3, dtype=torch.float32), "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def convert_images(images: list[torch.Tensor]) -> torch.Tensor:
    to_tensor = tf.ToTensor()
    decoded = []
    for image in images:
        decoded.append(to_tensor(Image.open(BytesIO(image.numpy().tobytes()))))
    return torch.stack(decoded)


def build_stream_batch(
    scene_item: dict[str, Any], args: argparse.Namespace, start_idx: int
) -> dict[str, Any] | None:
    total_needed = args.num_context + args.num_target
    end_idx = start_idx + total_needed
    if len(scene_item["images"]) < end_idx:
        return None

    poses = scene_item["cameras"][start_idx:end_idx]
    extrinsics, intrinsics = convert_poses(poses)
    images = convert_images(scene_item["images"][start_idx:end_idx])

    context_idx = torch.arange(args.num_context)
    target_idx = torch.arange(args.num_context, total_needed)

    example = {
        "context": {
            "extrinsics": extrinsics[context_idx],
            "intrinsics": intrinsics[context_idx],
            "image": images[context_idx],
            "near": torch.full((args.num_context,), 0.5, dtype=torch.float32),
            "far": torch.full((args.num_context,), 100.0, dtype=torch.float32),
            "index": torch.arange(start_idx, start_idx + args.num_context),
        },
        "target": {
            "extrinsics": extrinsics[target_idx],
            "intrinsics": intrinsics[target_idx],
            "image": images[target_idx],
            "near": torch.full((args.num_target,), 0.5, dtype=torch.float32),
            "far": torch.full((args.num_target,), 100.0, dtype=torch.float32),
            "index": torch.arange(start_idx + args.num_context, end_idx),
        },
        "scene": scene_item["key"],
    }
    example = apply_crop_shim(example, (args.image_height, args.image_width))
    return {
        "context": {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in example["context"].items()},
        "target": {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in example["target"].items()},
        "scene": [example["scene"]],
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model = build_model(args)
    device = next(model.parameters()).device

    chunk_paths = sorted((args.dataset_root / args.split).glob("*.torch"))
    if not chunk_paths:
        raise FileNotFoundError(f"No .torch chunks found in {args.dataset_root / args.split}")

    results: list[dict[str, Any]] = []

    for chunk_path in chunk_paths:
        chunk = torch.load(chunk_path)
        for scene_item in chunk:
            max_start = len(scene_item["images"]) - (args.num_context + args.num_target)
            if max_start < 0:
                continue

            for start_idx in range(0, max_start + 1, args.step_stride):
                if len(results) >= args.max_time_steps:
                    break

                batch = build_stream_batch(scene_item, args, start_idx)
                if batch is None:
                    continue

                batch = model.data_shim(batch)
                batch["context"] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["context"].items()}
                batch["target"] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["target"].items()}
                _, _, _, h, w = batch["target"]["image"].shape

                start_t = time.perf_counter()
                gaussians = model.encoder(batch["context"], model.global_step, deterministic=False)
                output = model.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )
                elapsed = time.perf_counter() - start_t

                step_psnr = compute_psnr(batch["target"]["image"][0], output.color[0]).mean().item()
                step_fps = args.num_target / elapsed
                timestep = {
                    "scene": scene_item["key"],
                    "time_step": len(results),
                    "window_start_frame": start_idx,
                    "window_end_frame": start_idx + args.num_context + args.num_target - 1,
                    "psnr": step_psnr,
                    "time_cost_sec": elapsed,
                    "fps": step_fps,
                    "num_context": args.num_context,
                    "num_target": args.num_target,
                }
                results.append(timestep)
                print(
                    f"time_step={timestep['time_step']:04d} scene={timestep['scene']} "
                    f"start={start_idx:04d} psnr={step_psnr:.4f} time={elapsed:.4f}s fps={step_fps:.2f}"
                )
            if len(results) >= args.max_time_steps:
                break
        if len(results) >= args.max_time_steps:
            break

    if not results:
        raise RuntimeError("No valid streaming time steps evaluated.")

    avg_psnr = sum(r["psnr"] for r in results) / len(results)
    avg_time = sum(r["time_cost_sec"] for r in results) / len(results)
    avg_fps = sum(r["fps"] for r in results) / len(results)

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "evaluated_time_steps": len(results),
        "num_context": args.num_context,
        "num_target": args.num_target,
        "step_stride": args.step_stride,
        "avg_psnr": avg_psnr,
        "avg_time_cost_sec": avg_time,
        "avg_fps": avg_fps,
        "time_steps": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))

    print("\n=== Streaming MonoSplat Summary ===")
    print(f"Time steps evaluated: {len(results)}")
    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg time cost (4 in -> 15 out): {avg_time:.4f} sec")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
