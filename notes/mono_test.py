#!/usr/bin/env python3
"""Streaming-style MonoSplat test script for RE10K.

This script evaluates a *5-second streaming episode* per scene.

For each second (T1..T5), it samples:
    - `num_context` random input frames from that 1-second window (default: 4)
    - 15 target frames using the same 1+2+4+8 aggregation policy implemented in
        `src/dataset/re10k_dataset_stream.py`.
"""

from __future__ import annotations
import os, sys
sys.path.append("/data0/xxy/code/MonoSplat/")
sys.path.append("/data0/xxy/code/MonoSplat/src")
import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw, ImageFont

from src.config import load_typed_root_config
from src.dataset.shims.crop_shim import apply_crop_shim
from src.misc.image_io import save_image, save_video
from src.evaluation.metrics import compute_psnr
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.dataset.re10k_dataset_stream import build_re10k_stream_splits


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
    parser.add_argument("--num-seconds", type=int, default=5, help="Seconds per streaming episode")
    parser.add_argument("--fps", type=int, default=30, help="Assumed FPS per second window")
    parser.add_argument("--seed", type=int, default=0, help="Seed for context-frame sampling")
    parser.add_argument("--max-time-steps", type=int, default=200)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notes/mono_test_outputs"),
        help="Where to save context images and rendered videos",
    )
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


def overlay_text_top_left(
    image: torch.Tensor,
    text: str,
    *,
    pad: int = 6,
) -> torch.Tensor:
    """Overlay `text` in the top-left of an image tensor (3,H,W) in [0,1]."""
    image = image.detach().float().clip(0, 1)
    np_img = (rearrange(image, "c h w -> h w c").cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(np_img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("assets/Inter-Regular.otf", 18)
    except OSError:
        font = ImageFont.load_default()

    # Simple outline for readability.
    x, y = pad, pad
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    out = torch.tensor(np.array(pil), dtype=torch.float32) / 255.0
    return rearrange(out, "h w c -> c h w")


def build_stream_batch_from_indices(
    *,
    scene_item: dict[str, Any],
    args: argparse.Namespace,
    context_indices: np.ndarray,
    target_indices: np.ndarray,
) -> dict[str, Any] | None:
    if len(context_indices) != args.num_context:
        return None
    if len(target_indices) != args.num_target:
        return None
    if len(scene_item["images"]) <= int(max(context_indices.max(), target_indices.max())):
        return None

    context_indices_t = torch.as_tensor(context_indices, dtype=torch.long)
    target_indices_t = torch.as_tensor(target_indices, dtype=torch.long)

    all_indices_t = torch.cat([context_indices_t, target_indices_t], dim=0)
    poses = scene_item["cameras"][all_indices_t]
    extrinsics_all, intrinsics_all = convert_poses(poses)

    context_images = [scene_item["images"][i.item()] for i in context_indices_t]
    target_images = [scene_item["images"][i.item()] for i in target_indices_t]

    example = {
        "context": {
            "extrinsics": extrinsics_all[: args.num_context],
            "intrinsics": intrinsics_all[: args.num_context],
            "image": convert_images(context_images),
            "near": torch.full((args.num_context,), 0.5, dtype=torch.float32),
            "far": torch.full((args.num_context,), 100.0, dtype=torch.float32),
            "index": context_indices_t,
        },
        "target": {
            "extrinsics": extrinsics_all[args.num_context :],
            "intrinsics": intrinsics_all[args.num_context :],
            "image": convert_images(target_images),
            "near": torch.full((args.num_target,), 0.5, dtype=torch.float32),
            "far": torch.full((args.num_target,), 100.0, dtype=torch.float32),
            "index": target_indices_t,
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
    if args.num_target != 15:
        raise ValueError(
            "This streaming evaluator expects --num-target=15 to match the "
            "RE10K streaming policy in src/dataset/re10k_dataset_stream.py."
        )
    if args.num_seconds != 5:
        raise ValueError(
            "This evaluator currently supports exactly --num-seconds=5 (T1..T5)."
        )
    model = build_model(args)
    device = next(model.parameters()).device

    chunk_paths = sorted((args.dataset_root / args.split).glob("*.torch"))
    if not chunk_paths:
        raise FileNotFoundError(f"No .torch chunks found in {args.dataset_root / args.split}")

    results: list[dict[str, Any]] = []

    rng = np.random.default_rng(args.seed)

    for chunk_path in chunk_paths:
        chunk = torch.load(chunk_path)
        for scene_item in chunk:
            splits = build_re10k_stream_splits(
                num_frames=len(scene_item["images"]),
                input_num=args.num_context,
                num_seconds=args.num_seconds,
                fps=args.fps,
                rng=rng,
            )
            if splits is None:
                continue

            # Accumulate all rendered frames for this scene into one video.
            scene_video_frames: list[torch.Tensor] = []

            for sec_idx in range(args.num_seconds):
                if len(results) >= args.max_time_steps:
                    break

                context_indices = splits["input_indices_per_sec"][sec_idx]
                target_indices = splits["target_indices_per_step"][sec_idx]

                # Time from "input" (batch construction) through rendering.
                # Excludes PSNR computation and any disk I/O.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_t0 = time.perf_counter()

                batch = build_stream_batch_from_indices(
                    scene_item=scene_item,
                    args=args,
                    context_indices=context_indices,
                    target_indices=target_indices,
                )
                if batch is None:
                    continue

                # Keep a CPU copy for later saving (excluded from timing).
                context_imgs_cpu = batch["context"]["image"][0].detach().cpu()  # [V,3,H,W]

                batch = model.data_shim(batch)
                batch["context"] = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch["context"].items()
                }
                batch["target"] = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch["target"].items()
                }
                _, _, _, h, w = batch["target"]["image"].shape

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

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - step_t0

                # PSNR computation (excluded from timing).
                per_frame_psnr_t = compute_psnr(batch["target"]["image"][0], output.color[0])
                per_frame_psnr = [float(x) for x in per_frame_psnr_t.detach().cpu().tolist()]
                step_psnr = float(per_frame_psnr_t.mean().item())
                step_fps = args.num_target / elapsed
                all_idx = np.concatenate([context_indices, target_indices])

                # Add PSNR overlay and accumulate into a per-scene video.
                rendered_frames = output.color[0].detach().cpu()  # [15,3,H,W]
                for frame_idx in range(rendered_frames.shape[0]):
                    annotated = overlay_text_top_left(
                        rendered_frames[frame_idx],
                        f"PSNR {per_frame_psnr[frame_idx]:.2f}",
                    )
                    scene_video_frames.append(annotated)

                # Save the (cropped) context input images used by the model.
                step_out_dir = args.output_dir / scene_item["key"] / f"timestep_{sec_idx + 1:02d}"
                for view_idx in range(context_imgs_cpu.shape[0]):
                    save_image(
                        context_imgs_cpu[view_idx],
                        step_out_dir
                        / f"context_{view_idx:02d}_frame_{int(context_indices[view_idx]):04d}.png",
                    )

                timestep = {
                    "scene": scene_item["key"],
                    "time_step": len(results),
                    "stream_second": sec_idx,
                    "window_start_frame": int(all_idx.min()),
                    "window_end_frame": int(all_idx.max()),
                    "context_indices": [int(x) for x in context_indices.tolist()],
                    "target_indices": [int(x) for x in target_indices.tolist()],
                    "per_frame_psnr": per_frame_psnr,
                    "per_target_frame": [
                        {"frame": int(target_indices[i]), "psnr": float(per_frame_psnr[i])}
                        for i in range(len(per_frame_psnr))
                    ],
                    "psnr": step_psnr,
                    "time_cost_sec": elapsed,
                    "fps": step_fps,
                    "num_context": args.num_context,
                    "num_target": args.num_target,
                }
                results.append(timestep)
                print(
                    f"time_step={timestep['time_step']:04d} scene={timestep['scene']} "
                    f"sec={sec_idx + 1} psnr={step_psnr:.4f} time={elapsed:.4f}s fps={step_fps:.2f} "
                    f"frames=[{timestep['window_start_frame']:04d},{timestep['window_end_frame']:04d}]"
                )

            reached_limit = len(results) >= args.max_time_steps

            # Save one video per scene (all evaluated seconds concatenated).
            if scene_video_frames:
                scene_video_path = args.output_dir / scene_item["key"] / "rendered_psnr.mp4"
                save_video(scene_video_frames, scene_video_path)

            if reached_limit:
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
        "num_seconds": args.num_seconds,
        "fps": args.fps,
        "seed": args.seed,
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
