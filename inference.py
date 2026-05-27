import os
import argparse
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.distributed as dist
from huggingface_hub import snapshot_download

import trimesh
from diffsynth import save_video
from training.wan_video_action_images import WanVideoActionImagesPipeline
from training.models import ModelConfig
from training.helpers.io import concat_video, combine_action_video, load_images_to_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from training.utils import fuse_multiview_heatmaps_to_7d_point_torch, intrinsics_transform


def resolve_ckpt_path(ckpt_path: str) -> str:
    if os.path.isfile(ckpt_path):
        return ckpt_path

    local_dir = os.path.join("checkpoints", ckpt_path.replace("/", "--"))

    if not dist.is_initialized() or dist.get_rank() == 0:
        snapshot_download(repo_id=ckpt_path, local_dir=local_dir)
    if dist.is_initialized():
        dist.barrier()

    for name in os.listdir(local_dir):
        if name.endswith(".ckpt"):
            return os.path.join(local_dir, name)

    raise FileNotFoundError(f"No .ckpt in {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Two-image inference with VGGT and WanVideoActionImagesPipeline")
    parser.add_argument("--images", type=str, nargs=2, required=True, help="Paths to two input images")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B", help="Model identifier")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--output", type=str, default="results/inference", help="Output directory")
    parser.add_argument("--visualize_actions", action="store_true")
    parser.add_argument("--height", type=int, default=512, help="Output video height")
    parser.add_argument("--width", type=int, default=512, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=41, help="Number of frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--cfg_parallel", action="store_true", help="Split CFG positive/negative branches across GPUs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate with different seeds")
    parser.add_argument("--task_type", type=str, default="i2v", help="Task type")
    parser.add_argument(
        "--view1_action",
        type=float,
        nargs="+",
        default=None,
        help="Mapped action for view1. Provide 7 values or 7*num_frames values.",
    )
    parser.add_argument(
        "--view2_action",
        type=float,
        nargs="+",
        default=None,
        help="Mapped action for view2. Provide 7 values or 7*num_frames values.",
    )
    parser.add_argument("--use_usp", action="store_true", help="Use Unified Sequence Parallel")
    parser.add_argument("--dynamic_cache_schedule", action="store_true", help="Enable should_run_model cache schedule")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for inference models")
    return parser.parse_args()


def setup_device_dtype():
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "cuda" in device and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return device, dtype


def load_vggt(device: str):
    model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir="./checkpoints").to(device)
    return model


def estimate_cameras(model: VGGT, images, device: str, dtype: torch.dtype, output_dir: str):
    # resize images to 518x518
    images = torch.nn.functional.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    if images.dim() != 4:
        raise ValueError("Expected VGGT images tensor of shape [V, C, H, W]")

    with torch.no_grad():
        images = images[None].to(device) / 255.0
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )
        trimesh.PointCloud(
            point_map_by_unprojection.reshape(-1, 3),
            colors=images.squeeze(0).permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy(),
        ).export(os.path.join(output_dir, "point_map_by_unprojection.ply"))

    extrinsic = extrinsic.squeeze(0).to(torch.float32)
    intrinsic = intrinsic.squeeze(0).to(torch.float32)
    return extrinsic, intrinsic, point_map_by_unprojection


def build_pipeline(args):
    model_configs = [
        ModelConfig(
            local_model_path="checkpoints",
            model_id=args.model_id,
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
            skip_download=True,
        ),
        ModelConfig(
            local_model_path="checkpoints",
            model_id=args.model_id,
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16*.pth",
            offload_device="cpu",
            skip_download=True,
        ),
        ModelConfig(
            local_model_path="checkpoints",
            model_id=args.model_id,
            origin_file_pattern="Wan*_VAE.pth",
            offload_device="cpu",
            skip_download=True,
        ),
    ]
    tokenizer_config = ModelConfig(
        local_model_path="checkpoints",
        model_id=args.model_id,
        origin_file_pattern="google/*",
        skip_download=True,
    )
    pipe = WanVideoActionImagesPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
        use_usp=args.use_usp,
        cfg_parallel=args.cfg_parallel,
        dynamic_cache_schedule=args.dynamic_cache_schedule,
        torch_compile=args.torch_compile,
    )

    # 2. Initialize additional modules
    # Match training cam encoder definition (AdaptiveAvgPool3d->Flatten->Linear + projector as identity)
    H_l = args.height // pipe.vae.upsampling_factor
    W_l = args.width // pipe.vae.upsampling_factor
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((48, 16, 16)),
            nn.Flatten(start_dim=2),
            nn.Linear(32 * 32 * 12, dim),
        )
        block.projector = nn.Linear(dim, dim)
        for module in block.cam_encoder:
            if isinstance(module, nn.Linear):
                module.weight = nn.Parameter(torch.zeros_like(module.weight))
                module.bias = nn.Parameter(torch.zeros_like(module.bias))
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    if args.ckpt_path is not None:
        state_dict = torch.load(resolve_ckpt_path(args.ckpt_path), map_location="cpu")
        pipe.dit.load_state_dict(state_dict, strict=True)

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
        device = f"cuda:{local_rank}"
    else:
        device = "cuda"
    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    pipe.eval()
    torch.set_grad_enabled(False)
    return pipe


def prepare_inputs(
    args,
    images: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    video_segments: Optional[torch.Tensor] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
):
    images = torch.nn.functional.interpolate(
        images, size=(args.height, args.width), mode="bilinear", align_corners=False
    )
    images = images / 255.0 * 2 - 1

    if video_segments is not None:
        video = torch.nn.functional.interpolate(
            video_segments.reshape(-1, 3, video_segments.shape[-2], video_segments.shape[-1]),
            size=(args.height, args.width),
            mode="bilinear",
            align_corners=False,
        ).reshape(2, args.num_frames, 3, args.height, args.width)
        video = video / 255.0 * 2 - 1
        video = torch.cat([video[0], video[1]], dim=0).unsqueeze(0)
    else:
        # make a fake video of 1, num_frames * 2, 3, H, W
        video = torch.zeros((1, args.num_frames * 2, 3, args.height, args.width), dtype=torch.float32)
        video[:, 0, ...] = images[0]
        video[:, args.num_frames, ...] = images[1]
    video = video.permute(0, 2, 1, 3, 4)

    action_7d = torch.zeros((1, args.num_frames, 7), dtype=torch.float32) + 100.0
    action_7d[:, :, -1] = 1
    action_5d = None
    if args.view1_action is not None or args.view2_action is not None:
        if args.view1_action is None or args.view2_action is None:
            raise ValueError("Both --view1_action and --view2_action must be provided together.")

        def _parse_view_action(action_list: List[float], name: str) -> torch.Tensor:
            values = torch.tensor(action_list, dtype=torch.float32)
            if values.numel() == 7:
                values = values.unsqueeze(0).repeat(args.num_frames, 1)
            elif values.numel() == args.num_frames * 7:
                values = values.reshape(args.num_frames, 7)
            else:
                raise ValueError(f"{name} expects 7 values or {args.num_frames * 7} values, got {values.numel()}.")
            return values

        view1_action = _parse_view_action(args.view1_action, "--view1_action")
        view2_action = _parse_view_action(args.view2_action, "--view2_action")
        action_5d = torch.cat([view1_action, view2_action], dim=0).unsqueeze(0)

    intr_cond = torch.zeros((1, args.num_frames * 2, 3, 3), dtype=torch.float32)  # [1, 2T, 3, 3]
    intrinsics = intrinsics_transform(intrinsics, (518, 518), (args.height, args.width))  # [2, 3, 3]
    intrinsics[..., 0, 0] *= -1
    intrinsics[..., 1, 1] *= -1
    intr_cond[:, : args.num_frames, ...] = intrinsics[0]
    intr_cond[:, args.num_frames :, ...] = intrinsics[1]

    # Invert extrinsics: [R|t] -> [R^T | -R^T @ t]
    # extrinsics shape: [2, 3, 4]
    R = extrinsics[..., :3, :3]  # [2, 3, 3]
    t = extrinsics[..., :3, 3:4]  # [2, 3, 1]
    R_inv = R.transpose(-1, -2)  # [2, 3, 3]
    t_inv = -torch.matmul(R_inv, t)  # [2, 3, 1]
    extrinsics_inv = torch.cat([R_inv, t_inv], dim=-1)  # [2, 3, 4]
    # flip y and z
    matrix = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    matrix[0, 0] = -1
    matrix[1, 1] = -1
    extrinsics_inv = extrinsics_inv @ matrix
    extr_cond = torch.zeros((1, args.num_frames * 2, 3, 4), dtype=torch.float32)  # [1, 2T, 3, 4]
    extr_cond[:, : args.num_frames, ...] = extrinsics_inv[0]
    extr_cond[:, args.num_frames :, ...] = extrinsics_inv[1]
    camera = extr_cond.reshape(1, -1, 12)
    # extr_cond is homogeneous. pad [0, 0, 0, 1] to the last column
    pad = torch.zeros((1, args.num_frames * 2, 1, 4), dtype=torch.float32)  # [1, 2T, 1, 4]
    extr_cond = torch.cat([extr_cond, pad], dim=2)
    extr_cond[:, :, 3, 3] = 1

    if device is not None and dtype is not None:
        video = video.to(device=device, dtype=dtype)
        camera = camera.to(device=device, dtype=dtype)
        action_7d = action_7d.to(device=device, dtype=dtype)
        if action_5d is not None:
            action_5d = action_5d.to(device=device, dtype=dtype)
        extr_cond = extr_cond.to(device=device, dtype=dtype)
        intr_cond = intr_cond.to(device=device, dtype=dtype)

    return video, camera, action_7d, action_5d, extr_cond, intr_cond


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def export_action_point_cloud_from_pred_video(
    pred_video: List[Image.Image],
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    output_dir: str,
    base_text: str,
    sample_idx: int,
    *,
    near: float = 0.5,
    far: float = 1.0,
    num_depth_samples: int = 512,
    apply_edge_smoothing: bool = False,
):
    T = len(pred_video)
    raw_video = torch.stack([torch.from_numpy(np.array(frame)) for frame in pred_video], dim=0)
    heatmaps_1 = raw_video[T // 4 : T // 2]
    heatmaps_2 = raw_video[3 * T // 4 :]
    heatmaps = torch.stack([heatmaps_1, heatmaps_2], dim=1).to(torch.float32)
    extr_reshaped = extrinsics[..., :3, :].reshape(2, -1, 3, 4).permute(1, 0, 2, 3)
    intr_reshaped = intrinsics.reshape(2, -1, 3, 3).permute(1, 0, 2, 3)

    action_points = fuse_multiview_heatmaps_to_7d_point_torch(
        heatmaps,
        extr_reshaped,
        intr_reshaped,
        near=near,
        far=far,
        num_depth_samples=num_depth_samples,
        apply_edge_smoothing=apply_edge_smoothing,
    )

    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, T // 4))
    colors = colors[:, :3]
    mesh = trimesh.PointCloud(action_points[:, :3], colors=colors)
    mesh.export(os.path.join(output_dir, f"point_cloud_{base_text}_{sample_idx}.ply"))


def main():
    args = parse_args()
    pipe = build_pipeline(args)

    device, dtype = setup_device_dtype()
    set_random_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    images, video_segments = load_images_to_square(args.images, args.task_type, args.num_frames)  # [V, C, H, W]
    vggt_model = load_vggt(device).to(device)
    extrinsics, intrinsics, point_map_by_unprojection = estimate_cameras(vggt_model, images, device, dtype, args.output)
    video, camera, action_7d, action_5d, extrinsics, intrinsics = prepare_inputs(
        args,
        images,
        extrinsics,
        intrinsics,
        video_segments=video_segments,
        device=pipe.device,
        dtype=torch.bfloat16,
    )
    base_text = args.prompt.replace(" ", "_") if args.prompt else "noprompt"

    for sample_idx in range(args.num_samples):
        if args.torch_compile:
            torch.compiler.cudagraph_mark_step_begin()
        if dist.is_initialized():
            dist.barrier()
        current_seed = args.seed + sample_idx
        set_random_seed(current_seed)
        pred_video = pipe(
            prompt=[args.prompt],
            negative_prompt="",
            task_type=args.task_type,
            video=video,
            camera=camera,
            action_7d=action_7d,
            action_5d=action_5d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=current_seed,
            tiled=False,
            tile_size=(args.height // 16, args.width // 16),
            tile_stride=(args.height // 32, args.width // 32),
            enable_usp=args.use_usp,
            cfg_parallel=args.cfg_parallel,
        )
        if dist.is_initialized() and dist.get_rank() != 0:
            continue

        if args.task_type == "i2v":
            video_frames = concat_video(pred_video, split_num=2)
        else:
            video_frames = combine_action_video(pred_video)
            T = video_frames.shape[0]
            video_view_1 = video_frames[: T // 2]
            video_view_2 = video_frames[T // 2 :]
            video_frames = np.concatenate([video_view_1, video_view_2], axis=2)

        if args.visualize_actions:
            export_action_point_cloud_from_pred_video(
                pred_video=pred_video,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                output_dir=args.output,
                base_text=base_text,
                sample_idx=sample_idx,
            )

        video_frames = [Image.fromarray(frame) for frame in video_frames]
        output_path = os.path.join(args.output, f"video_{base_text}_{sample_idx}.mp4")
        save_video(video_frames, output_path, fps=30, quality=8)
        print(f"Saved video to {output_path}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
