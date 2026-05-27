import sys
import types
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from training.models.wan_video_dit import WanModel
from training.models.model_manager import ModelManager

from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.pipelines.base import BasePipeline
from diffsynth.prompters import WanPrompter
from diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from diffsynth.models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from diffsynth.models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from training.models import ModelConfig
from training.utils import (
    project_actions_7d_to_5d_torch_batch,
    project_action_5d_to_rgb_torch,
    get_plucker_embeddings_torch,
)


def _encode_video_bv(pipe, video_bvcthw: torch.Tensor, tiler_kwargs: dict) -> torch.Tensor:
    """Encode [B, V, C, T, H, W] by flattening B*V; returns [B, V, C_l, T_l, H_l, W_l]."""
    b, v, c, t, h, w = video_bvcthw.shape
    flat = video_bvcthw.reshape(b * v, c, t, h, w)
    lat = pipe.encode_video(flat, **tiler_kwargs)
    return lat.reshape(b, v, lat.shape[1], lat.shape[2], lat.shape[3], lat.shape[4])


def _pack_latents_bv(latents_bvct: torch.Tensor) -> torch.Tensor:
    """Pack [B, V, C, T_l, H_l, W_l] -> [B, C, V*T_l, H_l, W_l] (view-major)."""
    b, v, c, t, h, w = latents_bvct.shape
    return latents_bvct.permute(0, 2, 1, 3, 4, 5).reshape(b, c, v * t, h, w).contiguous()


def _pack_cam_emb_bv(cam_bvtchw: torch.Tensor) -> torch.Tensor:
    """Pack [B, V, T_l, C, H_l, W_l] -> [B, V*T_l, C, H_l, W_l] (view-major)."""
    b, v, t, c, h, w = cam_bvtchw.shape
    return cam_bvtchw.reshape(b, v * t, c, h, w).contiguous()


class WanVideoActionImagesPipeline(BasePipeline):

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.float16,
        tokenizer_path=None,
        dynamic_cache_schedule=False,
        torch_compile=False,
    ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.dynamic_cache_schedule = dynamic_cache_schedule
        self.use_cfg_parallel = False

    def initialize_parallel(self, use_usp: bool = False, cfg_parallel: bool = False):
        if not (use_usp or cfg_parallel):
            return

        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

        if not dist.is_initialized():
            missing_env = [name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ]
            if missing_env:
                raise ValueError(
                    "Distributed inference requires launching with torchrun. "
                    f"Missing environment variables: {', '.join(missing_env)}"
                )
            dist.init_process_group(backend="nccl", init_method="env://")

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        init_distributed_environment(rank=rank, world_size=world_size, local_rank=local_rank)

        if cfg_parallel:
            if use_usp:
                if world_size % 2 != 0:
                    raise ValueError("--cfg_parallel with --use_usp requires an even number of processes/GPUs.")
                sp_degree = world_size // 2
            else:
                if world_size != 2:
                    raise ValueError("--cfg_parallel without --use_usp requires exactly 2 processes/GPUs.")
                sp_degree = 1

            initialize_model_parallel(
                classifier_free_guidance_degree=2,
                sequence_parallel_degree=sp_degree,
                ring_degree=1,
                ulysses_degree=sp_degree,
            )
            self.use_cfg_parallel = True
            if rank == 0:
                print(f"CFG Parallelism initialized: cfg_degree=2, " f"sequence_parallel_degree={sp_degree}")
        else:
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=1,
                ulysses_degree=world_size,
            )

        torch.cuda.set_device(local_rank)

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype=None,
        device=None,
        dynamic_cache_schedule=False,
        torch_compile=False,
    ):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoActionImagesPipeline(
            device=device,
            torch_dtype=torch_dtype,
            dynamic_cache_schedule=dynamic_cache_schedule,
            torch_compile=torch_compile,
        )
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(
            input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return latents

    def _broadcast_tensor(self, tensor, src: int = 0):
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor

        metadata = [None]
        if dist.get_rank() == src:
            tensor = tensor.to(self.device)
            metadata = [(tuple(tensor.shape), tensor.dtype)]
        dist.broadcast_object_list(metadata, src=src)

        shape, dtype = metadata[0]
        if dist.get_rank() != src:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.broadcast(tensor, src=src)
        return tensor

    def encode_videos_distributed(
        self,
        inputs: dict,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        distributed=True,
    ):
        """
        Encode multiple videos/images latents across ranks.

        - If torch.distributed is not initialized (or world_size == 1), falls back to local sequential encoding.
        - If distributed, assigns each input to a source rank (idx % world_size). **Phase 1:** each rank encodes only
          its assigned keys with no collective in between, so up to N GPUs can overlap in wall time. **Phase 2:**
          ordered broadcast so every rank gets the full dict (same tensors as before).
        - If N < number_of_inputs, some ranks run multiple encodes **sequentially on that GPU**; different ranks
          still overlap (e.g. N=4 and 8 inputs: each rank does 2 encodes, four-way overlap for the first wave).
        """
        import torch.distributed as dist

        # Keep stable ordering across ranks for deterministic src assignment.
        items = list(inputs.items())

        if not distributed or not dist.is_initialized() or dist.get_world_size() == 1:
            return {
                k: self.encode_video(v, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) for k, v in items
            }

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Phase 1: parallel encode (no broadcast between items — avoids serializing GPUs).
        local = {}
        for idx, (k, v) in enumerate(items):
            if idx % world_size == rank:
                local[k] = self.encode_video(v, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # Phase 2: each tensor has a single owner rank; broadcast in fixed order for deterministic collectives.
        outputs = {}
        for idx, (k, _v) in enumerate(items):
            src = idx % world_size
            latents = local[k] if rank == src else None
            outputs[k] = self._broadcast_tensor(latents, src=src)
        return outputs

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def decode_videos_distributed(
        self,
        inputs: dict,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        """
        Decode multiple latent segments across ranks (mirrors `encode_videos_distributed`).

        Phase 1: each rank decodes only its assigned keys (idx % world_size) with no collective in between.
        Phase 2: ordered broadcast so every rank gets the full dict of decoded frames.
        """
        import torch.distributed as dist

        items = list(inputs.items())

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return {
                k: self.decode_video(v, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) for k, v in items
            }

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        local = {}
        for idx, (k, v) in enumerate(items):
            if idx % world_size == rank:
                local[k] = self.decode_video(v, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        dist.barrier()

        outputs = {}
        for idx, (k, _v) in enumerate(items):
            src = idx % world_size
            frames = local[k] if rank == src else None
            outputs[k] = self._broadcast_tensor(frames, src=src)
        return outputs

    def should_run_model(self, index, current_timestep, prev_predictions):
        if not self.dynamic_cache_schedule:
            return True

        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        if self.skip_countdown == 1:
            self.skip_countdown = 0
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]
        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def prepare_action_images_inference_latents(
        self,
        video: torch.Tensor,
        camera: torch.Tensor,
        action_7d: torch.Tensor,
        action_5d: Optional[torch.Tensor],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        task_type: Optional[str],
        tiler_kwargs: dict,
        seed,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, List[Tuple[int, int]]]:
        """Encode ActionImages inputs and build noise, latent canvas, masks, camera emb, and segment spans for inference.

        Supports arbitrary num_views: video is [B, 3, V*T, H, W] with views concatenated along time
        (same layout as CombDataset).
        """
        # Infer view count from concatenated time axis
        B, _, N, H, W = video.shape  # [B, 3, V*T, H, W]
        T = action_7d.shape[1]  # [B, T, 7]
        if N % T != 0:
            raise ValueError(
                f"video length N={N} must be divisible by action_7d length T={T} "
                f"(expected layout [B, 3, num_views*T, H, W])."
            )
        num_views = N // T  # V

        # Split multi-view video into per-view tensors
        video_bvcthw = (
            video.reshape(B, -1, num_views, T, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()
        )  # [B, V, 3, T, H, W]

        # Build per-view Plucker camera embeddings and reshape to [B, V, 3, T, H, W]
        extrinsics_3x4 = camera.reshape(B, -1, 3, 4)  # [B, V*T, 3, 4]
        plucker_emb = get_plucker_embeddings_torch(extrinsics_3x4, intrinsics, (H, W))  # [B, V*T, H, W, 6]
        plucker_emb = plucker_emb.permute(0, 4, 1, 2, 3).contiguous()  # [B, 6, V*T, H, W]
        direction, moment = plucker_emb[:, :3], plucker_emb[:, 3:]  # [B, 3, V*T, H, W] each
        moment = moment / 3.0
        direction_bvcthw = (
            direction.reshape(B, 3, num_views, T, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()
        )  # [B, V, 3, T, H, W]
        moment_bvcthw = (
            moment.reshape(B, 3, num_views, T, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()
        )  # [B, V, 3, T, H, W]

        # Encode video and camera latents
        video_lat_bv = _encode_video_bv(self, video_bvcthw, tiler_kwargs)  # [B, V, C, T_l, H_l, W_l]
        T_l = int(video_lat_bv.shape[3])
        C = int(video_lat_bv.shape[2])
        H_l = int(video_lat_bv.shape[4])
        W_l = int(video_lat_bv.shape[5])

        dir_lat = _encode_video_bv(self, direction_bvcthw, tiler_kwargs)  # [B, V, C, T_l, H_l, W_l]
        mom_lat = _encode_video_bv(self, moment_bvcthw, tiler_kwargs)  # [B, V, C, T_l, H_l, W_l]
        cam_lat_bv = torch.cat([dir_lat, mom_lat], dim=2)  # [B, V, 2*C, T_l, H_l, W_l]
        cam_lat_bvt = cam_lat_bv.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, V, T_l, 2*C, H_l, W_l]

        no_action_input = (action_7d is None or torch.sum(action_7d**2) == 0) or task_type == "i2v"

        if no_action_input:
            # Video-only path: one latent segment per view
            latents_input = _pack_latents_bv(video_lat_bv)  # [B, C, V*T_l, H_l, W_l]
            latent_t = num_views * T_l
            seg_num = num_views
            seg_latent_spans = [(v * T_l, (v + 1) * T_l) for v in range(num_views)]
            cam_emb = _pack_cam_emb_bv(cam_lat_bvt)  # [B, V*T_l, 2*C, H_l, W_l]
            noise = self.generate_noise(
                (B, C, latent_t, H_l, W_l),
                seed=seed,
                device=self.device,
            )  # [B, C, V*T_l, H_l, W_l]
            masks = torch.zeros_like(latents_input, dtype=torch.bool)  # [B, C, V*T_l, H_l, W_l]
            for v in range(num_views):
                masks[:, :, v * T_l, ...] = 1
        else:
            # Action path: interleave video + action latents per view
            if action_5d is not None:
                if action_5d.dim() != 3 or action_5d.shape[0] != B or action_5d.shape[-1] != 7:
                    raise ValueError(f"action_5d must have shape [B, N, 7], got {tuple(action_5d.shape)} for B={B}.")
                if action_5d.shape[1] == num_views:
                    action_5d = action_5d[:, :, None, :].repeat(1, 1, T, 1).reshape(B, num_views * T, 7)  # [B, V*T, 7]
                elif action_5d.shape[1] != num_views * T:
                    raise ValueError(
                        f"action_5d length must be num_views ({num_views}) or num_views*T ({num_views * T}), "
                        f"got {action_5d.shape[1]}."
                    )
                action_5d = action_5d.to(device=video.device, dtype=video.dtype)
            else:
                action_7d_rep = action_7d.repeat(1, num_views, 1)  # [B, V*T, 7]
                action_5d = project_actions_7d_to_5d_torch_batch(action_7d_rep, extrinsics, intrinsics)  # [B, V*T, 7]
            action_video = project_action_5d_to_rgb_torch(action_5d, H, W)  # [B, V*T, H, W, 3]
            action_video = action_video.permute(0, 4, 1, 2, 3).contiguous()  # [B, 3, V*T, H, W]
            action_video = action_video * 2 - 1
            action_bvcthw = (
                action_video.reshape(B, 3, num_views, T, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()
            )  # [B, V, 3, T, H, W]
            action_lat_bv = _encode_video_bv(self, action_bvcthw, tiler_kwargs)  # [B, V, C, T_l, H_l, W_l]

            lat_stack = torch.stack([video_lat_bv, action_lat_bv], dim=2)  # [B, V, 2, C, T_l, H_l, W_l]
            latents_input = lat_stack.permute(0, 3, 1, 2, 4, 5, 6).reshape(B, C, num_views * 2 * T_l, H_l, W_l)
            cam_stack = torch.stack([cam_lat_bvt, cam_lat_bvt], dim=2)  # [B, V, 2, T_l, 2*C, H_l, W_l]
            cam_emb = cam_stack.reshape(
                B, num_views * 2 * T_l, cam_lat_bvt.shape[3], cam_lat_bvt.shape[4], cam_lat_bvt.shape[5]
            )
            latent_t = num_views * 2 * T_l
            seg_num = num_views * 2
            seg_latent_spans = [(i * T_l, (i + 1) * T_l) for i in range(num_views * 2)]
            noise = self.generate_noise(
                (B, C, latent_t, H_l, W_l),
                seed=seed,
                device=self.device,
            )  # [B, C, 2*V*T_l, H_l, W_l]
            masks = torch.zeros_like(latents_input, dtype=torch.bool)  # [B, C, 2*V*T_l, H_l, W_l]
            for block in range(num_views * 2):
                masks[:, :, block * T_l, ...] = 1
            # Refine conditioning mask by task type (video vs action segments)
            if task_type == "i2va":
                pass
            elif task_type == "v2a":
                for v in range(num_views):
                    masks[:, :, (v * 2) * T_l : (v * 2 + 1) * T_l, ...] = 1
            elif task_type == "a2v":
                for v in range(num_views):
                    masks[:, :, (v * 2 + 1) * T_l : (v * 2 + 2) * T_l, ...] = 1
            else:
                raise ValueError(f"Invalid task type: {task_type}")

        # Fill unmasked latent slots with noise
        latents_input = latents_input.to(device=self.device, dtype=self.torch_dtype)
        noise = noise.to(device=self.device, dtype=self.torch_dtype)
        cam_emb = cam_emb.to(device=self.device, dtype=self.torch_dtype)
        masks = masks.to(device=self.device)
        latents_input[~masks] = noise[~masks]
        return noise, latents_input, masks, cam_emb, seg_num, seg_latent_spans

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        # New ActionImages inputs following training forward()
        video: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
        action_7d: Optional[torch.Tensor] = None,
        action_5d: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        task_type: Optional[str] = "i2va",
        # Legacy/optional inputs
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        height=512,
        width=512,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(32, 32),
        tile_stride=(16, 16),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        # Unified Sequence Parallel
        enable_usp=False,
        cfg_parallel=False,
        **kwargs,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Follow training forward(): build ActionImages latents and camera embeddings
        assert (
            video is not None
            and camera is not None
            and extrinsics is not None
            and intrinsics is not None
            and (action_7d is not None or action_5d is not None)
        ), "video, camera, extrinsics, intrinsics and one of action_7d/action_5d are required"

        # Prepare action video from 7D actions
        self.load_models_to_device(["vae"])
        noise, latents_input, masks, cam_emb, seg_num, seg_latent_spans = self.prepare_action_images_inference_latents(
            video,
            camera,
            action_7d,
            action_5d,
            extrinsics,
            intrinsics,
            task_type,
            tiler_kwargs,
            seed,
        )

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        use_cfg_parallel = cfg_parallel and cfg_scale != 1.0
        cfg_group = None
        cfg_rank = None
        prompt_emb_local = None
        prompt_emb_posi = None
        prompt_emb_nega = None
        if use_cfg_parallel:
            from xfuser.core.distributed import (
                get_cfg_group,
                get_classifier_free_guidance_rank,
                get_classifier_free_guidance_world_size,
            )

            if get_classifier_free_guidance_world_size() != 2:
                raise ValueError("CFG parallelism expects classifier_free_guidance_degree == 2.")

            cfg_group = get_cfg_group()
            cfg_rank = get_classifier_free_guidance_rank()
            if cfg_rank == 0:
                prompt_emb_local = self.encode_prompt(negative_prompt, positive=False)
            elif cfg_rank == 1:
                prompt_emb_local = self.encode_prompt(prompt, positive=True)
            else:
                raise ValueError(f"Invalid CFG rank: {cfg_rank}")
        else:
            prompt_emb_posi = self.encode_prompt(prompt, positive=True)
            if cfg_scale != 1.0:
                prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        image_emb = {}
        extra_input = self.prepare_extra_input(latents_input)

        # Denoise
        self.load_models_to_device(["dit"])
        prev_predictions = []
        self.skip_countdown = 0
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            should_run_model = self.should_run_model(progress_id, timestep, prev_predictions)
            if should_run_model:
                dit_kwargs = dict(
                    timestep=timestep,
                    cam_emb=cam_emb,
                    **image_emb,
                    **extra_input,
                    enable_usp=enable_usp,
                )

                def run_dit(prompt_emb):
                    return model_fn_wan_video(self.dit, latents_input, **prompt_emb, **dit_kwargs)

                if use_cfg_parallel:
                    noise_pred_local = run_dit(prompt_emb_local)
                    noise_pred_nega, noise_pred_posi = cfg_group.all_gather(noise_pred_local, separate_tensors=True)
                else:
                    noise_pred_posi = run_dit(prompt_emb_posi)
                    noise_pred_nega = run_dit(prompt_emb_nega) if cfg_scale != 1.0 else None

                noise_pred = (
                    noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    if cfg_scale != 1.0
                    else noise_pred_posi
                )
                prev_predictions.append((timestep, noise_pred))
                if len(prev_predictions) > 2:
                    prev_predictions.pop(0)
            else:
                noise_pred = prev_predictions[-1][1]

            # Scheduler: update only the ~masks
            latents_input[~masks] = self.scheduler.step(noise_pred[~masks], timestep, latents_input[~masks])

        # Decode
        self.load_models_to_device(["vae"])
        # Decode segments with multi-rank overlap; each segment is still decoded on a single GPU.
        seg_latents_map = {}
        for seg_idx, (seg_start, seg_end) in enumerate(seg_latent_spans):
            seg_latents_map[f"seg_{seg_idx}"] = latents_input[:, :, seg_start:seg_end, ...]
        seg_frames_map = self.decode_videos_distributed(seg_latents_map, **tiler_kwargs)
        frames = torch.cat([seg_frames_map[f"seg_{i}"] for i in range(seg_num)], dim=2)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
        cfg_parallel=False,
        dynamic_cache_schedule=False,
        torch_compile=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                # "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]

        # Initialize pipeline
        pipe = WanVideoActionImagesPipeline(
            device=device,
            torch_dtype=torch_dtype,
            dynamic_cache_schedule=dynamic_cache_schedule,
            torch_compile=torch_compile,
        )
        pipe.initialize_parallel(use_usp=use_usp, cfg_parallel=cfg_parallel)

        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=(use_usp or cfg_parallel))
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )

        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=(use_usp or cfg_parallel))
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if torch_compile:
            pipe.prompter.text_encoder.forward = torch.compile(
                pipe.prompter.text_encoder.forward, mode="reduce-overhead", fullgraph=False, dynamic=False
            )
            pipe.vae.model.encoder.forward = torch.compile(
                pipe.vae.model.encoder.forward, mode="reduce-overhead", fullgraph=True, dynamic=False
            )

        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        return pipe

    def enable_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
            get_sequence_parallel_world_size,
            model_parallel_is_initialized,
        )
        from diffsynth.distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        if not model_parallel_is_initialized():
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            init_distributed_environment(rank=rank, world_size=world_size, local_rank=local_rank)
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=1,
                ulysses_degree=world_size,
            )
            torch.cuda.set_device(local_rank)
        if dist.get_rank() == 0:
            print(f"Unified Sequence Parallel initialized with {dist.get_world_size()} processes.")

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cam_emb: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    enable_usp: bool = False,
    **kwargs,
):
    if enable_usp:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)

    freqs = (
        torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    if enable_usp and dist.is_initialized() and dist.get_world_size() > 1:
        # chunk x
        chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
        pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
        chunks = [
            torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0) for chunk in chunks
        ]
        x = chunks[get_sequence_parallel_rank()]
        # The cam_emb needs to be chunked at the same level as x, so need to be moeved inside the model.

    for block in dit.blocks:
        x = block(x, context, cam_emb, t_mod, freqs, (f, h, w))

    x = dit.head(x, t)

    if enable_usp and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)
        x = x[:, :-pad_shape] if pad_shape > 0 else x

    x = dit.unpatchify(x, (f, h, w))
    return x
