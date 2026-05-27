import os
import glob
import random
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

from training.dataset.base import BaseDataset
from training.helpers.io import load_video_frames


class BridgeMVDataset(BaseDataset):
    """
    Bridge multi-view dataset implementation.

    Expected directory layout per episode (supports both raw and processed roots):

    episode_dir/
      - video/
          0.mp4, 1.mp4, 2.mp4, ...
      - vggt/
          <frame_index>/
              intrinsic.npy  # 3x3
              extrinsic.npy  # 3x4 (camera-to-world)
      - instruction.txt or caption.txt

    Notes:
    - Actions are currently returned as zeros aligned with sampled frame indices.
    - Camera intrinsics/extrinsics are loaded from vggt for sampled frames; if a frame is
      missing, the closest available frame index (by searching backward then forward) is used.
    - The same camera parameters are reused across views if only a single set is provided per frame.
    """

    def _resolve_dataset_root(self) -> str:
        # Allow passing either .../data/bridge or .../data/bridge/processed
        # Prefer the "processed" subdir if it exists.
        processed_dir = os.path.join(self.base_path, "processed")
        if os.path.isdir(processed_dir):
            return processed_dir
        return self.base_path

    def _load_dataset(self) -> None:
        root = self._resolve_dataset_root()
        # Episodes are subdirectories containing at least 2 view videos
        candidate_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        for episode_dir in candidate_dirs:
            video_dir = os.path.join(episode_dir, "video")
            if not os.path.isdir(video_dir):
                continue
            self.episodes.append(
                {
                    "path": episode_dir,
                }
            )

        print(f"BridgeMVDataset: Found {len(self.episodes)} episodes in {root}")

    # ====== camera and instruction helpers ======
    def get_camera_params(self, vggt_dir: str, frame_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        # bridge always has static camera parameters
        extrinsics = np.load(os.path.join(vggt_dir, "0", "extrinsic.npy"))  # V, 3, 4
        intrinsics = np.load(os.path.join(vggt_dir, "0", "intrinsic.npy"))  # V, 3, 3
        # transform intrinsics
        intrinsics[..., 0, 0] *= -1
        intrinsics[..., 1, 1] *= -1
        # transform extrinsics
        V = extrinsics.shape[0]  # number of views
        # V, 3, 4 to V, 4, 4 by adding 0001
        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3:4]  # [V, 3, 1]
        R_inv = R.swapaxes(-1, -2)
        t_inv = -np.matmul(R_inv, t)  # [V, 3, 1]
        extrinsics = np.concatenate([R_inv, t_inv], axis=-1)  # [V, 3, 4]
        # flip x and y
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = -1
        matrix[1, 1] = -1
        extrinsics = extrinsics @ matrix
        extrinsics = np.concatenate([extrinsics, np.zeros((V, 1, 4))], axis=1)  # V, 4, 4
        extrinsics[:, 3, 3] = 1.0

        # to [length, 4, 4]
        length = len(frame_indices)  # number of frames
        extrinsics = np.repeat(extrinsics[:, None, ...], length, axis=1).astype(np.float32)  # V, length, 4, 4
        intrinsics = np.repeat(intrinsics[:, None, ...], length, axis=1).astype(np.float32)  # V, length, 3, 3

        extrinsics_list = [extrinsics[i] for i in range(V)]
        intrinsics_list = [intrinsics[i] for i in range(V)]

        return np.stack(extrinsics_list, axis=0), np.stack(intrinsics_list, axis=0)

    def get_8d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        # NOTE: we don't have 8d actions for bridge
        if frame_indices is None:
            length = self.num_frames
        else:
            length = len(frame_indices)
        # [x, y, z, euler_x, euler_y, euler_z, openness]
        return np.zeros((length, 8), dtype=np.float32)

    # ====== abstract method impls ======
    def get_7d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        if frame_indices is None:
            length = self.num_frames
        else:
            length = len(frame_indices)
        # [x, y, z, euler_x, euler_y, euler_z, openness]
        return np.zeros((length, 7), dtype=np.float32)

    def get_all_views(self, episode_path: str):
        videos: List[torch.Tensor] = []
        extrinsics_per_view: List[np.ndarray] = []
        intrinsics_per_view: List[np.ndarray] = []
        raw_resolutions: List[Tuple[int, int]] = []

        frame_indices: Optional[List[int]] = None

        # Load available view videos
        video_dir = os.path.join(episode_path, "video")
        view_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

        # Sample frames consistently across all views
        for vp in view_video_paths:
            if not os.path.isfile(vp):
                continue
            video, frame_indices, raw_resolution = load_video_frames(
                vp,
                frame_indices=frame_indices,
                max_frames=self.num_frames,
                frame_interval=self.frame_interval,
                frame_process=self.frame_process,
            )
            videos.append(video)
            raw_resolutions.append(raw_resolution)

        # Load camera parameters from vggt for sampled frames
        vggt_dir = os.path.join(episode_path, "vggt")
        if frame_indices is None:
            # No frames (e.g., missing videos); create a trivial set
            frame_indices = list(range(self.num_frames))

        extrinsics_per_view, intrinsics_per_view = self.get_camera_params(vggt_dir, frame_indices)

        return videos, frame_indices, extrinsics_per_view, intrinsics_per_view, raw_resolutions, False

    def get_instruction(self, episode_info: Dict[str, Any]) -> str:
        episode_path = episode_info["path"]
        instr_files = os.path.join(episode_path, "instruction.txt")
        with open(instr_files, "r") as f:
            texts = f.readlines()
        if not texts:
            return ""
        # Return a random instruction to introduce slight variation
        return random.choice(texts)
