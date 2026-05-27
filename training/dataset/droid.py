import os
import glob
import json
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from training.dataset.base import BaseDataset
from training.helpers.io import load_video_frames


class DROIDMVDataset(BaseDataset):
    """
    DROID multi-view dataset implementation.

    Expected directory layout per episode under base_path (supports passing .../data/droid or .../data/droid/processed):

    episode_dir/
      - video/
          ext1.mp4, ext2.mp4, ...
      - camera.json               # per-view static intrinsics/extrinsics (camera-to-world 4x4, intrinsics 3x3)
      - instruction.txt           # one instruction per line or a single line
      - action.npz                # optional; if missing, zeros are returned

    Notes:
    - Camera parameters are static across frames; duplicated to match sampled frame count.
    - `action.npz` is expected to contain a key "action_7d" or similar (fallbacks supported). If unavailable, zeros are used.
    - Actions are sub-sampled and aligned to the sampled video frame indices.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cam2base = json.load(open(os.path.join(self.base_path, "cam2base_extrinsic_superset.json")))
        self.camera_serials = json.load(open(os.path.join(self.base_path, "camera_serials.json")))

    def _resolve_dataset_root(self) -> str:
        processed_dir = os.path.join(self.base_path, "processed")
        if os.path.isdir(processed_dir):
            return processed_dir
        return self.base_path

    def _load_dataset(self) -> None:
        root = self._resolve_dataset_root()
        candidate_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        for episode_dir in candidate_dirs:
            video_dir = os.path.join(episode_dir, "video")
            if not os.path.isdir(video_dir):
                continue
            # require at least one mp4
            if not glob.glob(os.path.join(video_dir, "*.mp4")):
                continue
            self.episodes.append({"path": episode_dir})

        print(f"DROIDMVDataset: Found {len(self.episodes)} episodes in {root}")

    # ====== helpers ======
    def _read_camera_json(
        self, camera_json_path: str, view_video_paths: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        sample_name = os.path.basename(os.path.dirname(camera_json_path))
        camera_serials = self.camera_serials[sample_name]
        has_global_extrinsics = False

        with open(camera_json_path, "r") as f:
            cam = json.load(f)
        view_names = [os.path.basename(vp).split(".")[0] for vp in view_video_paths]
        extrinsics_list: List[np.ndarray] = []
        intrinsics_list: List[np.ndarray] = []
        for name in view_names:
            extr = np.array(cam[name]["extrinsics"], dtype=np.float32)  # 4x4, camera 2 camera
            # find the camera serial which the keys contains name
            camera_serial_view = [k for k in camera_serials.keys() if name in k][0]
            if sample_name in self.cam2base:
                has_global_extrinsics = True
                camera_id = camera_serials[camera_serial_view]
                extr_6d = self.cam2base[sample_name][camera_id]
                extr_6d = np.array(extr_6d, dtype=np.float32)  # (6,)
                extr = np.eye(4)
                extr[:3, :3] = R.from_euler("xyz", extr_6d[3:]).as_matrix()
                extr[:3, 3] = extr_6d[:3]

            intr = np.array(cam[name]["intrinsics"], dtype=np.float32)  # 3x3
            extrinsics_list.append(extr)
            intrinsics_list.append(intr)
        return extrinsics_list, intrinsics_list, has_global_extrinsics

    def _load_actions_npz(self, action_npz_path: str, frame_indices: List[int]) -> np.ndarray:
        """
        Load 7D actions and align to provided frame indices.
        Expected keys in the npz (checked in order):
          - "action_7d"
          - "actions_7d"
          - "action"
          - "actions"
        Each should be shaped [T, 7]. If shape differs, attempts simple fixes.
        """
        actions = np.load(action_npz_path)["action"]
        # HACK: now it is of shape (T, 6)
        actions = np.concatenate([actions, np.ones((actions.shape[0], 1))], axis=1)

        actions = actions[frame_indices]
        return actions

    def get_8d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        # NOTE: we don't have 8d actions for droid
        if frame_indices is None:
            length = self.num_frames
        else:
            length = len(frame_indices)
        # [x, y, z, euler_x, euler_y, euler_z, openness]
        return np.zeros((length, 8), dtype=np.float32)

    # ====== abstract method impls ======
    def get_7d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        action_path = os.path.join(episode_path, "action.npz")
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))
        return self._load_actions_npz(action_path, frame_indices)

    def get_all_views(self, episode_path: str):
        videos: List[torch.Tensor] = []
        extrinsics_per_view: List[np.ndarray] = []
        intrinsics_per_view: List[np.ndarray] = []
        raw_resolutions: List[Tuple[int, int]] = []

        frame_indices: Optional[List[int]] = None

        # 1) load videos consistently across views
        video_dir = os.path.join(episode_path, "video")
        view_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
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

        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        # load camera.json and broadcast across frames
        camera_json_path = os.path.join(episode_path, "camera.json")
        extr0, intr0, has_global_extrinsics = self._read_camera_json(camera_json_path, view_video_paths)

        V = len(extr0)
        L = len(frame_indices)
        for i in range(V):
            extr_broadcast = np.repeat(extr0[i][None, ...], L, axis=0)  # L, 4, 4
            intr_broadcast = np.repeat(intr0[i][None, ...], L, axis=0)  # L, 3, 3
            extrinsics_per_view.append(extr_broadcast)
            intrinsics_per_view.append(intr_broadcast)

        return videos, frame_indices, extrinsics_per_view, intrinsics_per_view, raw_resolutions, has_global_extrinsics

    def get_instruction(self, episode_info: Dict[str, Any]) -> str:
        episode_path = episode_info["path"]
        instr_path = os.path.join(episode_path, "instruction.txt")
        with open(instr_path, "r") as f:
            return f.readlines()[0].strip()
