import os
import torch
import imageio
import numpy as np
import random
import json
import glob
import pickle
from functools import lru_cache
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from einops import rearrange

from training.dataset.base import BaseDataset
from training.helpers.io import load_video_frames


class RLBenchMVDataset(BaseDataset):
    """
    RLBench multi-view dataset implementation.

    This dataset handles RLBench data with multiple camera views,
    providing video frames, camera poses, and action sequences.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_descriptions = {}
        for task_dir in glob.glob(os.path.join(self.base_path, "*")):
            if not os.path.isdir(task_dir):
                continue
            task_name = os.path.basename(task_dir)
            for variation_dir in glob.glob(os.path.join(task_dir, "variation*")):
                variation_name = os.path.basename(variation_dir)
                desc_file = os.path.join(variation_dir, "variation_descriptions.pkl")
                with open(desc_file, "rb") as f:
                    descriptions = pickle.load(f)
                    self.task_descriptions[f"{task_name}_{variation_name}"] = descriptions

    def _load_dataset(self) -> None:
        """Load RLBench dataset structure."""
        # Collect all episodes from all tasks and variations
        for task_dir in glob.glob(os.path.join(self.base_path, "*")):
            if not os.path.isdir(task_dir):
                continue

            task_name = os.path.basename(task_dir)
            for variation_dir in glob.glob(os.path.join(task_dir, "variation*")):
                variation_name = os.path.basename(variation_dir)

                episodes_dir = os.path.join(variation_dir, "episodes")
                if os.path.exists(episodes_dir):
                    for episode_dir in glob.glob(os.path.join(episodes_dir, "episode*")):
                        if os.path.isdir(episode_dir):
                            self.episodes.append(
                                {
                                    "path": episode_dir,
                                    "task": task_name,
                                    "variation": variation_name,
                                    "episode": os.path.basename(episode_dir),
                                }
                            )

        print(f"Found {len(self.episodes)} episodes")

    def get_camera_params(
        self, camera_params_path: str, frame_indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load camera intrinsics and extrinsics."""
        with open(camera_params_path, "r") as f:
            camera_data = json.load(f)

        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        intrinsics_list = []
        extrinsics_list = []

        for idx in frame_indices:
            frame_key = f"{idx:04d}"
            if frame_key in camera_data:
                intrinsics = np.array(camera_data[frame_key]["intrinsics"])
                extrinsics = np.array(camera_data[frame_key]["extrinsics"])
            else:
                # Use the last available frame if index is out of range
                last_key = max(camera_data.keys())
                intrinsics = np.array(camera_data[last_key]["intrinsics"])
                extrinsics = np.array(camera_data[last_key]["extrinsics"])

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        return np.stack(extrinsics_list), np.stack(intrinsics_list)

    @lru_cache(maxsize=5)
    def _load_8d_actions_base(self, episode_path: str) -> np.ndarray:
        """Load base 8d actions from file with LRU cache (maxsize=5)."""
        action_path = os.path.join(episode_path, "actions.npy")
        return np.load(action_path)  # Load directly without any modification

    def get_8d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        """Load 8d action data from RLBench format with caching."""
        actions = self._load_8d_actions_base(episode_path)
        actions = actions[::4]
        if frame_indices is not None:
            actions = actions[frame_indices]
        return actions

    def get_7d_action(self, episode_path: str, frame_indices: Optional[List[int]]) -> torch.Tensor:
        """Load action data from RLBench format."""
        actions = self.get_8d_action(episode_path, frame_indices=None)
        quat = actions[:, 3:7]
        euler = R.from_quat(quat).as_euler("xyz", degrees=False)
        actions = np.concatenate([actions[:, :3], euler, actions[:, 7:]], axis=1)
        actions = actions[frame_indices]
        return actions

    def get_all_views(self, episode_path: str):
        videos = []
        extrinsics = []
        intrinsics = []
        raw_resolutions = []
        frame_indices = None

        # Load multi-view videos and camera parameters
        for view_dir in glob.glob(os.path.join(episode_path, "view*")):
            view_path = view_dir
            if os.path.exists(view_path):
                # Load video
                video_path = os.path.join(view_path, "rgb", "video.mp4")
                if os.path.exists(video_path):
                    video, frame_indices, raw_resolution = load_video_frames(
                        video_path,
                        frame_indices=frame_indices,
                        max_frames=self.num_frames,
                        frame_interval=self.frame_interval,
                        frame_process=self.frame_process,
                    )
                    videos.append(video)
                    raw_resolutions.append(raw_resolution)

                # Load camera parameters
                camera_params_path = os.path.join(view_path, "camera_params.json")
                if os.path.exists(camera_params_path):
                    extr, intr = self.get_camera_params(camera_params_path, frame_indices=frame_indices)
                    extrinsics.append(extr)
                    intrinsics.append(intr)

        return videos, frame_indices, extrinsics, intrinsics, raw_resolutions, True

    def get_instruction(self, episode_info: Dict[str, Any]) -> str:
        """Get instruction for the task."""
        task_variation_key = f"{episode_info['task']}_{episode_info['variation']}"
        return random.choice(self.task_descriptions[task_variation_key])
