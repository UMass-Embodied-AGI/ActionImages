import os
import json
import pickle
import hashlib
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from torchvision.transforms import v2
import imageio
import random
from einops import rearrange
from training.utils import get_relative_pose_batch, CenterCropToAspect, convert_intrinsics_after_center_crop_resize


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for multi-view video datasets.

    This template provides a common interface and shared functionality
    for datasets used in multi-view video generation tasks.
    """

    def __init__(
        self,
        base_path: str,
        num_frames: int = 81,
        frame_interval: int = 1,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ):
        """
        Initialize the base dataset.

        Args:
            base_path: Root path to the dataset
            num_frames: Number of frames to sample from each video
            frame_interval: Interval between sampled frames
            height: Target height for video frames
            width: Target width for video frames
            **kwargs: Additional dataset-specific parameters
        """
        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.height = height
        self.width = width

        # Initialize frame processing pipeline
        self.frame_process = v2.Compose(
            [
                CenterCropToAspect(height, width),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Initialize dataset-specific data structures
        self.episodes = []

        # Load dataset (with simple hash-based cache under ./cache)
        if not self._try_load_cache():
            self._load_dataset()
            self._save_cache()

    # ==== BEGIN: abstract methods ====
    @abstractmethod
    def _load_dataset(self) -> None:
        pass

    @abstractmethod
    def get_8d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> torch.Tensor:
        # Default to null 8D actions
        if frame_indices is None:
            length = self.num_frames
        else:
            length = len(frame_indices)
        # [x, y, z, qw, qx, qy, qz, openness]
        return np.zeros((length, 8), dtype=np.float32)

    @abstractmethod
    def get_7d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def get_instruction(self, **kwargs) -> str:
        pass

    @abstractmethod
    def get_all_views(self, **kwargs) -> str:
        pass

    # ======== END ========

    def select_views(self, candidate_indices: List[int], view_indices: List[int]) -> List[int]:
        return [candidate_indices[i] for i in view_indices]

    @abstractmethod
    def getitem(self, index: int) -> Dict[str, Any]:
        pass

    def getitem(self, index: int) -> Dict[str, Any]:
        """Get a single RLBench data sample."""
        episode_info = self.episodes[index]
        episode_path = episode_info["path"]

        # Get text description
        text = self.get_instruction(episode_info)

        # ==== Load videos from different views ====
        videos, frame_indices, extrinsics, intrinsics, raw_resolutions, has_global_extrinsics = self.get_all_views(
            episode_path
        )
        intrinsics = convert_intrinsics_after_center_crop_resize(
            intrinsics, raw_resolutions, [(self.height, self.width)] * len(intrinsics)
        )

        # ==== Load action data aligned with video frames ====
        actions_8d = self.get_8d_action(episode_path, frame_indices=frame_indices)
        actions_7d = self.get_7d_action(episode_path, frame_indices=frame_indices)

        # ==== view selection start ====
        # Randomly select condition and target views
        if len(videos) == 1:
            view_indices = [0, 0]
        else:
            view_indices = random.sample(range(len(videos)), 2)
        video_cond, video_target = self.select_views(videos, view_indices)
        # Concatenate condition and target videos
        combined_video = torch.cat([video_cond, video_target], dim=1)  # C, 2*T, H, W

        # ==== Calculate relative camera poses ====
        extrinsics_cond, extrinsics_target = self.select_views(extrinsics, view_indices)
        intrinsics_cond, intrinsics_target = self.select_views(intrinsics, view_indices)
        base_extrinsics = extrinsics_cond[0]  # 3, 4
        relative_poses_target = get_relative_pose_batch(base_extrinsics, extrinsics_target)  # T, 3, 4
        relative_poses_cond = get_relative_pose_batch(base_extrinsics, extrinsics_cond)  # T, 3, 4
        # Concatenate condition and target relative poses
        relative_poses_target = rearrange(torch.from_numpy(relative_poses_target), "t c d -> t (c d)")
        relative_poses_cond = rearrange(torch.from_numpy(relative_poses_cond), "t c d -> t (c d)")
        camera_poses = torch.cat([relative_poses_cond, relative_poses_target], dim=0)
        camera_poses = camera_poses.to(torch.float32)  # T * 2, 12
        extrinsics = torch.from_numpy(np.concatenate([extrinsics_cond, extrinsics_target], axis=0))  # T * 2, 3, 4
        intrinsics = torch.from_numpy(np.concatenate([intrinsics_cond, intrinsics_target], axis=0))  # T * 2, 3, 3

        # ==== to torch ====
        actions_7d = torch.from_numpy(actions_7d)
        actions_8d = torch.from_numpy(actions_8d)

        return {
            "text": text,  # str
            "video": combined_video,  # 3, T * 2, H, W
            "camera": camera_poses,  # T * 2, 12
            "extrinsics": extrinsics,  # T * 2, 3, 4 - camera to world
            "intrinsics": intrinsics,  # T * 2, 3, 3
            "action_7d": actions_7d,  # T, 7
            "action_8d": actions_8d,  # T, 8
            "path": episode_path,  # str
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item with retry logic for robustness."""
        # return self.getitem(index)
        for _ in range(50):
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error in getitem: {e}")
                index = random.randint(0, len(self.episodes) - 1)
        raise Exception(f"Failed to get item after 10 attempts: {index}")

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.episodes)

    # ======== Simple caching helpers ========
    def _get_cache_key(self) -> str:
        """Compute a stable hash key from dataset identity and init parameters."""
        key_obj = {
            "class": self.__class__.__name__,
            "base_path": os.path.abspath(self.base_path),
            "num_frames": self.num_frames,
            "frame_interval": self.frame_interval,
            "height": self.height,
            "width": self.width,
        }
        payload = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _get_cache_dir(self) -> str:
        return os.path.join(".", "cache")

    def _get_cache_path(self) -> str:
        cache_name = f"{self.__class__.__name__}-{self._get_cache_key()}.pkl"
        return os.path.join(self._get_cache_dir(), cache_name)

    def _try_load_cache(self) -> bool:
        try:
            cache_path = self._get_cache_path()
            if os.path.isfile(cache_path):
                with open(cache_path, "rb") as f:
                    self.episodes = pickle.load(f)
                return isinstance(self.episodes, list) and len(self.episodes) >= 0
            return False
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            cache_dir = self._get_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            with open(self._get_cache_path(), "wb") as f:
                pickle.dump(self.episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Silently ignore caching errors to avoid disrupting dataset loading
            pass


class CombDataset(torch.utils.data.Dataset):
    """
    Combine multiple datasets and sample from them according to provided ratios.

    Input specs format: list of strings like "{name}@{ratio}", e.g., ["bridge@0.7", "rlbench@0.3"].

    - Supported names: "bridge", "rlbench" (extendable).
    - The dataset length is defined as the sum of underlying dataset lengths.
    - __getitem__ ignores the incoming index for fairness and samples a source
      dataset by ratio, then a random episode within that dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_specs: str,
        num_frames: int = 81,
        frame_interval: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> None:
        super().__init__()

        if "@" not in dataset_specs:
            dataset_specs = f"{dataset_specs}@1.0"

        dataset_specs = [s.strip() for s in dataset_specs.split(",") if s.strip()]

        # Lazy import to avoid circular imports with BaseDataset subclasses
        from training.dataset.bridge import BridgeMVDataset
        from training.dataset.rlbench import RLBenchMVDataset
        from training.dataset.droid import DROIDMVDataset

        name_to_ctor = {
            "bridge": lambda: BridgeMVDataset(
                base_path=os.path.join(dataset_path, "bridge"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
            ),
            "rlbench": lambda: RLBenchMVDataset(
                base_path=os.path.join(dataset_path, "rlbench"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
            ),
            "droid": lambda: DROIDMVDataset(
                base_path=os.path.join(dataset_path, "droid"),
                num_frames=num_frames,
                frame_interval=4,
                height=height,
                width=width,
            ),
        }

        datasets: List[torch.utils.data.Dataset] = []
        ratios: List[float] = []
        names: List[str] = []

        for spec in dataset_specs:
            if "@" not in spec:
                raise ValueError(f"Invalid dataset spec: {spec}. Expected format 'name@ratio'.")
            name, ratio_str = spec.split("@", 1)
            name = name.strip().lower()
            try:
                ratio = float(ratio_str)
            except Exception:
                raise ValueError(f"Invalid ratio in dataset spec: {spec}")
            if ratio <= 0:
                continue  # skip non-positive ratios

            if name not in name_to_ctor:
                raise ValueError(f"Unknown dataset name '{name}' in spec '{spec}'.")

            datasets.append(name_to_ctor[name]())
            ratios.append(ratio)
            names.append(name)

        if not datasets:
            raise ValueError("No valid datasets constructed from specs.")

        # Normalize ratios and build cumulative distribution for sampling
        ratio_sum = float(sum(ratios))
        if ratio_sum <= 0:
            raise ValueError("Sum of ratios must be positive.")
        probs = [r / ratio_sum for r in ratios]
        cum_probs = []
        c = 0.0
        for p in probs:
            c += p
            cum_probs.append(c)
        # Ensure the last value is exactly 1.0 to avoid floating issues
        cum_probs[-1] = 1.0

        self._datasets = datasets
        self._names = names
        self._probs = probs
        self._cum_probs = cum_probs
        self._lengths = [len(ds) for ds in datasets]
        self._total_length = int(sum(self._lengths))

    def __len__(self) -> int:
        # Use the aggregate size as epoch length. Trainer will sample indices randomly.
        return max(self._total_length, 1)

    def _uniform_from_index(self, index: int, salt: int = 0) -> float:
        """Deterministically map an integer index (+ optional salt) to [0, 1).

        This allows __getitem__ to be purely index-driven, which plays nicely
        with DistributedSampler to avoid cross-rank duplication.
        """
        # Use a stable 64-bit hash and scale to [0, 1)
        key = f"{index}_{salt}".encode("utf-8")
        h = hashlib.blake2b(key, digest_size=8).digest()
        u64 = int.from_bytes(h, byteorder="big", signed=False)
        return (u64 & ((1 << 64) - 1)) / float(1 << 64)

    def _choose_dataset_index(self, r: Optional[float] = None) -> int:
        # If r is not provided, fall back to RNG (non-DDP/inference cases)
        if r is None:
            r = random.random()
        # Binary search over cumulative probabilities
        lo, hi = 0, len(self._cum_probs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= self._cum_probs[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Deterministically map the sampler-provided index to a dataset id
        # so that different ranks (with disjoint indices from DistributedSampler)
        # do not collide on the same underlying sample.
        ds_idx = self._choose_dataset_index(r=self._uniform_from_index(index, salt=0))
        ds = self._datasets[ds_idx]

        # Deterministically pick an item from the chosen dataset based on index
        # while still looking "random" but remaining index-driven.
        local_len = len(ds)
        if local_len == 0:
            # Fallback: try another dataset
            for alt_idx, alt_ds in enumerate(self._datasets):
                if len(alt_ds) > 0:
                    ds_idx = alt_idx
                    ds = alt_ds
                    local_len = len(ds)
                    break
        if local_len == 0:
            raise IndexError("All underlying datasets are empty.")

        # Use a second deterministic uniform mapped to [0, local_len)
        r_local = self._uniform_from_index(index, salt=1)
        local_index = int(r_local * local_len)
        if local_index >= local_len:
            local_index = local_len - 1
        sample = ds[local_index]

        # Optionally tag the sample with its source dataset name
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["_source_dataset"] = self._names[ds_idx]
        return sample
