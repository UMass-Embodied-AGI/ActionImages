import random
import torch
import argparse
import numpy as np
from tqdm import tqdm
from training.dataset.rlbench import RLBenchMVDataset
from training.dataset.bridge import BridgeMVDataset
from training.dataset.droid import DROIDMVDataset
from training.utils import (
    project_actions_7d_to_5d_batch,
    project_action_5d_to_rgb,
    visualize_action_video,
    project_actions_7d_to_5d_torch_batch,
    project_action_5d_to_rgb_torch,
    get_plucker_embeddings,
    get_plucker_embeddings_torch,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="rlbench", choices=["rlbench", "bridge", "droid"])
    parser.add_argument("--save_dir", type=str, default="tmp/action_video")
    args = parser.parse_args()
    backend = args.backend
    num_frames = args.num_frames
    height = args.height
    width = args.width
    print(f"Using backend: {backend}")

    if args.dataset == "rlbench":
        dataset = RLBenchMVDataset(base_path="./data/rlbench", num_frames=num_frames, height=height, width=width)
    elif args.dataset == "bridge":
        dataset = BridgeMVDataset(base_path="./data/bridge", num_frames=num_frames, height=height, width=width)
    elif args.dataset == "droid":
        dataset = DROIDMVDataset(
            base_path="./data/droid", num_frames=num_frames, height=height, width=width, frame_interval=4
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    print(f"Dataset length: {len(dataset)}")

    import os

    os.makedirs(args.save_dir, exist_ok=True)
    moment_max = 0
    for i in tqdm(range(min(100, len(dataset)))):  # Test first 5 samples
        index = random.randint(0, len(dataset) - 1)
        sample = dataset.__getitem__(index)
        print(f"Sample {i} keys: {sample.keys()}")
        print(f"Video shape: {sample['video'].shape}")
        print(f"Camera shape: {sample['camera'].shape}")
        print(f"Extrinsics shape: {sample['extrinsics'].shape}")
        print(f"Intrinsics shape: {sample['intrinsics'].shape}")
        print(f"Action 7D shape: {sample['action_7d'].shape}")
        print(f"Text: {sample['text'][:50]}...")

        action_7d = sample["action_7d"]
        extrinsics = sample["extrinsics"]
        intrinsics = sample["intrinsics"]
        camera = sample["camera"]
        T = action_7d.shape[0]
        resolution = (height, width)

        if backend == "torch":
            # Convert to torch tensors and add batch dimension
            action_7d_torch = action_7d.unsqueeze(0)  # 1, T, 7
            extrinsics_torch = extrinsics.unsqueeze(0)  # 1, num_views * T, 4, 4
            intrinsics_torch = intrinsics.unsqueeze(0)  # 1, num_views * T, 3, 3

            # Test Plücker embeddings (convert to 3x4 format)
            extrinsics_3x4_torch = camera.reshape(1, -1, 3, 4)
            plucker_torch = get_plucker_embeddings_torch(extrinsics_3x4_torch, intrinsics_torch, resolution)
            print(f"Plücker torch shape: {plucker_torch.shape}")
            plucker = plucker_torch.squeeze(0).numpy()

            # Repeat action_7d to match camera dimensions
            num_views = extrinsics_torch.shape[1] // T
            action_7d_torch = action_7d_torch.repeat(1, num_views, 1)  # 1, num_views * T, 7

            print(f"Action 7D torch shape: {action_7d_torch.shape}")
            action_5d_torch = project_actions_7d_to_5d_torch_batch(action_7d_torch, extrinsics_torch, intrinsics_torch)
            print(f"Action 5D torch shape: {action_5d_torch.shape}")
            action_video_torch = project_action_5d_to_rgb_torch(
                action_5d_torch, sample["video"].shape[-2], sample["video"].shape[-1]
            )
            print(f"Action video torch shape: {action_video_torch.shape}")

            # Convert back to numpy for visualization (remove batch dimension)
            action_video = action_video_torch.squeeze(0).numpy()
            action_video = action_video * 2 - 1

        else:  # numpy backend
            # Test Plücker embeddings (convert to 3x4 format)
            extrinsics_3x4 = camera.reshape(-1, 3, 4)  # num_views * T, 3, 4
            intrinsics = intrinsics.cpu().numpy()
            plucker = get_plucker_embeddings(extrinsics_3x4, intrinsics, resolution)  # T, H, W, 6
            print(f"Plücker shape: {plucker.shape}")

            # get T, extrinsics.shape[0] // T times, ...
            action_7d = np.stack([action_7d] * (extrinsics.shape[0] // T), axis=0).reshape(-1, 7)
            print(f"Action 7D shape: {action_7d.shape}")
            action_5d = project_actions_7d_to_5d_batch(action_7d, extrinsics, intrinsics)
            print(f"Action 5D shape: {action_5d.shape}")
            action_video = project_action_5d_to_rgb(action_5d, sample["video"].shape[-2], sample["video"].shape[-1])
            action_video = action_video * 2 - 1
            print(f"Action video shape: {action_video.shape}")

        # === Process Plücker Visualization ===
        direction, moment = plucker[..., :3], plucker[..., 3:]
        moment_max = max(moment_max, np.abs(moment).max())
        moment = moment / 3.0
        print(f"video min max: {sample['video'].min()}, {sample['video'].max()}")
        print(f"action video min max: {action_video.min()}, {action_video.max()}")
        print(f"direction min max: {direction.min()}, {direction.max()}")
        print(f"moment min max: {moment.min()}, {moment.max()}")

        visualize_action_video(
            action_video,
            sample["video"],
            save_path=os.path.join(args.save_dir, f"action_video_{args.dataset}_{i}.mp4"),
            other_videos=[direction, moment],
        )

    print(f"Moment max: {moment_max}")
