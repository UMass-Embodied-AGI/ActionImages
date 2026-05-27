# RLBench-specific camera and action visualization utilities

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import glob
import cv2
import imageio
from scipy.spatial.transform import Rotation as R

from camera import CameraPoseVisualizer, visualize_cameras
from action import visualize_action_on_video


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_path", type=str, help="the path of the episode directory")
    parser.add_argument("--stride", type=int, default=1, help="stride for sampling frames")
    parser.add_argument("--max_frames", type=int, default=50, help="maximum number of frames to visualize")

    parser.add_argument("--fps", type=int, default=20, help="frames per second for action video")
    # Camera visualization parameters
    parser.add_argument("--base_xval", type=float, default=0.08)
    parser.add_argument("--zval", type=float, default=0.15)
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--y_min", type=float, default=-1.5)
    parser.add_argument("--y_max", type=float, default=1.5)
    parser.add_argument("--z_min", type=float, default=0)
    parser.add_argument("--z_max", type=float, default=2.5)
    return parser.parse_args()


def load_rlbench_cameras(episode_path, stride=1, max_frames=50):
    """Load camera parameters from RLBench format"""
    view_dirs = sorted(glob.glob(os.path.join(episode_path, "view*")))

    all_cameras = {}

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)
        camera_params_path = os.path.join(view_dir, "camera_params.json")

        if not os.path.exists(camera_params_path):
            print(f"Warning: {camera_params_path} not found")
            continue

        print(f"Loading {camera_params_path}")
        with open(camera_params_path, "r") as f:
            data = json.load(f)

        # Extract frame keys and sort them
        frame_keys = sorted([k for k in data.keys() if k.isdigit()], key=lambda x: int(x))

        # Sample frames with stride
        sampled_keys = frame_keys[::stride][:max_frames]

        cameras = []
        intrinsics_list = []

        for frame_key in sampled_keys:
            extrinsics = np.array(data[frame_key]["extrinsics"])
            intrinsics = np.array(data[frame_key]["intrinsics"])
            # extrinsics should be camera-to-world transformation
            cameras.append(extrinsics)
            intrinsics_list.append(intrinsics)

        all_cameras[view_name] = {"extrinsics": np.array(cameras), "intrinsics": np.array(intrinsics_list)}

    return all_cameras


def load_rlbench_action_data(episode_path):
    """Load action data from RLBench format"""
    action_path = os.path.join(episode_path, "actions.npy")
    key_frames_path = os.path.join(episode_path, "key_frames.npy")

    # action_3d is a [T, 8] array: [x, y, z, qx, qy, qz, qw, openness]
    action_3d = np.load(action_path)
    key_frames = np.load(key_frames_path) // 4  # This array maps each action to the corresponding video frame

    return action_3d, key_frames


def load_video_frames(video_path):
    """Load video frames from MP4 file"""
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()

    return video_frames


def visualize_actions_and_cameras(episode_path, args):
    """
    Visualize both actions on video and camera poses in 3D space.
    """
    print("=== Action Visualization ===")

    # Load action data
    action_3d, _ = load_rlbench_action_data(episode_path)

    # Load all camera data (including dynamic extrinsics)
    all_cameras = load_rlbench_cameras(episode_path, args.stride, args.max_frames)

    # Load all views
    view_dirs = sorted(glob.glob(os.path.join(episode_path, "view*")))

    all_video_frames = {}
    all_intrinsics = {}
    all_extrinsics = {}

    for view_dir in view_dirs:
        view_name = os.path.basename(view_dir)
        all_intrinsics[view_name] = all_cameras[view_name]["intrinsics"]
        all_extrinsics[view_name] = all_cameras[view_name]["extrinsics"]

        # Load video frames
        video_path = os.path.join(view_dir, "rgb", "video.mp4")
        video_frames = load_video_frames(video_path)
        all_video_frames[view_name] = video_frames

    print(f"Loaded {len(all_video_frames)} views for action visualization")

    # Visualize actions on video
    visualize_action_on_video(
        all_video_frames=all_video_frames,
        action_3d=action_3d,
        all_intrinsics=all_intrinsics,
        all_extrinsics=all_extrinsics,
        stride=args.stride,
        output_path="tmp/rlbench_action_output.mp4",
        fps=args.fps,
    )

    print("\n=== Camera Pose Visualization ===")
    # Load and visualize cameras
    all_cameras = load_rlbench_cameras(episode_path, args.stride, args.max_frames)

    # Visualize all cameras
    visualize_cameras(all_cameras, args)


if __name__ == "__main__":
    args = get_args()

    print("=== RLBench Visualization: Camera Poses & Actions ===")
    visualize_actions_and_cameras(args.episode_path, args)

    print("Visualization completed!")
