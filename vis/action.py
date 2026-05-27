import os
import cv2
import numpy as np
import imageio
from scipy.spatial.transform import Rotation as R


def project_point(point_3d, intrinsics, world_to_cam):
    """Project a 3D point (in world coordinates) onto the 2D image plane."""
    point_3d_hom = np.append(point_3d, 1.0)
    cam_coords = world_to_cam @ point_3d_hom

    if cam_coords[2] <= 0:
        return None

    pixel = intrinsics @ cam_coords[:3]
    pixel = pixel[:2] / pixel[2]
    return tuple(pixel.astype(int))


def get_gradient_color(i, total):
    """Compute a vivid gradient color between blue and red for the i-th point among total."""
    ratio = i / (total - 1) if total > 1 else 0
    hue = int((1 - ratio) * 240)
    hsv_color = np.uint8([[[hue // 2, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr_color)


def _prepare_camera_transforms(all_extrinsics):
    """Prepare world-to-camera transformations for each view."""
    world_to_cams = {}
    for view_name, extrinsics in all_extrinsics.items():
        world_to_cams[view_name] = np.linalg.inv(extrinsics)
    return world_to_cams


def _project_action_to_views(action, action_idx, view_names, all_intrinsics, world_to_cams):
    """Project a 3D action to all camera views and return projection info."""
    pos = action[:3]
    quat = action[3:7]
    openness = action[7]

    projections = {}

    for view_name in view_names:
        intrinsics = all_intrinsics[view_name]
        world_to_cam = world_to_cams[view_name]

        frame_intrinsics = intrinsics[min(action_idx, len(intrinsics) - 1)]
        frame_world_to_cam = world_to_cam[min(action_idx, len(world_to_cam) - 1)]

        pos_pixel = project_point(pos, frame_intrinsics, frame_world_to_cam)
        if pos_pixel is not None:
            rot_matrix = R.from_quat(quat).as_matrix()
            direction = rot_matrix[:, 0]
            tip_pos = pos + 0.1 * direction
            tip_pixel = project_point(tip_pos, frame_intrinsics, frame_world_to_cam)
            projections[view_name] = (pos_pixel, tip_pixel, openness)

    return projections


def _draw_action_trajectory(frame, drawn_actions):
    """Draw trajectory lines connecting action positions."""
    if len(drawn_actions) >= 2:
        for i in range(len(drawn_actions) - 1):
            start_point = drawn_actions[i][1]
            end_point = drawn_actions[i + 1][1]
            color = get_gradient_color(i, len(drawn_actions))
            cv2.line(frame, start_point, end_point, color, thickness=2)


def _draw_action_arrows(frame, drawn_actions, color=(0, 0, 255), dot_color=(0, 255, 0)):
    """Draw rotation arrows and position dots for actions."""
    for _, pos_pixel, tip_pixel, _ in drawn_actions:
        if pos_pixel is not None and tip_pixel is not None:
            cv2.arrowedLine(frame, pos_pixel, tip_pixel, color, thickness=2, tipLength=0.3)
            cv2.circle(frame, pos_pixel, 3, dot_color, -1)


def _highlight_gripper_changes(frame, drawn_actions, f_idx):
    """Highlight actions with gripper openness changes."""
    if not drawn_actions:
        return

    last_action = drawn_actions[-1]
    if last_action[0] == f_idx:
        if len(drawn_actions) > 1:
            prev_openness = drawn_actions[-2][3]
            curr_openness = last_action[3]
            if abs(curr_openness - prev_openness) > 1e-3:
                cv2.circle(frame, last_action[1], 8, (0, 255, 255), thickness=2)
        else:
            cv2.circle(frame, last_action[1], 8, (0, 255, 255), thickness=2)


def _process_frame_for_view(frame, drawn_actions, view_name, f_idx, is_dual_hand=False):
    """Process a single frame for a specific view."""
    frame_copy = frame.copy()

    if is_dual_hand:
        # drawn_actions is a list of [left_hand_actions, right_hand_actions]
        left_actions, right_actions = drawn_actions

        # Draw left hand in blue/green
        _draw_action_trajectory(frame_copy, left_actions)
        _draw_action_arrows(frame_copy, left_actions, color=(255, 0, 0), dot_color=(0, 255, 0))  # Blue arrow, green dot
        _highlight_gripper_changes(frame_copy, left_actions, f_idx)

        # Draw right hand in red/cyan
        _draw_action_trajectory(frame_copy, right_actions)
        _draw_action_arrows(
            frame_copy, right_actions, color=(0, 0, 255), dot_color=(255, 255, 0)
        )  # Red arrow, cyan dot
        _highlight_gripper_changes(frame_copy, right_actions, f_idx)
    else:
        _draw_action_trajectory(frame_copy, drawn_actions)
        _draw_action_arrows(frame_copy, drawn_actions)
        _highlight_gripper_changes(frame_copy, drawn_actions, f_idx)

    cv2.putText(frame_copy, view_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame_copy


def visualize_action_on_video(
    all_video_frames, action_3d, all_intrinsics, all_extrinsics, stride=1, output_path="tmp/action_output.mp4", fps=20
):
    """
    Visualize robot actions on video frames from all views and save as MP4 video.

    Args:
        all_video_frames: Dict of video frames by view name
        action_3d: Action data - either [T, 8] for single hand or list of two [T, 8] arrays for dual hands
                   Format: [x, y, z, qx, qy, qz, qw, openness]
        all_intrinsics: Dict of camera intrinsics by view name
        all_extrinsics: Dict of camera extrinsics by view name
        stride: Stride for sampling frames
        output_path: Output path for the MP4 video
        fps: Frames per second for the video
    """
    # Check if we have dual hands
    is_dual_hand = isinstance(action_3d, list)

    if is_dual_hand:
        num_actions = action_3d[0].shape[0]
        print(f"Processing {num_actions} actions for BOTH HANDS across views")
    else:
        num_actions = action_3d.shape[0]
        print(f"Processing {num_actions} actions for single hand across views")

    view_names = sorted(all_video_frames.keys())
    num_video_frames = len(all_video_frames[view_names[0]])
    print(f"Total video frames: {num_video_frames}")

    world_to_cams = _prepare_camera_transforms(all_extrinsics)
    processed_frames = []

    if is_dual_hand:
        # For dual hands, maintain separate drawn actions for left and right
        drawn_actions_per_view = {view_name: [[], []] for view_name in view_names}  # [left, right]
    else:
        drawn_actions_per_view = {view_name: [] for view_name in view_names}

    for f_idx in range(0, num_video_frames, stride):
        action_idx = f_idx // stride

        if action_idx < num_actions:
            if is_dual_hand:
                # Project both hands
                for hand_idx in [0, 1]:  # left=0, right=1
                    action = action_3d[hand_idx][action_idx]
                    projections = _project_action_to_views(
                        action, action_idx, view_names, all_intrinsics, world_to_cams
                    )

                    for view_name, (pos_pixel, tip_pixel, openness) in projections.items():
                        drawn_actions_per_view[view_name][hand_idx].append((f_idx, pos_pixel, tip_pixel, openness))
            else:
                # Project single hand
                action = action_3d[action_idx]
                projections = _project_action_to_views(action, action_idx, view_names, all_intrinsics, world_to_cams)

                for view_name, (pos_pixel, tip_pixel, openness) in projections.items():
                    drawn_actions_per_view[view_name].append((f_idx, pos_pixel, tip_pixel, openness))

        view_frames = []
        for view_name in view_names:
            frame = all_video_frames[view_name][f_idx]
            drawn_actions = drawn_actions_per_view[view_name]
            processed_frame = _process_frame_for_view(frame, drawn_actions, view_name, f_idx, is_dual_hand)
            view_frames.append(processed_frame)

        concatenated_frame = np.hstack(view_frames)
        frame_rgb = cv2.cvtColor(concatenated_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame_rgb)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, processed_frames, fps=fps)
    print(f"MP4 video saved as {output_path}")
