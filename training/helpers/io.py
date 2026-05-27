import random
from typing import Callable, List, Optional, Tuple

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image


def load_video_frames(
    video_path: str,
    frame_indices: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
    frame_interval: int = 1,
    frame_process: Callable = lambda x: x,
) -> Tuple[torch.Tensor, List[int], Tuple[int, int]]:
    """Load frames from video file."""
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    W, H = reader.get_meta_data()["size"]
    if max_frames is None:
        max_frames = total_frames

    if frame_indices is not None:
        frame_indices = frame_indices[:max_frames]
    elif total_frames <= max_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (max_frames - total_frames)
    else:
        start_idx = random.randint(0, max(0, total_frames - max_frames * frame_interval))
        frame_indices = [min(start_idx + i * frame_interval, total_frames - 1) for i in range(max_frames)]

    frames = []
    for idx in frame_indices:
        frame = reader.get_data(idx)
        frame = Image.fromarray(frame)
        frame = frame_process(frame)
        frames.append(frame)

    reader.close()
    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "T C H W -> C T H W")
    return frames, frame_indices, (H, W)


def center_crop_square(image: Image.Image) -> Image.Image:
    if image.width > image.height:
        left = (image.width - image.height) // 2
        return image.crop((left, 0, left + image.height, image.height))
    top = (image.height - image.width) // 2
    return image.crop((0, top, image.width, top + image.width))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = torch.Tensor(np.array(image, dtype=np.float32))
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def load_images_to_square(image_paths: List[str], task_type: str, num_frames: int):
    images = []
    video_segments = []
    for image_path in image_paths:
        if image_path.endswith(".mp4"):
            reader = imageio.get_reader(image_path)
            total_frames = reader.count_frames()
            first_frame = reader.get_data(0)
            first_frame = pil_to_tensor(center_crop_square(Image.fromarray(first_frame)))

            segment = None
            if task_type == "v2a":
                frames = []
                for idx in range(num_frames):
                    frame_idx = min(idx, total_frames - 1)
                    frame = reader.get_data(frame_idx)
                    frame = pil_to_tensor(center_crop_square(Image.fromarray(frame)))
                    frames.append(frame)
                segment = torch.stack(frames, dim=0)
            reader.close()
            image = first_frame
            video_segments.append(segment)
        else:
            image = Image.open(image_path)
            image = pil_to_tensor(center_crop_square(image))
            video_segments.append(None)
        images.append(image)

    max_size = max(image.shape[0] for image in images)
    max_size = max(max_size, max(image.shape[1] for image in images))

    processed_images = []
    for image in images:
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        if image.shape[2] < max_size or image.shape[3] < max_size:
            image = torch.nn.functional.interpolate(
                image, size=(max_size, max_size), mode="bilinear", align_corners=False
            )
        image = image.squeeze(0)
        processed_images.append(image)

    processed_video_segments = []
    for segment in video_segments:
        if segment is None:
            processed_video_segments.append(None)
            continue
        segment = segment.permute(0, 3, 1, 2)
        if segment.shape[2] < max_size or segment.shape[3] < max_size:
            segment = torch.nn.functional.interpolate(
                segment, size=(max_size, max_size), mode="bilinear", align_corners=False
            )
        processed_video_segments.append(segment)

    if task_type == "v2a" and all(segment is not None for segment in processed_video_segments):
        processed_video_segments = torch.stack(processed_video_segments, dim=0)
    else:
        processed_video_segments = None

    return torch.stack(processed_images, dim=0), processed_video_segments


def combine_action_video(pred_video, get_max=False):
    """
    Create an action-combined video from input frames by:
    - Splitting time into 4 equal folds (truncate remainder)
    - For folds (1,2) and (3,4), horizontally concatenating [first | second | 0.5*first+0.5*second]
    - Concatenating the two results along time

    Args:
        pred_video: Iterable of PIL.Image frames

    Returns:
        comb_video: List of PIL.Image frames after combination
    """
    pred_np = np.stack([np.array(frame) for frame in pred_video], axis=0)
    total_frames = pred_np.shape[0]
    usable_frames = (total_frames // 4) * 4
    if usable_frames == 0:
        return list(pred_video)

    pred_np = pred_np[:usable_frames]
    fold_size = usable_frames // 4
    first = pred_np[0:fold_size]
    second = pred_np[fold_size : 2 * fold_size]
    third = pred_np[2 * fold_size : 3 * fold_size]
    forth = pred_np[3 * fold_size : 4 * fold_size]

    if get_max:
        second[..., :2] = second[..., :2] * (second[..., :2] == np.max(second[..., :2], axis=0, keepdims=True))
        forth[..., :2] = forth[..., :2] * (forth[..., :2] == np.max(forth[..., :2], axis=0, keepdims=True))

    blend_12 = ((first.astype(np.float32) * 0.5) + (second.astype(np.float32) * 0.5)).round().astype(np.uint8)
    blend_34 = ((third.astype(np.float32) * 0.5) + (forth.astype(np.float32) * 0.5)).round().astype(np.uint8)

    hcat_12 = np.concatenate([first, second, blend_12], axis=2)
    hcat_34 = np.concatenate([third, forth, blend_34], axis=2)

    comb_video = np.concatenate([hcat_12, hcat_34], axis=0)
    return comb_video


def combine_action_video_i2a(pred_video, seg_frame_counts):
    """
    Same layout idea as combine_action_video, but segment lengths follow inference decode
    (i2a: 1 + T_l + 1 + T_l latent segments -> unequal pixel frame counts).

    Args:
        pred_video: Iterable of PIL.Image frames (concatenated segment decode order).
        seg_frame_counts: Length-4 list of pixel T per segment [n0, n1, n2, n3].

    Returns:
        uint8 array [T_combined, H, W, 3] suitable for save_video / PIL.
    """
    pred_np = np.stack([np.array(frame) for frame in pred_video], axis=0)
    total = int(sum(seg_frame_counts))
    if pred_np.shape[0] < total:
        pred_np = pred_np[: pred_np.shape[0]]
        total = pred_np.shape[0]
    if total == 0 or len(seg_frame_counts) != 4:
        return pred_np

    chunks = []
    off = 0
    for n in seg_frame_counts:
        n = int(n)
        end = min(off + n, pred_np.shape[0])
        chunks.append(pred_np[off:end])
        off = end
    first, second, third, forth = chunks

    def _align_pair(a, b):
        L = max(a.shape[0], b.shape[0])
        if a.shape[0] < L:
            pad = np.repeat(a[-1:], L - a.shape[0], axis=0)
            a = np.concatenate([a, pad], axis=0)
        if b.shape[0] < L:
            pad = np.repeat(b[-1:], L - b.shape[0], axis=0)
            b = np.concatenate([b, pad], axis=0)
        return a[:L], b[:L]

    f1, s1 = _align_pair(first, second)
    f2, s2 = _align_pair(third, forth)
    blend_12 = ((f1.astype(np.float32) * 0.5) + (s1.astype(np.float32) * 0.5)).round().astype(np.uint8)
    blend_34 = ((f2.astype(np.float32) * 0.5) + (s2.astype(np.float32) * 0.5)).round().astype(np.uint8)
    hcat_12 = np.concatenate([f1, s1, blend_12], axis=2)
    hcat_34 = np.concatenate([f2, s2, blend_34], axis=2)
    return np.concatenate([hcat_12, hcat_34], axis=0)


def concat_video(pred_video, split_num=2):
    fold_size = len(pred_video) // split_num
    comb_video = [pred_video[i * fold_size : (i + 1) * fold_size] for i in range(split_num)]
    comb_video = np.concatenate(comb_video, axis=2)
    return comb_video
