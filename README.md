<p align="center">
  <h1 align="center">Action Images: End-to-End Policy Learning via Multiview Video Generation</h1>
  <p align="center">
    arXiv 2026
  </p>
  <p align="center">
    <a href="https://haoyuzhen.com">Haoyu Zhen</a><sup>*</sup>,
    <a href="https://zixiangao.github.io/">Zixian Gao</a><sup>*</sup>,
    <a href="https://qiaosun22.github.io/">Qiao Sun</a>,
    <a href="https://ermu2001.github.io/me.io/">Yilin Zhao</a>,
    <a href="https://yyuncong.github.io/">Yuncong Yang</a>,
    <a href="https://yilundu.github.io/">Yilun Du</a>,
    <a href="https://psguo.github.io/">Pengsheng Guo</a>,
    <a href="https://zswang666.github.io/">Tsun-Hsuan Wang</a>,
    <a href="https://ylqiao.net/">Yi-Ling Qiao</a>,
    <a href="https://people.csail.mit.edu/ganchuang/">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2604.06168">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://actionimages.github.io' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/anyeZHY/ActionImages' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=flat&logo=Hugging%20face&logoColor=yellow' alt='Model Hugging Face'>
    </a>
    <a href='https://huggingface.co/datasets/anyeZHY/ActionImages-RLBench' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Dataset-RLBench-orange?style=flat&logo=Hugging%20face&logoColor=orange' alt='Dataset RLBench'>
    </a>
  </p>
</p>

We propose **Action Images**, an end-to-end framework for robotic policy learning that takes multi-view images and text instructions to jointly generate RGB videos and action trajectories, enabling direct policy learning through multiview video generation.

<p align="center">
    <img src="asset/teaser.png" alt="Logo" width="190%">
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#news">News</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#training">Training</a>
        <ul>
          <li>
            <a href="#pre-training-or-full-fine-tuning">Pre-training or Full Fine-tuning</a>
          </li>
          <li>
            <a href="#configuration">Configuration</a>
          </li>
        </ul>
    </li>
    <li>
      <a href="#inference">Inference</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## News
- [2026-05-26] We have released the training and inference code, along with the [model checkpoint](https://huggingface.co/anyeZHY/ActionImages) and [RLBench dataset](https://huggingface.co/datasets/anyeZHY/ActionImages-RLBench) on Hugging Face!
- [2026-04-06] Action Images is on [arXiv](https://arxiv.org/abs/2604.06168)!
- [2026-04-06] Check out our [project website](https://actionimages.github.io) for more demos and results.

## Installation
Create a conda environment and install the required packages:
```bash
conda create -n actionimages python=3.11
conda activate actionimages

git clone https://github.com/UMass-Embodied-AGI/ActionImages.git
cd ActionImages
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

## Data Preparation

Action Images supports multi-view robotic datasets including RLBench, Bridge, and DROID.

### RLBench

Download the processed RLBench data from [anyeZHY/ActionImages-RLBench](https://huggingface.co/datasets/anyeZHY/ActionImages-RLBench) into `./data/rlbench`, unzip every `.tar.gz` in that folder, then delete the archives. Example:

```bash
mkdir -p ./data/rlbench
hf download anyeZHY/ActionImages-RLBench --repo-type dataset --local-dir ./data/rlbench
```

To preview raw RLBench samples, run `python vis/vis_rlbench.py`. To add a custom dataset, subclass `BaseDataset` in [`training/dataset/base.py`](training/dataset/base.py). Before training, you can sanity-check the dataloader with:

```bash
python training/dataset/test_dataset.py --dataset rlbench --backend torch  # or numpy
```

### Bridge

> **TODO:** Release Bridge preprocessing script to convert raw Bridge data into the layout expected by [`BridgeMVDataset`](training/dataset/bridge.py).

## Training

### Pre-training or Full Fine-tuning

The training code supports distributed training with multiple GPUs via DeepSpeed ZeRO. Wan backbone weights are downloaded automatically on first run.

To train Action Images, run:
```bash
bash scripts/train.sh <num_gpus>
```

To fine-tune from a released checkpoint, download [anyeZHY/ActionImages](https://huggingface.co/anyeZHY/ActionImages) and add `--init_ckpt_path` or `--resume_ckpt_path` in [`scripts/train.sh`](scripts/train.sh):
```bash
hf download anyeZHY/ActionImages --local-dir ./checkpoints/ActionImages
torchrun ... train.py --init_ckpt_path ./checkpoints/ActionImages/checkpoint.ckpt ...
```

### Configuration

Key training arguments (see [`training/args.py`](training/args.py) for the full list):

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `rlbench` | Dataset: `rlbench`, `bridge`, or `droid` |
| `--num_frames` | `41` | Number of video frames per sample |
| `--height` / `--width` | `512` | Output resolution |

Multi-dataset co-training is supported via `--dataset_name` with per-dataset sampling ratios (`name@ratio`), e.g. `rlbench@0.5,bridge@0.3,droid@0.2`. A single dataset name defaults to `@1.0`.

Edit [`scripts/train.sh`](scripts/train.sh) to modify learning rate, batch size, checkpoint frequency, and W&B logging.

## Inference

Run multi-view inference with two input images and a text prompt. Outputs are saved under `results/` by default.

**Image-to-video-action (`i2va`)** — joint RGB video and action generation:
```bash
torchrun --nproc_per_node=8 inference.py \
  --images asset/xarm-left.jpg asset/xarm-right.jpg \
  --ckpt_path anyeZHY/ActionImages \
  --prompt "place the black cup in the blue bowl" \
  --task_type i2va \
  --use_usp \
  --num_inference_steps 50 \
  --cfg_parallel \
  --torch_compile \
  --view1_action 350 130 350 120 350 80 1 \
  --view2_action 325 190 375 180 325 100 1
```

**`--view1_action` / `--view2_action` format** (7 values per view, matching the RGB action image channels):

| Index | Name | Description |
|-------|------|-------------|
| 0–1 | `red x`, `red y` | Gripper position (R channel) |
| 2–3 | `green x`, `green y` | Gripper orientation / normal direction (G channel) |
| 4–5 | `blue x`, `blue y` | Gripper up direction (B channel) |
| 6 | `openness` | `1` = open, `0` = grasp |

- Pixel coordinates use the **top-left corner** of the image as origin `(0, 0)`, with `x` rightward and `y` downward.
- Provide **7 values** (same action repeated for all frames) or 7 × num_frames values (per-frame trajectory).

Optional flags:
- `--use_usp`: Unified Sequence Parallel for multi-GPU inference
- `--cfg_parallel`: Split CFG branches across GPUs
- `--dynamic_cache_schedule`: Faster inference via cache scheduling
- `--torch_compile`: Enable `torch.compile` for speedup
- `--task_type`: `i2v` (video only) or `i2va` (video + action)

> [!NOTE]
> Inference uses [VGGT](https://github.com/facebookresearch/vggt) to estimate camera poses from the two input images. The model weights are downloaded automatically on first run.

> **TODO:** Release Blender rendering script at [`inference/render_blender.py`](inference/render_blender.py) to visualize predicted actions / point clouds in a 3D scene.

## Citation
If you find our work useful, please consider citing:
```bibtex
@article{zhen2026actionimages,
  title={Action Images: End-to-End Policy Learning via Multiview Video Generation},
  author={Haoyu Zhen and Zixian Gao and Qiao Sun and Yilin Zhao and Yuncong Yang and Yilun Du and Pengsheng Guo and Tsun-Hsuan Wang and Yi-Ling Qiao and Chuang Gan},
  year={2026},
  eprint={2604.06168},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.06168},
}
```

## Acknowledgement
We would like to thank the following works for their code and models:
- Video generation: [Wan](https://github.com/Wan-Video/Wan2.1), [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster), [DiffSynth](https://github.com/modelscope/DiffSynth-Studio) and [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
- Camera estimation: [VGGT](https://github.com/facebookresearch/vggt)
- Datasets: [RLBench](https://github.com/stepjam/RLBench), [Bridge](https://rail-berkeley.github.io/) and [DROID](https://droid-dataset.github.io/)
