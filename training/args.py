import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Extended TrainingArguments for ActionImages training with additional custom parameters.
    """

    dataset_path: str = field(default="./data", metadata={"help": "The path of the Dataset."})
    dataset_name: str = field(
        default="rlbench",
        metadata={
            "help": "Dataset name(s): rlbench, bridge, or droid. "
            "Use name@ratio for mixed sampling (e.g. rlbench@0.5,bridge@0.3,droid@0.2); "
            "a single name without @ defaults to ratio 1.0."
        },
    )
    output_dir: str = field(
        default="./checkpoints",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of subprocesses to use for data loading."})

    # Model paths
    model_id: str = field(default="Wan-AI/Wan2.2-TI2V-5B", metadata={"help": "Path of model id."})

    # VAE tiling options
    tiled: bool = field(
        default=False, metadata={"help": "Whether enable tile encode in VAE. This option can reduce VRAM required."}
    )
    tile_size_height: int = field(default=34, metadata={"help": "Tile size (height) in VAE."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size (width) in VAE."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride (height) in VAE."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride (width) in VAE."})

    # Training configuration
    steps_per_epoch: int = field(default=500, metadata={"help": "Number of steps per epoch."})
    num_frames: int = field(default=41, metadata={"help": "Number of frames."})
    height: int = field(default=512, metadata={"help": "Image height."})
    width: int = field(default=512, metadata={"help": "Image width."})
    full_param: bool = field(default=False, metadata={"help": "Whether to train all parameters."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the backbone model."})

    # Legacy argument mappings (for backward compatibility)
    accumulate_grad_batches: Optional[int] = field(
        default=None,
        metadata={
            "help": "Legacy: The number of batches in gradient accumulation. Maps to gradient_accumulation_steps."
        },
    )
    max_epochs: Optional[int] = field(
        default=None, metadata={"help": "Legacy: Number of epochs. Maps to num_train_epochs."}
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "Legacy: Path to save the model. Maps to output_dir."}
    )

    # Gradient checkpointing
    use_gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing."})
    use_gradient_checkpointing_offload: bool = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing offload."}
    )

    # Checkpoint configuration
    metadata_file_name: str = field(default="metadata.csv", metadata={"help": "Name of the metadata file."})
    resume_ckpt_path: Optional[str] = field(default=None, metadata={"help": "Path to resume checkpoint."})
    init_ckpt_path: Optional[str] = field(default=None, metadata={"help": "Path to initialize checkpoint."})
    checkpoint_every_n_steps: int = field(default=1000, metadata={"help": "Save checkpoint every N training steps."})
    checkpoint_save_top_k: int = field(
        default=-1, metadata={"help": "Number of best checkpoints to keep. -1 saves all checkpoints."}
    )
    checkpoint_monitor: str = field(
        default="train_loss", metadata={"help": "Metric to monitor for best checkpoint selection."}
    )

    def __post_init__(self):
        # Handle legacy argument mappings
        if self.accumulate_grad_batches is not None:
            self.gradient_accumulation_steps = self.accumulate_grad_batches
        if self.max_epochs is not None:
            self.num_train_epochs = self.max_epochs
        if self.output_path is not None:
            self.output_dir = self.output_path

        # Call parent __post_init__
        super().__post_init__()


def parse_args():
    """Parse arguments using HfArgumentParser with dataclass"""
    parser = transformers.HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()
    return training_args
