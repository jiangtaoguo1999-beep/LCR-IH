from dataclasses import dataclass
import torch


@dataclass
class Args:
    """Compatibility alias for the COCO evaluation config module."""

    from config_finetune_dvi2k import Args as _BaseArgs
    locals().update(_BaseArgs.__dict__)
