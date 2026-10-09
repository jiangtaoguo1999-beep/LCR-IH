from dataclasses import dataclass
import torch


@dataclass
class Args:
    """Compatibility alias for the finetune config module."""

    # Reuse the DIV2K config settings by default.
    from config_finetune3_6_dvi2k import Args as _BaseArgs

    # Import all attributes from the canonical configuration class.
    locals().update(_BaseArgs.__dict__)

    # Ensure the dataclass is valid: this alias intentionally does not define
    # a new custom schema, but acts as a runtime import shim.
