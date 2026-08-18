"""wordgpt - Word-level GPT training toolkit."""

__version__ = "0.1.0"

from wordgpt.config import (
    get_config_dir,
    get_default_config_path,
    get_platform,
    ensure_config,
    convert_wsl_to_windows,
    convert_windows_to_wsl,
)

__all__ = [
    "get_config_dir",
    "get_default_config_path",
    "get_platform",
    "ensure_config",
    "convert_wsl_to_windows",
    "convert_windows_to_wsl",
]
