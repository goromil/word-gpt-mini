"""
Configuration management for wordgpt.

Installs config to ~/.config/wordgpt/ with platform-appropriate paths.
"""

import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path


def _is_wsl() -> bool:
    """Detect if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower() or "microsoft" in subprocess.getoutput("uname -r").lower()
    except Exception:
        return False


def _is_windows() -> bool:
    """Detect if running on Windows."""
    return os.name == "nt"


def get_platform() -> str:
    """Return 'wsl', 'windows', or 'linux'."""
    if _is_windows():
        return "windows"
    if _is_wsl():
        return "wsl"
    return "linux"


def get_config_dir() -> Path:
    """Return ~/.config/wordgpt directory (XDG convention)."""
    home = Path.home()
    cfg = home / ".config" / "wordgpt"
    return cfg


def get_config_dir_windows() -> Path:
    """Return Windows equivalent: %USERPROFILE%\\.config\\wordgpt."""
    home = Path.home()
    return home / ".config" / "wordgpt"


def get_default_config_path() -> Path:
    """Return the default config file path for the current platform."""
    return get_config_dir() / "gpt_train.json"


def convert_wsl_to_windows(wsl_path: str) -> str:
    """Convert /mnt/X/... to X:\\... for Windows config."""
    p = Path(wsl_path)
    if str(p).startswith("/mnt/"):
        parts = str(p).split("/")
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}"
    return wsl_path


def convert_windows_to_wsl(win_path: str, drive_map: dict = None) -> str:
    """Convert X:\\... to /mnt/x/... for WSL config."""
    m = re.match(r"^([A-Za-z]):\\(.*)$", win_path)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return win_path


def _get_template_path() -> Path:
    """Find the template config file from the package data."""
    pkg_dir = Path(__file__).resolve().parent
    root = pkg_dir.parent.parent
    if (root / "gpt_train.json.tmpl").exists():
        return root / "gpt_train.json.tmpl"
    if (root / "gpt_train_draft.json.tmpl").exists():
        return root / "gpt_train_draft.json.tmpl"
    return None


def _apply_platform_paths(cfg: dict, platform: str) -> dict:
    """Adjust paths in config dict for the target platform."""
    paths = cfg.get("paths", {})
    if not paths:
        return cfg

    if platform == "windows":
        for k, v in paths.items():
            if isinstance(v, str) and v.startswith("/mnt/"):
                paths[k] = convert_wsl_to_windows(v)
    elif platform in ("wsl", "linux"):
        for k, v in paths.items():
            if isinstance(v, str) and re.match(r"^[A-Za-z]:\\", v):
                paths[k] = convert_windows_to_wsl(v)

    cfg["paths"] = paths
    return cfg


def ensure_config(force: bool = False, platform: str = None) -> Path:
    """
    Ensure config directory and file exist.
    If ~/.config/wordgpt/gpt_train.json doesn't exist, copy from template.
    Adjust paths for current platform.
    Returns the config file path.
    """
    cfg_dir = get_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "gpt_train.json"

    if not cfg_file.exists() or force:
        tmpl = _get_template_path()
        if tmpl:
            plat = platform or get_platform()
            with open(tmpl, "r") as f:
                cfg = json.load(f)
            cfg = _apply_platform_paths(cfg, plat)
            with open(cfg_file, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"  Created: {cfg_file}")
        else:
            if not cfg_file.exists():
                with open(cfg_file, "w") as f:
                    json.dump({"model": {}, "training": {}, "tokenizer": {}, "paths": {}}, f, indent=2)
                print(f"  Created empty: {cfg_file}")

    return cfg_file


def _cli_setup():
    """CLI entry point: gpt_setup_config [--force]"""
    import argparse
    parser = argparse.ArgumentParser(description="Setup wordgpt config in ~/.config/wordgpt/")
    parser.add_argument("--force", action="store_true", help="Recreate config from template")
    parser.add_argument("--show", action="store_true", help="Just print config location and exit")
    parser.add_argument("--platform", choices=["wsl", "windows"],
                        help="Force platform for path conversion (default: auto-detect)")
    args = parser.parse_args()

    cfg_dir = get_config_dir()
    cfg_file = cfg_dir / "gpt_train.json"

    print("=" * 60)
    print("  wordgpt Configuration")
    print("=" * 60)
    platform = args.platform or get_platform()
    print(f"  Platform:        {platform}")
    print(f"  Config directory: {cfg_dir}")
    print(f"  Config file:     {cfg_file}")
    print(f"  Log directory:   {cfg_dir / 'logs'}")
    print("=" * 60)
    print()

    if args.show:
        return

    cfg_path = ensure_config(force=args.force, platform=platform)

    if cfg_path.exists():
        print(f"\n  Config ready at: {cfg_path}")
        print(f"  Edit paths in this file to point to your data/checkpoint directories.")
        print()
        print(f"  Quick start:")
        if platform in ("wsl", "linux"):
            print(f"    source ~/miniconda3/etc/profile.d/conda.sh")
            print(f"    conda activate ai")
            print(f"    pip install -e /path/to/word-gpt-mini")
            print(f"    gpt_nipc_train {cfg_path}")
        else:
            print(f"    conda activate ai")
            print(f"    pip install -e C:\\path\\to\\word-gpt-mini")
            print(f"    gpt_nipc_train {cfg_path}")
    else:
        print(f"\n  WARNING: No template found. Created empty config at {cfg_path}")
        print(f"  Copy your config there and edit paths.")


if __name__ == "__main__":
    _cli_setup()
