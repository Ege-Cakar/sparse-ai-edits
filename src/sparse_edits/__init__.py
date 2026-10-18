import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv


def cli(description, add_args=None):
    """Load .env into the environment, parse --config plus step-specific args; returns (config dict, args)."""
    load_dotenv()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    if add_args:
        add_args(p)
    args = p.parse_args()
    with open(args.config) as f:
        return yaml.safe_load(f), args


def path(cfg, *parts):
    """Path under data_dir; creates the parent directory."""
    p = Path(cfg["data_dir"], *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
