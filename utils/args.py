import argparse
from pathlib import Path

from utils.config import PipelineConfig


def parse_args() -> tuple[Path, PipelineConfig]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path.cwd() / "working")
    args = parser.parse_args()
    return args.workdir, PipelineConfig.from_yaml(args.config)
