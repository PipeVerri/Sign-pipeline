import argparse
from pathlib import Path
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path.cwd() / "working")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return args.workdir, config
