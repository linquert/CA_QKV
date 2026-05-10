from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo root or Colab cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.trainer import Trainer
from src.utils import load_yaml, ensure_dir, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if "output_dir" not in config:
        config["output_dir"] = os.path.join(ROOT, "outputs", config.get("experiment_name", "run"))
    ensure_dir(config["output_dir"])
    save_json(config, os.path.join(config["output_dir"], "resolved_config.json"))

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
