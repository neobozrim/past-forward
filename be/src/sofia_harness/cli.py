from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from .runner import run
from .scans import discover_scans, inspect_scan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sofia newspaper OCR evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="discover and technically inspect source scans")
    inspect.add_argument("--config", default="config.yaml")
    execute = sub.add_parser("run", help="crop regions and run independent OCR reads")
    execute.add_argument("--config", default="config.yaml"); execute.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["_path"] = str(Path(args.config).resolve())
    if args.command == "inspect":
        scans = discover_scans(config["dataset"]["images"], config["dataset"].get("patterns", ["*.png"]))
        print(json.dumps({"scans_available":len(scans), "sample":[inspect_scan(p) for p in scans[:3]]}, indent=2))
    else:
        print(run(config, args.dry_run).resolve())


if __name__ == "__main__": main()
