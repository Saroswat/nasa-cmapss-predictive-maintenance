from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .data import add_train_rul, download_dataset, load_fd001
from .modeling import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmapss-maintenance",
        description="NASA C-MAPSS remaining-life and maintenance modelling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download NASA C-MAPSS data")
    download.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Run the FD001 experiment")
    run.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    run.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    run.add_argument("--estimators", type=int, default=300)
    run.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        location = download_dataset(args.data_dir, force=args.force)
        print(f"NASA C-MAPSS data ready in {location.resolve()}")
        return

    if not args.skip_download:
        download_dataset(args.data_dir)
    train, test, truth = load_fd001(args.data_dir)
    config = ExperimentConfig(n_estimators=args.estimators)
    train = add_train_rul(train, cap=config.rul_cap)
    result = run_experiment(train, test, truth, config)
    from .reporting import save_artifacts

    save_artifacts(result, args.output_dir)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
