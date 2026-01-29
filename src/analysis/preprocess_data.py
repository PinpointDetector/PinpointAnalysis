import argparse

from analysis.utils.preprocessing_utils import (
    create_parquet_from_root,
    create_rebinned_parquet_file,
)
from analysis.utils.utils import get_root_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pixel-size",
        type=float,
        required=False,
        help="Rebinned pixel size in um.",
    )
    parser.add_argument(
        "-c",
        "--chunks",
        nargs="*",
        type=int,
        required=False,
    )
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    # Create parquet files from root files
    for run in args.runs:
        if len(args.chunks) == 0:
            run_path = get_root_path() / f"{run:05d}"
            files = list(run_path.glob(f"{run:05d}_*.root"))
            chunks = [int(f.stem.split("_")[-1]) for f in files]
        else:
            chunks = args.chunks
        for chunk in chunks:
            create_parquet_from_root(run=run, chunk=chunk, recreate=args.recreate)
            if args.pixel_size:
                create_rebinned_parquet_file(
                    run=run,
                    chunk=chunk,
                    pixel_size=args.pixel_size,
                    recreate=args.recreate,
                )
