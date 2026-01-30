import argparse
import logging

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
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["info", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Create parquet files from root files
    for run in args.runs:
        logging.info(f"Processing run {run}.")
        if args.chunks is None:
            run_path = get_root_path() / f"{run:05d}"
            files = list(run_path.glob(f"{run:05d}_*.root"))
            chunks = [int(f.stem.split("_")[-1]) for f in files]
        else:
            chunks = args.chunks
        for chunk in chunks:
            logging.info(f"Processing chunk {chunk}.")
            logging.info("Creating parquet files.")
            create_parquet_from_root(run=run, chunk=chunk, recreate=args.recreate)
            if args.pixel_size:
                logging.info("Creating rebinned parquet files.")
                create_rebinned_parquet_file(
                    run=run,
                    chunk=chunk,
                    pixel_size=args.pixel_size,
                    recreate=args.recreate,
                )


if __name__ == "__main__":
    main()
