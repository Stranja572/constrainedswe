import os
import shutil
import sys
import urllib.request
from argparse import ArgumentParser
from functools import partial, wraps
from pathlib import Path
import tarfile 

from data import get_task_dir
from utils import config_logger, get_logger

logg = config_logger(
    None, "%(asctime)s [%(levelname)s] %(message)s", level=2, use_stdout=True
)

def get_remote_path(bm: str):
    REMOTE_DATA_PATHS = {
        "scl": "https://zenodo.org/records/10631963/files",
    }
    return REMOTE_DATA_PATHS[bm]

def add_args(parser: ArgumentParser):
    parser.add_argument(
        "--to",
        type=str,
        required=True,
        help="Location to download the data to. If the location does not exist, it will be created.",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=[
            "scl",
        ],
        help="Benchmarks to download.",
    )

    return parser

def download_safe(
    remote_path: str, local_path: str, key: str = "file", verbose: bool = True
) -> str:
    if not os.path.exists(local_path):
        try:
            if verbose:
                logg.info(f"Downloading {key} from {remote_path} to {local_path}...")
            with urllib.request.urlopen(remote_path) as response, open(
                local_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            logg.error(f"Unable to download {key} - {e}")
            sys.exit(1)
    return local_path


def main(args):
    args.to = Path(args.to).absolute()
    logg.info(f"Download Location: {args.to}")
    logg.info("")

    logg.info("[BENCHMARKS]")
    benchmarks = args.benchmarks or []
    for bm in benchmarks:
        logg.info(f"Downloading {bm}...")
        task_dir = Path(get_task_dir(bm, database_root=args.to))
        os.makedirs(task_dir, exist_ok=True)

        if bm == "scl":
            fi_list = {
                "data.tar.gz": "data.tar.gz",
            }
        else:
            raise Exception

        for fi in fi_list:
            local_path = task_dir / fi_list[fi]
            remote_base = get_remote_path(bm)
            remote_path = f"{remote_base}/{fi}"
            download_safe(remote_path, local_path, key=f"{bm}/{fi}")

        if bm == "scl":
            file = tarfile.open(local_path)
            file.extractall(task_dir)
            file.close()
            os.replace(f"{task_dir}/data/annotation/scl/balanced.csv", f"{task_dir}/balanced.csv")
            os.remove(local_path)
            shutil.rmtree(f"{task_dir}/data/")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
