import importlib
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch_xla.distributed.xla_multiprocessing as xmp


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "PyTorch TPU distributed training launch "
            "helper utility that will spawn up "
            "multiple distributed processes"
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--num_cores", type=int, default=1, help="Number of TPU cores to use (1 or 8)."
    )

    return parser.parse_known_args()


def main():
    args, unknown = parse_args()

    script_fpath = str(Path(__file__).parent.resolve())
    sys.path.append(script_fpath)
    mod_name = "run_clm"
    mod = importlib.import_module(mod_name)

    # Patch sys.argv
    sys.argv = (
        [f"{script_fpath}/{mod_name}.py"]
        + unknown
        + ["--tpu_num_cores", str(args.num_cores)]
    )

    xmp.spawn(mod._mp_fn, args=(), nprocs=args.num_cores)


if __name__ == "__main__":
    main()
