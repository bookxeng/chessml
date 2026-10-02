"""Print what this machine offers for training: python -m chessml_train.env_check"""
from __future__ import annotations

import os
import platform
import sys


def main() -> None:
    import torch

    print(f"python   {sys.version.split()[0]} ({platform.system()})")
    print(f"cpus     {os.cpu_count()}")
    print(f"torch    {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"cuda     {torch.version.cuda}, {props.name}, {props.total_memory / 2**30:.1f} GiB")
        x = torch.randn(1024, 1024, device="cuda")
        print(f"matmul   ok ({(x @ x).sum().item():.1f})")
    else:
        print("cuda     not available (CPU only)")


if __name__ == "__main__":
    main()
