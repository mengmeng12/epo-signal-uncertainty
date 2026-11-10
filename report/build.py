#!/usr/bin/env python3
"""Build the collaborative paper from repository result files."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Reuse previously generated figures and tables.",
    )
    args = parser.parse_args()
    env = os.environ.copy()
    tex = Path("/Library/TeX/texbin")
    if tex.exists():
        env["PATH"] = str(tex) + os.pathsep + env.get("PATH", "")
    if not shutil.which("latexmk", path=env["PATH"]):
        raise SystemExit(
            "Install a TeX distribution with latexmk (for example MacTeX or TeX Live)."
        )
    if not args.skip_assets:
        subprocess.run(
            [sys.executable, str(HERE / "scripts/generate_assets.py")],
            cwd=HERE,
            env=env,
            check=True,
        )
    (HERE / "build").mkdir(exist_ok=True)
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=build",
            "main.tex",
        ],
        cwd=HERE,
        env=env,
        check=True,
    )
    print(f'Paper: {HERE / "build/main.pdf"}')


if __name__ == "__main__":
    main()
