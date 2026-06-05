from __future__ import annotations

import os
import sys

from . import image_gs_implementation
from .streamlit_app.image_loader import load


def main() -> int:
    if len(sys.argv) > 1:
        paths = [os.path.abspath(p) for p in sys.argv[1:] if os.path.isfile(p)]
        if not paths:
            print("No valid image paths provided.")
            return 1

        for p in paths:
            arr = load(p)
            results = image_gs_implementation.process(arr)
            print(f"{os.path.basename(p)} → {len(results)} output(s)")
            for i, r in enumerate(results):
                h, w = r.shape[:2]
                print(f"  [{i + 1}] {w}×{h} {r.dtype}")
        return 0
    else:
        from streamlit.web import cli

        sys.argv = [
            "streamlit",
            "run",
            os.path.join(os.path.dirname(__file__), "streamlit_app/app.py"),
        ]
        return cli.main()


if __name__ == "__main__":
    sys.exit(main())
