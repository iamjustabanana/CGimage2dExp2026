from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1:
    # CLI mode: process local files directly, no Streamlit
    from src.streamlit_app.image_loader import load
    from src.input_image.process import process_batch

    paths = [os.path.abspath(p) for p in sys.argv[1:] if os.path.isfile(p)]
    if not paths:
        print("No valid image paths provided.")
        sys.exit(1)

    images = [load(p) for p in paths]
    print(f"Loaded {len(images)} image(s)")

    results = process_batch(images)
    print(f"Got {len(results)} output image(s)")

    for i, arr in enumerate(results):
        h, w = arr.shape[:2]
        print(f"  Output {i + 1}: {w}×{h} {arr.dtype}")
else:
    # Streamlit mode: launch the interactive UI
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "src/streamlit_app/app.py"),
    ]
    sys.exit(cli.main())
