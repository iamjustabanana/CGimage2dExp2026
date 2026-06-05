from __future__ import annotations

import numpy as np

from .input_image.process import process_image as _process_image


def process(image: np.ndarray) -> list[np.ndarray]:
    result = _process_image(image)

    # mock result
    return [result.copy() for _ in range(10)]
