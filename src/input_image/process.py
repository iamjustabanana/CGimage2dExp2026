from __future__ import annotations

import numpy as np


def process_batch(images: list[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for img in images:
        result.extend([img.copy() for _ in range(10)])
    return result
