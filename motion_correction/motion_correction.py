"""Main module."""
from PIL import Image
from typing import List
import numpy as np
import numpy.typing import NDArray


def clip(x, x_min, x_max):
    return min(x_max, max(x, x_min))


def apply_transformation(original:NDArray, transform:NDArray):
 """Applies a transformation matrix to single channel flim array or intensity array
    Arguments:
        original: array(n_rows,n_cols,n_frames,n_nanotimes**)   **nanotimes optional
        transform: array(2, n_rows, n_cols, n_frames)  where the first dimension is (row offset, col offset) in pixels
    Returns:
        transformed array (same shape as original)
    """

    final = np.zeros_like(original)

    for f in range(final.shape[2]):  # frames

        for idx in np.ndindex(final.shape[:2]):
            dest_idx = (clip(idx[0] + round(transform[0, *idx, f]), 0, final.shape[0]-1),
                        clip(idx[1] + round(transform[1, *idx, f]), 0, final.shape[1]-1))
            final[*dest_idx, f] += original[*idx, 0]

    return final

