"""Main module."""
from PIL import Image
from typing import List, Optional
import numpy as np
import numpy.typing import NDArray

#This defines the API for the python package

'''Data dimensions:
flim_data_stack: (width, height, n_channels, n_frames, n_nanotimes)
intensity_data_stack: (width,height,n_frames)
'''



def get_intensity_stack(
    flim_data_stack: NDArray[np.uint8],
    channel: int
) -> NDArray[np.uint8]:
    """Converts a FLIM data stack (output of pqreader) to a stack of intensity frames for a single channel
    Arguments:
        flim_data_stack: A numpy array defining the FLIM data stack to be corrected (loaded from pqreader module). 
        channel: Channel to use for intensity
    Returns:
        intensity_stack: Array of dimension (n_frames,width,height)
    """

    NotImplemented
    #return intensity_stack

def calculate_correction(
        intensity_data_stack: NDArray[np.uint8],
        reference_frame: int = 0,
        local_algorithm: str | None = None, 
        local_params: dict | None = None,
        global_algorithm: str | None = None,
        global_params: dict | None = None,
        ) -> NDArray[np.float64]:
    """Calculates motion correction for given intensity data stack based on chosen algorithms
    Arguments:
        intensity_data_stack: A numpy array defining the intensity frames to use for correction (output of get_intensity_stack)
        reference_frame: The reference frame to use for 
        local_algorithm: Optional, Name of local correction algorithm to apply, valid options include XXX 
        local_params: Optional, A dictionary defining parameters to apply to the local algorithm, refer to docs
        global_algorithm: Optional, Name of global correction algorithm to apply, valid options include XXX 
        global_params: Optional, A dictionary defining parameters to use with global correction algorithm
    Returns:
        transform_matrix: Array of dimension (2, width, height, frames) defining the vertical and horizontal displacements to be applied to each pixel of each frame
    """

    NotImplemented

    #return transform_matrix


def apply_correction_intensity(
    intensity_data_stack: NDArray[np.uint8],
    transform_matrix: NDArray[np.float64]
):
    """Applies a transformation matrix to an intensity data stack
    Arguments:
        intensity_data_stack: A numpy array defining the intensity frames to use for correction (output of get_intensity_stack)
        transform_matrix: Array of dimension (2, width, height, frames) defining the vertical and horizontal displacements to be applied to each pixel of each frame
    Returns:
        corrected_intensity_data_stack: A numpy array of same shape as intensity_data_stack
        metrics: A dictionary of metrics relating to the correction of each frame. Each value in the dictionary is a 1d numpy array of length n_frames.
    """

    NotImplemented
    #return transformed_intensity_data_stack, metrics


def apply_correction_flim(
    flim_data_stack: NDArray[np.uint8],
    transform_matrix: NDArray[np.float64],
    exclude: list[int] | None = None
):
    """Applies a transformation matrix to a flim data stack
    Arguments:
        flim_data_stack: A numpy array defining the FLIM data stack to be corrected (loaded from pqreader module). 
        transform_matrix: Array of dimension (2, width, height, frames) defining the vertical and horizontal displacements to be applied to each pixel of each frame
        exclude: Optional, list of frames to exclude from final output
    Returns:
        corrected_flim_data_stack: A numpy array of same shape as flim_data_stack
    """

    final = np.zeros_like(flim_data_stack)

    for f in range(final.shape[2]):  # frames
        for idx in np.ndindex(final.shape[:2]):
            dest_idx = (clip(idx[0] + round(transform_matrix[0, *idx, f]), 0, final.shape[0]-1),
                        clip(idx[1] + round(transform_matrix[1, *idx, f]), 0, final.shape[1]-1))
            final[*dest_idx, f] += flim_data_stack[*idx, 0]
    if exclude:
        final = np.delete(final,exclude,axis=3)
    corrected_flim_data_stack = final
    return corrected_flim_data_stack
    #return transformed_flim_data_stack


def get_aggregated_intensity_image(
    intensity_data_stack: NDArray[np.uint8],
    exclude: list[int] | None = None
) -> NDArray[np.uint8]:
    """Converts an intensity stack to a single intensity image representing the sum of all frames in the stack
    Arguments:
        intensity_data_stack: A numpy array defining the intensity frames
        exclude: Optional, frames to exclude from aggregation
    Returns:
        intensity_frame: A single frame, shape (width,height)
    """

    NotImplemented

    #return intensity_frame



def clip(x, x_min, x_max):
    return min(x_max, max(x, x_min))
