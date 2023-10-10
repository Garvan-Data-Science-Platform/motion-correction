"""Main module."""

from typing import TypedDict
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import torch
import skimage.metrics as skm
from .algorithms import align_morphic_cpu, align_phase, align_optical_ilk, align_optical_poly, align_optical_tvl1, align_morphic_gpu, flow_warp


# This defines the API for the python package

'''Data dimensions:
flim_data_stack: (width, height, n_channels, n_frames, n_nanotimes)
intensity_data_stack: (width,height,n_frames)
'''


class Metric(TypedDict):
    original: NDArray[np.float32]
    global_corrected: NDArray[np.float32]
    corrected: NDArray[np.float32]


class Metrics(TypedDict):
    ncc: Metric
    spm: Metric
    mse: Metric
    nrm: Metric
    ssi: Metric


class CorrectionResults(TypedDict):
    corrected_intensity_data_stack: NDArray[np.uint16]
    global_corrected_intensity_data_stack: NDArray[np.uint16]
    metrics: Metrics
    combined_transforms: NDArray[np.float32]
    local_transforms: NDArray[np.float32]
    global_transforms: NDArray[np.float32]


def get_intensity_stack(
    flim_data_stack: NDArray[np.uint8],
    channel: int
) -> NDArray[np.uint8]:
    """Converts a FLIM data stack (output of pqreader) to a stack of intensity frames for a single channel

    :param flim_data_stack: A numpy array defining the FLIM data stack to be corrected (loaded from pqreader module).
    :param channel: Channel to use for intensity
    :return intensity_stack: Array of dimension (width,height,n_frames)

    """

    intensity_stack = flim_data_stack[:, :, channel, :, :].sum(axis=-1)

    return intensity_stack


def calculate_correction(
        intensity_data_stack: NDArray[np.uint8],
        reference_frame: int = 0,
        local_algorithm: str | None = None,
        local_params: dict | None = None,
        global_algorithm: str | None = None,
        global_params: dict | None = None,
) -> CorrectionResults:
    """Calculates motion correction for given intensity data stack based on chosen algorithms

    :param intensity_data_stack: A numpy array defining the intensity frames to use for correction (output of get_intensity_stack)
    :param reference_frame: The reference frame to use
    :param local_algorithm: Optional, Name of local correction algorithm to apply, valid options include 'morphic', 'optical_tvl1', 'optical_poly', 'optical_ilk'
    :param local_params: Optional, A dictionary defining parameters to apply to the local algorithm, refer to docs
    :param global_algorithm: Optional, Name of global correction algorithm to apply, valid options include 'phase'
    :param global_params: Optional, A dictionary defining parameters to use with global correction algorithm

    :return CorrectionResults: dictionar with keys:
            - **global_corrected_intensity_data_stack**: original intensity data stack with global corrections applied
            - **corrected_intensity_data_stack**: original intensity data stack with : original intensity data stack with global and local corrections applied
            - **metrics**: dict with fields for each metric (ncc, mse, nrm, ssi), with each entry looking like
                - {...
                'ncc': {
                        'original': numpy array of length num_frames, metric values for original frames
                        'global_corrected' : numpy array of length num_frames, metric values for globally corrected frames
                        'corrected: numpy array of length num_frames, metric values for corrected frames
                }, ...
               }
        - **combined_transforms**: Array of dimension (2, width, height, frames) defining the combined local and global vertical and horizontal displacements to be applied to each pixel of each frame
        - **local_transforms**: Array of dimension (2, width, height, frames) defining the local transformation
        - **global_transforms**: Array of dimension (2, width, height, frames) defining the global transformation

    """
    assert 0 <= reference_frame < intensity_data_stack.shape[2]
    local_transforms = np.zeros((2, *intensity_data_stack.shape), dtype=np.float32)
    global_transforms = np.zeros((2, *intensity_data_stack.shape), dtype=np.float32)
    combined_transforms = np.zeros((2, *intensity_data_stack.shape), dtype=np.float32)
    intensity_data_stack_corrected = np.zeros_like(intensity_data_stack)
    intensity_data_stack_global_corrected = np.zeros_like(intensity_data_stack)

    ref_frame = intensity_data_stack[:, :, reference_frame]
    num_frames = intensity_data_stack.shape[2]
    metrics = {}
    for metric in ["ncc", 'mse', 'nrm', 'ssi']:
        metrics[metric] = {
            'original': np.zeros((num_frames,), dtype=np.float32),
            'corrected': np.zeros((num_frames,), dtype=np.float32),
            'global_corrected': np.zeros((num_frames,), dtype=np.float32)
        }

    for i in (pbar := tqdm(range(num_frames))):
        pbar.set_description("Aligning frames")
        if i == reference_frame:
            intensity_data_stack_corrected[:, :, i] = intensity_data_stack[:, :, i]
            continue

        tst_frame = intensity_data_stack[:, :, i]
        if global_algorithm == 'phase':
            aligned, frame_transform_global = align_phase(ref_frame, tst_frame)
        else:
            aligned, frame_transform_global = tst_frame, np.zeros_like(tst_frame, dtype=np.float32)

        intensity_data_stack_global_corrected[:, :, i] = aligned

        if local_algorithm == "optical_poly":
            intensity_data_stack_corrected[:, :, i], frame_transform_local = align_optical_poly(ref_frame, aligned)
        elif local_algorithm == "optical_ilk":
            intensity_data_stack_corrected[:, :, i], frame_transform_local = align_optical_ilk(ref_frame, aligned)
        elif local_algorithm == "optical_tvl1":
            intensity_data_stack_corrected[:, :, i], frame_transform_local = align_optical_tvl1(ref_frame, aligned)
        elif local_algorithm == "morphic":
            if torch.cuda.is_available():
                intensity_data_stack_corrected[:, :, i], frame_transform_local = align_morphic_gpu(ref_frame, aligned)
            else:
                intensity_data_stack_corrected[:, :, i], frame_transform_local = align_morphic_cpu(ref_frame, aligned)
        else:
            intensity_data_stack_corrected[:, :, i], frame_transform_local = aligned, np.zeros_like(tst_frame, dtype=np.float32)

        metrics['ncc']['original'][i] = _ncc(ref_frame, intensity_data_stack[:, :, i])
        metrics['ncc']['corrected'][i] = _ncc(ref_frame, intensity_data_stack_corrected[:, :, i])
        metrics['ncc']['global_corrected'][i] = _ncc(ref_frame, intensity_data_stack_global_corrected[:, :, i])

        metrics['mse']['original'][i] = skm.mean_squared_error(ref_frame, intensity_data_stack[:, :, i])
        metrics['mse']['corrected'][i] = skm.mean_squared_error(ref_frame, intensity_data_stack_corrected[:, :, i])
        metrics['mse']['global_corrected'][i] = skm.mean_squared_error(ref_frame, intensity_data_stack_global_corrected[:, :, i])

        metrics['nrm']['original'][i] = skm.normalized_root_mse(ref_frame, intensity_data_stack[:, :, i])
        metrics['nrm']['corrected'][i] = skm.normalized_root_mse(ref_frame, intensity_data_stack_corrected[:, :, i])
        metrics['nrm']['global_corrected'][i] = skm.normalized_root_mse(ref_frame, intensity_data_stack_global_corrected[:, :, i])

        metrics['ssi']['original'][i] = skm.structural_similarity(ref_frame, intensity_data_stack[:, :, i])
        metrics['ssi']['corrected'][i] = skm.structural_similarity(ref_frame, intensity_data_stack_corrected[:, :, i])
        metrics['ssi']['global_corrected'][i] = skm.structural_similarity(ref_frame, intensity_data_stack_global_corrected[:, :, i])

        local_transforms[:, :, :, i] = frame_transform_local
        global_transforms[:, :, :, i] = frame_transform_global

    combined_transforms = local_transforms + global_transforms

    correction_results = {
        'global_corrected_intensity_data_stack': intensity_data_stack_global_corrected,
        'corrected_intensity_data_stack': intensity_data_stack_corrected,
        'metrics': metrics,
        'combined_transforms': combined_transforms,
        'local_transforms': local_transforms,
        'global_transforms': global_transforms

    }

    return correction_results


def apply_correction_flim(
    flim_data_stack: NDArray[np.uint8],
    transform_matrix: NDArray[np.float64],
    exclude: list[int] | None = None,
    delete: bool = False
):
    """Applies a transformation matrix to a flim data stack

    :param flim_data_stack: A numpy array defining the FLIM data stack to be corrected (loaded from pqreader module).
    :param transform_matrix: Array of dimension (2, width, height, frames) defining the x and y displacements to be applied to each pixel of each frame
    :param exclude: Optional, list of frames to exclude from final output
    :param delete: Optional, delete input data stack to save memory
    :return corrected_flim_data_stack: A numpy array of same shape as flim_data_stack

    """

    # flim_data_dict, shape = flim_data_stack
    num_rows, num_cols, num_channels, num_frames, num_nanotimes = flim_data_stack.shape
    # coords = flim_data_dict[:5, :]
    # data = flim_data_dict[5, :]
    # coo = sparse.COO(coords, data, shape)
    # gcxs = sparse.GCXS.from_coo(coo, compressed_axes=[2, 3, 4])

    if delete:
        del flim_data_stack

    corrected_flim_data_stack = np.zeros_like(flim_data_stack)
    for frame_idx in (pbar := tqdm(range(num_frames))):
        pbar.set_description("Aligning raw data")
        for ch in range(num_channels):
            # frame = gcxs[:, :, ch, frame_idx, :].todense().astype(np.float32)
            frame = flim_data_stack[:, :, ch, frame_idx, :].astype(np.float32)
            flow = transform_matrix[:, :, :, frame_idx]
            warped, warped_int = flow_warp(frame, flow)  # Z x H x W
            corrected_flim_data_stack[:, :, ch, frame_idx, :] = np.moveaxis(warped_int.astype(np.uint8), [0, 1, 2], [2, 0, 1])

    if exclude:
        corrected_flim_data_stack = np.delete(corrected_flim_data_stack, exclude, axis=3)

    return corrected_flim_data_stack
    # return transformed_flim_data_stack


def get_aggregated_intensity_image(
    flim_data_stack: NDArray[np.uint8],
    channel: int,
) -> NDArray[np.uint8]:
    """Converts an intensity stack to a single intensity image representing the sum of all frames in the stack

    :param intensity_data_stack: A numpy array defining the intensity frames
    :param channel: Channel to use
    :return intensity_frame: A single frame, shape (width,height)

    """

    intensity_frame = flim_data_stack[:, :, channel, :, :].sum(axis=(2, 3))

    return intensity_frame


def _clip(x, x_min, x_max):
    return min(x_max, max(x, x_min))


def _ncc(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
