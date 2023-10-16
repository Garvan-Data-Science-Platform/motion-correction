"""Main module."""

from typing import TypedDict
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import torch
import skimage.metrics as skm
from .algorithms import _CorrectionAlgorithm
import torch.nn.functional as F


# This defines the API for the python package

'''Data dimensions:
flim_data_stack: (width, height, n_channels, n_frames, n_nanotimes)
intensity_data_stack: (width,height,n_frames)
'''

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    print("No GPU detected, or CUDA toolkit not installed. https://developer.nvidia.com/cuda-downloads")


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
        local_algorithm: _CorrectionAlgorithm | None = None,
        global_algorithm: _CorrectionAlgorithm | None = None,
) -> CorrectionResults:
    """Calculates motion correction for given intensity data stack based on chosen algorithms

    :param intensity_data_stack: A numpy array defining the intensity frames to use for correction (output of get_intensity_stack)
    :param reference_frame: The reference frame to use
    :param local_algorithm: Optional, CorrectionAlgorithm to use, imported from motion_correction.algorithms
    :param global_algorithm: Optional, CorrectionAlgorithm to use, imported from motion_correction.algorithms

    :return CorrectionResults: dictionary with keys:
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
    print("LOCAL", local_algorithm)
    for i in (pbar := tqdm(range(num_frames))):
        pbar.set_description("Aligning frames")
        if i == reference_frame:
            intensity_data_stack_corrected[:, :, i] = intensity_data_stack[:, :, i]
            continue

        tst_frame = intensity_data_stack[:, :, i]
        if global_algorithm:
            assert global_algorithm.algorithm_type == 'global'
            aligned, frame_transform_global = global_algorithm.align(ref_frame, tst_frame)
        else:
            aligned, frame_transform_global = tst_frame, np.zeros_like(tst_frame, dtype=np.float32)

        intensity_data_stack_global_corrected[:, :, i] = aligned

        if local_algorithm:
            assert local_algorithm.algorithm_type == "local"
            intensity_data_stack_corrected[:, :, i], frame_transform_local = local_algorithm.align(ref_frame, aligned)
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

        def norm2uint8(data):
            return ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

        metrics['ssi']['original'][i] = skm.structural_similarity(
            norm2uint8(ref_frame), norm2uint8(intensity_data_stack[:, :, i]), data_range=255)
        metrics['ssi']['corrected'][i] = skm.structural_similarity(
            norm2uint8(ref_frame), norm2uint8(intensity_data_stack_corrected[:, :, i]), data_range=255)
        metrics['ssi']['global_corrected'][i] = skm.structural_similarity(
            norm2uint8(ref_frame), norm2uint8(intensity_data_stack_global_corrected[:, :, i]), data_range=255)

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
            warped, warped_int = _flow_warp(frame, flow)  # Z x H x W
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


def _flow_warp(frame, flow, padding_mode='reflection'):
    # frame: H x W x C, flow: 2 (y, x) x H x W
    rows, cols = frame.shape[0], frame.shape[1]
    col_coords, row_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='xy')
    assert frame.shape[:2] == flow.shape[-2:]

    frame_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # 1 x C x H x W

    grid = np.array([col_coords + flow[1, :, :], row_coords + flow[0, :, :]], dtype=np.float32)
    grid[0, :, :] = 2.0 * grid[0, :, :] / (cols-1) - 1.0
    grid[1, :, :] = 2.0 * grid[1, :, :] / (rows-1) - 1.0
    grid_tensor = torch.tensor(grid).permute(1, 2, 0).unsqueeze(0).float().to(device)   # 1 x H x W x 2 (x, y)

    warped = F.grid_sample(frame_tensor, grid_tensor, padding_mode=padding_mode, align_corners=True)
    # warped_int = cascade_round_tensor_hist(warped.squeeze(0).permute((1, 2, 0))).permute((2, 0, 1)).unsqueeze(0)
    warped_int = _cascade_round_tensor(warped)
    return torch.squeeze(warped).cpu().numpy(), torch.squeeze(warped_int).cpu().numpy().astype(np.uint16)

    # warped = cascade_round_tensor(
    #     F.grid_sample(frame_tensor, grid_tensor, padding_mode=padding_mode, align_corners=True))
    # return torch.squeeze(warped).cpu().numpy().astype(np.uint16)


def _cascade_round_tensor(tensor):
    tsr_sum = tensor.sum(dim=0, keepdim=True)
    lwr_tsr = tensor.floor()
    lwr_sum = lwr_tsr.sum(axis=0, keepdims=True)
    count_tensor = tsr_sum - lwr_sum
    random_mask = torch.rand(*tensor.shape, device=tensor.device) * tensor.shape[0] <= count_tensor

    return (lwr_tsr + random_mask).int()


def _hist_laxis_tensor(data, n_bins, range_limits):
    # Move data to CUDA if available
    if torch.cuda.is_available():
        data = data.cuda()

    # Setup bins and determine the bin location for each element for the bins
    N = data.shape[-1]
    bins = torch.linspace(range_limits[0], range_limits[1], n_bins + 1, device="cuda")
    data2D = data.view(-1, N)
    idx = torch.searchsorted(bins, data2D.contiguous(), right=True) - 1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx == -1) | (idx == n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins * torch.arange(data2D.shape[0]).unsqueeze(1).to(data.device) + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins * data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = torch.bincount(scaled_idx.view(-1), minlength=limit + 1)[:-1]
    counts = counts.view(data.shape[:-1] + (n_bins,))
    return counts


def _cascade_round_tensor_hist(tensor, num_bins=50):
    rows, cols = tensor.shape[:2]
    array = tensor.cuda() if torch.cuda.is_available() else torch.tensor(tensor)
    flr_arr = array.floor()
    arr_sum = array.sum(dim=-1, keepdim=True)
    flr_sum = flr_arr.sum(dim=-1, keepdim=True)
    dif_sum = arr_sum - flr_sum

    # Estimate histogram
    residuals = array - flr_arr
    hist = _hist_laxis_tensor(residuals, n_bins=num_bins, range_limits=(0, 1))
    cum_sum = hist.flip(dims=[-1, ]).cumsum(dim=-1)

    indices = torch.argmax((cum_sum > dif_sum).int(), dim=-1)
    bins = torch.linspace(0, 1, num_bins + 1, device='cuda')
    thresholds = bins[num_bins - indices.flatten()].reshape((rows, cols, 1))
    random_mask = residuals > thresholds

    return (flr_arr + random_mask).int()
