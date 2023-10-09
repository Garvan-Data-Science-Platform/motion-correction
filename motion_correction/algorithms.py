
import cv2
import torch
import pyimof
import numpy as np
from numba import njit
from sys import platform
from skimage.transform import warp
from image_registration import chi2_shift
from image_registration.fft_tools import shift
from dipy.align.metrics import CCMetric as ccmetric_cpu
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration as morphic_cpu

import torch.nn.functional as F
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
else:
    print("No GPU detected, or CUDA toolkit not installed. https://developer.nvidia.com/cuda-downloads")


@njit
def stream_one_frame(corrected_frames, LineStartMarker, LineStopMarker, FrameMarker, current_ts=0, ts_index=0):
    num_rows, num_cols, num_channels, num_nanotimes = corrected_frames.shape
    total_entries = int(corrected_frames.sum()*2 + 1 + num_rows * 2)

    # Initialize arrays
    sync = np.zeros(total_entries, dtype=np.uint32)
    chan = np.zeros(total_entries, dtype=np.uint32)
    tcspc = np.zeros(total_entries, dtype=np.uint32)

    idx = 0
    for r in range(num_rows):
        # LineStartMarker
        sync[idx] = current_ts
        chan[idx] = 15
        tcspc[idx] = LineStartMarker
        idx += 1

        for ch in range(num_channels):
            for c in range(num_cols):
                ts = current_ts + c
                if ts >= 65536 * (1 + ts_index):
                    ts_index += 1
                    sync[idx] = ts
                    chan[idx] = 15
                    tcspc[idx] = 0
                    idx += 1

                num_non_zeros = np.sum(corrected_frames[r, c, ch, :])
                sync[idx:idx+num_non_zeros] = np.repeat(ts, num_non_zeros)
                chan[idx:idx+num_non_zeros] = np.repeat(ch + 1, num_non_zeros)
                tcspc[idx:idx+num_non_zeros] = np.repeat(np.arange(num_nanotimes), corrected_frames[r, c, ch, :])
                idx += num_non_zeros

        current_ts += num_cols

        if current_ts >= 65536 * (1 + ts_index):
            ts_index += 1
            sync[idx] = current_ts
            chan[idx] = 15
            tcspc[idx] = 0
            idx += 1

        sync[idx] = current_ts
        chan[idx] = 15
        tcspc[idx] = LineStopMarker
        idx += 1

    sync[idx] = current_ts
    chan[idx] = 15
    tcspc[idx] = FrameMarker
    idx += 1

    sync = sync[:idx]
    chan = chan[:idx]
    tcspc = tcspc[:idx]

    return sync, chan, tcspc, current_ts, ts_index


def cascade_round_tensor(tensor):
    tsr_sum = tensor.sum(dim=0, keepdim=True)
    lwr_tsr = tensor.floor()
    lwr_sum = lwr_tsr.sum(axis=0, keepdims=True)
    count_tensor = tsr_sum - lwr_sum
    random_mask = torch.rand(*tensor.shape, device=tensor.device) * tensor.shape[0] <= count_tensor

    return (lwr_tsr + random_mask).int()


def hist_laxis_tensor(data, n_bins, range_limits):
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


def cascade_round_tensor_hist(tensor, num_bins=50):
    rows, cols = tensor.shape[:2]
    array = tensor.cuda() if torch.cuda.is_available() else torch.tensor(tensor)
    flr_arr = array.floor()
    arr_sum = array.sum(dim=-1, keepdim=True)
    flr_sum = flr_arr.sum(dim=-1, keepdim=True)
    dif_sum = arr_sum - flr_sum

    # Estimate histogram
    residuals = array - flr_arr
    hist = hist_laxis_tensor(residuals, n_bins=num_bins, range_limits=(0, 1))
    cum_sum = hist.flip(dims=[-1, ]).cumsum(dim=-1)

    indices = torch.argmax((cum_sum > dif_sum).int(), dim=-1)
    bins = torch.linspace(0, 1, num_bins + 1, device='cuda')
    thresholds = bins[num_bins - indices.flatten()].reshape((rows, cols, 1))
    random_mask = residuals > thresholds

    return (flr_arr + random_mask).int()


def flow_warp(frame, flow, padding_mode='reflection'):
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
    warped_int = cascade_round_tensor(warped)
    return torch.squeeze(warped).cpu().numpy(), torch.squeeze(warped_int).cpu().numpy().astype(np.uint16)

    # warped = cascade_round_tensor(
    #     F.grid_sample(frame_tensor, grid_tensor, padding_mode=padding_mode, align_corners=True))
    # return torch.squeeze(warped).cpu().numpy().astype(np.uint16)


def align_phase(fixed_img, moving_img):
    """
    Aligns two images using phase correlation-based alignment.

    Parameters:
    - fixed_img (ndarray): The reference image.
    - moving_img (ndarray): The image to be aligned.

    Returns:
    - aligned (ndarray): The aligned version of the moving image.
    - transform (ndarray): A 2xN array representing the translation in the y and x dimensions.

    Note:
    This function computes the translation between two images using phase correlation
    and applies the translation to align the moving image with the fixed image.
    """

    xoff, yoff = chi2_shift(fixed_img, moving_img, return_error=False,
                            upsample_factor='auto')
    # xpad, ypad = int(np.ceil(np.abs(xoff))), int(np.ceil(np.abs(yoff)))
    # tst_frame_padded = np.pad(moving_img, ((ypad, ypad), (xpad, xpad)), mode='symmetric')

    aligned = shift.shiftnd(moving_img, (-yoff, -xoff))
    # aligned = aligned[ypad//2:moving_img.shape[0]+ypad//2, xpad//2:moving_img.shape[1]+xpad//2]

    transform = np.zeros((2, *fixed_img.shape), dtype=np.float32)
    transform[0, ...] = yoff
    transform[1, ...] = xoff

    return aligned, transform


def align_optical_ilk(fixed_img, moving_img):
    """
    Aligns two images using the Iterative Lucas-Kanade (ILK) optical flow algorithm.

    Parameters:
    - fixed_img (ndarray): The reference image.
    - moving_img (ndarray): The image to be aligned.

    Returns:
    - aligned (ndarray): The aligned version of the moving image.
    - transform (ndarray): A 2xN array representing the optical flow vectors (u, v).

    Note:
    This function estimates the optical flow between two images using the ILK algorithm
    and applies the estimated flow to align the moving image with the fixed image.
    """

    ref_img = (fixed_img - np.min(fixed_img)) / (np.max(fixed_img) - np.min(fixed_img))
    tst_img = (moving_img - np.min(moving_img)) / (np.max(moving_img) - np.min(moving_img))
    ref_img = (ref_img * 255).astype(np.uint8)
    tst_img = (tst_img * 255).astype(np.uint8)
    u, v = pyimof.solvers.ilk(ref_img, tst_img)  # optical_flow_ilk(ref_img, tst_img)
    row_coords, col_coords = np.meshgrid(
        np.arange(tst_img.shape[0]), np.arange(tst_img.shape[1]), indexing='ij')

    aligned = warp(moving_img,
                   np.array([row_coords + v, col_coords + u]),
                   mode='symmetric', preserve_range=True)
    transform = np.stack((v, u), axis=0)
    return aligned, transform


def align_optical_tvl1(fixed_img, moving_img):
    """
    Aligns two images using the TV-L1 optical flow algorithm.

    Parameters:
    - fixed_img (ndarray): The reference image.
    - moving_img (ndarray): The image to be aligned.

    Returns:
    - aligned (ndarray): The aligned version of the moving image.
    - transform (ndarray): A 2xN array representing the optical flow vectors (u, v).

    Note:
    This function estimates the optical flow between two images using the TV-L1 algorithm
    and applies the estimated flow to align the moving image with the fixed image.
    """

    ref_img = (fixed_img - np.min(fixed_img)) / (np.max(fixed_img) - np.min(fixed_img))
    tst_img = (moving_img - np.min(moving_img)) / (np.max(moving_img) - np.min(moving_img))
    ref_img = (ref_img * 255).astype(np.uint8)
    tst_img = (tst_img * 255).astype(np.uint8)

    u, v = pyimof.solvers.tvl1(ref_img, tst_img)  # optical_flow_tvl1(ref_img, tst_img)

    row_coords, col_coords = np.meshgrid(
        np.arange(tst_img.shape[0]), np.arange(tst_img.shape[1]), indexing='ij')

    aligned = warp(moving_img,
                   np.array([row_coords + v, col_coords + u]),
                   mode='symmetric', preserve_range=True)
    transform = np.stack((v, u), axis=0)

    return aligned, transform


def align_optical_poly(fixed_img, moving_img):
    """
    Aligns two images using polynomial expansion-based optical flow.

    Parameters:
    - fixed_img (ndarray): The reference image.
    - moving_img (ndarray): The image to be aligned.

    Returns:
    - aligned (ndarray): The aligned version of the moving image.
    - transform (ndarray): A 2xN array representing the optical flow vectors (u, v).

    Note:
    This function estimates the optical flow between two images using polynomial expansion
    and applies the estimated flow to align the moving image with the fixed image.
    """

    ref_img = (fixed_img - np.min(fixed_img)) / (np.max(fixed_img) - np.min(fixed_img))
    tst_img = (moving_img - np.min(moving_img)) / (np.max(moving_img) - np.min(moving_img))
    ref_img = (ref_img * 255).astype(np.uint8)
    tst_img = (tst_img * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(ref_img, tst_img, None,
                                        pyr_scale=0.5, levels=3, winsize=12,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    u = flow[..., 0]
    v = flow[..., 1]

    row_coords, col_coords = np.meshgrid(
        np.arange(tst_img.shape[0]), np.arange(tst_img.shape[1]), indexing='ij')

    aligned = warp(moving_img,
                   np.array([row_coords + v, col_coords + u]),
                   mode='wrap', preserve_range=True)

    transform = np.stack((v, u), axis=0)

    return aligned, transform


def align_morphic_cpu(fixed_img, moving_img, sigma_diff=20, radius=15):
    """
    Aligns two images using CPU-based MORPHIC image registration.

    Parameters:
    - fixed_img (ndarray): The reference image.
    - moving_img (ndarray): The image to be aligned.
    - sigma_diff (float): The standard deviation for difference calculation (default: 20).
    - radius (int): The radius for the cross-correlation computation (default: 15).

    Returns:
    - aligned (ndarray): The aligned version of the moving image.
    - transform (ndarray): A 3x3 transformation matrix.

    Note:
    This function performs image registration using the MORPHIC method on the CPU.
    """

    metric = ccmetric_cpu(2, sigma_diff=sigma_diff, radius=radius)
    level_iters = [10, 10, 5]
    sdr = morphic_cpu(metric, level_iters)
    mapping = sdr.optimize(fixed_img, moving_img)
    aligned = mapping.transform(moving_img)
    return aligned, np.moveaxis(mapping.backward, [0, 1, 2], [1, 2, 0])


def align_morphic_gpu(fixed_img, moving_img, sigma_diff=20, radius=15):
    """
    Aligns two images using GPU-based MORPHIC image registration.

    Parameters:
    - fixed_img (ndarray): The reference image.
    - moving_img (ndarray): The image to be aligned.
    - sigma_diff (float): The standard deviation for difference calculation (default: 20).
    - radius (int): The radius for the cross-correlation computation (default: 15).

    Returns:
    - aligned (ndarray): The aligned version of the moving image.
    - transform (ndarray): A 3x3 transformation matrix.

    Note:
    This function performs image registration using the MORPHIC method on the GPU.
    """
    from .cudipy.align.imwarp import SymmetricDiffeomorphicRegistration as morphic_gpu
    import cupy as cp
    from .cudipy.align.metrics import CCMetric as ccmetric_gpu

    metric = ccmetric_gpu(2, sigma_diff=sigma_diff, radius=radius)
    level_iters = [10, 10, 5]
    sdr = morphic_gpu(metric, level_iters)
    mapping = sdr.optimize(cp.asarray(fixed_img), cp.asarray(moving_img))
    warped_moving = mapping.transform(cp.asarray(moving_img))
    return cp.asnumpy(warped_moving), cp.asnumpy(cp.moveaxis(mapping.backward, [0, 1, 2], [1, 2, 0]))
