
import numpy as np
import warnings

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration as morphic_cpu
from dipy.align.metrics import CCMetric as ccmetric_cpu
from image_registration.fft_tools import shift
from image_registration import chi2_shift
from skimage.transform import warp
from sys import platform

import pyimof
import torch
import cv2
from numpy.typing import NDArray


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class _CorrectionAlgorithm:

    """
    Base class for image correction algorithms.
    """

    def init(self):
        self.algorithm_type = "local"

    def align(self, ref_frame: NDArray[np.uint16], target_frame: NDArray[np.uint16]):
        """
        Aligns two images using an image correction algorithm.

        :param ref_frame: The reference image.
        :param target_frame: The image to be aligned.
        :return aligned: (ndarray) The aligned version of the moving image.
        :return transform: (ndarray) A 2xN array representing the translation in the y and x dimensions.
        """
        raise NotImplementedError("Align method not implemented")


class Phase(_CorrectionAlgorithm):
    """
    Aligns two images using phase correlation-based alignment.
    """

    def __init__(self):
        self.algorithm_type = 'global'

    def align(self, fixed_img, moving_img):
        """
        Aligns two images using phase correlation-based alignment.

        :param fixed_img: (ndarray) The reference image.
        :param moving_img: (ndarray) The image to be aligned.
        :param sigma_diff: (float) The standard deviation for difference calculation (default: 20).
        :param radius: (int) The radius for the cross-correlation computation (default: 15).
        :return aligned: (ndarray) The aligned version of the moving image.
        :return transform: (ndarray) A 3x3 transformation matrix.

        Note:
        This function computes the translation between two images using phase correlation
        and applies the translation to align the moving image with the fixed image.
        """

        xoff, yoff = chi2_shift(fixed_img, moving_img, return_error=False,
                                upsample_factor='auto')
        # xpad, ypad = int(np.ceil(np.abs(xoff))), int(np.ceil(np.abs(yoff)))
        # tst_frame_padded = np.pad(moving_img, ((ypad, ypad), (xpad, xpad)), mode='symmetric')

    # aligned = shift.shiftnd(moving_img, (-yoff, -xoff))
    # aligned = aligned[ypad//2:moving_img.shape[0]+ypad//2, xpad//2:moving_img.shape[1]+xpad//2]

        transform = np.zeros((2, *fixed_img.shape), dtype=np.float32)
        transform[0, ...] = yoff
        transform[1, ...] = xoff

        row_coords, col_coords = np.meshgrid(
            np.arange(moving_img.shape[0]), np.arange(moving_img.shape[1]), indexing='ij')

        aligned = warp(moving_img,
                       np.array([row_coords + transform[0], col_coords + transform[1]]),
                       mode='symmetric', preserve_range=True)

        return aligned, transform


class OpticalILK(_CorrectionAlgorithm):

    def __init__(self):
        self.algorithm_type = 'local'

    def align(self, fixed_img, moving_img):
        """
        Aligns two images using the Iterative Lucas-Kanade (ILK) optical flow algorithm.

        :param fixed_img: (ndarray) The reference image.
        :param moving_img: (ndarray) The image to be aligned.
        :param sigma_diff: (float) The standard deviation for difference calculation (default: 20).
        :param radius: (int) The radius for the cross-correlation computation (default: 15).
        :return aligned: (ndarray) The aligned version of the moving image.
        :return transform: (ndarray) A 3x3 transformation matrix.

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


class OpticalTVL1(_CorrectionAlgorithm):
    """
    Aligns two images using the TV-L1 optical flow algorithm.
    """

    def __init__(self):
        self.algorithm_type = 'local'

    def align(self, fixed_img, moving_img):
        """

        :param fixed_img: (ndarray) The reference image.
        :param moving_img: (ndarray) The image to be aligned.
        :param sigma_diff: (float) The standard deviation for difference calculation (default: 20).
        :param radius: (int) The radius for the cross-correlation computation (default: 15).
        :return aligned: (ndarray) The aligned version of the moving image.
        :return transform: (ndarray) A 3x3 transformation matrix.

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


class OpticalPoly(_CorrectionAlgorithm):
    """
    Aligns two images using polynomial expansion-based optical flow.
    """

    def __init__(self):
        self.algorithm_type = 'local'

    def align(self, fixed_img, moving_img):
        """

        :param fixed_img: (ndarray) The reference image.
        :param moving_img: (ndarray) The image to be aligned.
        :param sigma_diff: (float) The standard deviation for difference calculation (default: 20).
        :param radius: (int) The radius for the cross-correlation computation (default: 15).
        :return aligned: (ndarray) The aligned version of the moving image.
        :return transform: (ndarray) A 3x3 transformation matrix.
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


class Morphic(_CorrectionAlgorithm):
    """
    Aligns two images using MORPHIC image registration. Uses GPU if available
    :param sigma_diff: (float) The standard deviation for difference calculation (default: 20)
    :param radius: (int) The radius for the cross-correlation computation (default: 15).
    """

    def __init__(self, sigma_diff: float = 20, radius=15):
        self.sigma_diff = sigma_diff
        self.radius = radius
        self.algorithm_type = 'local'

    def align(self, fixed_img, moving_img):
        """
        Aligns two images using MORPHIC image registration.

        :param fixed_img: (ndarray) The reference image.
        :param moving_img: (ndarray) The image to be aligned.
        :param sigma_diff: (float) The standard deviation for difference calculation (default: 20).
        :param radius: (int) The radius for the cross-correlation computation (default: 15).
        :return aligned: (ndarray) The aligned version of the moving image.
        :return transform: (ndarray) A 3x3 transformation matrix.
        """
        if torch.cuda.is_available():
            return self.align_morphic_gpu(fixed_img, moving_img)
        else:
            return self.align_morphic_cpu(fixed_img, moving_img)

    def align_morphic_cpu(self, fixed_img, moving_img):

        metric = ccmetric_cpu(2, sigma_diff=self.sigma_diff, radius=self.radius)
        level_iters = [10, 10, 5]
        sdr = morphic_cpu(metric, level_iters)
        mapping = sdr.optimize(fixed_img, moving_img)
        aligned = mapping.transform(moving_img)
        return aligned, np.moveaxis(mapping.backward, [0, 1, 2], [1, 2, 0])

    def align_morphic_gpu(self, fixed_img, moving_img):

        from .cudipy.align.imwarp import SymmetricDiffeomorphicRegistration as morphic_gpu
        import cupy as cp
        from .cudipy.align.metrics import CCMetric as ccmetric_gpu

        metric = ccmetric_gpu(2, sigma_diff=self.sigma_diff, radius=self.radius)
        level_iters = [10, 10, 5]
        sdr = morphic_gpu(metric, level_iters)
        mapping = sdr.optimize(cp.asarray(fixed_img), cp.asarray(moving_img))
        warped_moving = mapping.transform(cp.asarray(moving_img))
        return cp.asnumpy(warped_moving), cp.asnumpy(cp.moveaxis(mapping.backward, [0, 1, 2], [1, 2, 0]))
