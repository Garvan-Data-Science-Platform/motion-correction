from motion_correction.motion_correction import _flow_warp, calculate_correction
from motion_correction.algorithms import _CorrectionAlgorithm
import os
import sparse
import numpy as np
from tqdm import tqdm
from enum import Enum
from sparse import GCXS
from pathlib import Path
import matplotlib.pyplot as plt
from motion_correction.pqreader import load_ptfile
from .utility import join_path, save_sequence_images
from numba import njit

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class SimMetric(Enum):
    NCC = 'ncc'
    SPM = 'spm'
    MSE = 'mse'
    NRM = 'nrm'
    SSI = 'ssi'


class FlimAligner:
    """
    A class for aligning and correcting FLIM (Fluorescence Lifetime Imaging Microscopy) data.

    Attributes:
        ptfile (str): The path to the input PT3/PTU file.
        transforms (numpy.ndarray): Transformation matrices estimated from intensity frame alignment.
        global_method (AlignMethod): The global alignment method (e.g., AlignMethod.PHASE).
        local_method (AlignMethod): The local alignment method (e.g., AlignMethod.OPTICAL_POLY).
        channel (int): The channel index selected for intensity frame alignment.
        shape (tuple): The shape of the FLIM stack data.
        flim_dict (dict): A dictionary containing the indices and values of non-zero FLIM data entries.
        flim_frames (numpy.ndarray): FLIM intensity frames before correction.
        flim_frames_corrected (numpy.ndarray): Corrected FLIM intensity frames.
        curve_fit (numpy.ndarray): Curve decay curve histogramed data.
        curve_fit_corrected (numpy.ndarray): Corrected decay curve histogramed data.
        curve_fit_corrected_int (numpy.ndarray): Corrected curve histogramed data as integers.
        save_dir (str): The directory where results will be saved.
        sim_metric (SimMetric): The similarity metric for frame alignment (e.g., SimMetric.NCC).
        old_sim (numpy.ndarray): Similarity values before alignment for all intensity frames.
        new_sim (numpy.ndarray): Similarity values after alignment for all intensity frames.
        meta (dict): Metadata associated with the FLIM data.

    Methods:
        set_methods(global_method=None, local_method=None): Set global and local alignment methods.
        set_sim_metric(sim=SimMetric.NCC): Set the similarity metric for alignment.
        get_intensity_stack(pt_file, is_raw=False): Load and prepare FLIM intensity data.
        apply_correction_intensity(ptfile, ref_frame_idx=0): Apply intensity correction to FLIM frames.
        export_results(save_dir=None): Export corrected results to files.
        apply_correction_flim(): Apply FLIM data correction.

    """

    def __init__(self):
        """
        Initializes a FlimAligner object with default attribute values.
        """
        self.ptfile = None
        self.transforms = None
        self.global_method = None
        self.local_method = None
        self.channel = 0
        self.shape = None
        self.flim_dict = None
        self.flim_frames = None
        self.flim_frames_corrected = None
        self.curve_fit = None
        self.curve_fit_corrected = None
        self.curve_fit_corrected_int = None
        self.save_dir = None
        self.sim_metric = SimMetric.NCC
        self.old_sim = None
        self.new_sim = None
        self.meta = None

    def set_methods(self, global_method: _CorrectionAlgorithm | None = None, local_method: _CorrectionAlgorithm | None = None):
        """
        Set the global and local alignment methods.

        Args:
            global_method (AlignMethod): The global alignment method.
            local_method (AlignMethod): The local alignment method.
        """
        if type(global_method) is _CorrectionAlgorithm:
            assert global_method.algorithm_type == "global"
            global_method = global_method.value

        if type(local_method) is _CorrectionAlgorithm:
            assert global_method.algorithm_type == "local"
            local_method = local_method.value

        self.global_method = global_method
        self.local_method = local_method

    def set_sim_metric(self, sim=SimMetric.NCC):
        """
        Set the similarity metric for frame alignment.

        Args:
            sim (SimMetric): The similarity metric to use.
        """
        if isinstance(sim, SimMetric):
            self.sim_metric = sim
        else:
            raise ValueError

    def set_channel(self, channel: int):
        self.channel = channel

    def get_intensity_stack(self, pt_file, is_raw=False):
        """
        Load and prepare FLIM intensity data from a PT3 file.

        Args:
            pt_file (str): The path to the PT3 file.
            is_raw (bool): Whether the data is raw intensity data.
        """
        self.ptfile = pt_file
        data, self.meta = load_ptfile(self.ptfile, is_raw)
        if isinstance(data, tuple):
            self.flim_dict = data[0]
            self.shape = data[1]
        else:
            self.flim_frames = data[:, :, self.channel, :].sum(axis=-1).astype(np.int64)
            del data
            self.shape = self.flim_frames.shape

    def apply_correction_intensity(self, ptfile, ref_frame_idx=0):
        """
        Apply intensity correction to FLIM frames.

        Args:
            ptfile (str): The path to the PT3 file.
            ref_frame_idx (int): The index of the reference frame for alignment.
        """
        try:
            self.get_intensity_stack(ptfile, is_raw=False)
        except FileNotFoundError as e:
            print(e.errno)

        assert 0 <= ref_frame_idx < self.shape[2]

        results = calculate_correction(self.flim_frames, ref_frame_idx, self.local_method, self.global_method)

        self.transforms = results["combined_transforms"]
        self.old_sim = results["metrics"][self.sim_metric.value]["original"]
        self.new_sim = results["metrics"][self.sim_metric.value]["corrected"]
        self.flim_frames_corrected = results["corrected_intensity_data_stack"]

    def export_results(self, save_dir=None):
        """
        Export corrected results to files.

        Args:
            save_dir (str): The directory where results will be saved.
        """
        if save_dir is None:
            self.save_dir = os.path.join(os.getcwd(), 'save_dir')
        else:
            self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        name = os.path.basename(self.ptfile)

        save_sequence_images(join_path(self.save_dir, f'{name}_original.mp4'),
                             np.moveaxis(self.flim_frames, [0, 1, 2], [1, 2, 0]))
        save_sequence_images(join_path(self.save_dir, f'{name}_aligned.mp4'),
                             np.moveaxis(self.flim_frames_corrected, [0, 1, 2], [1, 2, 0]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        subfig0 = axes[0].imshow(np.sum(self.flim_frames, axis=2))
        plt.colorbar(subfig0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title('original')
        subfig1 = axes[1].imshow(np.sum(self.flim_frames_corrected, axis=2))
        plt.colorbar(subfig1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title('aligned')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{name}_intensity_images.svg"), format='svg')
        plt.close()

        plt.plot(self.old_sim, linewidth=2, label="Original")
        plt.plot(self.new_sim, linewidth=2, label="Aligned")
        plt.ylabel(self.sim_metric)
        plt.xlabel('Frame')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{name}_sim_plot.svg"), format='svg')
        plt.close()
        print(f"Visualization results have been saved to {self.save_dir}")

    def apply_correction_flim(self):
        """
        Apply FLIM data correction to the loaded data.

        This method applies FLIM data correction to the loaded raw FLIM data.
        It uses the transformations estimated in function apply_correction_intensity to correct the data.

        Note:
            This method assumes that the FLIM data has already been loaded using the `get_intensity_stack` method.

        """
        tuple_data, _ = load_ptfile(self.ptfile, is_raw=True)  # shape: H x W x C x F x nanotime
        flim_data_dict, shape = tuple_data
        coords = flim_data_dict[:5, :]
        data = flim_data_dict[5, :]
        coo = sparse.COO(coords, data, shape)
        gcxs = GCXS.from_coo(coo, compressed_axes=[2, 3, 4])
        del flim_data_dict

        num_rows, num_cols, num_channels, num_frames, num_nanotimes = shape
        self.curve_fit = np.zeros((num_nanotimes, num_rows, num_cols), dtype=np.uint16)
        self.curve_fit_corrected = np.zeros((num_nanotimes, num_rows, num_cols), dtype=np.float32)
        self.curve_fit_corrected_int = np.zeros((num_nanotimes, num_rows, num_cols), dtype=np.uint16)

        header_variables = np.array([
            self.meta['imghdr'][1], self.meta['imghdr'][6],
            self.meta['imghdr'][7], self.meta['imghdr'][3],
            self.meta['imghdr'][4], self.meta['imghdr'][2]], dtype=np.uint64)
        ImgHdr_PixX = header_variables[1]
        ImgHdr_LineStart = header_variables[3]
        ImgHdr_LineStop = header_variables[4]
        ImgHdr_Frame = header_variables[5]

        LineStartMarker = 2 ** (ImgHdr_LineStart - 1)
        LineStopMarker = 2 ** (ImgHdr_LineStop - 1)
        FrameMarker = 2 ** (ImgHdr_Frame - 1)

        timestamps = np.array([], dtype=np.uint32)
        detectors = np.array([], dtype=np.uint32)
        nanotimes = np.array([], dtype=np.uint32)

        ts_index = 0
        current_ts = 0
        corrected_frames = np.zeros((num_rows, num_cols, num_channels, num_nanotimes), dtype=np.uint16)
        for frame_idx in (pbar := tqdm(range(num_frames))):
            pbar.set_description("Aligning raw data")
            for ch in range(num_channels):
                frame = gcxs[:, :, ch, frame_idx, :].todense().astype(np.float32)
                self.curve_fit += frame.astype(np.uint16).transpose(2, 1, 0)
                flow = self.transforms[:, :, :, frame_idx]
                warped, warped_int = _flow_warp(frame, flow)  # Z x H x W
                self.curve_fit_corrected += warped
                self.curve_fit_corrected_int += warped_int
                corrected_frames[:, :, ch, :] = np.moveaxis(warped_int, [0, 1, 2], [2, 0, 1])

            sync, chan, tcspc, current_ts, ts_index = _stream_one_frame(
                corrected_frames, LineStartMarker, LineStopMarker, FrameMarker, current_ts, ts_index)
            timestamps = np.concatenate((timestamps, sync), axis=0)
            detectors = np.concatenate((detectors, chan), axis=0)
            nanotimes = np.concatenate((nanotimes, tcspc), axis=0)

        # save stream data to pt3 file
        time_bit = 16
        dtime_bit = 12
        t3records = np.left_shift(detectors.astype(np.uint32), time_bit + dtime_bit) \
            | np.left_shift(nanotimes.astype(np.uint32), time_bit) \
            | timestamps.astype(np.uint16)

        filename = os.path.join(self.save_dir, os.path.basename(self.ptfile)[:-4]+"_corrected.pt3")
        print(f'Data exported to {filename}')
        with open(filename, 'wb') as f:
            for m in ['header', 'dispcurve', 'params', 'repeatgroup', 'hardware', 'router', 'ttmode']:
                f.write(np.array(self.meta[m]).tobytes())
            f.write(np.array(self.meta['imghdr']))
            f.write(t3records.astype(np.uint32))


@njit
def _stream_one_frame(corrected_frames, LineStartMarker, LineStopMarker, FrameMarker, current_ts=0, ts_index=0):
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
