from PIL import Image
from typing import List
import numpy as np


def __reverse_get_pt3_data_frame(flim_data_stack, meta):

    header_variables = np.array(
        [meta['imghdr'][1], meta['imghdr'][6],
         meta['imghdr'][7], meta['imghdr'][3],
         meta['imghdr'][4], meta['imghdr'][2]], dtype=np.uint64)
    ImgHdr_PixX = header_variables[1]
    ImgHdr_LineStart = header_variables[3]
    ImgHdr_LineStop = header_variables[4]
    ImgHdr_Frame = header_variables[5]

    LineStartMarker = 2 ** (ImgHdr_LineStart - 1)
    LineStopMarker = 2 ** (ImgHdr_LineStop - 1)
    FrameMarker = 2 ** (ImgHdr_Frame - 1)
    num_pixel_X = ImgHdr_PixX

    (num_rows, num_cols, num_channels, num_frames, num_nanotimes) = flim_data_stack.shape

    size = int(flim_data_stack.sum() + num_frames + num_rows*2*num_frames)
    timestamps = np.zeros(size, dtype=np.uint32)
    detectors = np.zeros(size, dtype=np.uint32)
    nanotimes = np.zeros(size, dtype=np.uint32)

    idx = 0

    currentTimestamp = 0

    timestamp_index = 0
    print(timestamp_index)

    def check_insert_index(ts):
        nonlocal idx
        nonlocal timestamp_index
        nonlocal timestamps
        nonlocal detectors
        nonlocal nanotimes

        if ts >= 65536 * (1+timestamp_index):
            timestamp_index += 1
            # lengthen arrays
            timestamps = np.concatenate([timestamps, [0]])
            detectors = np.concatenate([detectors, [0]])
            nanotimes = np.concatenate([nanotimes, [0]])

            timestamps[idx] = ts
            detectors[idx] = 15
            nanotimes[idx] = 0
            idx += 1

    # For each Frame F:
    for f in range(num_frames):
        # For each line L:
        for l in range(num_rows):
            # LineStartMarker
            check_insert_index(currentTimestamp)
            timestamps[idx] = currentTimestamp
            detectors[idx] = 15
            nanotimes[idx] = LineStartMarker
            idx += 1
            for ch in range(num_channels):
                for c in range(num_cols):
                    for n in range(num_nanotimes):
                        for i in range(flim_data_stack[l, c, ch, f, n]):
                            ts = currentTimestamp + c
                            check_insert_index(ts)
                            timestamps[idx] = ts
                            detectors[idx] = ch + 1
                            nanotimes[idx] = n
                            idx += 1

            currentTimestamp += num_pixel_X
            check_insert_index(currentTimestamp)
            timestamps[idx] = currentTimestamp
            detectors[idx] = 15
            nanotimes[idx] = LineStopMarker
            idx += 1
        # currentTimestamp += 1
        # check_insert_index(currentTimestamp)
        timestamps[idx] = currentTimestamp
        detectors[idx] = 15
        nanotimes[idx] = FrameMarker
        idx += 1

    return timestamps, detectors, nanotimes


def __reverse_process_t3records(detectors, timestamps, nanotimes, time_bit=16, dtime_bit=12):

    t3records = np.left_shift(detectors.astype(np.uint32),
                              time_bit + dtime_bit) | np.left_shift(nanotimes.astype(np.uint32),
                                                                    time_bit) | timestamps.astype(np.uint16)  # Must be 16 to truncate

    return t3records


def write_pt3(meta, flim_data_stack, filename):
    """Write a FLIM data stack to a .pt3 file

    :param meta: Must be the same meta dictionary from pqreader.load_ptfile function
    :param flim_data_stack: A numpy array, usually after applying motion correction algorightms
    :param filename: Output filename
    """
    timestamps, detectors, nanotimes = __reverse_get_pt3_data_frame(flim_data_stack, meta)
    t3records = __reverse_process_t3records(detectors, timestamps, nanotimes)

    with open(filename, 'wb') as f:
        for m in ['header', 'dispcurve', 'params', 'repeatgroup', 'hardware', 'router', 'ttmode']:
            f.write(np.array(meta[m]).tobytes())
        f.write(np.array(meta['imghdr']))
        f.write(t3records.astype(np.uint32))
