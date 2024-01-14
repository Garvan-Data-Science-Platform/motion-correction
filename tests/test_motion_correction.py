#!/usr/bin/env python

"""Tests for `motion_correction` package."""

import pytest

from motion_correction import motion_correction
import numpy as np


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    assert motion_correction is not None
    assert True


def test_demo(response):
    # %%
    from motion_correction.desktop.flim_aligner import FlimAligner, SimMetric

    flim_aligner = FlimAligner()

    # %%
    from motion_correction.algorithms import (
        Phase,
        Morphic,
    )

    phase = Phase()
    morphic = Morphic(sigma_diff=20, radius=15)
    flim_aligner.set_methods(global_method=phase, local_method=morphic)

    # %%
    flim_aligner.set_sim_metric(sim=SimMetric.MSE)

    # %%
    flim_aligner.set_channel(1)

    pt_file_path = "./tests/2_frames.pt3"
    flim_aligner.apply_correction_intensity(pt_file_path)

    flim_aligner.export_results(save_dir=None)

    from motion_correction.desktop.utility import plot_sequence_images
    import matplotlib.pyplot as plt

    plot_sequence_images(flim_aligner.flim_frames.transpose(2, 1, 0))
    plot_sequence_images(flim_aligner.flim_frames_corrected.transpose(2, 1, 0))

    fig, axes = plt.subplots()
    axes.plot(flim_aligner.old_sim, label="original")
    axes.plot(flim_aligner.new_sim, label="corrected")
    axes.set_ylabel(flim_aligner.sim_metric)
    axes.set_xlabel("Frame")
    plt.legend(loc="best")
    # plt.show()

    flim_aligner.apply_correction_flim()

    col_idx = 105
    row_idx = 180
    blk_sz = 5
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(
        flim_aligner.curve_fit[
            :, row_idx : row_idx + blk_sz, col_idx : col_idx + blk_sz
        ].sum(axis=(1, 2))
    )
    axes[0].set_title("original")
    axes[1].plot(
        flim_aligner.curve_fit_corrected[
            :, row_idx : row_idx + blk_sz, col_idx : col_idx + blk_sz
        ].sum(axis=(1, 2))
    )
    axes[1].set_title("corrected")
    axes[2].plot(
        flim_aligner.curve_fit_corrected_int[
            :, row_idx : row_idx + blk_sz, col_idx : col_idx + blk_sz
        ].sum(axis=(1, 2))
    )
    axes[2].set_title("corrected int")
    # plt.show()

    class RectangularROI:
        def __init__(
            self, fig, decay_data, image_AX, decay_AX, decay_fig, tau_resolution
        ):
            self.fig = fig
            self.decay_data = decay_data
            self.image_ax = image_AX
            self.ori_img = self.image_ax.get_array()
            self.decay_ax = decay_AX
            self.decay_fig = decay_fig
            self.tau_resolution = tau_resolution
            self.image_ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
            self.image_ax.figure.canvas.mpl_connect(
                "button_release_event", self.on_release
            )
            self.xs = None
            self.ys = None

        def on_press(self, event):
            self.x0 = event.xdata
            self.y0 = event.ydata

        def on_release(self, event):
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.x_indices = np.int_(
                np.ceil(np.abs(np.sort(np.array([self.x1, self.x0]))))
            )  # [x1, x2]
            self.y_indices = np.int_(
                np.ceil(np.abs(np.sort(np.array([self.y0, self.y1]))))
            )  # [y1, y2]
            self.ys = np.sum(
                self.decay_data[
                    self.y_indices[0] : self.y_indices[1],
                    self.x_indices[0] : self.x_indices[1],
                    :,
                ],
                axis=0,
            )
            self.ys = np.sum(self.ys, axis=0)
            self.xs = (
                np.linspace(0, decay_data.shape[2], decay_data.shape[2], dtype=np.int32)
                * self.tau_resolution
            )
            self.decay_ax.set_data(self.xs, self.ys)
            self.decay_fig.set_ylim(ymin=0, ymax=np.max(self.ys) * 10)
            #         self.decay_ax.fig.canvas.draw()
            self.shown_img = self.ori_img.copy()
            self.shown_img[
                self.y_indices[0], self.x_indices[0] : self.x_indices[1]
            ] = 1000
            self.shown_img[
                self.y_indices[1], self.x_indices[0] : self.x_indices[1]
            ] = 1000
            self.shown_img[
                self.y_indices[0] : self.y_indices[1], self.x_indices[0]
            ] = 1000
            self.shown_img[
                self.y_indices[0] : self.y_indices[1], self.x_indices[1]
            ] = 1000
            self.image_ax.set_array(self.shown_img)
            self.fig.redraw()

    intensity_image = flim_aligner.flim_frames_corrected.sum(axis=-1)
    decay_data = flim_aligner.curve_fit_corrected.transpose((1, 2, 0))
    tau_resolution = 1.0

    fig = plt.figure(figsize=(9, 4))

    image_ax = fig.add_subplot(121)
    image_AX = image_ax.imshow(intensity_image, cmap="viridis")
    fig.colorbar(image_AX)
    image_ax.set_aspect("auto")
    plt.title("Draw a rectangle on intensity image")

    plot_decay_data = decay_data[0, 0, :]
    tau = (
        np.linspace(0, decay_data.shape[2], decay_data.shape[2], dtype=np.int32)
        * tau_resolution
    )
    decay_fig = fig.add_subplot(122)
    (decay_AX,) = decay_fig.plot(
        tau, plot_decay_data, "k-", label="Selected ROI Histogram", linewidth=1
    )
    plt.yscale(value="log")
    plt.axis([0, np.max(tau), 0, np.max(plot_decay_data) * 10])
    plt.xlabel("Time [ns]")
    plt.ylabel("Intensity [counts]")
    plt.title("TCSPC Decay")
    plt.grid(True)
    plt.legend()

    plt.sca(decay_fig)
    _ = RectangularROI(fig, decay_data, image_AX, decay_AX, decay_fig, tau_resolution)
    plt.tight_layout()
    # plt.show()
