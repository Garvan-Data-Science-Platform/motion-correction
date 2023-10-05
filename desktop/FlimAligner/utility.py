import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from typing import List, Union, Tuple, Optional
from IPython.core.display import display, HTML


def ncc(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def display_images(images: List[Union[str, np.ndarray]],
                   rows: Optional[int] = None, cols: Optional[int] = None, colorbar=True,
                   figsize: Tuple[int, int] = (12, 12)) -> None:
    """
    Display a list of images in a grid layout using Matplotlib.

    Args:
        images (list): List of image paths or NumPy arrays.
        rows (int): Number of rows in the grid (default: None, automatic calculation).
        cols (int): Number of columns in the grid (default: None, automatic calculation).
        figsize (tuple): Size of the figure (default: (10, 10)).

    Note:
    This function displays a grid of images using Matplotlib. It supports both image paths
    and NumPy arrays as input.

    """
    num_images = len(images)

    if num_images < 1:
        print("No images to display.")
        return

    if rows is None and cols is None:
        # Calculate the number of rows and columns automatically.
        if num_images <= 4:
            rows, cols = 1, num_images
        else:
            cols = 4
            rows = (num_images + 3) // 4
    elif rows is None:
        # Calculate the number of rows automatically based on columns.
        rows = (num_images + cols - 1) // cols
    elif cols is None:
        # Calculate the number of columns automatically based on rows.
        cols = (num_images + rows - 1) // rows

    # Create a new figure with the specified size.
    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        axs = plt.subplot(rows, cols, i + 1)

        if isinstance(img, str):
            # If the image is a file path, load it using matplotlib.
            img = plt.imread(img)

        if isinstance(img, np.ndarray):
            # If the image is a NumPy array, display it as an image.
            subfig = plt.imshow(img)
            if colorbar:
                plt.colorbar(subfig, ax=axs, fraction=0.046, pad=0.04)
            plt.axis('off')
        else:
            print(f"Skipping image {i + 1} as it is not a valid image.")

    plt.tight_layout()
    plt.show()


def plot_grids(u: np.ndarray, v: np.ndarray, ax: Optional[plt.Axes] = None, **kwargs) -> None:
    """
    Plot grid lines on a Matplotlib Axes object.

    Parameters:
    - u (np.ndarray): Horizontal grid lines.
    - v (np.ndarray): Vertical grid lines.
    - ax (plt.Axes, optional): Matplotlib Axes object to plot on (default: None).

    Note:
    This function plots grid lines on a Matplotlib Axes object using horizontal and vertical
    arrays u and v.

    """
    
    ax = ax or plt.gca()
    segs1 = np.stack((u, v), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_sequence_images(image_array):
    """
    Display a sequence of images as an animation/video in a Jupyter notebook.

    Args:
        image_array (numpy.ndarray): An array of images with shape (num_images, height, width, num_channels).

    Note:
    This function displays a sequence of images as an animation in a Jupyter notebook.
    """
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array), interval=33, repeat_delay=1, repeat=True)
    display(HTML(anim.to_html5_video()))
    plt.close()


def save_sequence_images(file_path, image_array, fps=15):
    """
    Save a sequence of images as a video file.

    Args:
        file_path (str): The file path to save the video.
        image_array (numpy.ndarray): An array of images with shape (num_images, height, width, num_channels).
        fps (int): Frames per second for the video (default: 15).

    Note:
    This function saves a sequence of images as a video file at the specified file_path.
    """
    dpi = 300.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array), interval=33, repeat_delay=1, repeat=False)
    #     Writer = animation.Writers['ffmpeg']
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(file_path, writer)
    plt.close()
#     anim.save(file_path, dpi=300, writer=PillowWriter(fps=25))


def join_path(*args):
    return os.path.join(*args).replace("\\", "/")