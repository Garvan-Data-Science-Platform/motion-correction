FLIM Motion Correction
======================================

Features
--------

* Various algorithms to correct motion in FLIM (.ptu and .pt3) files

Installation
------------

To install motion_correction, run this command in your terminal:

.. highlight:: shell

.. code-block:: console

    $ pip install motion_correction

This is the preferred method to install motion_correction, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

GPU Support
+++++++++++

Some algorithms run faster with a GPU. 
`CUDA Toolkit 11.2 - 11.8 <https://developer.nvidia.com/cuda-11-8-0-download-archive>`_ must be installed to run on GPU.

Alternatively,
.. code-block:: console

    $ conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit`

.. highlight:: python

Usage
-----

Refer to this this `tutorial notebook <https://github.com/Garvan-Data-Science-Platform/motion-correction/blob/main/examples/Tutorial.ipynb>`_ for correcting FLIM images in a desktop environment.

To use motion_correction in a script::

    from motion_correction import load_ptfile, write_pt3, get_intensity_stack, apply_correction_flim, calculate_correction

Load a pt3 or ptu file as a numpy array (flim_data_stack)::

    #The dimensions of flim_data_stack are (width,height,channels,repititions,nanotimes)
    flim_data_stack, meta = load_ptfile('input.pt3')

Convert flim data stack to an intensity stack for a single channel. Shape: (width,height,repititions)::

    intensity_stack = get_intensity_stack(flim_data_stack, 2)

Choose a local and/or global correction algorithm::

    from motion_correction.algorithms import Morphic
    morphic = Morphic(radius=16)

Correct intensity stack::

    results = calculate_correction(intensity_stack, 0, local_algorithm=morphic)

Apply correction to flim data::

    transform_matrix = results['combined_transforms']
    corrected_flim_data_stack = apply_correction_flim(flim_data_stack, transform_matrix)

Write to .pt3::
    
    write_pt3(meta, corrected_flim_data_stack, "output.pt3")




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   modules
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
