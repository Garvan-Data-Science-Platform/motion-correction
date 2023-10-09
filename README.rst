=================
motion_correction
=================


.. image:: https://img.shields.io/pypi/v/motion_correction.svg
        :target: https://pypi.python.org/pypi/motion_correction

.. image:: https://readthedocs.org/projects/motion-correction/badge/?version=latest
        :target: https://motion-correction.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Motion correction algorithms for FLIM data.

* Free software: MIT license
* Documentation: https://motion-correction.readthedocs.io.

GPU Support
-----------

Some algorithms run faster with a GPU. 
`CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>` must be installed to run on GPU.
Alternatively if using conda, `conda install -c nvidia cuda-toolkit`

Features
--------

* Various algorithms to correct FLIM (.ptu and .pt3) files
* Refer to `Tutorial.ipynb <https://github.com/Garvan-Data-Science-Platform/motion-correction/blob/main/desktop/Tutorial.ipynb>` for an example of the library in use

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
