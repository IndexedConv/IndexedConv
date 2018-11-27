.. role:: hidden
    :class: hidden-section

utils
=============================


.. automodule:: indexedconv.utils
.. currentmodule:: indexedconv.utils


Datasets
--------

:hidden:`HDF5Dataset`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HDF5Dataset
    :members:

:hidden:`NumpyDataset`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NumpyDataset
    :members:

Transforms
----------

:hidden:`NumpyToTensor`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NumpyToTensor
    :members:

:hidden:`SquareToHexa`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SquareToHexa
    :members:

Image processing
----------------

.. autoclass:: PCA
    :members:
.. autofunction:: normalize
.. autofunction:: build_hexagonal_position
.. autofunction:: square_to_hexagonal
.. autofunction:: square_to_hexagonal_index_matrix

Indexed functions
-----------------

.. autofunction:: build_kernel
.. autofunction:: neighbours_extraction
.. autofunction:: create_index_matrix
.. autofunction:: pool_index_matrix
.. autofunction:: prepare_mask
.. autofunction:: img2mat
.. autofunction:: mat2img

Utilities
---------

.. autofunction:: get_gpu_usage_map
.. autofunction:: compute_total_parameter_number
