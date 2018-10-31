Indexed Convolution
===================

The indexed operations allow the user to perform convolution and pooling on non-Euclidian grids of data given that the neighbors pixels of each pixel is known and provided.

It gives an alternative to masking or resampling the data in order to apply standard Euclidian convolution.
This solution has been developed in order to apply convolutional neural networks to data from physics experiments that propose specific pixels arrangements.

It is used in the `GammaLearn project <https://lapp-gitlab.in2p3.fr/GammaLearn/>`_ for the Cherenkov Telescope Array.


Here you will find the code for the indexed operations as well as applied examples. The current implementation has been done for pytorch.

`Documentation may be found online. <https://indexedconv.readthedocs.io/en/latest/>`

Install
=======

Install from IndexedConv folder:

.. code-block:: bash

    python setup.py install


Requirements
------------

.. code-block:: bash

    "torch>=0.4",
    "numpy",
    "tensorboardx",
    "matplotlib",


Authors
=======

The development of the indexed convolution is born from a collaboration between physicists and computer scientists.

- Luca Antiga, Orobix
- Mikael Jacquemont, LAPP (CNRS), LISTIC (USMB)
- Thomas Vuillaume, LAPP (CNRS)


Contributing
============

All contributions are welcome.    
Start by contacting the authors, either directly by email or by creating a GitHub issue.
