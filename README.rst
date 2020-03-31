Indexed Convolution
===================

The indexed operations allow the user to perform convolution and pooling on non-Euclidian grids of data given that the neighbors pixels of each pixel is known and provided.

It gives an alternative to masking or resampling the data in order to apply standard Euclidian convolution.
This solution has been developed in order to apply convolutional neural networks to data from physics experiments that propose specific pixels arrangements.

It is used in the `GammaLearn project <https://lapp-gitlab.in2p3.fr/GammaLearn/>`_ for the Cherenkov Telescope Array.


Here you will find the code for the indexed operations as well as applied examples. The current implementation has been done for pytorch.

`Documentation may be found online. <https://indexed-convolution.readthedocs.io/en/latest/>`_

.. image:: https://travis-ci.org/IndexedConv/IndexedConv.svg?branch=master
    :target: https://travis-ci.org/IndexedConv/IndexedConv
.. image:: https://readthedocs.org/projects/indexed-convolution/badge/?version=latest
    :target: https://indexed-convolution.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://anaconda.org/gammalearn/indexedconv/badges/installer/conda.svg
    :target: https://anaconda.org/gammalearn/indexedconv
    
Install
=======

Install from IndexedConv folder:

.. code-block:: bash

    python setup.py install
    
Install with pip:

.. code-block:: bash

    pip install indexedconv

Install with conda:

    conda install -c gammalearn indexedconv


Requirements
------------

.. code-block:: bash

    "torch>=1.4",
    "torchvision",
    "numpy",
    "matplotlib",
    "h5py",
    "sphinxcontrib-katex"


Running an experiment
=====================
For example, to train the network with indexed convolution on the CIFAR10 dataset transformed to hexagonal:

.. code-block:: bash

    python examples/cifar_indexed.py main_folder data_folder experiment_name --hexa --batch 125 --epochs 300 --seeds 1 2 3 4 --device cpu

In order to train on the AID dataset, it must be downloaded and can be found `here <https://captain-whu.github.io/AID/>`_.

Authors
=======

The development of the indexed convolution is born from a collaboration between physicists and computer scientists.

- Luca Antiga, Orobix
- Mikael Jacquemont, LAPP (CNRS), LISTIC (USMB)
- Thomas Vuillaume, LAPP (CNRS)

References
==========
If you want to use IndexedConv, please cite:

.. image:: https://zenodo.org/badge/150430897.svg
   :target: https://zenodo.org/badge/latestdoi/150430897

Publication
-----------
Jacquemont, M.; Antiga, L.; Vuillaume, T.; Silvestri, G.; Benoit, A.; Lambert, P. and Maurin, G. (2019). **Indexed Operations for Non-rectangular Lattices Applied to Convolutional Neural Networks**. In Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP, ISBN 978-989-758-354-4, pages 362-371. DOI: 10.5220/0007364303620371

Contributing
============

All contributions are welcome.    

Start by contacting the authors, either directly by email or by creating a GitHub issue.
