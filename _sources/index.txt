Common Fate Model and Transform
===============================

This package is a python implementation of the Common Fate Transform and Model to be used for audio source separation as described in an ICASSP 2016 paper "Common Fate Model for Unison source Separation"

.. toctree ::

   usage

Modules API:

.. toctree::
   :maxdepth: 2

   commonfate
   commonfate.transform
   commonfate.model
   commonfate.decompose

Installation
============

.. code:: bash

    pip install commonfate


References
~~~~~~~~~~

If you use this package, please reference the following paper

.. code:: tex

   @inproceedings{stoeter2016cfm,
     TITLE = {{Common Fate Model for Unison source Separation}},
     AUTHOR = {St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Badeau, Roland and Edler, Bernd and Magron, Paul},
     BOOKTITLE = {{41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
     ADDRESS = {Shanghai, China},
     PUBLISHER = {{IEEE}},
     SERIES = {Proceedings of the 41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
     YEAR = {2016},
     KEYWORDS = {Non-Negative tensor factorization ; Sound source separation ; Common Fate Model},
   }
