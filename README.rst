lsstools
=========================================
Tools for large-scale structure analysis.

Usage
-----
See `perr <https://github.com/mschmittfull/perr>`_ for an example of how to use the code.


Installation
------------
The code requires `nbodykit <https://github.com/bccp/nbodykit>`_ version 0.3.4 or higher.

To install this in a new anaconda environment, use

.. code-block:: bash

  $ cd ~/anaconda/anaconda/envs
  $ conda create -n nbodykit-env -c bccp -c astropy python=2.7 nbodykit=0.3.4 bigfile pmesh 

To load this environment, use

.. code-block:: bash

  $ source activate nbodykit-env

To deactivate it, use

.. code-block:: bash

  $ source deactivate
