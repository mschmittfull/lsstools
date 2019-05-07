lsstools
=========================================
Tools for large-scale structure analysis.

Usage
-----
See `perr <https://github.com/mschmittfull/perr>`_ for an example of how to use the code.


Installation
------------
The code requires `nbodykit <https://github.com/bccp/nbodykit>`_ version 0.3.x or higher.

To install this in a new anaconda environment, use for example

.. code-block:: bash

  $ cd ~/anaconda/anaconda/envs
  $ conda create -n nbodykit-0.3.7-env -c bccp -c astropy python=2.7 nbodykit=0.3.7 bigfile  pmesh ujson

Newer versions of nbodykit should also work but are not tested. 

To activate the environment, use

.. code-block:: bash

  $ source activate nbodykit-0.3.7-env

To deactivate it, use

.. code-block:: bash

  $ source deactivate


Contributing
------------
To contribute, create a fork on github, make changes and commits, and submit a pull request on github.

To get consistent code style, run

.. code-block:: bash

  $ yapf -i *.py */*.py