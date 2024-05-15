QQuantLib.DL
============

The **Data Loading (DL) package** comprises modules responsible for loading data into quantum circuits. Two different module can be found in this package:

* :doc:`dl.data_loading`: this module contains basic functions for creating quantum circuits that loads input numpy arrays in the amplitude of the different quantum states.
* :doc:`dl.encoding_protocols`: this module implements the **Encoding** Python class that allows to encode input numpy arrays in amplitudes of quantum states using different encoding protocols.


.. toctree::
   :maxdepth: 1
   :caption: data loading
   :hidden:


   dl.data_loading.rst


.. toctree::
   :maxdepth: 1
   :caption: Encoding protocols
   :hidden:

   dl.encoding_protocols.rst
