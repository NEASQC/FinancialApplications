QQuantLib.qml4var
=================

This package implements the mandatory modules and functions to train a **Parametric Quantum Circuit (PQC)** that can be used as surrogate models of complex and time consuming financial **Cumulative Distribution Functions (CDF)** using the *myQLM EVIDEN* software.  

The following modules are presented:

* :doc:`qml4var.data_utils`: This module contains functions for generating suitable datasets for training **PQC**s for **CDF** evaluation.
* :doc:`qml4var.architectures`: This module implements a hardware efficient ansatz **PQC** and define the measurement observable. 
* :doc:`qml4var.plugins`: This module contains home made **myQLM Plugins** used for evaluating **PQC** s for a set of trainable and feature parameters.
* :doc:`qml4var.myqlm\_workflows`: This module implements workflows for evaluating the **PQC** s.
* :doc:`qml4var.losses`: This module implements several loss functions.
* :doc:`qml4var.adam`: This module implements the ADAM optimizer.

.. toctree::
   :maxdepth: 1
   :hidden:

   qml4var.data_utils.rst

.. toctree::
   :maxdepth: 1
   :hidden:

    qml4var.architectures.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   qml4var.plugins.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   qml4var.myqlm_workflows.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   qml4var.losses.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   qml4var.adam.rst

