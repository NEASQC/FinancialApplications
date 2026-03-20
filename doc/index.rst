.. index

.. only:: header

    NEASQC Project
    ==============

.. only:: html

    Welcome to the FinancialApplications documentation
    ==================================================

    .. image:: logo-neasqc.svg
        :scale: 50%
        :align: center
        :alt: NEASQC Project

    Documentation of the *Quantum Quantitative Finance Library* (**QQuantLib**) associated with the use case of *Financial Applications* of the Work Package **Machine Learning & Optimisation** of the NEASQC European project.

    **QQuantLib** is a Python library developed using **myQLM** (**EVIDEN** quantum software stack) containing following packages:

* :doc:`dl`: This is the *Data Loading (DL)* package which contains modules related to the loading of the data.
* :doc:`aa`: This is the *Amplitude Amplification (AA)* package which contains modules related to amplitude amplification operators.
* :doc:`pe`: This is the *Phase Estimation*  package which contains modules for phase estimation algorithms that can be used in amplitude estimation procedures. 
* :doc:`ae`: This is the *Amplitude Estimation* package which is devoted to different amplitude amplification algorithms.
* :doc:`finance`: This package implements different modules related to finance applications of *Amplitude Estimation* techniques.
* :doc:`qml4var`: This package contains modules for training **PQCs** for using as surrogate models for Financial **CDFs** for VaR computations.
* :doc:`qpu`: This package contains a module for selecting the different **EVIDEN** *Quantum Process Units* (**QPUs**) for simulating the different circuits created by the different modules of the **QQuantLib** library.
* :doc:`utils`: This package contains auxiliary modules used for all the beforementioned packages.


    NEASQC project has received funding from the European Union’s Horizon 2020 research and innovation programme under Grant Agreement No. 951821. https://www.neasqc.eu/

    Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

.. toctree::
   :maxdepth: 1
   :hidden:

   dl.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   aa.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   pe.rst


.. toctree::
   :maxdepth: 1
   :hidden:

   finance.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   qml4var.rst


.. toctree::
   :maxdepth: 1
   :hidden:

   qpu.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   utils.rst


