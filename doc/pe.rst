QQuantLib.PE
============

The **Phase Estimation (PE)** package includes modules for implementing *Quantum Phase Estimation* (**QPE**) algorithms. The following modules were implemented: 

* :doc:`pe.classical_qpe`: This module implements the Python class CQPE that implements the *QPE* algorithm that returns the eigenvalues of a given input unitary operator and an initial eigenstate (or a linear combination of them). The algorithm uses the Quantum Fourier Transformation Routine.
* :doc:`pe.iterative_quantum_pe`: This module implements the Python class IQPE that implements the *Kitaev Iterative Phase Estimation* algorithm that returns the eigenvalues of a given input unitary operator and an initial eigenstate (or a linear combination of them).


.. toctree::
   :maxdepth: 1
   :hidden:

   pe.classical_qpe.rst

.. toctree::
   :maxdepth: 1
   :hidden:


   pe.iterative_quantum_pe.rst 


