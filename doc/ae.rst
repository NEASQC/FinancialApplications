
QQuantLib.AE
============

The **Amplitude Estimation (AE)** package comprises modules that implement different **AE** algorithms as Python classes. For a given quantum unitary operator, usually called *oracle*, **AE** algorithms aim to find the **probability** of a given target state when the operator is applied. The modules presented in the package are:

* :doc:`ae.montecarlo_ae`: This module contains the Python class MCAE (for Monte Carlo Amplitude Estimation) that uses direct measurement for obtaining the desired estimation of the **probability**.
* :doc:`ae.ae_classical_qpe`: This module contains the Python class CQPEAE that implements the classical Quantum Phase Estimation algorithm.
* :doc:`ae.ae_iterative_quantum_pe`: This module contains the Python class IQPEAE that implements the *Kitaev* Iterative Quantum Phase Estimation algorithm.
* :doc:`ae.maximum_likelihood_ae`: This module contains the Python class MLAE that implements the Maximum Likelihood Amplitude Estimation algorithm.
* :doc:`ae.mlae_utils`: This module contains functions used for the: *QQuantLib.AE..maximum_likelihood_ae* module.
* :doc:`ae.iterative_quantum_ae`: This module contains the Python class IQAE that implements the Iterative Quantum Amplitude Estimation algorithm.
* :doc:`ae.miterative_quantum_ae`: This module contains the Python class mIQAE that implements a modification of the Iterative Quantum Amplitude Estimation algorithm that increases the performance over the IQAE.
* :doc:`ae.bayesian_ae`: This module contains the Python class BAYESQAE that implements the Bayesian Quantum Amplitude Estimation algorithm. 
* :doc:`ae.real_quantum_ae`: This module contains the Python class RQAE that implements the Real Quantum Amplitude Estimation algorithm. This algorithm estimates the **amplitude** of the given target state instead of its probability.
* :doc:`ae.mreal_quantum_ae`: This module contains the Python class mRQAE that implements a modification of the Real Quantum Amplitude Estimation algorithm that increases theoretical performance over the RQAE. This algorithm estimates the **amplitude** of the given target state instead of its probability.
* :doc:`ae.sreal_quantum_ae`: This module contains the Python class sRQAE that implements a modification of the Real Quantum Amplitude Estimation algorithm where the user can provide the number of shots the generated quantum circuits should be measured. This algorithm estimates the **amplitude** of the given target state instead of its probability.
* :doc:`ae.ereal_quantum_ae`: This module contains the Python class eRQAE that implements an extended Real Quantum Amplitude Estimation algorithm where the user can guide the evolution of the algorithm by providing a schedule. This algorithm estimates the **amplitude** of the given target state instead of its probability.
* :doc:`ae.ae_class`: This module contains the Python class AE which can be used as a selector of the different classes that the **AE** package has implemented.


.. toctree::
   :maxdepth: 1
   :hidden:

   ae.montecarlo_ae.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.ae_classical_qpe.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.ae_iterative_quantum_pe.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.maximum_likelihood_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.mlae_utils.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.iterative_quantum_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.miterative_quantum_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.bayesian_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.real_quantum_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.mreal_quantum_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.sreal_quantum_ae.rst 

.. toctree::
   :maxdepth: 1
   :hidden:

   ae.ereal_quantum_ae.rst 
 
.. toctree::
   :maxdepth: 1
   :hidden:

   ae.ae_class.rst

