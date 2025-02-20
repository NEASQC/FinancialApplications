QQuantLib.finance
=================

The **finance** package implements several Python modules related to financial industry problems. The following modules are presented:

* :doc:`finance.classical_finance`: This module implements several classical quantitative finance functions.
* :doc:`finance.probability_class`: This module implements the Python class DensityProbability that allows to the user configure a typical Black-Scholes (log-normal) probability density function by providing different financial parameters.
* :doc:`finance.payoff_class`: This module implements the Python class PayOff that allows to the user configure payoffs for several financial derivatives (like options or futures) by providing different financial parameters. 
* :doc:`finance.quantum_integration`: This module implements the function *q_solve_integral* that allows to the user codify into a quantum circuit a desired integral (using the different encoding procedures from **QQuantLib.DL.encoding_protocols** module) and solving it using the different **AE** algorithms implemented in the **QQuantLib.AE** package.
* :doc:`finance.ae_price_estimation`: This module implements the ae_price_estimation function that allows to the user configure a financial derivative price problem using typical financial parameters, codify the expected value integral in a quantum circuit, and solve it by using the different **AE** algorithms implemented in the **QQuantLib.AE** package. This module uses the *finance.quantum_integration* module.
* :doc:`finance.ae_price_estimation_step_payoff`: This module implements the ae_price_estimation_step_po function that allows to the user configure a financial derivative price problem using typical financial parameters, codify the expected value integral in a quantum circuit, and solve it by using the different **AE** algorithms implemented in the **QQuantLib.AE** package. This module uses the *finance.quantum_integration* module.  In this module, the positive part and the negative parts of the payoff will be loaded separately and the quantum estimations of the two parts are post-processed to get the desired price.
* :doc:`finance.cliquet_return_estimation`: This module implements the ae_cliquet_estimation function that allows to the user configure a price problem for cliquet options products using using their typical financial parameters. It codifies the desired expected value in a quantum circuit, and solve it by using the different **AE** algorithms implemented in the **QQuantLib.AE** package. This module uses the *finance.quantum_integration* module. Cliquet options can have positive or negatives values for the expected payoff. 
* :doc:`finance.cliquet_return_estimation_step_payoff`: This module implements the ae_cliquet_estimation function that allows to the user configure a price problem for cliquet options products using using their typical financial parameters. It codifies the desired expected value in a quantum circuit, and solve it by using the different **AE** algorithms implemented in the **QQuantLib.AE** package. This module uses the *finance.quantum_integration* module. Cliquet options can have positive or negatives values for the expected payoff. In this module, the positive part and the negative parts of the payoff will be loaded separately and the quantum estimations of the two parts are post-processed to get the desired price.


.. toctree::
   :maxdepth: 1
   :hidden:

   finance.classical_finance.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.probability_class.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.payoff_class.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.quantum_integration.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.ae_price_estimation.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.ae_price_estimation_step_payoff.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.cliquet_return_estimation.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   finance.cliquet_return_estimation_step_payoff.rst
