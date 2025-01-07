# Benchmark utilities

This folder contains five different packages which allow to the user execute benchmarks for testing the more important parts of the *QQuantLib*:

* **compare_ae_probability**: this package allows the user to test and compare the different quantum **AE** algorithms, from *QQuantLib*, easily. This can be done using the *probabiliy_estimation* module from the command line. How to use, results and more information can be found in the notebook *CompareAEalgorithmsOnPureProbability.ipynb* (located inside the folder).

* **q_ae_price**: this package allows the user to test and compare different quantum **AE** algorithms, from **QQuantLib**, for pricing different options derivatives (using the **Black-Scholes** model for stock evolution). How to use, summary of results and more information can be found in the notebook *Compare_AE_algorithms_On_PriceEstimation.ipynb* (located inside of the folder). Two different modules can be found in the package:

    * *benchmark_ae_option_price.py*: allows the user compute the option price using **AE** algorithms for an user defined option problem (python benchmark_ae_option_price.py -h for help) 

    * *benchmark_ae_option_price_step_po.py*: allows the user the compute the price of a derivative using different **AE** algorithms when the payoff function can take positive and negative values. In this case, the positive and negative parts of the payoff are loaded separately and two different estimations, using quantum **AE** algorithms, are obtained. These values should be post-processed to obtain the final desired value.

* **sine_integral**: this package allows the user to test the *QQuantLib.finance.quantum\_integration* module by estimating the defined integral of a sine function in two different domains. How to use, results and more information can be found in the notebook: *QAE_SineIntegration_WindowQPE.ipynb* (located inside the folder).

* **q_ae_cliquet**: this package allows the user to test and compare the different quantum **AE** algorithms, from *QQuantLib*, for pricing a type of exotic options: the *Cliquet Options* (under stock evolution using the *Black-Scholes* model). How to use, summary of results and more information can be found in the notebook *QAE_CliquetOptions.ipynb* (located inside of the folder). Two different modules can be found in the package:

    * *benchmark_cliquet.py*: computes the return of the configured Cliquet option using a properly configured *AE* algorithm (python benchmark_cliquet.py -h for help)

    * *benchmark_cliquet_step_po.py*: computes the return of the configured Cliquet option using a properly configured *AE* algorithm when the payoff function can take positive and negative values. In this case, the positive and negative parts of the payoff are loaded separately and two different estimations, using quantum **AE** algorithms, are obtained. These values should be post-processed to obtain the final desired value.


easily. This can be done using the *probabiliy_estimation* module from the command line. How to use, results and more information can be found in the notebook *CompareAEalgorithmsOnPureProbability.ipynb* (located inside the folder).

* **qml4var**: this package allows the user to test the *QQuantLib.qml4var* package. The following different modules (that can be executed from the command line) can be found:

    * *data_sets*: this module allows to the user build datasets for training a **PQC**. The user can select between a random or a properly configured **Black-Scholes** (a.k.a. log-normal) distribution function. The module builds and stores the train and test datasets.

    * *new_training*: allows the user to train a properly configured **PQC** using a given training dataset, a configured Adam optimizer and the $R_{L^2, {\bar L}^2}^{S_{\chi}}$ loss function.

    * *new_training.py*: allows the user to continue the training of a **PQC** by loading the last stored trainable weights. The Loss function will be: $R_{L^2, {\bar L}^2}^{S_{\chi}}$

    * *new_training_mse*: allows the user to train a properly configured **PQC** using a given training dataset, a configured Adam optimizer and the **Mean Square Error (MSE)** loss function.
    
