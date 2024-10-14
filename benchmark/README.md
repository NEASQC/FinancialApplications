# Benchmark utilities

This folder contains three different packages that allows to the user executes benchmarks for testing the more important parts of the *QQuantLib*:

* **compare_ae_probability**: this package allows the user to test and compare the different quantum **AE** algorithms, from *QQuantLib*, easily. This can be done using the *probabiliy_estimation* module from command line. More information, how to uses and summary of results can be found in the notebook *CompareAEalgorithmsOnPureProbability.ipynb* (located inside the folder).
* **q_ae_price**: this package allows the user to test and compare different quantum **AE** algorithms, from **QQuantLib**, for pricing different options derivatives (using the **Black-Scholes** model for stock evolution). More information, how to use and summary of results can be found in the notebook: *Compare_AE_algorithms_On_PriceEstimation.ipynb* (located inside of the folder). Two different modules can be found:
    * *benchmark_ae_option_price.py*: allows the user compute the option price using **AE** algorithms for an user defined option problem (python benchmark_ae_option_price.py -h for help) 
    * benchmark_ae_option_price_step_po.py: allows the user the compute the price of a derivative using different **AE** algorithms when the payoff function can take positive and negative values. In this case, the positive and negative parts of the payoff are loaded separately and two different estimations, using quantum **AE** algorithms, are obtained. These values should be post-processed to obtain the final desired value. 
* **qml4var**: this package allows the user to test the *QQuantLib.qml4var* package. The following different modules (that can be executed from command line) can be found:
    * *data_sets*: this module allows to the user build datasets for training a **PQC**. The user can select between a random or a properly configured **Black-Scholes** (a.k.a. log normal) distribution function. The module builds and stores the train and test datasets.
    * *new_training*: allows the user to train a properly configured **PQC** using a given training dataset, a configured Adam optimizer and the $R_{L^2, \\bar{L}^2}^{S_{\\chi}}$ loss function.
    * *new_training.py*: allows the user to continue the training of a **PQC** by loading the last stored trainable weights. The Loss function will be: $R_{L^2, \\bar{L}^2}^{S_{\\chi}}$
    * *new_training_mse*: allows the user to train a properly configured **PQC** using a given training dataset, a configured Adam optimizer and the **Mean Square Error (MSE)** loss function.
    
