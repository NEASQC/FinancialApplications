# The Notebooks

The notebooks presents and explains the different functionalities of the **QQuantLib** and how they can be used to solve typical problems encountered in the financial industry. 

The **Financial Applications** use case of the NEASQC *WP5* focusses on developing quantum solutions, algorithms and software libraries for two different Financial problems: *option pricing* and *VaR* computation.

## Option Pricing

For the option price problem the **Quantum Accelerated Monte Carlo (QAMC)** algorithm was our starting point (see our project deliverable D[https://www.neasqc.eu/wp-content/uploads/2022/10/NEASQC_D5.4-Evaluation_of_quantum_algorithms_for_pricing_and_computation_of_VaR_R1.0.pdf](5.4: Evaluation of quantum algorithms for pricing and computation of VaR) and references therein). The quantum speed up of the algorithm is located just at the end of **QAMC**, in the **quantum Amplitude Estimation (AE)** procedure. The **QQuantLib** allows the user to build the different mandatory blocks to simulate the last step of the **QAMC** using the **myQLM EVIDEN** software. In addition to different state-of-the-art **AE** techniques, the **QQuantLib** includes our new **Real Quantum Amplitude Estimation (RQAE)** developed in the framework of the NEASQC project that allows to estimate not only the amplitude of a state but also the sign. Several functionalities to use the different available **AE** algorithms, for solving the option pricing problem, were developed too.  

The following notebooks explain the part of the **QQuantLib** devoted to use **AE** algorithms to solve the option pricing problem:

* 00_AboutTheNotebooksAndQPUs.ipynb: This notebook explains how to use the **EVIDEN QPUs** in the framework of the **QQuantLib**.
* 01_Data_Loading_Module_Use.ipynb: This notebook serves as a tutorial for using the **QQuantLib.DL** package used for loading data into quantum states.
* 02_Amplitude_Amplification_Operators.ipynb: In this notebook the **QQuantLib.AA** package, that allows to build amplification operators (Grover-like), is presented. 
* 03_Maximum_Likelihood_Amplitude_Estimation_Class.ipynb: It explains how to use the **QQuantLib.AE.maximum_likelihood_ae** module that allows to the user implements easily the *Maximum Likelihood Amplitude Estimation* algorithm.
* 04_Classical_Phase_Estimation_Class.ipynb: This notebook explains how the classical **Quantum Phase Estimation** algorithm works and how to implement it using the **QQuantLib.PE.classical_qpe** module. Additionally, it is showed how this algorithm can be used for **Amplitude Estimation** by using the **QQuantLib.AE.ae_classical_qpe** module.
* 05_Iterative_Quantum_Phase_Estimation_Class.ipynb: This notebook explains how to use the **QQuantLib.PE.iterative_quantum_pe** module for phase estimation using the *Iterative Quantum Phase Estimation (IQPE)* algorithm. The **QQuantLib.AE.ae_iterative_quantum_pe** module that uses the *IQPE* for **AE** is presented too.
* 06_Iterative_Quantum_Amplitude_Estimation_class.ipynb: The state-of-the-art *Iterative Quantum Amplitude Estimation (IQAE)* algorithm is presented in this notebook which explain how to implement it using the **QQuantLib.AE.iterative_quantum_ae** module.
* 07_Real_Quantum_Amplitude_Estimation_class.ipynb: This notebook presents our new proposed algorithm *Real Quantum Amplitude Estimation (RQAE)* and the corresponding implementation using the **QQuantLib.AE.real_quantum_ae** module.
* 07-02_Improvements_on_Real_Quantum_Amplitude_Estimation.ipynb: Several modifications and improvements of the *IQAE* algorithm are presented in this notebook.
* 08_AmplitudeEstimation_Class.ipynb: the **QQuantLib.AE.ae_class** module, which gather easily all the **AE** algorithms implemented in the **QQuantLib**, is presented.
* 09_DataEncodingClass.ipynb: The **QQuantLib.DL.encoding_protocols** module is explained in this notebook. The different procedure for encoding probabilities and functions in quantum states are presented here. In addition to the classical encoding procedure (proposed for Grover and Terry in 2002) two new encoding proposals, that allows to encode negative defined functions, are presented here. 
* 10_ApplicationTo_Finance_01_IntegralComputing.ipynb: This notebook explains how the **QAMC** algorithm can be used for evaluating integrals and presents the **QQuantLib.finance.quantum_integration**.
* 11_ApplicationTo_Finance_02_ClassicalFinance.ipynb: This notebook presents the **QQuantLib.finance.classical_finance** module where several functions for classical option pricing are located.
* 12_ApplicationTo_Finance_03_AEPriceEstimation.ipynb: This notebook explains how to use the **QQuantLib.finance.ae_price_estimation** module for computing a price estimation of a given financial derivative under the **Black-Scholes** model. 
* 13_Benchmark_utils.ipynb: Other utilities developed in the **QQuantLib** are presented here.

## Quantum Machine Learning for VaR 
* 14_qml4var_Intro.ipynb
* 15_qml4var_DataSets.ipynb
* 16_qml4var_BuildPQC.ipynb
* 17_qml4var_pqc_evaluation.ipynb
* 18_qml4var_loss_computation.ipynb
* 19_qml4var_training.ipynb
* 20_PerformanceComparisons.ipynb
* 21_VaR_computation.ipynb
