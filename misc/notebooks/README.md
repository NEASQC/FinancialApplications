# The Notebooks

The notebooks present and explain the different functionalities of the **QQuantLib** and how they can be used to solve typical problems encountered in the financial industry. 

The **Financial Applications** use case of the NEASQC *WP5* focuses on developing quantum solutions, algorithms, and software libraries for two different Financial problems: *option pricing* and *VaR* computation.

## Option Pricing

For the option price problem, the **Quantum Accelerated Monte Carlo (QAMC)** algorithm was our starting point (see our project deliverable [D5.4: Evaluation of quantum algorithms for pricing and computation of VaR](https://www.neasqc.eu/wp-content/uploads/2022/10/NEASQC_D5.4-Evaluation_of_quantum_algorithms_for_pricing_and_computation_of_VaR_R1.0.pdf) and references therein). The quantum speed-up of the algorithm is located just at the end of **QAMC**, in the **quantum Amplitude Estimation (AE)** procedure. The **QQuantLib** allows the user to build the different mandatory blocks to simulate the last step of the **QAMC** using the **myQLM EVIDEN** software. In addition to other state-of-the-art **AE** techniques, the **QQuantLib** includes our new **Real Quantum Amplitude Estimation (RQAE)**, developed into the framework of the NEASQC project, that allows to estimate, not only the amplitude of a state but also the sign. Several functionalities, for solving the option pricing problem, using the different available **AE** algorithms were developed too.  

The following notebooks explain the part of the **QQuantLib** devoted to using **AE** algorithms to solve the option pricing problem:

* 00_AboutTheNotebooksAndQPUs.ipynb: This notebook explains how to use the **EVIDEN QPUs** in the framework of the **QQuantLib**.

* 01_Data_Loading_Module_Use.ipynb: This notebook serves as a tutorial for using the **QQuantLib.DL** package used package which allows the loading of classical data into quantum states.

* 02_Amplitude_Amplification_Operators.ipynb: In this notebook the **QQuantLib.AA** package, that allows to build amplification operators (Grover-like), is presented. 

* 03_Maximum_Likelihood_Amplitude_Estimation_Class.ipynb: It explains how to use the **QQuantLib.AE.maximum_likelihood_ae** module which allows the user to implement the *Maximum Likelihood Amplitude Estimation* algorithm easily.

* 04_Classical_Phase_Estimation_Class.ipynb: This notebook explains how the classical *Quantum Phase Estimation (QPE)* algorithm works and how to implement it using the **QQuantLib.PE.classical_qpe** module. Additionally, it is shown how the **QQuantLib.AE.ae_classical_qpe** module can be used for **AE** using the classical *QPE* algorithm.

* 05_Iterative_Quantum_Phase_Estimation_Class.ipynb: This notebook explains how to use the **QQuantLib.PE.iterative_quantum_pe** module for phase estimation using the *Iterative Quantum Phase Estimation (IQPE)* algorithm. Additionally, it is shown how the **QQuantLib.AE.ae_iterative_quantum_pe** module can be used for **AE** using the *IQPE* algorithm.

* 06_Iterative_Quantum_Amplitude_Estimation_class.ipynb: The state-of-the-art algorithm *Iterative Quantum Amplitude Estimation (IQAE)* is presented in this notebook which explains how to implement it using the **QQuantLib.AE.iterative_quantum_ae** module.

* 07_Real_Quantum_Amplitude_Estimation_class.ipynb: This notebook presents our new proposed algorithm *Real Quantum Amplitude Estimation (RQAE)* and the corresponding implementation using the **QQuantLib.AE.real_quantum_ae** module.

* 07-02_Improvements_on_Real_Quantum_Amplitude_Estimation.ipynb: Several modifications and improvements of the *RQAE* algorithm are presented in this notebook.

* 08_AmplitudeEstimation_Class.ipynb: the **QQuantLib.AE.ae_class** module, which gathers easily all the **AE** algorithms implemented in the **QQuantLib**, is presented.

* 09_DataEncodingClass.ipynb: The **QQuantLib.DL.encoding_protocols** module is explained in this notebook. The different procedures for encoding probabilities and functions in quantum states are presented here. In addition to the classical encoding procedure (proposed for Grover and Terry in 2002) two new encoding proposals, that allow to codify properly negative-defined functions in quantum states, are presented here. 

* 10_ApplicationTo_Finance_01_IntegralComputing.ipynb: This notebook explains how the **QAMC** algorithm can be used for evaluating integrals and presents the **QQuantLib.finance.quantum_integration**.

* 11_ApplicationTo_Finance_02_ClassicalFinance.ipynb: This notebook presents the **QQuantLib.finance.classical_finance** module where several functions for classical option pricing are located.

* 12_ApplicationTo_Finance_03_AEPriceEstimation.ipynb: This notebook explains how to use the **QQuantLib.finance.ae_price_estimation** module for computing a price estimation of a given financial derivative under the **Black-Scholes** model. 
* 13_Benchmark_utils.ipynb: Other utilities developed in the **QQuantLib** are presented here.

## Quantum Machine Learning for VaR 

One of the main important tasks in financial institutions is risk assessment particularly the computation of the **Value at Risk (VaR)** which measures the potential loss in case of an unlikely event. This metric requires computing a quantile once a **Cumulative Distribution Function** is available. 

The **QQuantLib.qml4var** package aims to train **Parametric Quantum Circuits (PQC)**, using **myQML EVIDEN** software, that can be used as surrogate models for complex and time-consuming financial **CDF**s. The **QQuantLib.qml4var** package includes a new loss function definition, $R_{L^2, {\bar L}^2}^{S_{\chi}}$, developed into the framework of the NEASQC (see Manzano A. (2024). Contributions to the pricing of financial derivatives contracts in commodity markets and the use of quantum computing in finance, Doctoral dissertation, Universidade da Coruña). This loss function trains **PQC** models with better behaviours along all domain range, than traditional ones when the sizes of training datasets are low. 

The following notebooks serve as tutorials for this part of the library:

* 14_qml4var_Intro.ipynb: Summary notebook where the *VaR* problem, the surrogate model approximation, the workflow and the new loss function are presented. 

* 15_qml4var_DataSets.ipynb: This notebook explains how to use the functions of the **QQuantLib.qml4var.data_utils** module for generating suitable datasets for training the **PQC**s.

* 16_qml4var_BuildPQC.ipynb: It explains how to use the **myQLM EVIDEN** library for building **PQC**s for posterior training. The **QQuantLib.qml4var.architectures** module, where a hardware efficient ansatz is encoded, is presented here.

* 17_qml4var_pqc_evaluation.ipynb: This notebook explains how to use the *Plugin EVIDEN* class for building workflows that allow evaluating the **PQC**s for computing the **CDF** and their corresponding feature derivatives for calculating the **Probability Density Function (PDF)** associated with it. The functionalities that allow these evaluations are gathered into the **QQuantLib.qml4var.myqlm_workflows** module.

* 18_qml4var_loss_computation.ipynb: The $R_{L^2, {\bar L}^2}^{S_{\chi}}$ loss function is presented in this notebook. Additionally, the workflow functions from **QQuantLib.qml4var.myqlm_workflows** module, which allow evaluation of this and other loss functions, are discussed here. 

* 19_qml4var_training.ipynb: The training workflow and the optimizer used, presented in the **QQuantLib.qml4var.adam** module, are explained in this notebook.

* 20_PerformanceComparisons.ipynb: This notebook present a performance comparison between **PQC**s trained using the $R_{L^2, {\bar L}^2}^{S_{\chi}}$ and the traditional **Mean Square Error**. 

* 21_VaR_computation.ipynb: This notebook explains how to use trained **PQC**s for *VaR* computation.
