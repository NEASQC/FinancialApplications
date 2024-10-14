# The Notebooks

The notebooks presents and explains the different functionalities of the **QQuantLib** and how they can be used to solve typical problems encountered in the financial industry. 

The **Financial Applications** use case of the NEASQC *WP5* focusses on developing quantum solutions, algorithms and software libraries for two different Financial problems: *option pricing* and *VaR* computation.

## Option Pricing

For the option price problem the **Quantum Accelerated Monte Carlo (QAMC)** algorithm was our starting point. The quantum speed up of the algorithm is located just at the end of **QAMC** in the **quantum Amplitude Estimation (AE)** procedure. In the **QQuantLib** several **AE** algorithms were developed using the *EVIDEN myqlm* library, including the new **Real Quantum Amplitude Estimation (RQAE)** developed in the framework of the NEASQC project. Additionally, several functionalities to use the different programed **AE** algorithms for solving the option pricing problem were developed too.  The notebooks from *01_Data_Loading_Module_Use.ipynb* to *12_ApplicationTo_Finance_03_AEPriceEstimation.ipynb* explain the part of the **QQuantLib** devoted to use **AE** algorithms to solve the option pricing problem.

## Quantum Machine Learning for VaR 
