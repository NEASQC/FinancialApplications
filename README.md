# Financial Applications

This repo is associated with the Financial Applications use case of the Machine Learning & Optimisation group of the NEASQC European project.
Its primary focus is the development of the Python library *Quantum Quantitative Finance Library* (**QQuantLib**), which encompasses various state-of-the-art quantum algorithms and techniques tailored for the financial industry. Many of these contributions were developed within the project framework.
The **QQuantLib** was programmed using the quantum software stack myQLM developed by EVIDEN.

## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

The mandatory Python libraries and packages for using the **QQuantLib** can be found into the *environment.yml* file.

## Library organisation 

The *Quantum Quantitative Finance Library* is structured as a standard Python library within the *QQuantLib* folder. It is organized into several packages:

1. **Data Loading (DL) package**: Located at *QQuantLib/DL*, this package comprises modules responsible for loading data into quantum circuits.
2. **Amplitude Amplification (AA) package**: Found at *QQuantLib/AA*, this package contains modules for creating amplitude amplification (or Grover-like) operators.
3. **Phase Estimation (PE) package**: Located at *QQuantLib/PE*, this package includes modules for phase estimation algorithms applicable in amplitude estimation procedures.
4. **Amplitude Estimation (AE) package**: Situated in *QQuantLib/AE*, this package focuses on implementing various amplitude estimation algorithms.
5. **Finance package**: Housed in *QQuantLib/finance*, this package implements modules for employing amplitude estimation techniques to solve financial problems.
6. **Quantum Machine Learning for VaR (qml4var)**: Located in  *QQuantLib/qml4var*, this package focus on implementing mandatory workflow for building and training **Parametric Quantum Circuits (PQC)** that can be used as a surrogate models for complex and time-consuming financial distributions. The trained surrogate models can be used for risk analysis and *VaR* computations.
6. **QPU package**: Located at *QQuantLib/qpu*, this package comprises modules for selecting different EVIDEN Quantum Process Units (**QPU**s) compatible with the QQuantLib library.
7. **Utils package**: Positioned at *QQuantLib/utils*, this package contains multiple modules with utilities utilized across other packages.

## Jupyter Notebooks

A series of Jupyter notebooks have been developed in the **misc/notebooks** folder as tutorials. These notebooks explain the functionality of the various packages and modules within the library, as well as demonstrate how to utilize them to solve typical problems encountered in the financial industry.

## Benchmark folder

In the benchmark folder, three Python packages are presented to assess the performance of various quantum algorithms:

1. **compare_ae_probability**: This package enables easy configuration of different amplitude estimation algorithms and their application to a simple amplitude estimation problem (this is getting the probability of a fixed state when a probability density array is loaded into a quantum circuit). For comparison between AE methods please review notebook CompareAEalgorithmsOnPureProbability.ipynb.
2. **q_ae_price**: This package simplifies the configuration of price estimation problems for different financial derivatives (call and put options, and futures) and solves them using various configurations of quantum amplitude estimation algorithms. For comparison between AE algorithms please refer to: *Compare_AE_algorithms_On_PriceEstimation.ipynb*
3. **qml4var**: This package allows to the user trains **PQC** that can be used as surrogate models of time consuming financial distribution functions.
4. **sine_integral**: this package allows the user to test the *QQuantLib.finance.quantum\_integration* module by estimating the defined integral of a sine function in two different domains.

## Acknowledgements

This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## Documentation

The html documentation of the **QQuantLib** library can be access at: https://neasqc.github.io/FinancialApplications
## Test it

You can test the library in binder using following link:

[Binder Link for QQuantLib](https://mybinder.org/v2/gh/NEASQC/FinancialApplications/HEAD)

