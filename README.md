# Financial Applications

This repo is associated to the use case of *Financial Applications* of **Machine Learning & Optimisation** group of use cases of the NEASQC european project. The main idea for this repo is the development of a QLM library, called *Quantum Quantitative Finance Library* (**QQuantLib** from now) that assemble different quantum algorithms and techniques for using in the financial industry.



## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

The *requirements.txt* file contains the mandatory python libraries for using present libraries.


## Library organisation 

The *Quantum Quantitative Finance Library* is deployed as typical Python library in the **QQuantLib** folder (we can use *import QQuantLib* for have access to the complete library). The library was organised in the following packages:
* *Data Loading* or **DL** package. Under **QQuantLib/DL**. This package contains modules related with the loading of the data into the quantum circuits.
    * data\_loading.py (QQuantLib.DL.data\_loading): this modules deals with functions for loading data in quantum circuits using two different kind of approximation: the brute force one where multi-controlled rotations are used directly and using a more efficient method based on *quantum multiplexors*.
* *Amplitude Amplification* or **AA** package. Under **QQuantLib/AA**. This package contains modules for creating amplitude amplification (or Grover-like) operators.
    * amplitude\_amplification.py (QQuantLib.AA.amplitude\_amplification). This module contains functions for creating mandatory operators for amplitude amplifications and grover-like operators as QLM AbstractGate or QRoutines.
* *Amplitude Estimation* or **AE** package. Under **QQuantLib/AE**. This package is devoted to the implementation of different amplitude amplification algorithms.
    * maximum\_likelihood\_ae.py (QQuantLib.AE.maximum\_likelihood\_ae). This package implements *Maximum Likelihood Amplitude estimation* (**MLAE**) algorithm. The algorithm was implemented as a python class called *MLAE* 
* *Phase Estimation* or **PE** package. Under **QQuantLib/PE**. This package contains modules for phase estimation algorithms that can be used in amplitude estimation procedure. 
    * iterative\_quantum\_pe.py (QQuantLib.PE.iterative\_quantum\_pe). This modules implements the *Kitaev Iterative Phase Estimation* (**IPE**) algorithm as python class called: *IterativeQuantumPE* 
    * phase\_estimation\_wqft.py (QQuantLib.PE.phase\_estimation\_wqft). This modules implements the classical Phase Estimation algorithm (with inverse of Quantum Fourier Transformation) as a python class called: *PhaseEstimationwQFT*.
* *utils* package. Under **QQuantLib/utils**. This package contains several modules with different utilities used for the before packages.
    * data\_extracting.py (QQuantLib.utils.data\_extracting). This module implements functions for creating QLM Programs from AbstractGates or QRoutines, creating their correspondent quantum circuits and jobs, simulating them and post processing the obtained results.
    * qlm\_solver.py (QQuantLib.utils.qlm\_solver). Module for calling the QLM solver.
    * utils.py (QQuantLib.utils.utils). Module with different auxiliary functions used for the other packages of the library.


## Jupyter Notebooks

The misc/notebooks folder contains jupyter notebooks that explain the use of the different packages and modules of the *QQuantLib* library.This notebooks are listed below:

* 01\_DataLoading\_Module\_Use.ipynb. The working of the *Data Loading* (**QQuantLib/DL**) package (and the data_loading module)  is explained in this module. Several examples of data loading in a quantum circuit are provided.
* 02\_AmplitudeAmplification\_Operators.ipynb. This notebook explains the *Amplitude Amplification* (**QQuantLib/AA**) package working. A carefully revision of the mandatory operators for creating grover-like operators is provided in this notebook. 
* 03\_MaximumLikelihood\_Class.ipynb. This notebook explains the **MLAE** algorithm  and the working of the correspondent module *maximum\_likelihood\_ae* from *Amplitude Estimation* package (QQuantLib.AE.maximum\_likelihood\_ae).
* 04\_PhaseEstimationwitQFT\_Class.ipynb. The classical Phase Estimation algorithm (with QFT) is explained in this notebook using the module *phase\_estimation\_wqft* from *Phase Estimation* package (QQuantLib.PE.phase\_estimation\_wqft)
 * 05\_Iterative\_QPE\_Class.ipynb. The Kitaev *IPE* algorithm is explained in this notebook using the module *iterative\_quantum\_pe* from *Phase Estimation* package (QQuantLib.PE.iterative\_quantum\_pe)
* 06_FinancialApplication_Example.ipynb. In this notebook the different developed packages of **QQuantLib** are used for develop an exaple of use for financial industry: calculations of expected value of a function. 

## Acknowledgements

This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## Documentation

The html documentation of the **QQuantLib** library can be access at: https://neasqc.github.io/FinancialApplications
## Test it

You can test the libray in binder using following link:

[Binder Link for QQuantLib](https://mybinder.org/v2/gh/NEASQC/FinancialApplications/HEAD)

