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
    * iterative\_quantum\_ae.py (QQuantLib.AE.iterative\_quantum\_ae). This package implements *Iterative Quantum Amplitude Estimation* (**IQAE**) algorithm. The algorithm was implemented as a python class called *IQAE* 
    * real\_quantum\_ae.py (QQuantLib.AE.real\_quantum\_ae). This package implements *Real Quantum Amplitude Estimation* (**RQAE**) algorithm. The algorithm was implemented as a python class called *RQAE* 
* *Phase Estimation* or **PE** package. Under **QQuantLib/PE**. This package contains modules for phase estimation algorithms that can be used in amplitude estimation procedure. 
    * iterative\_quantum\_pe.py (QQuantLib.PE.iterative\_quantum\_pe). This modules implements the *Kitaev Iterative Phase Estimation* (**IPE**) algorithm as python class called: *IQPE* 
    * phase\_estimation\_wqft.py (QQuantLib.PE.phase\_estimation\_wqft). This modules implements the classical Phase Estimation algorithm (with inverse of Quantum Fourier Transformation) as a python class called: *PE_QFT*.
* *utils* package. Under **QQuantLib/utils**. This package contains several modules with different utilities used for the before packages.
    * data\_extracting.py (QQuantLib.utils.data\_extracting). This module implements functions for creating QLM Programs from AbstractGates or QRoutines, creating their correspondent quantum circuits and jobs, simulating them and post processing the obtained results.
    * qlm\_solver.py (QQuantLib.utils.qlm\_solver). Module for calling the QLM solver.
    * utils.py (QQuantLib.utils.utils). Module with different auxiliary functions used for the other packages of the library.
    * classical\_finance.py (QQuantLib.utils.classical\_finance). Module with several functions from classical quantitative finance.


## Jupyter Notebooks

The misc/notebooks folder contains jupyter notebooks that explain the use of the different packages and modules of the *QQuantLib* library.This notebooks are listed below:

* 01\_DataLoading\_Module\_Use.ipynb. The working of the *Data Loading* (**QQuantLib/DL**) package (and the data_loading module)  is explained in this module. Several examples of data loading in a quantum circuit are provided.
* 02\_AmplitudeAmplification\_Operators.ipynb. This notebook explains the *Amplitude Amplification* (**QQuantLib/AA**) package working. A carefully revision of the mandatory operators for creating grover-like operators is provided in this notebook. 
* 03\_MaximumLikelihood\_Class.ipynb. This notebook explains the **MLAE** algorithm  and the working of the correspondent module *maximum\_likelihood\_ae* from *Amplitude Estimation* package (QQuantLib.AE.maximum\_likelihood\_ae).
* 04\_Classical\_Phase\_Estimation\_Class.ipynb. The classical Phase Estimation algorithm (with QFT) is explained in this notebook using the module *phase\_estimation\_wqft* from *Phase Estimation* package (QQuantLib.PE.phase\_estimation\_wqft)
 * 05\_Iterative\_Quantum\_Phase\_Estimation\_Class.ipynb. The Kitaev *IPE* algorithm is explained in this notebook using the module *iterative\_quantum\_pe* from *Phase Estimation* package (QQuantLib.PE.iterative\_quantum\_pe)
* 06\_Iterative\_Quantum\_Amplitude\_Estimation\_class.ipynb. The Iterative Quantum Amplitude Estimation is explained in this notebook using the iterative\_quantum\_ae from *Amplitude Estimation* package (QQuantLib.AE.iterative\_quantum\_ae). 
* 07\_Real\_Quantum\_Amplitude\_Estimation\_class.ipynb. The Real Quantum Amplitude Estimation algorithm and it associated class, from the module *real\_quantum\_ae* from  *Amplitude Estimation* package (QQuantLib.AE.real\_quantum\_ae), is presented in this notebook.
* 08\_ApplicationTo\_Finance\_01\_StandardApproach.ipynb. In this notebook we present how to use the diferent **amplitude estimation** classes of the library for computing the expectation of agiven function $f(x)$ when $x$ follows a probability density $p(x)$. 
* 09\_ApplicationTo\_Finance\_02\_Call\_Option\_BlackScholes.ipynb. This notebook uses the **amplitude estimation** classes of the library for solving the pricing problem of a *vanilla european call option* under the **Black Scholes** model.
* 10\_ApplicationTo\_Finance\_03\_StandardApproachProblems.ipynb. This notebook presents several problems of the **amplitude estimation** procedures when are applied to the computation of the expectation of a function. Additionally a new loading procedure is presented and the **RQAE** algorithm is used for solving the presented issues.
* 11\_ApplicationTo\_Finance\_03\_NewDataLoading.ipynb. This notebooks uses the modifications presented on previous one for solving several price estimation of **derivative contracts** using different **amplitude estimation** techniques.


## Acknowledgements

This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## Documentation

The html documentation of the **QQuantLib** library can be access at: https://neasqc.github.io/FinancialApplications
## Test it

You can test the libray in binder using following link:

[Binder Link for QQuantLib](https://mybinder.org/v2/gh/NEASQC/FinancialApplications/HEAD)

