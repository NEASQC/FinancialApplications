# Financial Applications

This repo is associated to the use case of *Financial Applications* of **Machine Learning & Optimisation** group of use cases of the NEASQC European project. The main idea for this repo is the development of a QLM library, called *Quantum Quantitative Finance Library* (**QQuantLib** from now) that assemble different quantum algorithms and techniques for using in the financial industry.



## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

The *requirements.txt* file contains the mandatory python libraries for using present libraries.


## Library organisation 

library *Quantum Quantitative Finance Library* is deployed as typical Python library in the **QQuantLib** folder (we can use *import QQuantLib* for have access to the complete library). The library was organised in the following packages:
* *Data Loading* or **DL** package. Under **QQuantLib/DL**. This package contains modules related with the loading of the data into the quantum circuits.
    * data\_loading.py (QQuantLib.DL.data\_loading): this modules deals with functions for loading data in quantum circuits using two different kind of approximation: the brute force one where multi-controlled rotations are used directly and using a more efficient method based on *quantum multiplexors*.
  *  encoding\_protocols.py (QQuantLib.DL.encoding\_protocols): in this module a class, called **Encoding**, was implemented for encoding numpy arrays into a quantum circuit. 
* *Amplitude Amplification* or **AA** package. Under **QQuantLib/AA**. This package contains modules for creating amplitude amplification (or Grover-like) operators.
    * amplitude\_amplification.py (QQuantLib.AA.amplitude\_amplification). This module contains functions for creating mandatory operators for amplitude amplifications and grover-like operators as QLM AbstractGate or QRoutines.
* *Amplitude Estimation* or **AE** package. Under **QQuantLib/AE**. This package is devoted to the implementation of different amplitude amplification algorithms.
    * maximum\_likelihood\_ae.py (QQuantLib.AE.maximum\_likelihood\_ae). This package implements *Maximum Likelihood Amplitude estimation* (**MLAE**) algorithm. The algorithm was implemented as a python class called *MLAE* 
    * iterative\_quantum\_ae.py (QQuantLib.AE.iterative\_quantum\_ae). This package implements *Iterative Quantum Amplitude Estimation* (**IQAE**) algorithm. The algorithm was implemented as a python class called *IQAE* 
    * real\_quantum\_ae.py (QQuantLib.AE.real\_quantum\_ae). This package implements *Real Quantum Amplitude Estimation* (**RQAE**) algorithm. The algorithm was implemented as a python class called *RQAE* 
    * ae\_classical\_qpe.py (QQuantLib.AE.ae\_classical\_qpe). This packages uses the cQPE class from QQuantLib.PE.classical\_qpe for solving amplitude estimation problems. The algorithm was implemented as a python class called *cQPE_AE*  
    * ae\_iterative\_quantum\_pe.py (QQuantLib.AE.ae\_iterative\_quantum\_pe). This packages uses the IQPE class from QQuantLib.PE.iterative\_quantum\_pe for solving amplitede estimations problem. The algorithm was implemented as a python class called *IQPE_AE*   
    * ae\_class.py: (QQuantLib.AE.ae\_clas). This module implements a python class, called **AE**, for selecting and configuring properly the different amplitude estimation algorithms implemented into the **QQuantLib.AE** package.
* *Phase Estimation* or **PE** package. Under **QQuantLib/PE**. This package contains modules for phase estimation algorithms that can be used in amplitude estimation procedure. 
    * iterative\_quantum\_pe.py (QQuantLib.PE.iterative\_quantum\_pe). This modules implements the *Kitaev Iterative Phase Estimation* (**IPE**) algorithm as python class called: *IQPE* 
    * classical\_qpe.py (QQuantLib.PE.classical\_qpe). This modules implements the classical Phase Estimation algorithm (with inverse of Quantum Fourier Transformation) as a python class called: *cQPE*.
* *utils* package. Under **QQuantLib/utils**. This package contains several modules with different utilities used for the before packages.
    * data\_extracting.py (QQuantLib.utils.data\_extracting). This module implements functions for creating QLM Programs from AbstractGates or QRoutines, creating their correspondent quantum circuits and jobs, simulating them and post processing the obtained results.
    * qlm\_solver.py (QQuantLib.utils.qlm\_solver). Module for calling the QLM solver.
    * utils.py (QQuantLib.utils.utils). Module with different auxiliary functions used for the other packages of the library.
* *finance* package. Under **QQuantLib/finance**. This package implements several modules for using amplitude estimation techniques for solving financial problems.
    * classical\_finance.py: (QQuantLib.finance.classical\_finance). Module with several functions from classical quantitative finance.
    * probability\_class.py: (QQuantLib.finance.probability\_class). Module where a class for defining and configuring typical financial probability densities are implemented.
    * payoff\_class.py: (QQuantLib.finance.payoff\_class). Module where a class for defining and configuring otpion derivative returns are implemented.
    * quantum\_integration.py: (QQuantLib.finance.quantum\_integration). Module for computing integrals using Amplitude Estimation techniques.
    * ae\_price\_estimation.py: (QQuantLib.finance.ae\_price\_estimation). Module for computing option price estimation uisng Amplitude Estimation techniques. 


## Jupyter Notebooks

The misc/notebooks folder contains jupyter notebooks that explain the use of the different packages and modules of the *QQuantLib* library.This notebooks are listed below:

* 01\_DataLoading\_Module\_Use.ipynb. The working of the *Data Loading* (**QQuantLib/DL**) package (and the data_loading module)  is explained in this module. Several examples of data loading in a quantum circuit are provided.
* 02\_AmplitudeAmplification\_Operators.ipynb. This notebook explains the *Amplitude Amplification* (**QQuantLib/AA**) package working. A carefully revision of the mandatory operators for creating grover-like operators is provided in this notebook. 
* 03\_MaximumLikelihood\_Class.ipynb. This notebook explains the **MLAE** algorithm  and the working of the correspondent module *maximum\_likelihood\_ae* from *Amplitude Estimation* package (QQuantLib.AE.maximum\_likelihood\_ae).
* 04\_Classical\_Phase\_Estimation\_Class.ipynb. The classical Phase Estimation algorithm (with QFT) is explained in this notebook using the module *classical\_qpe* from *Phase Estimation* package (QQuantLib.PE.classical\_qpe)
 * 05\_Iterative\_Quantum\_Phase\_Estimation\_Class.ipynb. The Kitaev *IPE* algorithm is explained in this notebook using the module *iterative\_quantum\_pe* from *Phase Estimation* package (QQuantLib.PE.iterative\_quantum\_pe)
* 06\_Iterative\_Quantum\_Amplitude\_Estimation\_class.ipynb. The Iterative Quantum Amplitude Estimation is explained in this notebook using the iterative\_quantum\_ae from *Amplitude Estimation* package (QQuantLib.AE.iterative\_quantum\_ae). 
* 07\_Real\_Quantum\_Amplitude\_Estimation\_class.ipynb. The Real Quantum Amplitude Estimation algorithm and it associated class, from the module *real\_quantum\_ae* from  *Amplitude Estimation* package (QQuantLib.AE.real\_quantum\_ae), is presented in this notebook.
* 08\_AmplitudeEstimation\_Class.ipynb. In this notebook the **AE** (QQuantLib.AE.ae\_class) class is presented. This class allows configuring and executing any of the *AE* algorithms of the **AE** package, given an input oracle, in an easy way.
* 09\_DataEncodingClass.ipynb. The **Encoding** class (QQuantLib.DL.encoding\_protocols) is presented in this notebook. This class deals with different encoding protocols used for creating quantum oracles suitable for computing integrals using amplitude estimation algorithms.
* 10\_ApplicationTo\_Finance\_01\_IntegralComputing.ipynb. This notebooks explains how to use the **q_solve_integral** (QQuantLib.finance.quantum\_integration) function for computing integrals using the different **AE** algorithms presented into the *AE* package.
* 11\_ApplicationTo\_Finance\_02\_ClassicalFinance.ipynb. This notebook presented several classical finance concepts related with derivatives price estimation. The **classical_finance** module, the **DensityProbability** and the **PayOff** classes  (QQuantLib.finance package) are introducing in this notebook.
* 12\_ApplicationTo\_Finance\_03\_AEPriceEstimation.ipynb. This notebook explains how to use the **ae_price_estimation** function (*from QQuantLib.finance.ae\_price\_estimatio*) for computing option price estimation with the help of amplitude estimation techniques.


## Acknowledgements

This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## Documentation

The html documentation of the **QQuantLib** library can be access at: https://neasqc.github.io/FinancialApplications
## Test it

You can test the library in binder using following link:

[Binder Link for QQuantLib](https://mybinder.org/v2/gh/NEASQC/FinancialApplications/HEAD)

