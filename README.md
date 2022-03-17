# Financial Applications

This repository contains several modules for implementing Quantum  Estimation and Amplification algorithms using QLM for future implementation of a financial toolkit. 

## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

For simplicity, an example of `setup.py` file is provided in this template.
Feel free to modify it if you have exotic build recipes.


## Modules 

The my\_lib/ folder is where the modules (python scripts) of present library were stored. 

The main modules of the library are: 
* **data\_loading**: This modules contains functions for loading data into a quantum circuit. Two different implementations of loading data in the amplitudes of the quantum state were implemented:
    * brute force mode: the data is loaded using multi-controlled rotations directly.
    * quantum multiplexors: the data is loaded using quantum multiplexors that is a much efficient way than the first method.
* **amplitude\_amplification**: This module implements all mandatory operators (that are implemented as QLM AbstractGate) for creating Grover-like operators based on loading data routines. 
* **maximum\_likelihood\_ae**: This module implements the Maximum Likelihood Amplitude Estimation (MLAE) algorithm. The algorithm was implemented as a python class that given an input oracle operator (a QLM AbstractgGate or QRoutine), creates its correspondent Grover-like operator and implements the MLAE algorithm providing the typical results.
* **iterative\_quantum\_pe**: This module implements the Iterative Quantum Phase Estimation (IQPE) algorithm for phase estimation. The algorithm was implemented as python class. Two different ways of use can be done:
    * Providing an Oracle and the class creates the correspondents *initial_state* and *grover* like operators and executes the IQPE algorithm providing the wanted phase.
    * Providing an *initial\_state* and *grover* operator, executing the IQPE algorithm and obtaining the wanted phase.

Additionally several auxiliary modules were developed:
* *data\_extracting*: this module contains functions to execute circuits and post-process results of QLM solver in proper way presentation.
* *utils*: this module contains several auxiliary functions that are used for all the other modules of the library.

## Jupyter Notebooks

The misc/notebooks folder contains examples demonstrating how the different modules of the library can be used. In general each notebook shows how to use each of the main modules:

* **01\_DataLoading\_Module\_Use**: contains different examples for using the functions of the data\_loading module. 
* **02\_AmplitudeAmplification\_Operators**: explains the theory behind the Grover-like operator for amplitude amplification using the functions of the module amplitude\_amplification. 
* **03\_AmplitudeAmplification\_Procedure**: shows how implement amplitude amplification procedure for calculating expected value of function when x follows a probability distribution ($'E_{x\\sim p}(f)'$) using the Grover-like operators programming in amplitude\_amplification module.
* **04\_MaximumLikelihood\_Class**: explains the MLAE algorithm and the use of the MLAE class programmed in the maximum\_likelihood\_ae module. 
* **05\_Iterative\_QPE\_Class**: explains the use of the IterativeQuantumPE class in the iterative\_quantum\_pe module for using the IQPE algorithm. 
## Acknowledgements

This work is supported by the [NEASQC](https://cordis.europa.eu/project/id/951821) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## Documentation
Customize the GitHub Workflow YAML file: *repo_name*/.github/workflow/sphinx\_doc.yml
There is a basic index.rst template under 'doc'. Modify it as you want.

Any modification under the folder 'doc' pushed to GitHub will trigger a rebuild of the documentation (using GitHub CI).
If the build is successful, then the resulting html documentation can be access at: https://neasqc.github.io/repo_name

Notes: 
  - You can follow the build process on the 'Actions' tab below the name of the repository.
  - neasqc.github.io does not immediately update. You may need to wait a minute or two before it see the changes.
