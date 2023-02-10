## Computing Integrals using AE

The *quantum_integration_check.py* is a script for testing the **q_solve_integral** module for pure integration.

The integral to solve is a pure *sin* and three different intervals can be used:

* positive\_function: $[0, \frac{\pi}{4}]$
* positive\_integral: $[\frac{3 \pi}{4}, \fra{9 \pi}{8}]$
* negative\_integral: $[\pi, \fra{5 \pi}{4}]$

Three different data encoding can be used for loading the desired intergral into the quantum final state: [0, 1, 2]. This configuration can be changed in the *json* files from **json_checks** folder. Additionally, this files can be edited for configuring the different AE algorithms that can be used. The deafult domain discretization is of 6 qubits (is hardcoded into the *problem* function of the script. 


## Command Line ussage

The *quantum_integration_check.py* can be executed from **CL**. For getting a help use:

* python  quantum\_integration\_check.py -h

## Tips and recomendations 

Some useful tips for executing are:

* The different AE algorithms can be selected by providing the corresponding argument: --MLAE, --IQAE, --RQAE ... 
* For each selected AE algorithm the correspondent *json* file will be used for the configuration. All the posible combinations will be generated for executing. 
* Different algorithms can be selected in one execution. You can provided following configuration 
    * python  quantum\_integration\_check.py --MLAE --IQAE 
* From all the posible configurations you can select one by providing the argument -id NumberOfExecution or you can select all of them by providing --all.    
* For executing the program the -exe argument MUST BE provided.
* For saving the results -save MUST BE provided


