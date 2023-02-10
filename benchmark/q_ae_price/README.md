## Computing expected values in finances 

The *benchmark\_ae\_estimation\_price.py* is a script for executing different price estimation on different financial derivatives using AE algorithms for computing the mandatory expected value functions.
This scripts calls the **ae\_price\_estimation** module for doing this computation.

In order to configure the complete AE estimation problem the different *json* files from the **jsons** folder should be modified.

## JSONs files

The *jsons* files allows to configure the complete Price Estimation Problem (PEP) to solve. Important files are:

* ae\_pe\_domain\_configuration.json: for configuring the desired domain of the problem
* ae\_pe\_density\_probability.json: for configuring the probability density used in PEP
* ae\_pe\_payoffs.json: for configuring the payoffs for the different derivatives.

The program will created a combination of all the possible PEP from this possible. 
Example: if the domain have 2 possible domains, 3 possible probability densities and 4 possible payoffs you will have: 2\*3\*4=24 possible PEP to solve.

For solving the PEP different AE algorithms can be used. The configuration can be provided by editing following jsons: 

* ae\_pe\_cqpeae\_configuration.json
* ae\_pe\_iqae\_configuration.json
* ae\_pe\_iqpeae\_configuration.json
* ae\_pe\_mcae\_configuration.json
* ae\_pe\_mlae\_configuration.json
* ae\_pe\_rqae\_configuration.json

**BE AWARE** These jsons have the configuration of the data loading into the key: **encoding**. Up to 3 types of encoding can be used. 

For each AE several configurations can be set up into the correspondent json. All possible configurations can be used. The final configuration will be a total combination of the PEP and the AE. Example: If the AE algorithm have 9 possible configurations and the total PEP problem have 24 possible configuration then the final number of total configurations will be: 9\*24=216 total configurations.

## Command Line usage

The *quantum_integration_check.py* can be executed from **CL**. For getting a help use:

* python  benchmark\_ae\_estimation\_price.py -h

## Tips and recommendations 

Some useful tips for executing are:

* The different AE algorithms can be selected by providing the corresponding argument: --MLAE, --IQAE, --RQAE ... 
* Different algorithms can be selected in one execution. You can provided following configuration 
    * python  benchmark\_ae\_estimation\_price.py --MLAE --IQAE 
* From all the possible configurations you can select one by providing the argument -id NumberOfExecution or you can select all of them by providing --all.    
* For executing the program the -exe argument MUST BE provided.
* For saving the results -save MUST BE provided


