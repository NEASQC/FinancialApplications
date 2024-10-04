"""
Architectures and Architecture class definition
"""

import copy
import numpy as np
import qat.lang.AQASM as qlm
from qat.core import Observable, Term

def hardware_efficient_ansatz(**kwargs):
    """
    Create a hardware efficient ansatz.

    Parameters
    ----------
    kwargs : kwargs
        Input dictionary for configuring the ansatz. Mandatory keys:
        features_number : number of features
        n_qubits_by_feature : number of qubits used for each feature
        n_layers : number of layers of the PQC
        base_frecuency : slope for feature normalization
        shift_feature : shift for feature normalization

    Returns
    -------
    pqc : QLM Program
        QLM Program with the ansatz
    weights_names : list
        list with the parameters corrresponding to the weights
    features_names : list
        list with the parameters corrresponding to the features
    """
    features_number = kwargs.get("features_number")
    n_qubits_by_feature = kwargs.get("n_qubits_by_feature")
    n_layers = kwargs.get("n_layers")
    base_frecuency = kwargs.get("base_frecuency", 1.0)
    shift_feature = kwargs.get("shift_feature", 1.0)
    n_qubits = n_qubits_by_feature * features_number
    # Begin the QLM program
    pqc = qlm.Program()
    qbits = pqc.qalloc(n_qubits)
    features_names = ["feature_{}".format(input_) for input_ in range(features_number)]
    features = [
        base_frecuency[i] * pqc.new_var(float, feature) + shift_feature[i]
        for i, feature in enumerate(features_names)
    ]


    # create lists with parametric weights
    weights_names = []
    weights = []
    for layer_ in range(n_layers):
        input_index = []
        for input_ in range(features_number):
            qubit_index = []
            for qubit_ in range(n_qubits_by_feature):
                weights_names.append(
                    "weights_{}_{}_{}".format(layer_, input_, qubit_))
                qubit_index.append(
                    pqc.new_var(float, "weights_{}_{}_{}".format(
                        layer_, input_, qubit_))
                )
            input_index.append(qubit_index)
        weights.append(input_index)


    # Creating Layers
    for layer_ in range(n_layers):
        for input_ in range(features_number):
            # Variational Layer
            for qubit_ in range(n_qubits_by_feature):
                # For each input reply along the number of qubits for input
                pqc.apply(
                    qlm.RX(weights[layer_][input_][qubit_]),
                    qbits[input_ * n_qubits_by_feature + qubit_]
                )
            for qubit_ in range(n_qubits_by_feature):
                # For each input reply the feature
                # along the number of qubits for input
                pqc.apply(
                    qlm.RY(features[input_]),
                    qbits[input_ * n_qubits_by_feature + qubit_]
                )

        # Complete entanglement layer
        for qubit_ in range(n_qubits-1):
            pqc.apply(qlm.X.ctrl(), qbits[qubit_], qbits[qubit_+1])
        if n_qubits > 1:
            pqc.apply(qlm.X.ctrl(), qbits[n_qubits-1], qbits[0])
    return pqc, weights_names, features_names

def z_observable(**kwargs):
    """
    Create an Observable.

    Parameters
    ----------
    kwargs : kwargs
        Input dictionary for configuring the ansatz

    Returns
    -------

    observable : QLM Observable
        QLM Observable
    """
    features_number = kwargs.get("features_number")
    n_qubits_by_feature = kwargs.get("n_qubits_by_feature")
    n_qubits = n_qubits_by_feature * features_number
    terms = [Term(1.0, "Z" * n_qubits, list(range(n_qubits)))]
    observable = Observable(nqbits=n_qubits, pauli_terms=terms)
    return observable

def normalize_data(min_value, max_value, min_x=[-0.5*np.pi], max_x=[0.5*np.pi]):
    """
    Feature Normalization.
    Parameters
    ----------
    min_value : list
        list with the minimum value for all the features
    max_value : list
        list with the maximum value for all the features
    min_x : list
        minimum value for encoding the feature in a rotation
    max_x : list
        maximum value for encoding the feature in a rotation
    Returns
    -------
    slope : np array
        with the slope for normalization of the features
    b0 : np array
        with shift for normalization of the features
    """
    max_value_ = np.array(max_value)
    min_value_ = np.array(min_value)
    min_x = np.array(min_x)
    max_x = np.array(max_x)
    slope = (max_x - min_x) / (max_value_-min_value_)
    b0 = min_x - slope * min_value_
    b1 = max_x - slope * max_value_
    return slope, b0

def compute_pdf_from_pqc(batch, parameters):
    """
    Given a QLM Batch with a PQC representing a Multivariate
    Cumulative Distribution Function (cdf) creates all the mandatory
    PQCs for computing the corresponding Probability Distribution
    Function, pdf. The returned is a QLM Batch with the jobs mandatory
    for computing the pdf

    Parameters
    ----------

    batch : QLM Batch
        QLM batch with the Jobs to execute
    parameters : list
        list with the name of the features for pdf computation

    Returns
    ------

    batch_ : QLM Batch
        QLM Batch with the jobs for pdf copmputation
    """
    batch_ = copy.deepcopy(batch)
    if len(batch_) != 1:
        raise ValueError("BE AWARE: Input batch MUST HAVE only 1 job")


    # Select the first job
    job = [batch_[0]]
    for feature in parameters:
        temp_list = []
        for step_job in job:
            temp_list = temp_list + step_job.differentiate(feature)
        # Overwrite the input job list with the new job list
        job = temp_list

    # Overwrite the jobs of the output Batch
    batch_.jobs = temp_list
    return batch_

def compute_gradient(batch, parameters):
    """
    Compile method of the plugin.

    Parameters
    ----------

    batch : QLM Batch
        QLM batch with the Jobs to execute
    parameters : list
        list with the name of the parameters for gradient computations

    Returns
    ------

    batch_ : QLM Batch
        QLM Batch with the jobs for computing graidents
    """

    #Deep Coopy of the input batch object
    batch_ = copy.deepcopy(batch)
    if len(batch_) != 1:
        raise ValueError("BE AWARE: Input batch MUST HAVE only 1 job")


    # Select the first job
    job = batch_[0]
    # Get the complete parameter names of the job
    circuit_parameters = list(job.get_variables())
    if not set(parameters).issubset(circuit_parameters):
        raise ValueError(
            "List of input parameters NOT contained in job parameters"
        )
    list_of_jobs = []
    for parameter in parameters:
        # For each parameter get the corresponding diferentiate jobs
        job_ = job.differentiate(parameter)
        for i_, step_job in enumerate(job_):
            step_job.meta_data = {
                "parameter" : parameter,
                "number" : i_,
                "gradient_circuit" : True
            }
            list_of_jobs.append(step_job)
    # list_of_jobs.append(job)
    batch_.jobs = list_of_jobs
    return batch_

def init_weights(weigths_names):
    """
    init weights of the PQC
    """

    init_weights = {v: np.random.uniform() for v in weigths_names}
    return init_weights
