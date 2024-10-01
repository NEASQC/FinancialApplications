"""
Definition of the functions that allows to build, execute and post-procces
QLM Batches for using in the training loops
"""
import numpy as np
from qat.core import Batch
from QQuantLib.qpu.select_qpu import select_qpu
from QQuantLib.qml4var.plugins import SetParametersPlugin, pdfPluging
from QQuantLib.qml4var.architectures import compute_pdf_from_pqc

def stack_execution(weights, x_sample, stack, **kwargs):
    """
    Given a stack, the weights, an input sample, a PQC and Observable
    (provided into the kewyword arguments) this function this function
    builds the corresponding QLM Batch of jobs and execute it.

    Parameters
    ----------
    weights : np array
        array with the weights of the PQC
    x_sample : np array
        input sample to provide to the PQC
    kwargs : keyword arguments
        The following keys are mandatory
        pqc : QLM Program
            qlm program with the implementation of the PQC
        observable : QLM Observable
            qlm observable with the Observable definition of the PQC
        weights_names : list
            list with the names of the parameters of the PQC corresponding
            to the weights
        features_names : list
            list with the names of the parameters of the PQC corresponding
            to the input features
        nbshots : int
            number of shots
    Returns
    -------
    results : QLM BatchResult
        QLM BatchResult with the results of the execution of the stack
    """
    # Prepare PQC
    pqc = kwargs.get("pqc")
    observable = kwargs.get("observable")
    weights_names = kwargs.get("weights_names")
    features_names = kwargs.get("features_names")
    nbshots = kwargs.get("nbshots")
    # Build Circuit
    circuit = pqc.to_circ()
    # Build Basic Job with parametric circuit
    job = circuit.to_job(
        nbshots=nbshots,
        observable=observable
    )
    job.meta_data = {"gradient_circuit" : False}
    # Build job for CDF
    cdf_batch = Batch([job])
    cdf_batch.meta_data = {
        "weights" : weights_names,
        "features" : features_names,
    }
    results = stack(weights, x_sample).submit(cdf_batch)
    return results

def cdf_workflow(weights, x_sample, **kwargs):
    """
    This function builds the mandatory stack for evaluating a PQC
    (provided in the kwargs) for the given weights and x_sample.
    Additionally, it executes the stack using the stack_execution
    function. So it computes the corresponding CDF value for the
    given weights and x_sample.

    Parameters
    ----------
    weights : np array
        array with the weights of the PQC
    x_sample : np array
        input sample to provide to the PQC
    kwargs : keyword arguments. See stack_execution function documentation.

    Returns
    -------
    results : float
        Value of the CDF, computed using the input PQC, for the given
        weights and x_sample.
    """
    # Get the QPU
    qpu_dict = kwargs.get("qpu_info")
    qpu = select_qpu(qpu_dict)
    # Build the execution stack for CDF
    stack_cdf = lambda weights_, features_: \
        SetParametersPlugin(weights_, features_) | qpu
    # Execute the stack
    workflow_cfg = {
        "pqc" : kwargs.get("pqc"),
        "observable" : kwargs.get("observable"),
        "weights_names" : kwargs.get("weights_names"),
        "features_names" : kwargs.get("features_names"),
        "nbshots" : kwargs.get("nbshots"),
    }
    results = stack_execution(weights, x_sample, stack_cdf, **workflow_cfg)
    results = results[0].value
    return results

def pdf_workflow(weights, x_sample, **kwargs):
    """
    Given a PQC that computes the CDF (provided in the kwargs), this
    function builds the mandatory stack for computing the corresponding
    PDF for the given weights and x_sample.
    Additionally, it executes the stack using the stack_execution
    function. So it computes the corresponding PDF value for the
    given weights and x_sample.

    Parameters
    ----------
    weights : np array
        array with the weights of the PQC
    x_sample : np array
        input sample to provide to the PQC
    kwargs : keyword arguments. See stack_execution function documentation.

    Returns
    -------
    results : float
        Value of the PDF, computed using the input PQC, for the given
        weights and x_sample.
    """
    # Get the QPU
    qpu_dict = kwargs.get("qpu_info")
    qpu = select_qpu(qpu_dict)
    # Build the execution stack
    features_names = kwargs.get("features_names")
    stack_pdf = lambda weights_, features_: \
        pdfPluging(features_names) | SetParametersPlugin(weights_, features_) | qpu
    # Execute the stack
    workflow_cfg = {
        "pqc" : kwargs.get("pqc"),
        "observable" : kwargs.get("observable"),
        "weights_names" : kwargs.get("weights_names"),
        "features_names" : kwargs.get("features_names"),
        "nbshots" : kwargs.get("nbshots"),
    }
    results = stack_execution(weights, x_sample, stack_pdf, **workflow_cfg)
    results = results[0].value
    return results

def workflow_execution(weights, data_x, workflow, dask_client=None):
    """
    Given an input weights, a complete dataset of features, and a
    properly configured workflow function (like cdf_workflow or
    pdf_workflow) executes the workflow for all the samples of the
    dataset

    Parameters
    ----------
    weights : np array
        array with the weights of the PQC
    data_x : np array
        array with the dataset of input features
    dask_client : dask client
        Dask client for speed up computations
    Returns
    -------
    y_data : depends on dask_client. If dask_client is None then returns
    a list with results of the workflow for all input dataset. If a
    dask_client is passed then returns a list of futures and a gather
    operation should be executed for retrieving the data.
    """
    if dask_client is None:
        y_data = [workflow(weights, x_) for x_ in data_x]
    else:
        y_data = [dask_client.submit(workflow, weights, x_, pure=False) for x_ in data_x]
    return y_data
