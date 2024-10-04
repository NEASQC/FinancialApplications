"""
This module contains the different myqlm workflows mandatory for
the different PQCs evaluations needed for a training proccess
"""
import numpy as np
from itertools import product
from qat.core import Batch
from QQuantLib.qpu.select_qpu import select_qpu
from QQuantLib.qml4var.plugins import SetParametersPlugin, pdfPluging
from QQuantLib.qml4var.architectures import compute_pdf_from_pqc
from QQuantLib.qml4var.losses import loss_function_qdml, mse

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

def qdml_loss_function(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Computes the qdml loss function.
    Parameters
    ----------
    Same parameters that workflow_for_qdml
    Returns
    -------
    loss_ : computed loss function value for the input data.

    """

    # Get the mandatory array for computing loss
    output_dict = workflow_for_qdml(
        weights, data_x, data_y, dask_client=dask_client, **kwargs)

    loss_ = loss_function_qdml(
        output_dict.get("data_y"),
        output_dict.get("y_predict_cdf"),
        output_dict.get("y_predict_pdf"),
        output_dict.get("x_integral"),
        output_dict.get("y_predict_pdf_domain"),
    )
    return loss_

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
    **kwargs : keyword arguments.In addition to kwargs provided to the
    stack_execution function the following one should be provided too:
        qpu_info : Python dictionary with the infor for configuring a QPU

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
    kwargs : keyword arguments. See cdf_workflow documentation

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

def workflow_for_cdf(weights, data_x, dask_client=None, **kwargs):
    """
    Workflow for proccessing the CDF for the input data_x
    Parameters
    ----------
    weights : numpy array
        Array with weights for PQC
    data_x : numpy array
        Array with dataset of the features
    dask_client : Dask client
        Dask client for speed up training. Not mandatory
    kwargs : keyword arguments. See cdf_workflow function documentation.

    Returns
    -------
    output_dict : dict
        dictionary with the computed arrays. Keys:
        data_y : input data_y data
        y_predict_cdf : CDF prediction for data_x
    """
    workflow_cfg = {
        "pqc" : kwargs.get("pqc"),
        "observable" : kwargs.get("observable"),
        "weights_names" : kwargs.get("weights_names"),
        "features_names" : kwargs.get("features_names"),
        "nbshots" : kwargs.get("nbshots"),
        "qpu_info" : kwargs.get("qpu_info")
    }
    # for computing CDF using PQC
    cdf_workflow_ = lambda w, x: cdf_workflow(w, x, **workflow_cfg)
    # Get CDF prediction for train data
    cdf_train_prediction = workflow_execution(
        weights, data_x, cdf_workflow_, dask_client=dask_client)
    if dask_client is None:
        cdf_train_prediction = np.array(cdf_train_prediction)
    else:
        cdf_train_prediction = np.array(dask_client.gather(cdf_train_prediction))
    output_dict = {
        "y_predict_cdf" : cdf_train_prediction
    }
    return output_dict

def workflow_for_pdf(weights, data_x, dask_client=None, **kwargs):
    """
    Workflow for proccessing the CDF for the input data_x
    Parameters
    ----------
    weights : numpy array
        Array with weights for PQC
    data_x : numpy array
        Array with dataset of the features
    dask_client : Dask client
        Dask client for speed up training. Not mandatory
    kwargs : keyword arguments. See pdf_workflow function documentation.
    Returns
    -------
    output_dict : dict
        dictionary with the computed arrays. Keys:
        y_predict_pdf : PDF prediction for data_x
    """
    workflow_cfg = {
        "pqc" : kwargs.get("pqc"),
        "observable" : kwargs.get("observable"),
        "weights_names" : kwargs.get("weights_names"),
        "features_names" : kwargs.get("features_names"),
        "nbshots" : kwargs.get("nbshots"),
        "qpu_info" : kwargs.get("qpu_info")
    }
    # for computing CDF using PQC
    pdf_workflow_ = lambda w, x: pdf_workflow(w, x, **workflow_cfg)
    # Get CDF prediction for train data
    pdf_train_prediction = workflow_execution(
        weights, data_x, pdf_workflow_, dask_client=dask_client)
    if dask_client is None:
        pdf_train_prediction = np.array(pdf_train_prediction)
    else:
        pdf_train_prediction = np.array(dask_client.gather(pdf_train_prediction))
    output_dict = {
        "y_predict_pdf" : pdf_train_prediction
    }
    return output_dict

def workflow_for_qdml(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Workflow for proccessing the data and obtain the mandatory arrays
    for computing the desired loss
    Parameters
    ----------
    weights : numpy array
        Array with weights for PQC.
    data_x : numpy array
        Array with dataset of the features. Shape: (-1, number of features)
    data_y : numpy array
        Array with targets (labes) dataset. Shape: (-1, 1)
    dask_client : Dask client
        Dask client for speed up training. Not mandatory
    **kwargs : keyword arguments.In addition to kwargs provided to the
    cdf_workflow function the following ones should be provided too:
        minval : list with the minimum values for the domain of all the features.
        maxval : list with the maximum values for the domain of all the features.
        points : number of points for a feature domain discretization
    Returns
    -------
    output_dict : dict
        dictionary with the computed arrays. Keys:
        data_y : input data_y data
        y_predict_cdf : CDF prediction for data_x
        y_predict_pdf : PDF prediction for data_x
        x_integral : domain discretization for computing integral
        y_predict_pdf_domain : PDF of the x_integral
    """

    workflow_cfg = {
        "pqc" : kwargs.get("pqc"),
        "observable" : kwargs.get("observable"),
        "weights_names" : kwargs.get("weights_names"),
        "features_names" : kwargs.get("features_names"),
        "nbshots" : kwargs.get("nbshots"),
        "qpu_info" : kwargs.get("qpu_info")
    }

    # for computing CDF using PQC
    cdf_workflow_ = lambda w, x: cdf_workflow(w, x, **workflow_cfg)
    # for computing PDF using PQC
    pdf_workflow_ = lambda w, x: pdf_workflow(w, x, **workflow_cfg)
    # Getting the Domain for discretization
    minval = kwargs.get("minval")
    maxval = kwargs.get("maxval")
    points = kwargs.get("points")
    x_integral = np.linspace(minval, maxval, points)
    domain_x = np.array(list(
        product(*[x_integral[:, i] for i in range(x_integral.shape[1])])
    ))
    # Get CDF prediction for train data
    cdf_train_prediction = workflow_execution(
        weights, data_x, cdf_workflow_, dask_client=dask_client)
    # Get PDF prediction for train data
    pdf_train_prediction = workflow_execution(
        weights, data_x, pdf_workflow_, dask_client=dask_client)
    # Get PDF prediction for domain
    pdf_domain_prediction = workflow_execution(
        weights, domain_x, pdf_workflow_, dask_client=dask_client)

    if dask_client is None:
        cdf_train_prediction = np.array(cdf_train_prediction)
        pdf_train_prediction = np.array(pdf_train_prediction)
        pdf_domain_prediction = np.array(pdf_domain_prediction)
    else:
        cdf_train_prediction = np.array(dask_client.gather(cdf_train_prediction))
        pdf_train_prediction = np.array(dask_client.gather(pdf_train_prediction))
        pdf_domain_prediction = np.array(dask_client.gather(pdf_domain_prediction))


    cdf_train_prediction = cdf_train_prediction.reshape((-1, 1))
    pdf_train_prediction = pdf_train_prediction.reshape((-1, 1))
    pdf_domain_prediction = pdf_domain_prediction.reshape((-1, 1))

    output_dict = {
        "y_predict_cdf" : cdf_train_prediction,
        "y_predict_pdf" : pdf_train_prediction,
        "x_integral" : domain_x,
        "y_predict_pdf_domain":pdf_domain_prediction,
        "data_y" : data_y

    }
    return output_dict

def qdml_loss_workflow(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Workflow for computing the qdml loss function.
    Parameters
    ----------
    Same parameters that workflow_for_qdml
    Returns
    -------
    loss_ : computed loss function value for the input data.

    """

    # Get the mandatory array for computing loss
    output_dict = workflow_for_qdml(
        weights, data_x, data_y, dask_client=dask_client, **kwargs)

    loss_ = loss_function_qdml(
        output_dict.get("data_y"),
        output_dict.get("y_predict_cdf"),
        output_dict.get("y_predict_pdf"),
        output_dict.get("x_integral"),
        output_dict.get("y_predict_pdf_domain"),
    )
    return loss_

def mse_workflow(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Computes the MSE function
    Parameters
    ----------
    Same parameters that workflow_for_qdml
    Returns
    -------
    mse__ : computed mse function value for the input data.

    """
    # Get the mandatory array for computing MSE
    output_dict = workflow_for_cdf(
        weights, data_x, dask_client=dask_client, **kwargs)
    mse_ = mse(data_y, output_dict["y_predict_cdf"])
    return mse_
