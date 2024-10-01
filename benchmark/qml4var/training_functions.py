"""
Functions for building training workflow
"""
import sys
import copy
import numpy as np
sys.path.append("../../")
from QQuantLib.qml4var.qlm_procces import workflow_execution, \
    cdf_workflow, pdf_workflow
from QQuantLib.qml4var.losses import loss_function_qdml, mse


def init_weights(weigths_names):
    """
    init weights of the PQC
    """

    init_weights = {v: np.random.uniform() for v in weigths_names}
    return init_weights

def workflow_for_qdml(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Workflow for proccessing the data and obtain the mandatory arrays
    for computing the desired loss
    Parameters
    ----------
    weights : numpy array
        Array with weights for PQC
    data_x : numpy array
        Array with dataset of the features
    data_y : numpy array
        Array with targets (labes) dataset
    dask_client : Dask client
        Dask client for speed up training. Not mandatory
    **kwargs

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
    # Get CDF prediction for train data
    cdf_train_prediction = workflow_execution(
        weights, data_x, cdf_workflow_, dask_client=dask_client)
    # Get PDF prediction for train data
    pdf_train_prediction = workflow_execution(
        weights, data_x, pdf_workflow_, dask_client=dask_client)
    # Get PDF prediction for domain
    pdf_domain_prediction = workflow_execution(
        weights, x_integral, pdf_workflow_, dask_client=dask_client)

    if dask_client is None:
        cdf_train_prediction = np.array(cdf_train_prediction)
        pdf_train_prediction = np.array(pdf_train_prediction)
        pdf_domain_prediction = np.array(pdf_domain_prediction)
    else:
        cdf_train_prediction = np.array(dask_client.gather(cdf_train_prediction))
        pdf_train_prediction = np.array(dask_client.gather(pdf_train_prediction))
        pdf_domain_prediction = np.array(dask_client.gather(pdf_domain_prediction))

    pdf_domain_prediction = pdf_domain_prediction.reshape(x_integral.shape)

    output_dict = {
        "y_predict_cdf" : cdf_train_prediction,
        "y_predict_pdf" : pdf_train_prediction,
        "x_integral" : x_integral,
        "y_predict_pdf_domain":pdf_domain_prediction,
        "data_y" : data_y

    }
    return output_dict

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

def workflow_for_cdf(weights, data_x, dask_client=None, **kwargs):
    """
    Workflow for proccessing the CDF for the input data_x
    Parameters
    ----------
    Same parameters that workflow_for_qdml
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
    Same parameters that workflow_for_qdml
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

def mse_function(weights, data_x, data_y, dask_client=None, **kwargs):
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

def numeric_gradient(weights, data_x, data_y, loss):
    """
    Compute the numeric gradient for some input loss function properly
    configured
    Parameters
    ----------
    weights : numpy array
        Array with weights for PQC
    data_x : numpy array
        Array with dataset of the features
    data_y : numpy array
        Array with targets (labes) dataset
    loss : function
        function for computing the loss properly configured
    """
    gradient_i = []
    epsilon = 1.0e-7
    for i, weight in enumerate(weights):
        #print(weight)
        new_weights = copy.deepcopy(weights)
        new_weights[i] = weight + epsilon
        loss_plus = loss(new_weights, data_x, data_y)
        #print(new_weights)
        #print(loss_plus)
        new_weights = copy.deepcopy(weights)
        new_weights[i] = weight - epsilon
        loss_minus = loss(new_weights, data_x, data_y)#, input_x, input_y)
        #print(new_weights)
        #print(loss_minus)
        gradient_i = gradient_i + [(loss_plus-loss_minus) / (2.0 * epsilon)]
        #print(gradient_i)
    return gradient_i

#def training(
#        train_x=None, train_y=None, weights_dict=None, optimizer=None,
#        initial_time=0, dask_client=None, **kwargs):
#    """
#    Parameters
#    ----------
#    train_x : numpy array
#        Array with features dataset
#    train_y : numpy array
#        Array with targets (labes) dataset
#    optimizer : function
#        function with the optimizer
#    dask_client : Dask client
#        Dask client for speed up training. Not mandatory
#    **kwargs
#    pqc : QLM Program
#        QLM Program with the PQC
#    observable : QLM Observable
#        QLM Observable with the observable for the pqc
#    """
#    # Load PQC in QLM Program
#
#
#    # Redefining the proccess for computing predcitions for CDF
#    # dask_procces_dataset_cdf = lambda x: procces_dataset_cdf(
#    #     x,
#    #     train_x,
#    #     train_y,
#    #     dask_client=dask_client,
#    #     **kwargs
#    # )
#    procces_data_ = lambda x: procces_data(
#        x, train_x, train_y, dask_client=dask_client, **kwargs)
#    # dask_procces_dataset_cdf_pdf = lambda x: procces_dataset_cdf_pdf(
#    #     x,
#    #     train_x,
#    #     train_y,
#    #     dask_client=dask_client,
#    #     **kwargs
#    # )
#    # Redefining the numeric gradient computations
#    numeric_gradient_ = lambda x: numeric_gradient(x, loss_qdml)
#
#    # mse_ = lambda x: compute_loss(
#    #     x,
#    #     dask_procces_dataset_cdf,
#    #     wraper_mse
#    # )
#    optimizer_info = kwargs.get("optimizer_info", None)
#    optimizer_info.update({"file_to_save": kwargs.get("file_to_save", None)})
#    print(optimizer_info)
#
#    weights = optimizer(
#        weights_dict=weights_dict,
#        loss_function=loss_qdml,
#        metric_function=None,
#        gradient_function=numeric_gradient_,
#        initial_time=initial_time,
#        **optimizer_info
#    )
#    return dict(zip(weights_dict.keys(), weights))
#
