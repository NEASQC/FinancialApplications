"""
Functions for computing losses
"""
import copy
import numpy as np

def trapezoidal_rule(x_domain,y_range):
    """
    Computes the integral using the trapezoidal_rule
    Parameters
    ----------
    x_domain : numpy array
        domain for computing the integral
    y_range : numpy array
        range for computing the integral
    Returns
    -------
    integral : float
        integral computed using the trapezoidal rule
    """
    dx = np.diff(x_domain)
    integral = np.dot((y_range[:-1]+y_range[1:])/2, dx)
    return integral

def loss_function_qdml(
    labels, predict_cdf, predict_pdf,
    x_quad, predict_quad, loss_weights=[1.0, 5.0]):
    """
    Computes the function for Quantum Differential Machine Learning
    Parameters
    ----------
    labels : numpy array
        numpy array with the labels
    predict_cdf : numpy array
        numpy array with the predictions for the CDF
    predict_pdf : numpy array
        numpy array with the predictions for the PDF
    x_quad : numpy array
        numpy array with the domain for computing the integral of the PDF
    predict_quad : numpy array
        numpy array with the PDF for computing the integral of the PDF
    Returns
    -------
    loss_ : float
        evaluation of the loss function for QDML
    """

    alpha_0 = loss_weights[0]
    alpha_1 = loss_weights[1]
    # Loss Computation
    #Typical DL error
    error_ = (predict_cdf - labels.reshape(predict_cdf.shape))
    loss_1 = np.mean(error_ ** 2)
    #print("\t loss_1 : {}".format(loss_1))
    mean = -2 * np.mean(predict_pdf)
    #print("\t mean : {}".format(mean))
    integral = trapezoidal_rule(x_quad[:, 0], predict_quad[:, 0] * predict_quad[:, 0])
    #print("\t integral: {}".format(integral))
    loss_ = alpha_0 * loss_1 + alpha_1 * (mean + integral)
    #print("\t loss: {}".format(loss))
    return loss_

def mse(labels, prediction):
    """
    Compute Mean Square Error
    Parameters
    ----------
    labels : numpy array
        numpy array with the labels
    prediction : numpy array
        numpy array with the predictions
    """
    error_ = (prediction - labels.reshape(prediction.shape))
    mse_v = np.mean(error_ ** 2)
    return mse_v


def compute_loss(weights, produce_results, loss_function):
    """
    Workflow for computing loss:
    Parameters
    ----------
    weights : list
        list with the weights for the PQC
    produce_results : python function
        Function for producing mandatory inputs for computing loss
    loss_function : python function
        Function for computing loss function
    """

    output_dict = produce_results(weights)
    loss_ = loss_function(**output_dict)
    return loss_
