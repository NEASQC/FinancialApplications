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
        numpy array with the labels. Shape: (-1, 1)
    predict_cdf : numpy array
        numpy array with the predictions for the CDF. Shape: (-1, 1)
    predict_pdf : numpy array
        numpy array with the predictions for the PDF. Shape: (-1, 1)
    x_quad : numpy array
        numpy array with the domain for computing the integral of the PDF
        Shape: (-1, number of features)
    predict_quad : numpy array
        numpy array with the PDF for computing the integral of the PDF
        Shape: (-1, 1)
    Returns
    -------
    loss_ : float
        evaluation of the loss function for QDML
    """

    alpha_0 = loss_weights[0]
    alpha_1 = loss_weights[1]
    # Loss Computation
    #Typical DL error
    if predict_cdf.shape != labels.shape:
        raise ValueError("predict_cdf and labels have different shape!!")
    error_ = (predict_cdf - labels)
    loss_1 = np.mean(error_ ** 2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!!")
    #print("\t loss_1 : {}".format(loss_1))
    mean = -2 * np.mean(predict_pdf)
    #print("\t mean : {}".format(mean))
    square_for_integral = predict_quad ** 2

    if x_quad.shape[1] == 1:
        # Typical 1-D trapezoidal integration
        integral = np.trapz(y=square_for_integral[:, 0], x=x_quad[:, 0])
    elif x_quad.shape[1] == 2:
        # 2-D Trapezoidal integration
        x_domain, y_domain = np.meshgrid(
            np.unique(x_quad[:, 0]),
            np.unique(x_quad[:, 1])
        )
        square_for_integral = square_for_integral.reshape(x_domain.shape)
        integral = np.trapz(
            np.trapz(y=square_for_integral, x=x_domain),
            x=y_domain[:, 0]
        )
    else:
        # MonteCarlo approach
        integral = np.sum(square_for_integral) * np.prod(
            x_quad.max(axis=0) - x_quad.min(axis=0)
        ) / square_for_integral.size
    #integral = trapezoidal_rule(x_quad[:, 0], predict_quad[:, 0] * predict_quad[:, 0])
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
