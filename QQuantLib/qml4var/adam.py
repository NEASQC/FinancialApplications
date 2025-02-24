""""
Adam
"""

import itertools
import numpy as np
import pandas as pd


def save_stuff(
        weights, weights_names, t_, loss_, metric_mse_=None, file_to_save=None):
    """
    Save stuff
    """
    pdf = pd.DataFrame(weights, index=weights_names).T
    pdf["t"] = t_
    pdf["loss"] = loss_
    pdf["metric_mse"] = metric_mse_
    if file_to_save is not None:
        pdf.to_csv(
            file_to_save, sep=";",
            index=True, mode='a', header=False
        )


def batch_generator(iterable, batch_size=1):
    iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break

def initialize_adam(parameters):
    """
    Initialize the parameters of ADAM
    """

    v = np.zeros(len(parameters))
    s = np.zeros(len(parameters))

    return v, s

# Update parameters using Adam
def update_parameters_with_adam(
    x, grads, s, v, t, learning_rate=0.01,
    beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update the parameters of ADAM
    """
    s = beta1 * s + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * grads ** 2
    s_hat = s / (1.0 - beta1 ** (t + 1))
    v_hat = v / (1.0 - beta2 ** (t + 1))
    x = x - learning_rate * s_hat / (np.sqrt(v_hat) + epsilon)
    return x, s, v

def adam_optimizer_loop(
        weights_dict, loss_function, metric_function, gradient_function,
        batch_generator, initial_time=0, **kwargs):
    """
    Parameters
    ----------
    weights_dict : dict
        dictionary with the weights to fit
    loss_function : function
        function for computing the loss function
    metric_function : fuction
        function for computing the metric function
    gradient_function : function
        function for computing the gradient of the loss function
    batch_generator : function
        function for generating batches of the trainin data.
    initial_time : int
        Initial time step
    kwargs : keyword arguments
        arguments for configuring optimizer. For ADAM:

    store_folder : kwargs, str
        Folder for saving results. If None not saving
    epochs : kwargs, int
        Maximum number of iterations
    tolerance : kwargs, float
        Tolerance to achieve
    n_counts_tolerance : kwargs, int
        Number of times the tolerance should be achieved in consecutive
        iterations
    print_step : kwargs, int
        Print_step for printing evolution of training
    learning_rate : kwargs,float
        Learning_rate for ADAM
    beta1 : kwargs, float
        beta1 for ADAM
    beta2 : kwargs, float
        beta2 for ADAM
    """
    # Get Weights
    weights = list(weights_dict.values())
    weights_names = list(weights_dict.keys())
    # Init Adam
    s_, v_ = initialize_adam(weights)#.keys())
    # ADAM time parameter
    t_ = initial_time
    # Tolerance steps
    n_tol = 0

    # Deal with save Folder
    file_to_save = kwargs.get("file_to_save", None)
    # Configure Stop
    epochs = kwargs.get("epochs", None)
    tolerance = kwargs.get("tolerance", None)
    n_counts_tolerance = kwargs.get("n_counts_tolerance", None)
    # Configure printing info
    print_step = kwargs.get("print_step", None)
    # Configure Adam
    learning_rate = kwargs.get("learning_rate", None)
    beta1 = kwargs.get("beta1", None)
    beta2 = kwargs.get("beta2", None)

    # Compute Initial Loss and Metric
    loss_0 = loss_function(weights)
    if metric_function is None:
        metric_mse_0 = None
    else:
        metric_mse_0 = metric_function(weights)
        print("Loss Function at t={}: {}".format(t_, loss_0))
        print("MSE at t={}: {}".format(t_, metric_mse_0))
        save_stuff(
            weights, weights_names, t_, loss_0, metric_mse_0, file_to_save)


    for t_ in range(t_, epochs):
        for batch in batch_generator:
            # Get the Batches
            batch_x = batch[0]
            batch_y = batch[1]
            # Compute gradient on batches
            loss_gradient = np.array(gradient_function(
                weights, batch_x, batch_y
            ))
            # Update Weights
            weights, s_, v_ = update_parameters_with_adam(
                weights, loss_gradient, s_, v_, t_,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2
            )
        loss_t = loss_function(weights)
        delta = -(loss_t - loss_0)
        loss_0 = loss_t
        # print("t= {} delta= {}".format(t_, delta))
        # print("t= {} n_tol= {}".format(t_, n_tol))
        if delta < tolerance:
            n_tol = n_tol + 1
        else:
            n_tol = 0
        if t_ % print_step == 0:
            # Compute loss
            if metric_function is None:
                metric_mse_t = None
            else:
                metric_mse_t = metric_function(weights)
            print("\t MSE at t={}: {}".format(t_, metric_mse_t))
            print("\t Iteracion: {}. Loss: {}".format(t_, loss_t))
            save_stuff(
                weights, weights_names, t_, loss_0, metric_mse_t, file_to_save)

        if n_tol >= n_counts_tolerance:
            print("Achieved Convergence. Delta: {}".format(delta))
            if metric_function is None:
                metric_mse_t = None
            else:
                metric_mse_t = metric_function(weights)
            save_stuff(
                weights, weights_names, t_, loss_0, metric_mse_t, file_to_save)
            break
    print("Maximum number of iterations achieved.")
    if metric_function is None:
        metric_mse_t = None
    else:
        metric_mse_t = metric_function(weights)
    save_stuff(
        weights, weights_names, t_, loss_t, metric_mse_t, file_to_save)
    return weights

# def adam_optimizer(
#         weights_dict, loss_function, metric_function, gradient_function,
#         batch_generator, initial_time=0, **kwargs):
#     """
#     Parameters
#     ----------
#     weights_dict : dict
#         dictionary with the weights to fit
#     loss_function : function
#         function for computing the loss function
#     metric_function : fuction
#         function for computing the metric function
#     gradient_function : function
#         function for computing the gradient of the loss function
#     initial_time : int
#         Initial time step
#     **kwargs : keyword arguments
#         arguments for configuring optimizer. For ADAM:
#         store_folder : folder for saving results. If None not saving
#         n_iter : maximum number of iterations
#         tolerance : tolerance to achieve
#         n_counts_tolerance : number of times the tolerance should be
#         achieved in consecutive iterations
#         print_step : print_step for printing evolution of training
#         learning_rate : learning_rate for ADAM
#         beta1 : beta1 for ADAM
#         beta2 : beta2 for ADAM
#     """
#     # Get Weights
#     weights = list(weights_dict.values())
#     weights_names = list(weights_dict.keys())
#     # Init Adam
#     s_, v_ = initialize_adam(weights)#.keys())
#     # ADAM time parameter
#     t_ = initial_time
#     # Tolerance steps
#     n_tol = 0
# 
#     # Deal with save Folder
#     file_to_save = kwargs.get("file_to_save", None)
#     # Configure Stop
#     n_iter = kwargs.get("n_iter", 200)
#     tolerance = kwargs.get("tolerance", 1.0e-4)
#     n_counts_tolerance = kwargs.get("n_counts_tolerance", 10)
# 
#     # Configure printing info
#     print_step = kwargs.get("print_step", 10)
# 
#     # Configure Adam
#     learning_rate = kwargs.get("learning_rate", 0.01)
#     beta1 = kwargs.get("beta1", 0.9)
#     beta2 = kwargs.get("beta2", 0.999)
# 
# 
#     # Compute Initial Loss and Metric
#     loss_0 = loss_function(weights)
#     if metric_function is None:
#         metric_mse_0 = None
#     else:
#         metric_mse_0 = metric_function(weights)
# 
#     print("Loss Function at t={}: {}".format(t_, loss_0))
#     print("MSE at t={}: {}".format(t_, metric_mse_0))
#     if file_to_save is not None:
#         pdf = pd.DataFrame(weights, index=weights_names).T
#         pdf["t"] = t_
#         pdf["loss"] = loss_0
#         pdf["metric_mse"] = metric_mse_0
#         pdf.to_csv(
#             file_to_save, sep=";",
#             index=True, mode='a', header=False
#         )
# 
#     contine_loop = True
#     while contine_loop:
#         for batch in batch_generator:
#             # Get the Batches
#             batch_x = batch[0]
#             batch_y = batch[1]
#             # Compute gradient on batches
#             loss_gradient = np.array(gradient_function(
#                 weights, batch_x, batch_y
#             ))
#             # Update Weights
#             weights, s_, v_ = update_parameters_with_adam(
#                 weights, loss_gradient, s_, v_, t_,
#                 learning_rate=learning_rate,
#                 beta1=beta1,
#                 beta2=beta2
#             )
#         loss_t = loss_function(weights)
#         delta = -(loss_t - loss_0)
# 
#         loss_0 = loss_t
#         # print("t= {} delta= {}".format(t_, delta))
#         # print("t= {} n_tol= {}".format(t_, n_tol))
# 
#         if delta < tolerance:
#             n_tol = n_tol + 1
#         else:
#             n_tol = 0
#         if t_ % print_step == 0:
#             # Compute loss
#             if metric_function is None:
#                 metric_mse_t = None
#             else:
#                 metric_mse_t = metric_function(weights)
#             print("\t MSE at t={}: {}".format(t_, metric_mse_t))
#             print("\t Iteracion: {}. Loss: {}".format(t_, loss_t))
#             if file_to_save is not None:
#                 pdf = pd.DataFrame(weights, index=weights_names).T
#                 pdf["t"] = t_
#                 pdf["loss"] = loss_0
#                 pdf["metric_mse"] = metric_mse_0
#                 pdf.to_csv(
#                     file_to_save, sep=";",
#                     index=True, mode='a', header=False
#                 )
# 
#         if t_ >= n_iter:
#             print("Maximum number of iterations achieved.")
#             contine_loop = False
#             if metric_function is None:
#                 metric_mse_t = None
#             else:
#                 metric_mse_t = metric_function(weights)
#             if file_to_save is not None:
#                 pdf = pd.DataFrame(weights, index=weights_names).T
#                 pdf["t"] = t_
#                 pdf["loss"] = loss_0
#                 pdf["metric_mse"] = metric_mse_0
#                 pdf.to_csv(
#                     file_to_save, sep=";",
#                     index=True, mode='a', header=False
#                 )
#         if n_tol >= n_counts_tolerance:
#             print("Achieved Convergence. Delta: {}".format(delta))
#             contine_loop = False
#             if metric_function is None:
#                 metric_mse_t = None
#             else:
#                 metric_mse_t = metric_function(weights)
#             if file_to_save is not None:
#                 pdf = pd.DataFrame(weights, index=weights_names).T
#                 pdf["t"] = t_
#                 pdf["loss"] = loss_0
#                 pdf["metric_mse"] = metric_mse_0
#                 pdf.to_csv(
#                     file_to_save, sep=";",
#                     index=True, mode='a', header=False
#                 )
#         t_ = t_ + 1
#     return weights
