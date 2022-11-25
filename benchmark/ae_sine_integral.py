"""
In this module the function for computing sin integral are provided.

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import time
import sys
import numpy as np
import pandas as pd
sys.path.append("../../")
from benchmark.benchmark_utils import list_of_dicts_from_jsons
from QQuantLib.finance.quantum_integration import q_solve_integral
from QQuantLib.utils.qlm_solver import get_qpu



def sine_integral(n_qbits, interval, ae_dictionary):
    """
    Function for solving the sine integral between two input values:

    n_qbits : int
        for discretization of the input domain in 2ⁿ intervals
    interval: int
        Interval for integration: Only can be:
            0 : [0,3π/8]
            1 : [3π/4, 9π/8]
            2 : [π, 5π/4]
    ae_dictionary : dict
        dictionary with the complete amplitude estimation
        algorithm configuration

    Return
    ----------

    metrics : pandas DataFrame
        DataFrame with the metrics of the benchmark
    pdf : pandas DataFrame
        DataFrame with the complete information of the benchmark
    """


    start_time = time.time()

    start = [0.0, 3.0*np.pi/4.0, np.pi]
    end = [3.0*np.pi/8.0, 9.0*np.pi/8.0, 5.0*np.pi/4.0]
    #The sine function
    function = np.sin

    if interval not in [0, 1, 2]:
        raise ValueError("interval MUST BE 0, 1 or 2")

    #Getting the domain integration limits
    a_ = start[interval]
    b_ = end[interval]
    #Computing exact integral
    exact_integral = np.cos(a_) - np.cos(b_)
    #Discretizing the domain integration
    domain_x = np.linspace(a_, b_, 2 ** n_qbits + 1)
    #Discretization of the sine function
    f_x = []
    #x_ = []
    for i in range(1, len(domain_x)):
        step_f = (function(domain_x[i]) + function(domain_x[i-1]))/2.0
        f_x.append(step_f)
        #x_.append((domain_x[i] + domain_x[i-1])/2.0)
    f_x = np.array(f_x)
    #x_ = np.array(x_)
    #Normalisation constant
    normalization = np.max(np.abs(f_x)) + 1e-8
    #Normalization of the Riemann array
    f_norm_x = f_x/normalization
    #Encoding dictionary
    encoding_dict = {
        "array_function" : f_norm_x,
        "array_probability" : None,
        "encoding" : 2
    }
    #Updating the ae configuration with the encoding configuration
    ae_dictionary.update(encoding_dict)
    #EXECUTE COMPUTATION
    solution, solver_object = q_solve_integral(**ae_dictionary)
    #Amplitude Estimation computed integral estimator
    estimator_s = normalization * (b_ - a_) * solution / (2 ** n_qbits)
    #Metrics computation
    absolute_error = np.abs(estimator_s["ae"] - exact_integral)
    relative_error = absolute_error / exact_integral
    oracle_calls = solver_object.oracle_calls
    end_time = time.time()
    elapsed_time = end_time - start_time

    ae_dictionary.pop('array_function')
    ae_dictionary.pop('array_probability')

    pdf = pd.DataFrame([ae_dictionary])
    pdf["interval"] = interval
    pdf["n_qbits"] = n_qbits
    pdf["a_"] = a_
    pdf["b_"] = b_
    wanted_columns = pdf.columns
    #Store the integral from q_solve_integral
    pdf = pd.concat([pdf, solution], axis=1)
    #the desired integral
    integral_columns = ["integral_" + col for col in solution.columns]
    pdf[integral_columns] = estimator_s
    pdf["exact_integral"] = exact_integral
    #Sum of Riemann array
    pdf["riemann_sum"] = (b_ - a_) * np.sum(f_x) / (2 ** n_qbits)
    pdf["normalization"] = normalization
    #Error vs exact integral
    pdf["absolute_error_exact"] = absolute_error
    pdf["relative_error_exact"] = relative_error
    #Error vs Riemann Sum
    pdf["absolute_error_sum"] = np.abs(pdf["ae"] - pdf["riemann_sum"])
    #Error by Riemann aproximation to Integral
    pdf["absolute_riemann_error"] = np.abs(pdf["riemann_sum"] - pdf["exact_integral"])
    pdf["oracle_calls"] = oracle_calls
    pdf["elapsed_time"] = elapsed_time
    pdf["quantum_time"] = solver_object.quantum_time

    columns_metrics = [
        "absolute_error_exact", "relative_error_exact", "absolute_error_sum",
        "absolute_riemann_error", "oracle_calls",
        "elapsed_time", "quantum_time"
    ]
    metrics = pdf[
        list(wanted_columns) + columns_metrics
    ]
    return metrics, pdf
