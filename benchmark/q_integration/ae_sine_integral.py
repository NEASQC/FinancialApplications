"""
Mandatory code for softaware implemetation of the Benchmark Test Case
of AE kernel
"""
import sys
sys.path.append("../..")
import os
import re
import json
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from get_qpu import get_qpu

from QQuantLib.utils.benchmark_utils import combination_for_list
from QQuantLib.finance.quantum_integration import q_solve_integral


def save(save, save_name, input_pdf, save_mode):
    """
    For saving panda DataFrames to csvs

    Parameters
    ----------

    save: bool
        For saving or not
    save_nam: str
        name for file
    input_pdf: pandas DataFrame
    save_mode: str
        saving mode: overwrite (w) or append (a)
    """
    if save:
        with open(save_name, save_mode) as f_pointer:
            input_pdf.to_csv(
                f_pointer,
                mode=save_mode,
                header=f_pointer.tell() == 0,
                sep=';'
            )

def sine_integral(n_qbits, interval, ae_dictionary):
    """
    Function for solving the sine integral between two input values:

    Parameters
    ----------

    n_qbits : int
        for discretization of the input domain in 2ⁿ intervals
    interval: int
        Interval for integration: Only can be:
            0 : [0,3π/8]
            1 : [π, 5π/4]
    ae_dictionary : dict
        dictionary with the complete amplitude estimation
        algorithm configuration

    Return
    ----------

    pdf : pandas DataFrame
        DataFrame with the complete information of the benchmark
    """

    #Local copy for AE configuration dictionary
    ae_dictionary_ = deepcopy(ae_dictionary)

    start_time = time.time()

    #Section 2.1: Function for integration
    function = np.sin

    #Section 2.1: Integration Intervals
    start = [0.0, np.pi]
    end = [3.0*np.pi/8.0, 5.0*np.pi/4.0]
    if interval not in [0, 1]:
        raise ValueError("interval MUST BE 0 or 1")
    a_ = start[interval]
    b_ = end[interval]
    #Section 2.1: Computing exact integral
    exact_integral = np.cos(a_) - np.cos(b_)

    #Section 2.2: Domain discretization
    domain_x = np.linspace(a_, b_, 2 ** n_qbits + 1)

    #Section 2.3: Function discretization
    f_x = []
    for i in range(1, len(domain_x)):
        step_f = (function(domain_x[i]) + function(domain_x[i-1]))/2.0
        f_x.append(step_f)
        #x_.append((domain_x[i] + domain_x[i-1])/2.0)
    f_x = np.array(f_x)

    #Section 2.4: Array Normalisation
    normalization = np.max(np.abs(f_x)) + 1e-8
    f_norm_x = f_x/normalization

    #Sections 2.5 and 2.6: Integral computation using AE techniques

    #Section 3.2.3: configuring input dictionary for q_solve_integral
    q_solve_configuration = {
        "array_function" : f_norm_x,
        "array_probability" : None,
        "encoding" : 2
    }
    #Now added the AE configuration.
    #The ae_dictionary_ has a local copy of the AE configuration.
    q_solve_configuration.update(ae_dictionary_)

    #The q_solve_integral needs a QPU object.
    #q_solve_configuration["qpu"] = get_qpu(q_solve_configuration["qpu"])

    #Compute the integral using AE algorithms!!
    solution, solver_object = q_solve_integral(**q_solve_configuration)

    #Section 3.2.3: eq (3.7). It is an adapatation of eq (2.22)
    estimator_s = normalization * (b_ - a_) * solution / (2 ** n_qbits)

    #Section 2.7: Getting the metrics
    absolute_error = np.abs(estimator_s["ae"] - exact_integral)
    relative_error = absolute_error / exact_integral
    oracle_calls = solver_object.oracle_calls

    end_time = time.time()
    elapsed_time = end_time - start_time

    #ae_dictionary_.pop('array_function')
    #ae_dictionary_.pop('array_probability')

    #Section 4.2: Creating the output pandas DataFrame for using
    #properly the KERNEL_BENCHMARK class

    #Adding the complete AE configuration
    pdf = pd.DataFrame([ae_dictionary_])

    #Adding information about the computed integral
    pdf["interval"] = interval
    pdf["n_qbits"] = n_qbits
    pdf["a_"] = a_
    pdf["b_"] = b_

    #Adding the output from q_solve_integral
    pdf = pd.concat([pdf, solution], axis=1)

    #Adding the AE computation of the integral
    integral_columns = ["integral_" + col for col in solution.columns]
    pdf[integral_columns] = estimator_s

    #Adding information about the integral that must be computed
    pdf["exact_integral"] = exact_integral
    pdf["riemann_sum"] = (b_ - a_) * np.sum(f_x) / (2 ** n_qbits)

    #Adding the normalization constant
    pdf["normalization"] = normalization

    #Error vs exact integral
    pdf["absolute_error_exact"] = absolute_error
    pdf["relative_error_exact"] = relative_error

    #Error vs Riemann Sum
    pdf["IntegralAbsoluteError"] = np.abs(pdf["integral_ae"] - pdf["riemann_sum"])

    #Error by Riemann aproximation to Integral
    pdf["absolute_riemann_error"] = np.abs(
        pdf["riemann_sum"] - pdf["exact_integral"])
    pdf["oracle_calls"] = oracle_calls
    pdf["elapsed_time"] = elapsed_time
    pdf["run_time"] = solver_object.run_time
    pdf["quantum_time"] = solver_object.quantum_time

    #pdf will have a complete output for trazability.
    #Columns for the metric according to 2.7 and 2.8 will be:
    #[absolute_error_sum, oracle_calls,
    #elapsed_time, run_time, quantum_time]
    return pdf

def select_ae(ae_method):
    """
    Function for selecting the AE algorithm used in the benchmark

    Parameters
    ----------

    ae_method : string
       Amplitude Estimation method used in the benchmark

    Returns
    _______

    ae_configuration : dict
        Dictionary with the complete configuration of the Amplitude

    """

    folder_json = os.getcwd()
    #folder_json = re.sub(
    #    r"WP3_Benchmark/(?=WP3_Benchmark/)*.*", "WP3_Benchmark/", folder_json)
    folder_json = folder_json + "/jsons"
    lista_ae_ = []
    if ae_method == "MLAE":
        lista_ae_.append(folder_json+"/integral_mlae_configuration.json")
    elif ae_method == "IQAE":
        lista_ae_.append(folder_json+"/integral_iqae_configuration.json")
    elif ae_method == "mIQAE":
        lista_ae_.append(folder_json+"/integral_miqae_configuration.json")
    elif ae_method == "RQAE":
        lista_ae_.append(folder_json+"/integral_rqae_configuration.json")
    elif ae_method == "CQPEAE":
        lista_ae_.append(folder_json+"/integral_cqpeae_configuration.json")
    elif ae_method == "IQPEAE":
        lista_ae_.append(folder_json+"/integral_iqpeae_configuration.json")
    elif ae_method == "MCAE":
        lista_ae_.append(folder_json+"/integral_mcae_configuration.json")
    else:
        raise ValueError(
            "ae_method MUST BE: MLAE, IQAE, RQAE, CQPEAE or IQPEAE")

    ae_list_ = []
    for ae_json_ in lista_ae_:
        with open(ae_json_) as json_file_:
            ae_list_ = ae_list_ + json.load(json_file_)
    #Creates the complete configuration for AE solvers
    final_list_ = combination_for_list(ae_list_)
    return final_list_
    #if len(final_list_) > 1:
    #    text = "There are more than 1 AE algorithm configuration. "\
    #        "FOR BENCHMARK only 1 configuration should be given. "\
    #        "Please change the correspondent json!!"
    #    raise ValueError(text)
    #ae_configuration = final_list_[0]
    #return ae_configuration


def run_id(n_qbits, interval, ae_configuration, qpu, repetitions, save_name):

    list_of_pdfs = []
    ae_configuration.update({"qpu": get_qpu(qpu)})
    for i in range(repetitions):
        step_pdf = sine_integral(
            n_qbits,
            interval,
            ae_configuration
        )
        list_of_pdfs.append(step_pdf)
        save(True, save_name, step_pdf, "a")
    pdf = pd.concat(list_of_pdfs)
    pdf.reset_index(drop=True, inplace=True)
    return pdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    #Arguments for execution
    parser.add_argument(
        "-n_qbits",
        dest="n_qbits",
        type=int,
        help="Number of qbits for interval discretization.",
        default=None,
    )
    parser.add_argument(
        "-interval",
        dest="interval",
        type=int,
        help="Integration Interval. valid Values 0 or 1",
        default=None,
    )
    parser.add_argument(
        "-repetitions",
        dest="repetitions",
        type=int,
        help="Number of repetitions the integral will be computed."+
        "Default: 1",
        default=1,
    )
    #AE algorithm configuration arguments
    parser.add_argument(
        "-ae_type",
        dest="ae_type",
        type=str,
        default=None,
        help="AE algorithm for integral kernel: "+
        "[MLAE, IQAE, RQAE, MCAE, CQPEAE, IQPEAE]",
    )
    #For information about the configuation
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the AE algorihtm configuration."
    )
    #QPU argument
    parser.add_argument(
        "-qpu",
        dest="qpu",
        type=str,
        default="python",
        help="QPU for simulation: See function get_qpu in get_qpu module",
    )
    #Saving results arguments
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving results",
    )
    parser.add_argument(
        "--folder",
        dest="folder_path",
        type=str,
        help="Path for storing folder",
        default="./",
    )
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="For counting elements on the list",
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    #Execution argument
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()

    final_list = select_ae(args.ae_type)
    #ae_configuration.update({"qpu":get_qpu(args.qpu)})

    if args.print:
        if args.id is not None:
            print(final_list[args.id])
        else:
            print(final_list)

    if args.count:
        print(len(final_list))

    folder = "/home/cesga/gferro/Codigo/FinancialApplications/benchmark/q_integration/"

    if args.execution:
        if args.id is not None:
            pdf = run_id(
                args.n_qbits,
                args.interval,
                final_list[args.id],
                args.qpu,
                args.repetitions,
                folder + args.ae_type + "_"+str(args.id)+".csv"
            )
    #     list_of_pdfs = []
    #     for i in range(args.repetitions):
    #         step_pdf = sine_integral(
    #             args.n_qbits,
    #             args.interval,
    #             ae_configuration
    #         )
    #         list_of_pdfs.append(step_pdf)
    #     pdf = pd.concat(list_of_pdfs)
    #     pdf.reset_index(drop=True, inplace=True)
    #     print(pdf)
    #     if args.save:
    #         if args.folder_path is None:
    #             raise ValueError("folder_name is None!")
    #         if not os.path.exists(args.folder_path):
    #             os.mkdir(args.folder_path)
    #         base_name = args.ae_type + "_n_qbits_" + str(args.n_qbits) + \
    #             "_interval_" + str(args.interval) + ".csv"
    #         file_name = args.folder_path + "/" + base_name
    #         print(file_name)
    #         with open(file_name, "w") as f_pointer:
    #             pdf.to_csv(
    #                 f_pointer,
    #                 mode="w",
    #                 sep=';'
    #             )
