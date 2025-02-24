"""
Mandatory code for softaware implemetation of the Benchmark Test Case
of AE kernel
"""
import os
import re
import json
import time
import numpy as np
import pandas as pd
from copy import deepcopy

import sys

sys.path.append("../..")
from QQuantLib.utils.benchmark_utils import combination_for_list
from QQuantLib.utils.benchmark_utils import create_ae_pe_solution
from QQuantLib.finance.quantum_integration import q_solve_integral


from QQuantLib.qpu.select_qpu import select_qpu

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
    q_solve_configuration.update(ae_dictionary)

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
    max_oracle_depth = solver_object.max_oracle_depth

    end_time = time.time()
    elapsed_time = end_time - start_time

    #ae_dictionary_.pop('array_function')
    #ae_dictionary_.pop('array_probability')

    #Section 4.2: Creating the output pandas DataFrame for using
    #properly the KERNEL_BENCHMARK class

    #Adding the complete AE configuration
    pdf = pd.DataFrame([ae_dictionary])

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
    pdf["max_oracle_depth"] = max_oracle_depth
    pdf["elapsed_time"] = elapsed_time
    pdf["run_time"] = solver_object.run_time
    pdf["quantum_time"] = solver_object.quantum_time

    #pdf will have a complete output for trazability.
    #Columns for the metric according to 2.7 and 2.8 will be:
    #[absolute_error_sum, oracle_calls,
    #elapsed_time, run_time, quantum_time]
    return pdf

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

def run_id(
    n_qbits=None,
    interval=0,
    repetitions=None,
    ae_config=None,
    qpu=None,
    save_=False,
    folder_path=None,
    id_=None
    ):

    ae_config.update({"qpu":select_qpu(ae_config)})

    if save_:
        if folder_path is None:
            raise ValueError("folder_name is None!")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    ae_type = ae_config["ae_type"]
    base_name = ae_type + "_n_qbits_" + str(n_qbits) + \
        "_interval_" + str(interval) + "_id_" + str(id_) + ".csv"
    file_name = folder_path + "/" + base_name

    list_of_pdfs = []
    for i in range(repetitions):
        step_pdf = sine_integral(
            n_qbits,
            interval,
            ae_config
        )
        save(save_, file_name, step_pdf, "a")
        list_of_pdfs.append(step_pdf)
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
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    #AE algorithm configuration arguments
    parser.add_argument(
        "-json_ae",
        dest="json_ae",
        type=str,
        default=None,
        help="JSON AE algorithm configuration",
    )
    #QPU configuration
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="qpu/qpu.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "-folder",
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
    #For information about the configuation
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the AE algorihtm configuration."
    )
    #Saving results arguments
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving results",
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

    with open(args.json_ae) as json_file:
        ae_cfg = json.load(json_file)
    # Creates the complete configuration for AE solvers
    ae_list = combination_for_list(ae_cfg)

    with open(args.json_qpu) as json_file:
        noisy_cfg = json.load(json_file)
    qpu_list = combination_for_list(noisy_cfg)
    final_list = create_ae_pe_solution(ae_list, qpu_list)

    if args.count:
        print(len(final_list))
    if args.print:
        if args.id is not None:
            print(final_list[args.id])
        else:
            print(final_list)

    if args.execution:
        if args.id is not None:
            print(run_id(
                n_qbits=args.n_qbits,
                interval=args.interval,
                repetitions=args.repetitions,
                ae_config=final_list[args.id],
                folder_path=args.folder_path,
                #qpu=args.qpu,
                save_=args.save,
                id_=args.id
            ))
