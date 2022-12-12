"""
This module can be used for launchin benchamark using ae_sine_integral.py

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""
import sys
import json
import time
import numpy as np
import pandas as pd
sys.path.append("../../../")
from benchmark.benchmark_utils import combination_for_list
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.finance.quantum_integration import q_solve_integral

def sine_integral(n_qbits, interval, ae_dictionary):
    """
    Function for solving the sine integral between two input values:

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

    metrics : pandas DataFrame
        DataFrame with the metrics of the benchmark
    pdf : pandas DataFrame
        DataFrame with the complete information of the benchmark
    """


    start_time = time.time()

    #start = [0.0, 3.0*np.pi/4.0, np.pi]
    #end = [3.0*np.pi/8.0, 9.0*np.pi/8.0, 5.0*np.pi/4.0]
    start = [0.0, np.pi]
    end = [3.0*np.pi/8.0, 5.0*np.pi/4.0]
    #The sine function
    function = np.sin

    if interval not in [0, 1]:
        raise ValueError("interval MUST BE 0 or 1")

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
    pdf["absolute_error_sum"] = np.abs(pdf["integral_ae"] - pdf["riemann_sum"])
    #Error by Riemann aproximation to Integral
    pdf["absolute_riemann_error"] = np.abs(
        pdf["riemann_sum"] - pdf["exact_integral"])
    pdf["oracle_calls"] = oracle_calls
    pdf["elapsed_time"] = elapsed_time
    pdf["run_time"] = solver_object.run_time
    pdf["quantum_time"] = solver_object.quantum_time

    columns_metrics = [
        "absolute_error_exact", "relative_error_exact", "absolute_error_sum",
        "absolute_riemann_error", "oracle_calls",
        "elapsed_time", "run_time", "quantum_time"
    ]
    metrics = pdf[
        list(wanted_columns) + columns_metrics
    ]
    return metrics, pdf

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

    lista_ae_ = []
    if ae_method == "MLAE":
        lista_ae_.append("jsons/integral_mlae_configuration.json")
    elif ae_method == "IQAE":
        lista_ae_.append("jsons/integral_iqae_configuration.json")
    elif ae_method == "RQAE":
        lista_ae_.append("jsons/integral_rqae_configuration.json")
    elif ae_method == "CQPEAE":
        lista_ae_.append("jsons/integral_cqpeae_configuration.json")
    elif ae_method == "IQPEAE":
        lista_ae_.append("jsons/integral_iqpeae_configuration.json")
    else:
        raise ValueError("ae_method MUST BE: MLAE, IQAE, RQAE, CQPEAE or IQPEAE")

    ae_list_ = []
    for ae_json_ in lista_ae_:
        with open(ae_json_) as json_file_:
            ae_list_ = ae_list_ + json.load(json_file_)
    #Creates the complete configuration for AE solvers
    final_list_ = combination_for_list(ae_list_)
    if len(final_list_) > 1:
        text = "There are more than 1 AE algorithm configuration. "\
            "FOR BENCHMARK only 1 configuration should be given. "\
            "Please change the correspondent json!!"
        raise ValueError(text)
    ae_configuration = final_list_[0]
    del ae_configuration["integral"]
    del ae_configuration["number_of_tests"]
    return ae_configuration

def run_id(
    n_qbits,
    ae_problem,
    id_name,
    qlmaas=False,
    file_name=None,
    folder_name=None,
    save=False
):
    linalg_qpu = get_qpu(qlmaas)
    print(linalg_qpu)
    ae_problem.update({"qpu": linalg_qpu})
    ae_problem.update({"save": save})
    integral = ae_problem['integral']
    ae_problem.pop('integral')
    if save:
        if folder_name is None:
            raise ValueError("folder_name is None!")
        if file_name is None:
            file_name = ae_problem["ae_type"] + "_{}.csv".format(id_name)
        metrics_file_name = folder_name + 'metrics_n_qbits_' \
            + str(n_qbits) + '_' + file_name
        file_name = folder_name + 'n_qbits_' + str(n_qbits) + '_' + file_name
    list_of_metrics = []
    for i in range(ae_problem["number_of_tests"]):
        ae_problem.update({"file_name": file_name})
        metrics, pdf = sine_integral(n_qbits, integral, ae_problem)
        print(pdf)
        list_of_metrics.append(pdf)
        if save:
            with open(file_name, "a") as f_pointer:
                pdf.to_csv(
                    f_pointer,
                    mode="a",
                    header=f_pointer.tell() == 0,
                    sep=';'
                )
            #with open(metrics_file_name, "a") as m_pointer:
            #    metrics.to_csv(
            #        m_pointer,
            #        mode="a",
            #        header=m_pointer.tell() == 0,
            #        sep=';'
            #    )
    metrics_pdf = pd.concat(list_of_metrics)
    return metrics_pdf

def run_staff(
    n_qbits,
    ae_problem_list,
    file_name=None,
    folder_name=None,
    qlmaas=False,
    save=False
):
    list_of_pdfs = []
    for i, step in enumerate(ae_problem_list):
        step_pdf = run_id(
            n_qbits,
            step,
            i,
            file_name=file_name,
            folder_name=folder_name,
            qlmaas=qlmaas,
            save=save)
        list_of_pdfs.append(step_pdf)
    price_pdf = pd.concat(list_of_pdfs)
    return price_pdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        dest="folder_path",
        type=str,
        help="Path for storing folder",
        default="./",
    )
    parser.add_argument(
        "--name",
        dest="file_name",
        type=str,
        help="Name for storing csv. Only applies for --all",
        default=None,
    )
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="For counting elements on the list",
    )
    parser.add_argument(
        "--list",
        dest="list",
        default=False,
        action="store_true",
        help="For listing "
    )
    parser.add_argument(
        "--qlmass",
        dest="qlmass",
        default=False,
        action="store_true",
        help="For using or not QLM as a Service",
    )
    parser.add_argument(
        "--all",
        dest="all",
        default=False,
        action="store_true",
        help="For executing complete list",
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    parser.add_argument(
        "-n_qbits",
        dest="n_qbits",
        type=int,
        help="Number of qbits for interval discretization.",
        default=None,
    )
    parser.add_argument(
        "--MLAE",
        dest="mlae_var",
        default=False,
        action="store_true",
        help="For adding MLAE staff",
    )
    parser.add_argument(
        "--IQAE",
        dest="iqae_var",
        default=False,
        action="store_true",
        help="For adding IQAE staff",
    )
    parser.add_argument(
        "--RQAE",
        dest="rqae_var",
        default=False,
        action="store_true",
        help="For adding RQAE staff",
    )
    parser.add_argument(
        "--CQPEAE",
        dest="cqpeae_var",
        default=False,
        action="store_true",
        help="For adding CQPEAE staff",
    )
    parser.add_argument(
        "--IQPEAE",
        dest="iqpeae_var",
        default=False,
        action="store_true",
        help="For adding IQPEAE staff",
    )
    parser.add_argument(
        "--MCAE",
        dest="mcae_var",
        default=False,
        action="store_true",
        help="For adding MCAE staff",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving results",
    )
    args = parser.parse_args()
    print(args)

    lista_ae = []
    if args.mlae_var:
        lista_ae.append("jsons/integral_mlae_configuration.json")
    if args.iqae_var:
        lista_ae.append("jsons/integral_iqae_configuration.json")
    if args.rqae_var:
        lista_ae.append("jsons/integral_rqae_configuration.json")
    if args.cqpeae_var:
        lista_ae.append("jsons/integral_cqpeae_configuration.json")
    if args.iqpeae_var:
        lista_ae.append("jsons/integral_iqpeae_configuration.json")
    if args.mcae_var:
        lista_ae.append("jsons/integral_mcae_configuration.json")

    ae_list = []
    for ae_json in lista_ae:
        with open(ae_json) as json_file:
            ae_list = ae_list + json.load(json_file)

    #Creates the complete configuration for AE solvers
    final_list = combination_for_list(ae_list)

    if args.count:
        print(len(final_list))
    if args.list:
        if args.id is not None:
            print(final_list[args.id])
        else:
            print(final_list)
    if args.execution:
        if args.n_qbits is None:
            raise ValueError("n_qbits CAN NOT BE None")
        if args.all:
            print(
                run_staff(
                    args.n_qbits,
                    final_list,
                    file_name=args.file_name,
                    folder_name=args.folder_path,
                    qlmaas=args.qlmass,
                    save=args.save,
                )
            )
        else:
            if args.id is not None:
                print(
                    run_id(
                        args.n_qbits,
                        final_list[args.id],
                        args.id,
                        file_name=args.file_name,
                        folder_name=args.folder_path,
                        qlmaas=args.qlmass,
                        save=args.save,
                    )
                )
