
"""
This module contains a class for selecting data encoding protocols

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import warnings
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
sys.path.append("../../")
from benchmark.benchmark_utils import list_of_dicts_from_jsons
from QQuantLib.finance.quantum_integration import q_solve_integral
from QQuantLib.utils.qlm_solver import get_qpu


def problem(ae_problem, id_name, qlmaas=False, file_name=None, folder_name=None, save=False):
    n_ = 6
    list_a = [0.0, np.pi - np.pi / 4.0, np.pi]
    list_b = [np.pi / 4.0, np.pi + np.pi / 8.0, np.pi + np.pi / 4.0]
    list_c = ["positive_function", "positive_integral", "negative_integral"]
    opa = []
    for a_, b_, c_ in zip(list_a, list_b, list_c):
        domain_x = np.linspace(a_, b_, 2 ** n_)
        #discretized function
        f_x = np.sin(domain_x)
        f_x_normalisation = np.max(f_x) + 1e-8
        #normalised function
        norm_f_x = f_x / f_x_normalisation
        p_x = domain_x
        for prob in [True, False]:
            if prob:
                p_x_normalisation = np.sum(p_x) + 1e-8
                norm_p_x = p_x / p_x_normalisation
                #desired integral
                riemman = np.sum(p_x * f_x)
            else:
                p_x_normalisation = 1.0
                norm_p_x = None
                #desired integral
                riemman = np.sum(f_x)
            encoding_dict = {
                "a_" : a_,
                "b_" : b_,
                "n_" : n_,
                "result_type": c_,
                "Probability" : prob,
                "array_function" : norm_f_x,
                "array_probability" : norm_p_x,
                "f_x_normalisation": f_x_normalisation,
                "p_x_normalisation" : p_x_normalisation,
                "riemman" : riemman,
            }
            if not((not prob) and (int(ae_problem["encoding"]) == 1)):
                pdf, _, _ = run_id(
                    ae_problem,
                    id_name,
                    encoding_dict,
                    qlmaas=qlmaas,
                    file_name=file_name,
                    folder_name=folder_name,
                    save=save
                )
                opa.append(pdf)
    return opa

def run_id(ae_problem, id_name, encoding_problem, qlmaas=False, file_name=None, folder_name=None, save=False):

    ae_problem.update(encoding_problem)
    linalg_qpu = get_qpu(qlmaas)
    ae_problem.update({"qpu": linalg_qpu})

    #EXECUTE COMPUTATION
    solution, solver_object = q_solve_integral(**ae_problem)
    #Post Procces and Saving

    ae_problem.update({"file_name": file_name})
    pdf = pd.DataFrame([ae_problem])
    pdf = pd.concat([pdf, solution], axis=1)
    q_riemman = solution * encoding_problem["p_x_normalisation"] * encoding_problem["f_x_normalisation"]
    pdf[
        ["integral_" + col for col in q_riemman.columns]
    ] = q_riemman
    pdf["error_classical"] = abs(
        pdf["integral_ae"] - pdf["riemman"]
    )


    if solver_object is None:
        #Computation Fails Encoding 0 and RQAE
        pdf["schedule_pdf"] = [None]
        pdf["oracle_calls"] = [None]
        pdf["max_oracle_depth"] = [None]
    else:
        if solver_object.schedule_pdf is None:
            pdf["schedule_pdf"] = [None]
        else:
            pdf["schedule_pdf"] = [solver_object.schedule_pdf.to_dict()]
        pdf["oracle_calls"] = solver_object.oracle_calls
        pdf["max_oracle_depth"] = solver_object.max_oracle_depth


    if save:
        if folder_name is None:
            raise ValueError("folder_name is None!")
        if file_name is None:
            file_name = ae_problem["ae_type"] + "_{}.csv".format(id_name)
        file_name = folder_name + file_name
        with open(file_name, "a") as f_pointer:
            pdf.to_csv(f_pointer, mode="a", header=f_pointer.tell() == 0, sep=';')
    return pdf, solver_object

def run_staff(dict_list, file_name="Todo.csv", folder_name=None, qlmaas=False, save=False):
    """
    run all problems
    """
    #list_of_pdfs = []
    for i, step in enumerate(dict_list):
        problem(
            step,
            i,
            file_name=file_name,
            folder_name=folder_name,
            qlmaas=qlmaas,
            save=save
        )

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
        "--list", dest="list", default=False, action="store_true", help="For listing "
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
        lista_ae.append("json_tests/mlae_configuration.json")
    if args.iqae_var:
        lista_ae.append("json_tests/iqae_configuration.json")
    if args.rqae_var:
        lista_ae.append("json_tests/rqae_configuration.json")
    if args.cqpeae_var:
        lista_ae.append("json_tests/cqpeae_configuration.json")
    if args.iqpeae_var:
        lista_ae.append("json_tests/iqpeae_configuration.json")
    final_list = list_of_dicts_from_jsons(lista_ae)
    if args.count:
        print(len(final_list))
    if args.list:
        if args.id is not None:
            print(final_list[args.id])
        else:
            print(final_list)
    if args.execution:
        if args.all:
            run_staff(
                final_list,
                file_name=args.file_name,
                folder_name=args.folder_path,
                qlmaas=args.qlmass,
                save=args.save,
            )
        else:
            if args.id is not None:
                problem(
                    final_list[args.id],
                    args.id,
                    file_name=args.file_name,
                    folder_name=args.folder_path,
                    qlmaas=args.qlmass,
                    save=args.save,
                )
