
"""
This module contains a class for selecting data encoding protocols

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import warnings
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
sys.path.append("../")
from encoding_protocols import Encoding
from benchmark_utils import list_of_dicts_from_jsons
from QQuantLib.AE.ae_class import AE
from QQuantLib.utils.utils import text_is_none
from QQuantLib.utils.qlm_solver import get_qpu

def q_solve_integral(**kwargs):
    """
    Function for solving an integral using quantum amplitude estimationtechniques.

    Parameters
    ----------
    Mandatory keys of the input kwargs will be:

    array_function : numpy array
        numpy array wiht the desired function for encoding into the Quantum Circuit.
    encoding : int
        Selecting the encode protocol
    array_probability : numpy array
        numpy array wiht the desired probability for encoding into the
        Quantum Cirucit. It can be None (uniform distribution will be used)
    ae_type : string
        string with the desired AE algorithm:
        MLAE, CQPEAE, IQPEAE, IQAE, RQAE

    Other keys will be realted with circuit implementations of the encoding procedure
    or/and AE algorithm configuration (see QQQuantLib.AE.ae_class)
    """


    encoding = kwargs.get("encoding", None)
    ae_type = kwargs.get("ae_type", None)
    if (encoding == 0) and (ae_type == "RQAE"):
        string_error = (
            "RQAE method CAN NOT BE USED with encoding protocol: "+str(encoding)
        )

        warnings.warn(string_error)

        ae_estimation = pd.DataFrame(
            [None, None, None],
            index=["ae", "ae_l", "ae_u"],
        ).T
        return ae_estimation, None, None
    else:

        #Mandatory kwargs for encoding data
        array_function = kwargs.get("array_function", None)
        text_is_none(array_function, "array_function", variable_type=np.ndarray)
        array_probability = kwargs.get("array_probability", None)
        text_is_none(encoding, "encoding", variable_type=int)
        encoding_dict = {"multiplexor": kwargs.get("multiplexor", True)}
        #instantiate encoding class
        encode_class = Encoding(
            array_function=array_function,
            array_probability=array_probability,
            encoding=encoding,
            **encoding_dict
        )
        #execute run method of the encoding class
        encode_class.run()


        if encode_class.oracle is None:
            raise ValueError("Oracle was not created!!")

        #Mandatory kwargs for ae solver
        ae_dict = deepcopy(kwargs)
        #Delete keys from encoding
        for step in ["array_function", "array_probability", "encoding", "multiplexor"]:
            ae_dict.pop(step, None)
        ae_dict.pop("ae_type", None)
        #Instantiate AE solver
        solver_ae = AE(
            oracle=encode_class.oracle,
            target=encode_class.target,
            index=encode_class.index,
            ae_type=ae_type,
            **ae_dict)
        # run the amplitude estimation algorithm
        solver_ae.run()
        # Recover amplitude estimation from ae_solver
        if encoding == 0:
            ae_pdf = solver_ae.ae_pdf
        elif encoding == 1:
            if ae_type == "RQAE":
                #Amplitude is provided directly by this algorithm
                ae_pdf = solver_ae.ae_pdf
            else:
                #Other algorithms return probability
                ae_pdf = np.sqrt(solver_ae.ae_pdf)
        elif encoding == 2:
            if ae_type == "RQAE":
                #RQAE provides amplitude directly.
                ae_pdf = solver_ae.ae_pdf
            else:
                #Other algorithms return probability
                ae_pdf = np.sqrt(solver_ae.ae_pdf)
        else:
            raise ValueError("Not valid encoding key was provided!!!")
        #Now we need to deal with encoding normalisation
        ae_estimation = ae_pdf * encode_class.encoding_normalization
        return ae_estimation, solver_ae, encode_class


def run_id(ae_problem, id_name, qlmaas=False, file_name=None, folder_name=None, save=False):

    #domain defintion
    #Strict positive function
    a_ = 0
    b_ = np.pi / 4.0
    #a_ = np.pi - np.pi / 4.0
    #b_ = np.pi + np.pi / 8.0
    #number of qbits
    n_ = 6
    #domain discretization
    domain_x = np.linspace(a_, b_, 2 ** n_)
    #discretized probability distribution
    p_x = domain_x
    #discretized function
    f_x = np.sin(domain_x)
    f_x_normalisation = np.max(f_x) + 1e-8
    #normalised function
    norm_f_x = f_x / f_x_normalisation

    prob = True
    #normalisation constants
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
    ae_problem.update({
        "array_function" : norm_f_x,
        "array_probability" : norm_p_x,
    })
    linalg_qpu = get_qpu(qlmaas)
    ae_problem.update({"qpu": linalg_qpu})

    #EXECUTE COMPUTATION
    solution, solver_object, encode_object = q_solve_integral(**ae_problem)
    #Post Procces and Saving

    ae_problem.update({"file_name": file_name})
    ae_problem.update({"domain_a": a_})
    ae_problem.update({"domain_b": b_})
    ae_problem.update({"domain_n": n_})
    pdf = pd.DataFrame([ae_problem])
    pdf = pd.concat([pdf, solution], axis=1)
    q_riemman = solution * p_x_normalisation * f_x_normalisation
    pdf[
        ["integral_" + col for col in q_riemman.columns]
    ] = q_riemman
    pdf["p_x_normalisation"] = p_x_normalisation
    pdf["f_x_normalisation"] = f_x_normalisation
    pdf["riemman"] = riemman
    pdf["error_classical"] = abs(
        pdf["integral_ae"] - pdf["riemman"]
    )


    if (solver_object is None) and (encode_object is None):
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
    return pdf

def run_staff(dict_list, file_name="Todo.csv", folder_name=None, qlmaas=False, save=False):
    """
    run all problems
    """
    #list_of_pdfs = []
    for i, step in enumerate(dict_list):
        run_id(
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
        lista_ae.append("jsons_2/mlae_configuration.json")
    if args.iqae_var:
        lista_ae.append("jsons_2/iqae_configuration.json")
    if args.rqae_var:
        lista_ae.append("jsons_2/rqae_configuration.json")
    if args.cqpeae_var:
        lista_ae.append("jsons_2/cqpeae_configuration.json")
    if args.iqpeae_var:
        lista_ae.append("jsons_2/iqpeae_configuration.json")
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
                run_id(
                    final_list[args.id],
                    args.id,
                    file_name=args.file_name,
                    folder_name=args.folder_path,
                    qlmaas=args.qlmass,
                    save=args.save,
                )
