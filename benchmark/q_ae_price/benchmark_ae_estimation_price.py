"""
This module contains a class for selecting data encoding protocols

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import sys
import json
import pandas as pd
sys.path.append("../../")
from QQuantLib.utils.benchmark_utils import create_pe_problem, combination_for_list,\
create_ae_pe_solution
from QQuantLib.finance.ae_price_estimation import ae_price_estimation
from QQuantLib.utils.qlm_solver import get_qpu



def run_id(
    solve_ae_pe,
    id_name,
    file_name=None,
    folder_name=None,
    qpu=None,
    save=False
):
    """
    This function configure the mandatory dictionary neede for solvin
    an option price estimation problem using the ae_price_estimation
    function.

    Parameters
    ----------

    solve_ae_pe :  python dictionary
        The dictionary should have all the mandatory keys for creating
        a price estimation problem and solving it using a properly configured
        AE integrations technique.
    id_name: string
        name for giving to the estimation problem for saving purpouses
    file_name: string
        name for the file where results will be stored. If not given will
        be created using id_name and ae_type string.
    folder_name: string
        folder name for saving the results of the solution of the
        price estimation problem
    qlmaas: bool
        For usign a QLM as a Service for solving the price estimation
        problem
    save: bool
        For saving the results of the the price estimation problem as a
        csv

    Return
    ----------

    pdf : Pandas DataFrame
        DataFrame with all the information of the price estimation
        problem, the configuration of the ae and the obtained results
    """
    linalg_qpu = get_qpu(qpu)
    solve_ae_pe.update({"qpu": linalg_qpu})
    solve_ae_pe.update({"save": save})
    if save:
        if folder_name is None:
            raise ValueError("folder_name is None!")
        if file_name is None:
            file_name = solve_ae_pe["ae_type"] + "_{}.csv".format(id_name)
        file_name = folder_name + file_name
    solve_ae_pe.update({"file_name": file_name})
    print(solve_ae_pe)
    pdf = ae_price_estimation(**solve_ae_pe)
    return pdf

def run_staff(
    solve_ae_pe_list,
    file_name="Todo.csv",
    folder_name=None,
    qpu=None,
    save=False
):
    """
    This function executes sequentially a list of different AE_PriceP
    dictionaries.

    Parameters
    ----------

    solve_ae_pe_list :  list of python dictionary
        Each dictionary of the list should have all the mandatory keys for
        creating a price estimation problem and solving it using a properly
        configured AE integrations technique.
    file_name: string
        name for the file where results will be stored. If not given will
        be created using id_name and ae_type string.
    folder_name: string
        folder name for saving the results of the solution of the
        price estimation problem
    qlmaas: bool
        For usign a QLM as a Service for solving the price estimation
        problem
    save: bool
        For saving the results of the the price estimation problem as a
        csv

    Return
    ----------

    price_pdf : Pandas DataFrame
        DataFrame with all the information of the price estimation
        problem, the configuration of the ae and the obtained results
        for each element of the input dictionary
    """
    list_of_pdfs = []
    for i, step in enumerate(solve_ae_pe_list):
        step_pdf = run_id(step, i, file_name, folder_name, qpu, save)
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
        "--list", dest="list", default=False, action="store_true", help="For listing "
    )
    parser.add_argument(
        "-qpu",
        dest="qpu",
        type=str,
        default="python",
        help="QPU for simulation: [qlmass, python, c]",
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

    #First we need to load the dictionaries for the price estimation problems

    json_density = "jsons/ae_pe_density_probability.json"
    with open(json_density) as json_file:
        density_cfg = json.load(json_file)
    json_payoffs = "jsons/ae_pe_payoffs.json"
    with open(json_payoffs) as json_file:
        payoff_cfg = json.load(json_file)
    json_domain = "jsons/ae_pe_domain_configuration.json"
    with open(json_domain) as json_file:
        domain_cfg = json.load(json_file)
    #list wiht all the complete price estimation problems
    pe_problem = create_pe_problem(domain_cfg, payoff_cfg, density_cfg)

    lista_ae = []
    if args.mlae_var:
        lista_ae.append("jsons/ae_pe_mlae_configuration.json")
    if args.iqae_var:
        lista_ae.append("jsons/ae_pe_iqae_configuration.json")
    if args.rqae_var:
        lista_ae.append("jsons/ae_pe_rqae_configuration.json")
    if args.cqpeae_var:
        lista_ae.append("jsons/ae_pe_cqpeae_configuration.json")
    if args.iqpeae_var:
        lista_ae.append("jsons/ae_pe_iqpeae_configuration.json")
    if args.mcae_var:
        lista_ae.append("jsons/ae_pe_mcae_configuration.json")

    ae_list = []
    for ae_json in lista_ae:
        with open(ae_json) as json_file:
            ae_list = ae_list + json.load(json_file)

    #Creates the complete configuration for AE solvers
    ae_solver = combination_for_list(ae_list)
    final_list = create_ae_pe_solution(ae_solver, pe_problem)

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
                qpu=args.qpu,
                save=args.save,
            )
        else:
            if args.id is not None:
                print(run_id(
                    final_list[args.id],
                    args.id,
                    file_name=args.file_name,
                    folder_name=args.folder_path,
                    qpu=args.qpu,
                    save=args.save,
                ))
