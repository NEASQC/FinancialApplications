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
from QQuantLib.finance.ae_price_estimation_step_payoff import ae_price_estimation_step_po
from QQuantLib.qpu.select_qpu import select_qpu


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
    solve_ae_pe,
    id_name,
    repetitions,
    file_name="",
    folder_name="",
    #qpu=None,
    save_=False
):
    """
    This function configure the mandatory dictionary needed for solving
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
    repetitions : int
        number of times for executing the estimation problem
    file_name: string
        name for the file where results will be stored. If not given will
        be created using id_name and ae_type string.
    folder_name: string
        folder name for saving the results of the solution of the
        price estimation problem
    qlmaas: bool
        For usign a QLM as a Service for solving the price estimation
        problem
    save_: bool
        For saving the results of the the price estimation problem as a
        csv

    Return
    ----------

    pdf : Pandas DataFrame
        DataFrame with all the information of the price estimation
        problem, the configuration of the ae and the obtained results
    """
    qpu = select_qpu(solve_ae_pe)

    save_name = folder_name + str(id_name) + "_" + \
        solve_ae_pe["file"] + str(file_name) +  ".csv"
    solve_ae_pe.update({"qpu": qpu})
    final = []
    for i in range(repetitions):
        step_pdf = ae_price_estimation_step_po(**solve_ae_pe)
        final.append(step_pdf)
        save(save_, save_name, step_pdf, "a")
    final = pd.concat(final).reset_index(drop=True)
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="For counting elements on the list",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing "
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
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving results",
    )
    parser.add_argument(
        "-folder",
        dest="folder_path",
        type=str,
        help="Path for storing folder",
        default="./",
    )
    parser.add_argument(
        "-name",
        dest="file_name",
        type=str,
        help="Name for storing csv. Only applies for --all",
        default=None,
    )
    parser.add_argument(
        "-qpu",
        dest="qpu",
        type=str,
        default="python",
        help="QPU for simulation: [qlmass, python, c]",
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
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
        "-json_domain",
        dest="json_domain",
        type=str,
        default="jsons/domain_configuration.json",
        help="JSON with the domain configuration",
    )
    parser.add_argument(
        "-json_density",
        dest="json_density",
        type=str,
        default="jsons/density_probability.json",
        help="JSON with the probability density configuration",
    )
    parser.add_argument(
        "-json_payoff",
        dest="json_payoff",
        type=str,
        default="jsons/payoffs.json",
        help="JSON with the payoff configuration",
    )
    parser.add_argument(
        "-json_ae",
        dest="json_ae",
        type=str,
        default=None,
        help="JSON AE algorithm configuration",
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="jsons/qpu_ideal.json",
        help="JSON with the qpu configuration",
    )
    args = parser.parse_args()
    print(args)

    #First we need to load the dictionaries for the price estimation problems

    with open(args.json_domain) as json_file:
        domain_cfg = json.load(json_file)
    with open(args.json_density) as json_file:
        density_cfg = json.load(json_file)
    with open(args.json_payoff) as json_file:
        payoff_cfg = json.load(json_file)
    #list wiht all the complete price estimation problems
    pe_problem = create_pe_problem(domain_cfg, payoff_cfg, density_cfg)

    if args.json_ae is None:
        raise ValueError("AE algorithm configuration NOT provided!")

    with open(args.json_ae) as json_file:
        ae_cfg = json.load(json_file)
    # Creates the complete configuration for AE solvers
    ae_solver = combination_for_list(ae_cfg)
    final_list = create_ae_pe_solution(ae_solver, pe_problem)
    # Combine with qpu configuration
    with open(args.json_qpu) as json_file:
        noisy_cfg = json.load(json_file)
    qpu_list = combination_for_list(noisy_cfg)
    final_list = create_ae_pe_solution(final_list, qpu_list)

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
                final_list[args.id],
                args.id,
                args.repetitions,
                file_name=args.file_name,
                folder_name=args.folder_path,
                #qpu=args.qpu,
                save_=args.save,
            ))
