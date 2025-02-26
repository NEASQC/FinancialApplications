"""
Script for using AE algorithms in a pure estimation problem.
A probability is loading in the quantum circuit and the AE algorithm
tries to estimate the ampliutude of an input provided state.
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import itertools as it
from collections import ChainMap
sys.path.append("../../")
from QQuantLib.finance.probability_class import DensityProbability
import QQuantLib.DL.data_loading as dl
from QQuantLib.utils.utils import bitfield
from QQuantLib.AE.ae_class import AE
from QQuantLib.utils.benchmark_utils import create_ae_pe_solution
from QQuantLib.utils.benchmark_utils import combination_for_list
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

def get_domain(**kwargs):
    """
    Creates the domain
    """
    # Configure domain
    n_qbits = kwargs.get("n_qbits", None)
    x0 = kwargs.get("x0", 1.0)
    xf = kwargs.get("xf", 3.0)
    # Configure probability
    domain = np.linspace(x0, xf, 2**n_qbits)
    return domain, n_qbits

def get_probability_density(domain, **kwargs):
    """
    Creates the probability denisty function array
    """
    # Configure probability
    pc = DensityProbability(**kwargs)
    # Creating probability distribution
    p_x = pc.probability(domain)
    return p_x

def get_amplitude_estimation(**kwargs):
    """
    Function for loading a probability distribution in a quantum circuit
    and return the probability of a input state using AE algorithms
    """

    # n_qbits = kwargs.get("n_qbits", None)
    # x0 = kwargs.get("x0", 1.0)
    # xf = kwargs.get("xf", 3.0)

    # Configure domain
    domain, n_qbits = get_domain(**kwargs)

    # pc = DensityProbability(**kwargs)
    # Creating probability distribution
    # p_x = pc.probability(domain)

    # Configure and create probability array
    p_x = get_probability_density(domain, **kwargs)
    # Creating oracle
    oracle = dl.load_probability(p_x)
    # Getting the target id and the target state
    # State 21 is the state with maximum probability
    target_id = kwargs.get("target_id", 21)
    if target_id > 2 ** n_qbits -1:
        raise ValueError("target bigger than 2^n-1")
    target = bitfield(target_id, n_qbits)
    index = list(range(n_qbits))
    # Create AE object
    ae_obj = AE(
        oracle=oracle,
        target=target,
        index=index,
        **kwargs
    )
    # Runing AE object
    ae_obj.run()
    # Processing results
    pdf = pd.DataFrame.from_dict(kwargs, orient="index").T
    ae_type = kwargs.get("ae_type")
    pdf["Value"] = p_x[target_id]
    # The post-proccessng depends on the AE used.
    if ae_type in ["RQAE", "mRQAE", "eRQAE", "sRQAE"]:
        # In the RQAE based methods the estimation of the amplitude is
        # provided. But in this case we codify an amplitude so
        # we need to obtain an estimation of the probability

        result = ae_obj.ae_pdf
        p_l, p_u = get_probability_from_amplitude(
            ae_obj.ae_pdf["ae_l"].values, ae_obj.ae_pdf["ae_u"].values
        )
        result["ae_l"] = p_l
        result["ae_u"] = p_u
        result["ae"] = (result["ae_u"] + result["ae_l"]) / 2.0
    else:
        result = ae_obj.ae_pdf
    pdf = pd.concat([pdf, result], axis=1)
    pdf["oracle_calls"] = ae_obj.oracle_calls
    pdf["max_oracle_depth"] = ae_obj.max_oracle_depth
    pdf["schedule_pdf"] = [ae_obj.schedule_pdf.to_dict()]
    pdf["measured_epsilon"] = (pdf["ae_u"] - pdf["ae_l"]) / 2.0
    pdf["absolute_error"] = np.abs(pdf["ae"] - pdf["Value"])
    return pdf


def get_probability_from_amplitude(a_l, a_u):
    """
    Transforming Amplitude into probability for RQAE algorithm like
    """

    if ((a_u >= 0) and (a_l >= 0)):
        p_l = a_l ** 2
        p_u = a_u ** 2
        return p_l, p_u
    if ((a_u < 0) and (a_l < 0)):
        p_l = a_u ** 2
        p_u = a_l ** 2
        return p_l, p_u
    if (a_u * a_l) < 0:
        if np.abs(a_u) >= np.abs(a_l):
            p_l = 0.0
            p_u = a_u ** 2
            return p_l, p_u
        else:
            p_l = 0.0
            p_u = a_l ** 2
            return p_l, p_u

def run_id(
    repetitions,
    id_,
    save_,
    #qpu,
    base_name,
    save_folder,
    target,
    **ae_configuration
):
    #Domain configuration

    # domain_configuration = {
    #     'x0': 0.01,
    #     'xf': 6.0,
    #     'n_qbits': 6,
    # }

    # #Probability density configuration
    # probability_configuration = {
    #     'probability_type': 'Black-Scholes',
    #     's_0': 2,
    #     'risk_free_rate': 0.05,
    #     'maturity': 1.0,
    #     'volatility': 0.2,
    # }
    # ae_configuration.update(domain_configuration)
    # ae_configuration.update(probability_configuration)

    qpu = select_qpu(ae_configuration)
    ae_configuration.update({"qpu": qpu})
    ae_configuration.update({"target_id": target})

    save_name = save_folder + str(id_) + "_" + ae_configuration["file"] + str(base_name) +  ".csv"
    print(save_name)
    for i in range(repetitions):
        tick = time.time()
        step_pdf = get_amplitude_estimation(**ae_configuration)
        print(step_pdf)
        tack = time.time()
        elapsed = tack - tick
        step_pdf["elapsed_time"] = elapsed
        save(save_, save_name, step_pdf, "a")




#m_k = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#ae_configuration = {
#    #Amplitude Estimation selection
#    'ae_type': 'mIQAE', #IQAE, RQAE, CQPEAE, IQPEAE
#
#    #MLAE configuration
#    'schedule': [
#        m_k,
#        [100]*len(m_k)
#    ],
#    'delta' : 1.0e-7,
#    'ns' : 10000,
#
#    #CQPEAE configuration
#    'auxiliar_qbits_number': 14,
#
#    #IQPEAE configuration
#    'cbits_number': 10,
#
#    #IQAE & RQAQE
#    'epsilon': 0.01,
#    #IQAE
#    'alpha': 0.05,
#    #RQAE
#    'gamma': 0.05,
#    'q': 1.2,
#
#    #shots
#    'shots': 100,
#    #Multi controlled decomposition
#    'mcz_qlm': False,
#    'qpu': linalg_qpu,
#}


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
        "--all",
        dest="all",
        default=False,
        action="store_true",
        help="For executing complete list",
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing the AE algorihtm configuration."
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving staff"
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
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
        "-name",
        dest="base_name",
        type=str,
        help="Additional name for the generated files",
        default="",
    )
    # parser.add_argument(
    #     "-qpu",
    #     dest="qpu",
    #     type=str,
    #     default="python",
    #     help="QPU for simulation: See function get_qpu in get_qpu module",
    # )
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
        "-json_ae",
        dest="json_ae",
        type=str,
        default=None,
        help="JSON AE algorithm configuration",
    )
    parser.add_argument(
        "-folder",
        dest="folder_path",
        type=str,
        help="Path for storing folder",
        default="./",
    )
    parser.add_argument(
        "-target",
        dest="target",
        type=int,
        help="Target State. Default will be 21.",
        default=21,
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="jsons/qpu_ideal.json",
        help="JSON with the qpu configuration",
    )
    args = parser.parse_args()

    with open(args.json_domain) as json_file:
        domain_cfg = json.load(json_file)
    with open(args.json_density) as json_file:
        density_cfg = json.load(json_file)
    with open(args.json_ae) as json_file:
        ae_cfg = json.load(json_file)
    dp_list = combination_for_list(density_cfg)
    do_list = combination_for_list(domain_cfg)
    ae_solver = combination_for_list(ae_cfg)
    pe_problem = [
        dict(ChainMap(*list(x))) for x in it.product(dp_list, do_list)
    ]
    combination_list = create_ae_pe_solution(ae_solver, pe_problem)
    with open(args.json_qpu) as json_file:
        noisy_cfg = json.load(json_file)
    qpu_list = combination_for_list(noisy_cfg)
    combination_list = create_ae_pe_solution(combination_list, qpu_list)

    if args.print:
        if args.id is not None:
            print(combination_list[args.id])
            print(args)
        elif args.all:
            print(combination_list)
        else:
            print("Provide -id or --all")
    if args.count:
        print("Number of elements: {}".format(len(combination_list)))


    if args.execution:
        if args.id is not None:
            configuration = combination_list[args.id]
            run_id(
                args.repetitions, args.id, args.save, #args.qpu,
                args.base_name, args.folder_path, args.target,
                **configuration)
