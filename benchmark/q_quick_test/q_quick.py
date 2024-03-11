import sys
sys.path.append("../../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qat.lang.AQASM as qlm
#This cell loads the QLM solver. QPU = [qlmass, python, c]
from QQuantLib.utils.qlm_solver import get_qpu

from QQuantLib.finance.probability_class import DensityProbability
import QQuantLib.DL.data_loading as dl
from QQuantLib.utils.utils import bitfield
from QQuantLib.AE.ae_class import AE

from QQuantLib.utils.benchmark_utils import list_of_dicts_from_jsons
from get_qpu import get_qpu


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

def test(**kwargs):

    # Configure domain
    n_qbits = kwargs.get("n_qbits", None)
    x0 = kwargs.get("x0", 1.0)
    xf = kwargs.get("xf", 3.0)
    # Configure probability
    pc = DensityProbability(**kwargs)
    # Creating domain
    domain = np.linspace(x0, xf, 2**n_qbits)
    # Creating probability distribution
    p_x = pc.probability(domain)
    # Creating oracle
    oracle = dl.load_probability(p_x)
    # Getting the target id and the target state
    target_id = np.argmax(p_x)
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
    if ae_type in ["RQAE", "mRQAE", "eRQAE"]:
        pdf["Value"] = np.sqrt(np.max(p_x))
    else:
        pdf["Value"] = np.max(p_x)
    pdf = pd.concat([pdf, ae_obj.ae_pdf], axis=1)
    pdf["oracle_calls"] = ae_obj.oracle_calls
    pdf["schedule_pdf"] = [ae_obj.schedule_pdf.to_dict()]
    pdf["measured_epsilon"] = (pdf["ae_u"] - pdf["ae_l"]) / 2.0
    pdf["absolute_error"] = np.abs(pdf["ae"] - pdf["Value"])
    return pdf


def run_id(repetitions, id_, save_, qpu, base_name, save_folder, **ae_configuration):
    #Domain configuration

    domain_configuration = {
        'x0': 0.01,
        'xf': 6.0,
        'n_qbits': 6,
    }

    #Probability density configuration
    probability_configuration = {
        'probability_type': 'Black-Scholes',
        's_0': 2,
        'risk_free_rate': 0.05,
        'maturity': 1.0,
        'volatility': 0.2,
    }
    ae_configuration.update(domain_configuration)
    ae_configuration.update(probability_configuration)
    ae_configuration.update({"qpu": get_qpu(qpu)})
    
    save_name = save_folder + str(id_) + "_" + ae_configuration["file"] + "_" + str(base_name) +  ".csv"
    print(save_name)
    
    for i in range(repetitions):
        step_pdf = test(**ae_configuration)
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



#
    
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
    parser.add_argument(
        "-qpu",
        dest="qpu",
        type=str,
        default="python",
        help="QPU for simulation: See function get_qpu in get_qpu module",
    )
    parser.add_argument(
        "-json",
        dest="json",
        type=str,
        default="./quick.json",
        help="Json to use.",
    )
    parser.add_argument(
        "-folder",
        dest="folder_path",
        type=str,
        help="Path for storing folder",
        default="./",
    )
    args = parser.parse_args()

    combination_list = list_of_dicts_from_jsons([args.json])

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
                args.repetitions, args.id, args.save, args.qpu,
                args.base_name, args.folder_path, **configuration)






