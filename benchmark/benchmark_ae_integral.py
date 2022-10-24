"""
This module can be used for launchin benchamark using ae_sine_integral.py

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""
import sys
import json
import pandas as pd
sys.path.append("../")
from benchmark.benchmark_utils import combination_for_list
from benchmark.ae_sine_integral import sine_integral
from QQuantLib.utils.qlm_solver import get_qpu


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
    ae_problem.update({"qpu": linalg_qpu})
    ae_problem.update({"save": save})
    integral = ae_problem['integral']
    ae_problem.pop('integral')
    if save:
        if folder_name is None:
            raise ValueError("folder_name is None!")
        if file_name is None:
            file_name = ae_problem["ae_type"] + "_{}.csv".format(id_name)
        metrics_file_name = folder_name + 'metrics_n_qbits_' + str(n_qbits) + '_' + file_name
        file_name = folder_name + 'n_qbits_' + str(n_qbits) + '_' + file_name
    ListOfMetrics = []
    for i in range(ae_problem["number_of_tests"]):
        ae_problem.update({"file_name": file_name})
        metrics, pdf = sine_integral(n_qbits, integral, ae_problem)
        print(pdf)
        ListOfMetrics.append(pdf)
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
    metrics_pdf = pd.concat(ListOfMetrics)
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
