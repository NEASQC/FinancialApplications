"""
Create dictionaries for feeding the benchmark code
"""
import json
from collections import ChainMap
import itertools as it
import pandas as pd
from montecarlo_bc import MonteCarlo
from QQuantLib.utils.qlm_solver import get_qpu


def combination_for_dictionary(input_dict):
    """
    Creates a list of dictionaries with all the posible combination of
    the input dictionary.

    Parameters
    ----------
    input_dict : python dictionary
        python dictionary where each key value MUST be a list. For each
        value of a list a new dictioanry will be created

    Returns
    ----------
    list_of_dictionaries : list of python dictionaries
        A list with all posible combination of dictionaries from the
        input dictionary
    """

    list_of_dictionaries = [
        dict(zip(input_dict, x)) for x in it.product(*input_dict.values())
    ]
    return list_of_dictionaries


def combination_for_list(input_list):
    """
    For each dictionary of the list the function creates all posible
    combinations. All the posible combinations are concatenated.

    Parameters
    ----------
    input_list : list of python dictionary
        The values of each key of the each python dictionary MUST BE lists.

    Returns
    ----------
    list_of_combinations : list of python dictionaries
        A list with  the concatenation of all posible combinations for
        each dictionary of the input_list
    """
    list_of_combinations = []
    for step_dict in input_list:
        list_of_combinations = list_of_combinations + combination_for_dictionary(
            step_dict
        )
    return list_of_combinations


def create_pricep_list(probability_list, payoff_list, domain_list):
    """
    Creates a list of price problems (PriceP). A PriceP will be a
    python dictionary that allows to configure properly a price problem:
    this is a pay off under a density distribution for a especified domain.

    Parameters
    ----------
    probability_list: list
        list with different dictionaries for configure a density destribution.
    payoff_list: list
        list with different dictionaries for configure a pay off function.
    domain_list: list
        list with different dictionaries for configure a domain.

    Returns
    ----------
    pricep_list : list
        list with different PriceP python dictionaries.
    """
    # List of density probabilities dictionaries
    dp_list = combination_for_list(probability_list)
    # List for pay offs
    po_list = combination_for_list(payoff_list)
    # list of domain dictionaries
    do_list = combination_for_list(domain_list)
    # List of problems dictionaries
    # A problem will be defined as pay off under a densitiy probabiliy for
    # a especified domain
    pricep_list = [
        dict(ChainMap(*list(x))) for x in it.product(dp_list, do_list, po_list)
    ]
    return pricep_list


def create_pricep_list_fromjsons(
    json_density=None, json_payoffs=None, json_domain=None
):
    """
    Create a list of PriceP dictionaries using jsons.

    Parameters
    ----------
    json_density : string
        pathname of the json file with the probability density configuration.
    json_payoffs : string
        pathname of the json file with the payoff configuration.
    json_domain : string
        pathname of the json file with the domain configuration.

    Returns
    ----------
    pricep_list : list
        list with different PriceP python dictionaries.
    """

    if json_density is None:
        json_density = "jsons/density_probability.json"
    with open(json_density) as json_file:
        probability_dictionary = json.load(json_file)
    if json_payoffs is None:
        json_payoffs = "jsons/payoffs.json"
    with open(json_payoffs) as json_file:
        payoff_dictionary = json.load(json_file)
    if json_domain is None:
        json_domain = "jsons/domain_configuration.json"
    with open(json_domain) as json_file:
        domain_dictionary = json.load(json_file)
    pricep_list = create_pricep_list(
        probability_dictionary, payoff_dictionary, domain_dictionary
    )
    return pricep_list


def create_ae_pricep_list(ae_list, problem_list):
    """
    Creates a list of diferent Amplitude Estimation Price Problem
    (ae_pricep). An ae_price is a python dictionary with the complete
    configuration for solving a price problem of a pay-off, under a
    probability density for a given domain using a properly configured
    amplitude estimation method. An ae_price dictionary can be given
    as an input of the PriceEstimation class defined in the
    finance_benchmark.py

    Parameters
    ----------
    ae_list : list
        List with properly configured amplitude amplification methods
    problem_list : list
        List with different pricep problems (usually should be an output
        of the create_pricep_list or create_pricep_list_from_jsons functions.

    Returns
    ----------
    ae_pricep_list : list
        List where each element is a ae_pricep dictionary
        The list will have the combination of each posible amplitude
        estimation solver with all posible price problem list
    """
    ae_pricep_list = []
    for ae in ae_list:
        step_list = [dict(ChainMap(*list(x))) for x in it.product(problem_list, [ae])]
        ae_pricep_list = ae_pricep_list + step_list
    return ae_pricep_list


# def ae_combination(json_name):
#     with open(complete_path+json_name) as json_file:
#         ae_dictionary = json.load(json_file)
#     ae_list = combination_for_list(ae_dictionary)
#     return ae_list


def create_ae_pricep_list_fromjsons(ae_json_list, **kwargs):
    """
    Creates a ae_pricep dictionary list from jsons.

    Parameters
    ----------
    ae_list : list of json.
        List with name of json files with a complete configuration of an
        amplitude estimation method
    kwargs: other parameters.
        Valid parameters are the same that in the
        create_pricep_list_from_jsons function

    Returns
    ----------
    ae_pricep_list : list of python dictionaries
        Each dictionary is a complete price amplitude estimation problem.
        The list will have the combination of each posible amplitude
        estimation solver with all posible price problem list
    """
    problem_list = create_pricep_list_fromjsons(**kwargs)
    ae_list = []
    for ae_json in ae_json_list:
        with open(ae_json) as json_file:
            ae_list = ae_list + json.load(json_file)
    ae_list = combination_for_list(ae_list)
    ae_pricep_list = create_ae_pricep_list(ae_list, problem_list)
    return ae_pricep_list


def create_ae_pricep_list_fromjson(json_file):
    with open(json_file) as json_pointer:
        ae_list = json.load(json_pointer)
    ae_pricep_list = combination_for_list(ae_list)
    return ae_pricep_list


def run_staff(
    ae_pricep_list, file_name="Todo.csv", folder_name=None, qlmaas=False, save=False
):
    """
    This function executes sequentially a list of different AE_PriceP
    dictionaries.

    Parameters
    ----------

    ae_pricep_list :  python dictionary
        list with AE_PriceP python dictionaries. The dictionaries haves
        to contain all the complete configuration for given to the
        PriceEstimation class.
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
        DataFrame with the complete information and solution of each
        of the AE_PriceP inputs problems.
    """
    list_of_pdfs = []
    for i, step in enumerate(ae_pricep_list):
        step_clas = run_id(step, i, file_name, folder_name, qlmaas, save)
        list_of_pdfs.append(step_clas.pdf)
    price_pdf = pd.concat(list_of_pdfs)
    # if save:
    #     file_name = "Todo.csv"
    #     file_name = folder_name + file_name
    #     price_pdf.to_csv(file_name)
    return price_pdf


def run_id(
    ae_pricep, id_name, file_name=None, folder_name=None, qlmaas=False, save=False
):
    """
    This function solves a complete AE_PriceP input dictionary. This
    function calculates the price for a given payoff, under a given
    probability density in a given domain interval, using the provided
    configured amplitude estimation method.

    Parameters
    ----------

    ae_pricep :  python dictionary
        AE_PriceP python dictionary. The dictionary has to contain all
        the complete configuration for given to the PriceEstimation class.
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

    pe_object : PriceEstimation object
        PriceEstimation object configured with the input ae_pricep
        dictionary. The run method of the object is executed.

    """
    linalg_qpu = get_qpu(qlmaas)
    ae_pricep.update({"qpu": None})
    ae_pricep.update({"save": save})
    if save:
        if folder_name is None:
            raise ValueError("folder_name is None!")
        if file_name is None:
            file_name = ae_pricep["ae_type"] + "_{}.csv".format(id_name)
        file_name = folder_name + file_name
    ae_pricep.update({"file_name": file_name})
    print(file_name)
    print(ae_pricep)
    pe_object = MonteCarlo(**ae_pricep)
    pe_object.run()
    # if save:
    #     file_name = ae_pricep["ae_type"]
    #     file_name = file_name + "_{}.csv".format(id_name)
    #     file_name = folder_name + file_name
    #     pe_object.pdf.to_csv(file_name)
    return pe_object


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
        "-json",
        dest="json",
        type=str,
        help="json file with complete ae configuration.",
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
        "--all",
        dest="all",
        default=False,
        action="store_true",
        help="For executing complete list",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving results",
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
    args = parser.parse_args()
    print(args)

    if args.json is not None:
        final_list = create_ae_pricep_list_fromjson(args.json)

    else:
        final_list = create_ae_pricep_list_fromjsons(["jsons/mc_configuration.json"])

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
                save=args.save,
            )
        else:
            if args.id is not None:
                run_id(
                    final_list[args.id],
                    args.id,
                    file_name=None,
                    folder_name=args.folder_path,
                    save=args.save,
                )
