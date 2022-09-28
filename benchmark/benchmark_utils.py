"""
Uitls functions fro benchmark purpouses.
"""
import json
import itertools as it
from collections import ChainMap

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

def list_of_dicts_from_jsons(ae_json_list):
    """
    Creates a list of dictionaries from inputs jsons.

    Parameters
    ----------
    ae_list : list of json.
        List with name of json files with a complete configuration of an
        amplitude estimation method

    Returns
    ----------
    ae_pricep_list : list of python dictionaries
    """
    ae_list = []
    for ae_json in ae_json_list:
        with open(ae_json) as json_file:
            ae_list = ae_list + json.load(json_file)
    ae_list = combination_for_list(ae_list)
    return ae_list
    #ae_pricep_list = create_ae_pricep_list(ae_list, problem_list)
    #return ae_pricep_list
