"""
Utils functions from benchmark purpouses.

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro
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

def create_pe_problem(domain_cfg, payoff_cfg, density_cfg):
    """
    Create a list of price estimation problems. Each element is a python
    dictionary with a complete option price estimation problem.

    Parameters
    ----------
    domain_cfg : list of dictionaries
        Each dictionary has a domain configuration for a price estimation problem.
    payoffs_cfg : list of dictionaries
        Each dictionary has an option configuration for a price estimation problem.
    density_cfg : list of dictionaries
        Each dictionary has probability density configuration for a price estimation problem.

    Returns
    ----------
    pe_problem_list : list of dictionaries
        list with different price estimation problems.
    """

    # List of density probabilities dictionaries
    dp_list = combination_for_list(density_cfg)
    # List for pay offs
    po_list = combination_for_list(payoff_cfg)
    # list of domain dictionaries
    do_list = combination_for_list(domain_cfg)
    pe_problem_list = [
        dict(ChainMap(*list(x))) for x in it.product(dp_list, do_list, po_list)
    ]
    return pe_problem_list

def create_ae_pe_solution(ae_list, problem_list):
    """
    Creates a list of price estimation problems for solving with amplitude
    estimation (AE) techniques. Each element will have the complete
    information for generating a price estimation problem and the
    configuration for solving it using an AE algorithm. This is each element
    is a python dictionary that allows define a price estimation problem
    and solving it using a properly configure AE algorithm

    Parameters
    ----------
    ae_list : list
        List with properly configured AE solvers.
    problem_list : list
        List with different price estimation problems.

    Returns
    ----------
    solve_ae_pe_list : list
        List where each element is a ae_pricep dictionary
        The list will have the combination of each posible amplitude
        estimation solver with all posible price problem list
    """
    solve_ae_pe_list = []
    for ae in ae_list:
        step_list = [dict(ChainMap(*list(x))) for x in it.product(problem_list, [ae])]
        solve_ae_pe_list = solve_ae_pe_list + step_list
    return solve_ae_pe_list

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
