"""
Auxiliary functions for using with the demo notebook: NoisyModels.ipynb
"""
import sys
import numpy as np
sys.path.append("../../")
from QQuantLib.DL.encoding_protocols import Encoding
from QQuantLib.finance.probability_class import DensityProbability
from QQuantLib.finance.payoff_class import PayOff



def create_arrays(price_problem):
    """
    This function creates the mandatory arrays for configuring an option
    price estimation problem for notebook NoisyModels.ipynb

    Parameters
    ----------

    price_problem : dict
        Python dictionary with a complete dictionary for configuring
        the arrays for a option price estimation problem

    Returns
    -------

    domain : numpy array
        numpy array with the domain for the price estimation problem
    norm_pay_off : array
        numpy array with the normalised payoff
    norm_p_x : numpy array
        numpy array with the normalised probability density
    pay_off_normalisation : float
        normalization constant for the payoff
    p_x_normalisation : float
        normalization constant for the probability density

    """
    n_qbits = price_problem.get("n_qbits", None)
    x0 = price_problem.get("x0", 1.0)
    xf = price_problem.get("xf", 3.0)
    domain = np.linspace(x0, xf, 2**n_qbits)
    #Building the Probability distribution
    pc = DensityProbability(**price_problem)
    p_x = pc.probability(domain)
    #Normalisation of the probability distribution
    p_x_normalisation = np.sum(p_x) + 1e-8
    norm_p_x = p_x / p_x_normalisation
    #Building the option payoff
    po = PayOff(**price_problem)
    pay_off = po.pay_off(domain)
    #Normalisation of the pay off
    pay_off_normalisation = np.max(np.abs(pay_off)) + 1e-8
    norm_pay_off = pay_off / pay_off_normalisation
    return domain, norm_pay_off, norm_p_x, pay_off_normalisation, p_x_normalisation


def first_step(epsilon, ratio, gamma):
    """
    Configuration of the first step for a RQAE algorithm. This is an
    auxiliary function for notebook NoisyModels.ipynb

    Parameters
    ----------

    epsilon : float
        epsilon for RQAE
    ratio : float
        ratio (q) for RQAE
    gamma : float
        gamma for RQAE

    Returns
    -------

    shift : float
        shift for first step of RQAE
    n_i : int
        number of shots for first step of RQAE
    gamma_i : float
        failure probability for first step of RQAE
    theoretical_epsilon : float
        theoretical epsilon for first step of RQAE
    """
    epsilon = 0.5 * epsilon
    theoretical_epsilon = 0.5 * np.sin(np.pi / (2 * (ratio + 2))) ** 2
    k_max = int(
        np.ceil(
            np.arcsin(np.sqrt(2 * theoretical_epsilon))
            / np.arcsin(2 * epsilon)
            * 0.5
            - 0.5
        )
    )
    bigk_max = 2 * k_max + 1
    big_t = np.log(
        ratio
        * ratio
        * (np.arcsin(np.sqrt(2 * theoretical_epsilon)))
        / (np.arcsin(2 * epsilon))
    ) / np.log(ratio)
    gamma_i = gamma / big_t
    n_i = int(
        np.ceil(1 / (2 * theoretical_epsilon**2) * np.log(2 * big_t /  gamma))
    )
    epsilon_probability = np.sqrt(1 / (2 * n_i) * np.log(2 / gamma_i))
    shift = theoretical_epsilon / np.sin(np.pi / (2 * (ratio + 2)))
    return shift, n_i, gamma_i, theoretical_epsilon
