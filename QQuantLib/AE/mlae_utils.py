"""
This module contains mandatory functions used by the MLAE class

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import numpy as np


def likelihood(theta: float, m_k: int, n_k: int, h_k: int) -> float:
    r"""
    Calculates Likelihood from Suzuki paper. For h_k positive events
    of n_k total events, this function calculates the probability of
    this taking into account that the probability of a positive
    event is given by theta and by m_k
    The idea is use this function to minimize it for this reason it gives
    minus Likelihood

    Notes
    -----
    .. math::
        l_k(\theta|h_k) = \sin^2\left((2m_k+1)\theta\right)^{h_k} \
        \cos^2 \left((2m_k+1)\theta\right)^{n_k-h_k}

    Parameters
    ----------

    theta : float
        Angle (radians) for calculating the probability of measure a
        positive event.
    m_k : int
        number of times the grover operator was applied.
    n_k : int
        number of total events measured for the specific  m_k
    h_k : int
        number of positive events measured for each m_k

    Returns
    ----------

    float
        Gives the Likelihood p(h_k with m_k amplifications|theta)

    """
    theta_ = (2 * m_k + 1) * theta
    p_0 = np.sin(theta_) ** 2
    p_1 = np.cos(theta_) ** 2
    l_k = (p_0**h_k) * (p_1 ** (n_k - h_k))
    return l_k

def log_likelihood(theta: float, m_k: int, n_k: int, h_k: int) -> float:
    r"""
    Calculates log of the likelihood from Suzuki paper.

    Notes
    -----
    .. math::
        \log{l_k(\theta|h_k)} = 2h_k\log\big[\sin\left((2m_k+1) \
        \theta\right)\big] +2(n_k-h_k)\log\big[\cos\left((2m_k+1) \
        \theta\right)\big]

    Parameters
    ----------

    theta : float
        Angle (radians) for calculating the probability of measure a
        positive event.
    m_k : int
        number of times the grover operator was applied.
    n_k : int
        number of total events measured for the specific  m_k
    h_k : int
        number of positive events measured for each m_k

    Returns
    ----------

    float
        Gives the log Likelihood p(h_k with m_k amplifications|theta)

    """
    theta_ = (2 * m_k + 1) * theta
    p_0 = np.sin(theta_) ** 2
    p_1 = np.cos(theta_) ** 2
    l_k = h_k * np.log(p_0) + (n_k - h_k) * np.log(p_1)
    return l_k


def cost_function(angle: float, m_k: list, n_k: list, h_k: list) -> float:
    r"""
    This method calculates the -Likelihood of angle theta
    for a given schedule m_k,n_k

    Notes
    -----
    .. math::
        L(\theta,\mathbf{h}) = -\sum_{k = 0}^M\log{l_k(\theta|h_k)}

    Parameters
    ----------

    angle: float
        Angle (radians) for calculating the probability of measure a
        positive event.
    m_k : list of ints
        number of times the grover operator was applied.
    n_k : list of ints
        number of total events measured for the specific  m_k
    h_k : list of ints
        number of positive events measured for each m_k

    Returns
    ----------

    cost : float
        the aggregation of the individual likelihoods
    """
    log_cost = 0
    # for i in range(len(m_k)):
    for i, _ in enumerate(m_k):
        log_l_k = log_likelihood(angle, m_k[i], n_k[i], h_k[i])
        log_cost = log_cost + log_l_k
    return -log_cost
