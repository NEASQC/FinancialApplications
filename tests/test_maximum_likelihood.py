"""
Tests For maximum likelihood
"""
import numpy as np
import qat.lang.AQASM as qlm

from QQuantLib.utils.utils import get_histogram
from QQuantLib.DL.data_loading import load_probability, load_array, load_pf
from QQuantLib.AE.maximum_likelihood_ae import MLAE


# Prepare Data for loading
def launch_data(n_qbits):
    def p(x):
        return x * x

    def f(x):
        return np.sin(x)

    # The number of bins
    m_bins = 2**n_qbits
    lower_limit = 0.0
    upper_limit = 1.0
    x, p_x = get_histogram(p, lower_limit, upper_limit, m_bins)
    f_x = f(x)
    return x, f_x, p_x


def load_gates(p_x, f_x):
    p_gate = load_probability(p_x)
    f_gate = load_array(np.sqrt(f_x))
    pf_gate = load_pf(p_gate, f_gate)
    return p_gate, f_gate, pf_gate


def test_maximum_likelihood():

    x, f_x, p_x = launch_data(5)
    p_gate, f_gate, pf_gate = load_gates(p_x, f_x)

    m_k = list(range(7))
    n_k = [200] * len(m_k)
    schedule = [m_k, n_k]

    mlae_ = MLAE(pf_gate, target=[0], index=[pf_gate.arity - 1], schedule=schedule)
    estimated_a = mlae_.run()
    calculated_integral = mlae_.ae
    theoric_integral = np.sum(f_x * p_x)
    print(calculated_integral)
    print(theoric_integral)

    Delta = np.abs(calculated_integral - theoric_integral)
    print("##########")
    print(Delta)
    print("##########")
    assert Delta < 0.01
