"""
Tests For maximum likelihood
"""
import sys
import numpy as np
import qat.lang.AQASM as qlm
sys.path.append("../")
from my_lib.utils import get_histogram
from my_lib.data_loading import load_probability, load_array, load_pf
from my_lib.maximum_likelihood_ae import MLAE 

from qat.qpus import PyLinalg
linalg_qpu = PyLinalg()

#Prepare Data for loading
def launch_data(n_qbits):
    def p(x):
        return x*x
    def f(x):
        return np.sin(x)
    #The number of bins
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
    arg_dictionary = {
        'oracle': pf_gate,
        'list_of_mks': 6,
        'qpu': linalg_qpu,
        'delta': 1e-3,
        'default_nbshots' : 100,
        'iterations' : 100,
        'display' :  False,
        'nbshots' : 0
    }
    ml_qae = MLAE(**arg_dictionary)
    ml_qae.run_mlae()

    calculated_integral = np.cos(ml_qae.theta)**2
    theoric_integral = np.sum(p_x*f_x)
    #print('calculated_integral: {}'.format(calculated_integral))
    #print('theoric_integral: {}'.format(theoric_integral))

    Delta = np.abs(calculated_integral-theoric_integral)
    assert Delta > 0.0001
