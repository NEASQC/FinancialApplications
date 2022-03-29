"""
Tests For maximum likelihood
"""
import numpy as np
import qat.lang.AQASM as qlm

from QQuantLib.utils.utils import get_histogram
from QQuantLib.DL.data_loading import load_probability, load_array, load_pf
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.PE.iterative_quantum_pe import IterativeQuantumPE


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

def __test_iterative_pe():
    """
    long time test
    """

    x, f_x, p_x = launch_data(3)
    n_cbits = 7
    p_gate, f_gate, pf_gate = load_gates(p_x, f_x)
    q_gate = grover(pf_gate, [0], [pf_gate.arity-1])
    #We can do several circuit executions configuring input dictionary properly
    iqpe_dict = {
        'initial_state': pf_gate,
        'unitary_operator':q_gate, 
        'cbits_number' : n_cbits,
        'shots': 100
    }
    iqpe_ = IterativeQuantumPE(**iqpe_dict)
    iqpe_.iqpe()
    calculated_integral = np.cos(iqpe_.sumary.iloc[0]['theta_90'])**2
    print('calculated_integral: {}'.format(calculated_integral))
    theoric_integral = np.sum(p_x*f_x)
    print('theoric_integral: {}'.format(theoric_integral))
    delta = np.abs(calculated_integral-theoric_integral)
    print(delta)
    assert delta < 0.001

def test_iterative_pe():
    #Number Of Qbits
    n_qbits = 1
    #Number Of Classical Bits
    n_cbits = 2
    #Basic Initial circuit and unitary operator whose autovalue we want to compute
    initial_state = qlm.QRoutine()
    q_bits = initial_state.new_wires(n_qbits)
    for i in range(n_qbits):
        initial_state.apply(qlm.X, q_bits[i])
    unitary_operator = qlm.PH(np.pi/2.0)

    iqpe_dict = {
        'initial_state': initial_state,
        'unitary_operator':unitary_operator, 
        'cbits_number' : n_cbits,
        'shots': 100
    }
    iqpe = IterativeQuantumPE(**iqpe_dict)
    iqpe.iqpe()
    pdf = iqpe.sumarize(iqpe.final_results, ['Phi'])
    print(np.isclose(pdf.iloc[0]['Phi'], 0.25))
    assert np.isclose(pdf.iloc[0]['Phi'], 0.25)
