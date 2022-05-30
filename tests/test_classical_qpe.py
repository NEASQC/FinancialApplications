"""
test for classical QPE
"""
import numpy as np
import qat.lang.AQASM as qlm

from QQuantLib.utils.utils import get_histogram
from QQuantLib.DL.data_loading import load_probability, load_array, load_pf
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.PE.classical_qpe import cQPE 


#### Phase Estimation Test-01: Phase of S Gate ###

def test_pe_s_gate():

    n_qbits = 1
    initial_state = qlm.QRoutine()
    q_bits = initial_state.new_wires(n_qbits)
    for i in range(n_qbits):
        initial_state.apply(qlm.X, q_bits[i])
    unitary_operator = qlm.PH(np.pi/2.0)
    auxiliar_qbits_number = 2
    #We create a python dictionary for configuration of class
    qft_pe_dict = {
        'initial_state': initial_state,
        'unitary_operator': unitary_operator,
        'auxiliar_qbits_number' : auxiliar_qbits_number,
        'shots': 100
    }
    qft_pe = cQPE(**qft_pe_dict)
    qft_pe.pe_qft()
    phi_meas = qft_pe.final_results.iloc[
        qft_pe.final_results['Probability'].idxmax()
    ]['Phi']
    assert np.isclose(phi_meas, 0.25)


#### Phase Estimation Test-02: Phase of Controlled-T Gate ###

def test_pe_c_t_gate():

    n_qbits = 3
    initial_state = qlm.QRoutine()
    q_bits = initial_state.new_wires(n_qbits)
    for i in range(n_qbits):
        initial_state.apply(qlm.X, q_bits[i])
    #Create cT operator
    unitary_operator = qlm.QRoutine()
    uq_qbits = unitary_operator.new_wires(n_qbits)
    unitary_operator.apply(qlm.PH(np.pi/4.0).ctrl(), 0, 1)
    auxiliar_qbits_number = 3
    #We create a python dictionary for configuration of class
    qft_pe_dict = {
        'initial_state': initial_state,
        'unitary_operator': unitary_operator,
        'auxiliar_qbits_number' : auxiliar_qbits_number,
        'shots': 100
    }
    qft_pe = cQPE(**qft_pe_dict)
    qft_pe.pe_qft()
    phi_meas = qft_pe.final_results.iloc[
        qft_pe.final_results['Probability'].idxmax()
    ]['Phi']
    assert np.isclose(phi_meas, 0.125)

###Phase Estimation with classical QPE###
from QQuantLib.AE.ae_classical_qpe import cQPE_AE

def test_ae_w_qpe_qft():

    #Here we created the mandatory oracle
    n = 3
    N = 2**n
    x = np.arange(N)
    probability = x/np.sum(x)
    oracle = load_probability(probability)


    #This will be the target state for grover and the list of qbits affected by Grover operator
    target = [0, 0, 1]
    index = range(oracle.arity)

    ae_pe_qft_dict = {
        'auxiliar_qbits_number': 4,
        'shots': 10
    }

    ae_pe_qft = cQPE_AE(
        oracle=oracle,
        target=target,
        index=index,
        **ae_pe_qft_dict
    )

    a_estimated = ae_pe_qft.run()

    classical_result = probability[1]
    error = abs(ae_pe_qft.a-classical_result)
    assert error < 0.005





