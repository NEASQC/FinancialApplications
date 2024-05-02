"""
test for classical QPE
"""
import numpy as np
import qat.lang.AQASM as qlm
import sys
sys.path.append("../")

from QQuantLib.utils.utils import get_histogram
from QQuantLib.DL.data_loading import load_probability, load_array, load_pf
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.PE.iterative_quantum_pe import IQPE
from QQuantLib.AE.ae_iterative_quantum_pe import IQPEAE
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.qpu.get_qpu import get_qpu


#### Phase Estimation Test-01: Phase of S Gate ###


def test_pe_s_gate():

    n_qbits = 1
    initial_state = qlm.QRoutine()
    q_bits = initial_state.new_wires(n_qbits)
    for i in range(n_qbits):
        initial_state.apply(qlm.X, q_bits[i])
    unitary_operator = qlm.PH(np.pi / 2.0)
    auxiliar_qbits_number = 2
    # We create a python dictionary for configuration of class
    linalg_qpu = get_qpu("python")
    iqpe_dict = {
        "qpu": linalg_qpu,
        "initial_state": initial_state,
        "unitary_operator": unitary_operator,
        "auxiliar_qbits_number": auxiliar_qbits_number,
        "shots": 100,
    }
    iqpe = IQPE(**iqpe_dict)
    iqpe.iqpe()
    phi_meas = iqpe.final_results.iloc[iqpe.final_results["Frequency"].idxmax()]["Phi"]
    assert np.isclose(phi_meas, 0.25)
#
#
##### Phase Estimation Test-02: Phase of Controlled-T Gate ###
#
#
def test_pe_c_t_gate():

    n_qbits = 3
    initial_state = qlm.QRoutine()
    q_bits = initial_state.new_wires(n_qbits)
    for i in range(n_qbits):
        initial_state.apply(qlm.X, q_bits[i])
    # Create cT operator
    unitary_operator = qlm.QRoutine()
    uq_qbits = unitary_operator.new_wires(n_qbits)
    unitary_operator.apply(qlm.PH(np.pi / 4.0).ctrl(), 0, 1)
    auxiliar_qbits_number = 3
    # We create a python dictionary for configuration of class
    linalg_qpu = get_qpu("python")
    iqpe_dict = {
        "qpu": linalg_qpu,
        "initial_state": initial_state,
        "unitary_operator": unitary_operator,
        "auxiliar_qbits_number": auxiliar_qbits_number,
        "shots": 100,
    }
    iqpe = IQPE(**iqpe_dict)
    iqpe.iqpe()
    phi_meas = iqpe.final_results.iloc[iqpe.final_results["Frequency"].idxmax()]["Phi"]
    assert np.isclose(phi_meas, 0.125)


###Phase Estimation with IQPE###


def test_ae_w_iqpe():

    # Here we created the mandatory oracle
    n = 3
    N = 2**n
    x = np.arange(N)
    probability = x / np.sum(x)
    oracle = load_probability(probability)

    # This will be the target state for grover and the list of qbits affected by Grover operator
    target = [0, 0, 1]
    index = range(oracle.arity)

    linalg_qpu = get_qpu("python")
    ae_iqpe_dict = {
        "qpu": linalg_qpu,
        "auxiliar_qbits_number": 4,
        "shots": 10,
    }

    ae_iqpe = IQPEAE(oracle=oracle, target=target, index=index, **ae_iqpe_dict)

    a_estimated = ae_iqpe.run()

    classical_result = probability[1]
    error = abs(ae_iqpe.ae - classical_result)
    assert error < 0.005
