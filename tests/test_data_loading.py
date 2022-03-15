"""
Tests For data_loading functions
"""
import sys
import numpy as np
import qat.lang.AQASM as qlm
sys.path.append("../../")

from my_lib.data_loading import load_probability, uniform_distribution,\
load_array
from my_lib.data_extracting import get_results


from qat.qpus import PyLinalg
linalg_qpu = PyLinalg()

def test_load_probability():
    """
    Testing probability loading
    """

    n_qbits = 3
    array_lenght = 2**n_qbits
    x = np.arange(array_lenght)
    probability = x/np.sum(x)
    q_rout = load_probability(probability)
    results, _, _, _ = get_results(q_rout, linalg_qpu=linalg_qpu)
    test_result = np.isclose(
        results.sort_values("Int_lsb")["Probability"].values, 
        probability
    )

    assert test_result.all() == True

def test_load_array():
    """
    Testing array loading
    """
    n_qbits = 3
    array_lenght = 2**n_qbits
    x = np.arange(array_lenght)
    normalization_constant = np.max(x)
    f = x
    f_normalised = x/normalization_constant
    q_rout = qlm.QRoutine()
    register = q_rout.new_wires(n_qbits+1)
    q_rout.apply(uniform_distribution(n_qbits), register[:n_qbits])
    q_rout.apply(load_array(f_normalised), register)
    results, _, _, _ = get_results(q_rout, linalg_qpu=linalg_qpu)
    quantum_probabilities = results.sort_values("Int_lsb")["Probability"].values
    quantum_f = np.sqrt(quantum_probabilities)*np.sqrt(array_lenght)\
    *normalization_constant
    test_result = np.isclose(quantum_f[:array_lenght], f)
    assert test_result.all() == True

def test_complete_load():
    """
    Loading a function upon a non trivial probability distribution
    """

    n_qbits = 3
    array_lenght = 2**n_qbits
    x = np.arange(array_lenght)
    probability = x/np.sum(x)
    normalization_constant = np.max(x)
    f = x
    f_normalised = x/normalization_constant

    q_rout = qlm.QRoutine()
    register = q_rout.new_wires(n_qbits+1)
    q_rout.apply(load_probability(probability), register[:n_qbits])
    f_root = np.sqrt(f_normalised)
    q_rout.apply(load_array(f_root), register)
    results, _, _, _ = get_results(q_rout, linalg_qpu=linalg_qpu)
    quantum_probabilities = results.sort_values("Int_lsb")["Probability"].values
    quantum_result = quantum_probabilities*normalization_constant
    test_result = np.isclose(
        quantum_result[0:array_lenght],
        probability*f
    )
    assert test_result.all() == True





