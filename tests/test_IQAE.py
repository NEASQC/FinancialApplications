import sys
sys.path.append("../")
import numpy as np
import qat.lang.AQASM as qlm
from qat.core.console import display
from QQuantLib.qpu.get_qpu import get_qpu
from QQuantLib.AE.iterative_quantum_ae import IQAE
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import bitfield_to_int
import QQuantLib.DL.data_loading as dl


def test_iqae():
    n = 3
    x = np.arange(2**n)
    p = x / np.sum(x)

    oracle = qlm.QRoutine()
    register = oracle.new_wires(n)
    oracle.apply(dl.load_probability(p), register)

    target = [0, 0, 1]
    index = [0, 1, 2]

    epsilon = 0.01
    alpha = 0.05
    N_shots = 100
    linalg_qpu = get_qpu("python")
    iqae_dict = {
        "qpu": linalg_qpu,
        "epsilon": epsilon,
        "shots": N_shots,
        "alpha": alpha,
    }

    iqae = IQAE(oracle, target, index, **iqae_dict)

    a_estimated = iqae.run()

    assert (p[bitfield_to_int(target)] > iqae.ae_l) and (
        p[bitfield_to_int(target)] <= iqae.ae_u
    )
