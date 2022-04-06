
import numpy as np
import sys
sys.path.append("../")

from QQuantLib.AE.iterative_quantum_ae import IQAE
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import bitfield_to_int
import QQuantLib.DL.data_loading as dl

import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from qat.core.console import display

n = 3
x = np.arange(2**n)
p = x/np.sum(x)


oracle = qlm.QRoutine()
register = oracle.new_wires(n)
oracle.apply(dl.load_probability(p),register)

target = [0,0,1]
index = [0,1,2]

iqae = IQAE(oracle,target,index)


epsilon = 0.01
alpha = 0.05
N_shots = 100
probability = iqae.run(epsilon = epsilon,alpha = alpha, N = N_shots)
assert (p[bitfield_to_int(target)]>probability[0]) and (p[bitfield_to_int(target)]<=probability[1])

