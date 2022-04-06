
import numpy as np
import sys
sys.path.append("../")

from QQuantLib.AE.real_quantum_ae import RQAE
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import bitfield_to_int
import QQuantLib.DL.data_loading as dl

import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from qat.core.console import display

n = 3
x = np.arange(2**n)
f = -x/np.max(x)


oracle = qlm.QRoutine()
register = oracle.new_wires(n+1)
oracle.apply(dl.uniform_distribution(n),register[:n])
oracle.apply(dl.load_array(f),register)

target = [0,0,1,1]
index = [0,1,2,3]

rqae = RQAE(oracle,target,index)
q = 2
epsilon = 0.01
gamma = 0.05
#RQAE.display_information(q = q,epsilon = epsilon, gamma = gamma)
amplitude = rqae.run(q = q,epsilon = epsilon, gamma = gamma)

assert (f[bitfield_to_int(target)]>np.sqrt(2)**n*amplitude[0]) and (f[bitfield_to_int(target)]<=np.sqrt(2)**n*amplitude[1])



