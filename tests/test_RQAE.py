
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

def test_rqae():
    n = 3
    x = np.arange(2**n)
    f = -x/np.max(x)
    
    
    oracle = qlm.QRoutine()
    register = oracle.new_wires(n+1)
    oracle.apply(dl.uniform_distribution(n),register[:n])
    oracle.apply(dl.load_array(f),register)
    
    target = [0,0,1,1]
    index = [0,1,2,3]
    
    q = 2
    epsilon = 0.01
    gamma = 0.05
    rqae_dict = {
        'epsilon': epsilon,
        'q': q,
        'gamma': gamma
    }

    rqae = RQAE(oracle,target,index, **rqae_dict)
    a_real = f[bitfield_to_int(target)]

    a_estimated = rqae.run()
    a_low = rqae.a_l*np.sqrt(2)**n
    a_up = rqae.a_u*np.sqrt(2)**n

    
    assert (a_real > a_low) and (a_real <= a_up)

test_rqae()
