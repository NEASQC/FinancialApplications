"""
Tests For data_loading functions
"""
import sys
import numpy as np
import qat.lang.AQASM as qlm
sys.path.append("../")
from my_lib.utils import get_histogram
from my_lib.data_loading import load_probability, load_array, load_pf
from my_lib.data_extracting import get_results
from my_lib.amplitude_amplification import uphi0_gate, d0_gate, load_uphi_gate,\
load_q_gate

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

def load_gates(n_qbits):
    x, f_x, p_x = launch_data(n_qbits)
    p_gate = load_probability(p_x)
    f_gate = load_array(np.sqrt(f_x))
    pf_gate = load_pf(p_gate, f_gate)
    return p_gate, f_gate, pf_gate

def GetAngle(Array):
    Modulo = np.linalg.norm(Array)
    cosTheta = Array[0]/Modulo
    Theta0 = np.arccos(cosTheta)
    sinTheta = Array[1]/Modulo
    Theta1 = np.arcsin(sinTheta)
    #print(Modulo, cosTheta, sinTheta, Theta0, Theta1)
    return Theta0

def get_initial_state(pf_gate):
    phi_state, circuit, q_prog, job = get_results(
        pf_gate,
        linalg_qpu=linalg_qpu,
        shots=0
    )
    initial_state, circuit, q_p, job = get_results(
        pf_gate,
        linalg_qpu=linalg_qpu,
        shots=0,
        qubits=[pf_gate.arity-1]
    )
    return phi_state, initial_state, q_prog

def test_Uphi0():
    """
    For Testing uphi0_gate from amplitude_amplification
    """
    p_gate, f_gate, pf_gate = load_gates(5)
    phi_state, initial_state, q_prog = get_initial_state(pf_gate)
    u_phi0_gate = uphi0_gate(pf_gate.arity)
    registers = q_prog.registers
    q_prog.apply(u_phi0_gate, registers)
    u_phi0_state, circuit, _, _ = get_results(
        q_prog,
        linalg_qpu=linalg_qpu,
        shots=0
    )
    
    phi0_state_0 = np.array(
        [p for s, p in zip(phi_state['States'], phi_state['Amplitude']) \
        if s.bitstring[-1] == '0']
    )
    u_phi0_state_0 = np.array(
        [p for s, p in zip(u_phi0_state['States'], u_phi0_state['Amplitude']) \
        if s.bitstring[-1] == '0']
    )

    #Testing Final qbit |0> should be of different sign
    last_qbit_0 = np.isclose(phi0_state_0, -u_phi0_state_0).all()

    assert last_qbit_0 == True

    phi0_state_1 = np.array(
        [p for s, p in zip(phi_state['States'], phi_state['Amplitude']) \
        if s.bitstring[-1] == '1']
    )
    u_phi0_state_1 = np.array(
        [p for s, p in zip(u_phi0_state['States'], u_phi0_state['Amplitude']) \
        if s.bitstring[-1] == '1']
    )
    #Testing Final qbit |1> should be of same sign
    last_qbit_1 = np.isclose(phi0_state_1, u_phi0_state_1).all()
    assert last_qbit_1 == True

    assert (last_qbit_0 and last_qbit_1) == True

def test_D0():
    """
    For Testing d0_gate from amplitude_amplification
    """
    p_gate, f_gate, pf_gate = load_gates(5)
    phi_state, initial_state, q_prog = get_initial_state(pf_gate)
    registers = q_prog.registers
    q_prog.apply(d0_gate(pf_gate.arity), registers)
    u_d0_state, circuit, _, _ = get_results(
        q_prog,
        linalg_qpu=linalg_qpu,
        shots=0
    )

    #Testing: state |0> should change sign
    state_0 = np.isclose(
        phi_state['Amplitude'].loc[0],
        -u_d0_state['Amplitude'].loc[0]
    )
    #Testing: otherwise states keep sign
    non_state_0 = np.isclose(
        phi_state['Amplitude'].loc[1:],
        u_d0_state['Amplitude'].loc[1:]
    ).all()
    assert state_0 == True
    assert non_state_0 == True
    assert (state_0 and non_state_0) == True


def test_difusor():
    """
    For Testing load_uphi_gate from amplitude_amplification
    """
    p_gate, f_gate, pf_gate = load_gates(5)
    diffusor_gate = load_uphi_gate(pf_gate)
    phi_state, initial_state, q_prog = get_initial_state(pf_gate)
    registers = q_prog.registers
    q_prog.apply(diffusor_gate, registers)
    diffusor_state, circuit, _, _ = get_results(
        q_prog,
        linalg_qpu=linalg_qpu,
        shots=0
    )

    #Resulted state opposite to initial state
    opposite_states = np.isclose(
        diffusor_state['Amplitude'],
        -phi_state['Amplitude']
    )
    assert opposite_states.all() == True

def test_grover():
    """
    For Testing load_q_gate from amplitude_amplification
    """
    p_gate, f_gate, pf_gate = load_gates(5)
    q_gate = load_q_gate(pf_gate)
    phi_state, initial_state, q_prog = get_initial_state(pf_gate)
    registers = q_prog.registers
    q_prog.apply(q_gate, registers)
    grover_state, circuit, _, _ = get_results(
        q_prog,
        linalg_qpu=linalg_qpu,
        shots=0,
        qubits=[q_gate.arity-1]
    )

    #First get the Amplitudes for Phi state
    a0 = np.sqrt(initial_state.iloc[0]['Probability'])
    a1 = np.sqrt(initial_state.iloc[1]['Probability'])
    #Quantum state after loading data: |Psi>
    psi_state = np.array([a0, a1])
    #Angle between |Psi> and axis |Psi_0>
    theta = GetAngle(psi_state)
    #Create a Rotation of 2*theta
    c, s = np.cos(2*theta), np.sin(2.*theta)
    #Rotation matrix
    rotation = np.array(((c, -s), (s, c)))
    #Apply Ry(2*theta) to quantum state |Psi>
    rotated_state = np.dot(rotation, psi_state)    

    is_equal = np.isclose(
        rotated_state**2,
        grover_state['Probability']
    )

    assert is_equal.all() == True

    



