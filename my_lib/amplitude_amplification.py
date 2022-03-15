"""
Copyright 2022 CESGA
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains all functions needed for creating Grover-like
operators mandatory for using Quantum Amplitude Amplification
and Estimation as explained in the 2000 Brassard paper:

    Gilles Brassard, Peter Hoyer, Michele Mosca and Alain Tapp
    Quantum Amplitude Amplification and Estimation
    AMS Contemporary Mathematics Series, 305, 06-2000
    https://arxiv.org/abs/quant-ph/0005055v1


Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""

import qat.lang.AQASM as qlm

@qlm.build_gate("U_Phi_0", [int], arity=lambda n: n)
def uphi0_gate(nqbits):
    """
    Circuit generator for creating gate U_Phi_0 for Phase
    Amplification Algorithm. For a n+1 qbit system where the quantum
    state can be decomposed by:
        |Psi>_{n+1} =a*|Phi_{n}>|1>+b|Phi_{n}>|0>.
    The function implements a reflexion around the state |Phi_{n}>|1>.
    The operator implemented is:
        I-2|Phi_{n}>|0><0|<Phi_{n}|
    Parameters
    ----------

    nqbits : int
        Number of Qbits of the Abstract Gate

    Returns
    ----------

    q_rout : QLM Routine
        Quantum routine wiht the circuit implementation for operator:
            I-2|Phi_{n}>|0><0|<Phi_{n}|
    """
    q_rout = qlm.QRoutine()
    qbits = q_rout.new_wires(nqbits)
    q_rout.apply(qlm.X, qbits[-1])
    q_rout.apply(qlm.Z, qbits[-1])
    q_rout.apply(qlm.X, qbits[-1])
    return q_rout

@qlm.build_gate("D_0", [int], arity=lambda n: n)
def d0_gate(nqbits):
    """
    Circuit generator for create an Abstract Gate that implements a
    Reflexion around the state perpendicular to |0>_{n}.
    Implements operator:
        I-2|0>_{n}{n}<0| = X^{n}c^{n-1}ZX^{n}
    Parameters
    ----------
    nqbits : int
        Number of Qbits of the Abstract Gate
    Returns
    ----------
    q_rout : QLM Routine
        Quantum routine wiht the circuit implementation for operator:
            I-2|0>_{n}{n}<0|
    """
    q_rout = qlm.QRoutine()
    qbits = q_rout.new_wires(nqbits)
    for i in range(nqbits):
        q_rout.apply(qlm.X, qbits[i])
    #Controlled Z gate by n-1 first qbits
    c_n_z = qlm.Z.ctrl(nqbits-1)
    q_rout.apply(c_n_z, qbits[:-1], qbits[-1])
    for i in range(nqbits):
        q_rout.apply(qlm.X, qbits[i])
    return q_rout

def load_uphi_gate(gate):
    """
    Create gate U_Phi (Grover Difusor operator) mandatory for using
    Grover Alogrithm.
    The operator to implement is:
        I-2|Phi><Phi|.
    Where the state |Phi> is: |Phi>=Ô*|0>.
    Where Ô operator is represented by the input gate
    Parameters
    ----------
    gate : QLM Gate
        QLM Gate for creating diffusion Operator
    Outputs:
        * U_Phi: guantum gate that implements U_Phi gate (diffusor operator)
    """
    #The arity of the r_gate fix the number of qbits for the routine
    nqbits = gate.arity

    @qlm.build_gate("UPhi", [], arity=nqbits)
    def u_phi_gate():
        """
        Circuit generator for the u_phi_gate.
        Operation to be implemented: R*P*D_0*P^{+}R^{+}
        Returns
        ----------
        q_rout : Quantum Routinequantum
            Quantum Routine with the circuit implementation for operator:
            R*P*D_0*P^{+}R^{+}
        """
        q_rout = qlm.QRoutine()
        qbits = q_rout.new_wires(nqbits)
        q_rout.apply(gate.dag(), qbits)
        d_0 = d0_gate(nqbits)
        q_rout.apply(d_0, qbits)
        q_rout.apply(gate, qbits)
        return q_rout
    return u_phi_gate()

def load_q_gate(pr_gate):
    """
    Create complete AbstractGate for Grover-like algorithm
    The operator to implement is:
        uphi0_gate*u_phi_gate
    This operator implements a y Rotation around the input state:
        Ry(Theta)|Phi> where |Phi> is the input State
    Where Theta is given by the square root of the expected value of
    function f(x) (loaded by r_gate) under a probability density p(x)
    (loaded by p_gate):
        sin(Theta) = sqrt(Expected_p(x)[f(x)])
    Parameters
    ----------
    pr_gate : QLM AbstractGate
        QLM Gate for creating Grover-like operator Q
    Returns
    ----------
    Q_Gate : AbstractGate
        Customized AbstractGate for Amplitude Amplification Algorithm
    """
    nqbits = pr_gate.arity
    @qlm.build_gate("Q_Gate", [], arity=nqbits)
    def q_gate():
        """
        Function generator for creating an AbstractGate for implementation
        of the Amplification Amplitude Algorithm (Q)
        Returns
        ----------
        q_rout : quantum routine
            Routine for Amplitude Amplification Algorithm
        """
        q_rout = qlm.QRoutine()
        qbits = q_rout.new_wires(nqbits)
        q_rout.apply(uphi0_gate(nqbits), qbits)
        u_phi_gate = load_uphi_gate(pr_gate)
        q_rout.apply(u_phi_gate, qbits)
        return q_rout
    return q_gate()

def load_qn_gate(qlm_gate, n):
    """
    Create an AbstractGate by applying an input gate n times

    Parameters
    ----------
    
    qlm_gate : QLM gate
        QLM gate that will be applied n times
    n : int
        number of times the qlm_gate will be applied

    """
    @qlm.build_gate("Q^{}".format(n), [], arity=qlm_gate.arity)
    def q_n_gate():
        """
        Function generator for creating an AbstractGate for apply
        an input gate n times
        Returns
        ----------
        q_rout : quantum routine
            Routine for applying n times an input gate
        """
        q_rout = qlm.QRoutine()
        q_bits = q_rout.new_wires(qlm_gate.arity)
        for _ in range(n):
            q_rout.apply(qlm_gate, q_bits)
        return q_rout
    return q_n_gate()

def create_qprogram(quantum_gate):
    """
    Creates a Quantum Program from an input qlm gate or routine

    Parameters
    ----------

    quantum_gate : QLM gate or QLM routine

    Returns
    ----------
    q_prog: QLM Program.
        Quantum Program from input QLM gate or routine
    """
    q_prog = qlm.Program()
    qbits = q_prog.qalloc(quantum_gate.arity)
    q_prog.apply(quantum_gate, qbits)
    return q_prog
