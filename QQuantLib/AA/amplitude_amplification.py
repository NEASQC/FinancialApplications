"""
This module contains all functions needed for creating Grover-like
operators mandatory for using Quantum Amplitude Amplification
and Estimation as explained in the 2000 Brassard paper:

    Gilles Brassard, Peter Hoyer, Michele Mosca and Alain Tapp
    Quantum Amplitude Amplification and Estimation
    AMS Contemporary Mathematics Series, 305, 06-2000
    https://arxiv.org/abs/quant-ph/0005055v1

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import time
from copy import deepcopy
import numpy as np
import qat.lang.AQASM as qlm


def reflection(lista: np.ndarray):
    r"""
    This function returns a QLM AbstractGate that implement a reflection
    around the perpendicular state of a given state.

    Notes
    -----
    .. math::
        |\Psi\rangle=|\Psi_0\rangle+|\Psi_1\rangle\\
        \mathcal{reflection}(|\Psi_0\rangle)|\Psi\rangle=
        -|\Psi_0\rangle+|\Psi_1\rangle\\

    Parameters
    ----------
    lista: list of ints
        binary representation of the
        State that we want to rotate pi

    Returns
    ----------
    reflection_gate : QLM gate
    """
    number_qubits = len(lista)

    @qlm.build_gate("R_{" + str(lista) + "}", [], arity=number_qubits)
    def reflection_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)

        for i in range(number_qubits):
            if lista[i] == 0:
                routine.apply(qlm.X, register[-i - 1])
        routine.apply(qlm.Z.ctrl(len(lista) - 1), register)
        for i in range(number_qubits):
            if lista[i] == 0:
                routine.apply(qlm.X, register[-i - 1])
        return routine

    return reflection_gate()


def create_u0_gate(oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray):
    r"""
    This function creates a QLM AbstractGate that implements an oracle:
    a reflection around the perpendicular state to a given state.

    Notes
    -----
    .. math::
        If \hspace{1em} |\Psi\rangle=\alpha_0|\Psi_0\rangle+\alpha_1|\Psi_1\rangle
        \hspace{1em} where \hspace{1em} |\Psi_0\rangle \perp |\Psi_1\rangle
        \hspace{1em} then \hspace{1em}
        \mathcal{U0}(|\Psi_0\rangle)=\mathcal{I}-2|\Psi_0\rangle\langle\Psi_0|

    Parameters
    ----------
    oracle: QLM routine/gate
        oracle that created the state
    target: list of ints
        target state
    index: list of ints
        index for the qubits that define the register
    Returns
    ----------
    u0_gate : QLM gate
    """

    number_qubits = oracle.arity

    @qlm.build_gate("U_0_" + str(time.time_ns()), [], arity=number_qubits)
    def u0_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(reflection(target), [register[i] for i in index])
        return routine

    return u0_gate()


def create_u_gate(oracle: qlm.QRoutine):
    r"""
    This function creates a QLM AbstractGate that implements a grover Diffusor
    from an input state.

    Notes
    -----
    .. math::
        \mathcal{U}(|\Psi\rangle) = \mathcal{I}-2|\Psi\rangle\langle\Psi|

    Parameters
    ----------
    oracle: QLM routine/gate
        operator O

    Returns
    ----------
    u_gate : QLM gate
    """
    oracle_cp = deepcopy(oracle)
    number_qubits = oracle.arity

    @qlm.build_gate("U_" + str(time.time_ns()), [], arity=number_qubits)
    def u_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(oracle.dag(), register)
        routine.apply(reflection(np.zeros(number_qubits, dtype=int)), register)
        routine.apply(oracle, register)
        return routine

    return u_gate()


def grover(oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray):
    r"""
    This function creates a QLM AbstractGate that returns the grover
    operator associated to a given oracle. This function is a composition
    of QLM AbstractGates generate by functions U and U0.

    Notes
    -----
    .. math::
        \mathcal{G} = \mathcal{U}(|\Psi\rangle)\mathcal{U0}(|\Psi_0\rangle)

    Parameters
    ----------
    oracle : QLM routine/gate
    target : list of ints
        the state that we want to amplify
    index : list of ints
        index for the qubits that define the register

    Returns
    ----------
    grover_gate : QLM gate
    """
    oracle_cp = deepcopy(oracle)
    number_qubits = oracle_cp.arity

    @qlm.build_gate("G_" + str(time.time_ns()), [], arity=number_qubits)
    def grover_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(create_u0_gate(oracle_cp, target, index), register)
        routine.apply(create_u_gate(oracle_cp), register)
        return routine

    return grover_gate()
