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


@qlm.build_gate("m_ph_" + str(time.time_ns()), [float], arity=2)
def phase_multiplexor_base(theta):
    """
    Implement an initial multiplexor for a controlled phase gate.

    Parameters
    ----------

    angle : float
        Phase angle to apply

    Returns
    _______

    routine : QLM routine
        QLM routine with the implementation of the basis multiplexor
        for the controlled phase gate

    """
    routine = qlm.QRoutine()
    # This will be a 2-qbits gate
    register = routine.new_wires(2)
    # routine.apply(qlm.CNOT, register[0], register[1])
    routine.apply(qlm.PH(-theta), register[1])
    # Apply the CNOT
    routine.apply(qlm.CNOT, register[0], register[1])
    # Apply the Phase gate (+)
    routine.apply(qlm.PH(theta), register[1])
    return routine


def recursive_multiplexor(input_gate):
    """
    Create a new multiplexor from an input gate.
    In this case takes the input gate adds a new qubit and creates a new
    multiplexor by applying the input gate, a c-NOT and the input gate again

    Parameters
    ----------

    input_gate : QLM routine
        QLM routine with the gate we want for multiplexor

    Returns
    _______

    routine : QLM routine
        QLM routine with a multiplexor of the input_gate

    """
    routine = qlm.QRoutine()
    input_arity = input_gate.arity
    # Create the qbits for the input gate
    old_qbits = routine.new_wires(input_arity)
    # Add a new qbit for multiplexion
    new_qbit = routine.new_wires(1)
    # routine.apply(qlm.CNOT, old_qbits[input_arity-1], new_qbit)
    routine.apply(input_gate, [old_qbits[: input_arity - 1], new_qbit])
    routine.apply(qlm.CNOT, old_qbits[input_arity - 1], new_qbit)
    routine.apply(input_gate, [old_qbits[: input_arity - 1], new_qbit])
    return routine


@qlm.build_gate("m_mcph_" + str(time.time_ns()), [float, int], arity=lambda x, y: y)
def multiplexor_controlled_ph(angle, number_qubits):
    """
    Multiplexor implementation for a Multi-Controlled-phase gate

    Parameters
    ----------

    angle : float
        Desired angle for Controlled-Phase application
    number_qubits : int
        Number of qubits for the multi-controlled phase gate

    Returns
    _______

    routine : QLM routine
        QLM routine with the implementation of a multi-controlled phase gate

    """
    routine = qlm.QRoutine()
    register = routine.new_wires(number_qubits)
    # Angle for each Phase gate
    angle = angle / (2 ** (number_qubits - 1))
    for i, _ in enumerate(register):
        # print('i:', i)
        if i == 0:
            # In the first qubit we need a Phase rotation
            routine.apply(qlm.PH(angle), register[i])
        elif i == 1:
            # In the second qubit we need the base gate for the multiplexor
            routine.apply(qlm.CNOT, register[i - 1], register[i])
            multiplexor = phase_multiplexor_base(angle)
            # print(register[:i])
            routine.apply(multiplexor, register[: i + 1])
        else:
            # For other qubits we need to create the new multiplexor
            # from the before step multiplexor
            routine.apply(qlm.CNOT, register[i - 1], register[i])
            multiplexor = recursive_multiplexor(multiplexor)
            routine.apply(multiplexor, register[: i + 1])
    return routine


@qlm.build_gate("m_mcz_" + str(time.time_ns()), [int], arity=lambda x: x)
def multiplexor_controlled_z(number_qubits):
    """
    Multiplexor implementation for a multi-controlled-Z gate

    Parameters
    ----------


    number_qubits : int
        Number of qubits for the multi-controlled phase gate gate

    Returns
    _______

    routine : QLM routine
        QLM routine with the implementation of a multi-controlled Z gate
    """
    routine = qlm.QRoutine()
    register = routine.new_wires(number_qubits)
    gate = multiplexor_controlled_ph(np.pi, number_qubits)
    routine.apply(gate, register)
    return routine


def reflection(lista: np.ndarray, mcz_qlm=True):
    r"""
    This function returns a QLM AbstractGate that implement a reflection
    around the perpendicular state of a given state.

    Notes
    -----
    .. math::
        |\Psi\rangle=|\Psi_0\rangle+|\Psi_1\rangle

    .. math::
        \mathcal{reflection}(|\Psi_0\rangle)|\Psi\rangle=
        -|\Psi_0\rangle+|\Psi_1\rangle\\

    Parameters
    ----------
    lista: list of ints
        binary representation of the
        State that we want to rotate pi
    mcz_qlm: bool
        If True QLM construction for multi-controlled Z will be used.

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
        if mcz_qlm:
            routine.apply(qlm.Z.ctrl(len(lista) - 1), register)
        else:
            mcz_multiplexor = multiplexor_controlled_z(len(lista))
            routine.apply(mcz_multiplexor, register)
        for i in range(number_qubits):
            if lista[i] == 0:
                routine.apply(qlm.X, register[-i - 1])
        return routine

    return reflection_gate()


def create_u0_gate(
    oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray, mcz_qlm=True
):
    r"""
    This function creates a QLM AbstractGate that implements an oracle:
    a reflection around the perpendicular state to a given state.

    Notes
    -----
    .. math::
        If \hspace{1em} |\Psi\rangle=\alpha_0|\Psi_0\rangle+ \
        \alpha_1|\Psi_1\rangle \hspace{1em} where \hspace{1em} \
        |\Psi_0\rangle \perp |\Psi_1\rangle \hspace{1em} then \
        \hspace{1em} \mathcal{U0}(|\Psi_0\rangle)=\mathcal{I}-2 \
        |\Psi_0\rangle\langle\Psi_0|

    Parameters
    ----------
    oracle: QLM routine/gate
        oracle that created the state
    target: list of ints
        target state
    index: list of ints
        index for the qubits that define the register
    mcz_qlm: bool
        If True QLM construction for multi-controlled Z will be used.
    Returns
    ----------
    u0_gate : QLM gate
    """

    number_qubits = oracle.arity

    @qlm.build_gate("U_0_" + str(time.time_ns()), [], arity=number_qubits)
    def u0_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(reflection(target, mcz_qlm), [register[i] for i in index])
        return routine

    return u0_gate()


def create_u_gate(oracle: qlm.QRoutine, mcz_qlm=True):
    r"""
    This function creates a QLM AbstractGate that implements a grover
    diffusion operator from an input state.

    Notes
    -----
    .. math::
        \mathcal{U}(|\Psi\rangle) = \mathcal{I}-2|\Psi\rangle\langle\Psi|

    Parameters
    ----------
    oracle: QLM routine/gate
        operator O
    mcz_qlm: bool
        If True QLM construction for multi-controlled Z will be used.

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
        routine.apply(reflection(np.zeros(number_qubits, dtype=int), mcz_qlm), register)
        routine.apply(oracle, register)
        return routine

    return u_gate()


def grover(oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray, mcz_qlm=True):
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
    mcz_qlm: bool
        If True QLM construction for multi-controlled Z will be used.

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
        routine.apply(create_u0_gate(oracle_cp, target, index, mcz_qlm), register)
        routine.apply(create_u_gate(oracle_cp, mcz_qlm), register)
        return routine

    return grover_gate()


def grover_extended(oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray, mcz_qlm=True):
    r"""
    This function creates a QLM AbstractGate that returns the grover
    extended operator associated to a given oracle.

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
    mcz_qlm: bool
        If True QLM construction for multi-controlled Z will be used.

    Returns
    ----------
    grover_gate : QLM gate
    """
    oracle_cp = deepcopy(oracle)
    number_qubits = oracle_cp.arity
    @qlm.build_gate("G'_" + str(time.time_ns()), [], arity=number_qubits)
    def grover_extended_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        #refection on |0>
        routine.apply(
            reflection(np.zeros(number_qubits, dtype=int), mcz_qlm),
            register
        )
        routine.apply(oracle_cp, register)
        routine.apply(
            reflection(target, mcz_qlm),
            [register[i] for i in index]
        )
        routine.apply(oracle_cp.dag(), register)
        return routine

    return grover_extended_gate()
