"""
This module contains all the functions in order to load data into the
quantum state.
There are two implementations for the loading of a function:

    * one based on brute force
    * one based on multiplexors.

The implementation of the multiplexors is a non-recursive version of:

    V.V. Shende, S.S. Bullock, and I.L. Markov.
    Synthesis of quantum-logic circuits.
    IEEE Transactions on Computer-Aided Design of Integrated Circuits
    and Systems, 25(6):1000–1010, Jun 2006
    arXiv:quant-ph/0406176v5

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import time
import numpy as np
import qat.lang.AQASM as qlm
from QQuantLib.utils.utils import mask, fwht, left_conditional_probability, expmod

# Loading uniform distribution
@qlm.build_gate("UD", [int], arity=lambda x: x)
def uniform_distribution(number_qubits: int):
    r"""
    Function to load a uniform distribution in a quantum circuit.

    Notes
    -----
    .. math::
        \mathcal{H}^{\otimes n}|\Psi\rangle


    Parameters
    ----------
    number_qubits : int
        Arity of the output gate.
    """
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(number_qubits):
        routine.apply(qlm.H, quantum_register[i])
    return routine


@qlm.build_gate("LP", [int, int, float], arity=lambda x, y, z: x)
def load_angle(number_qubits: int, index: int, angle: float):
    r"""
    Creates an QLM Abstract Gate that apply a rotation of a given angle
    into a auxiliary qubit controlled by a given state of the measurement basis.
    Direct QLM multi controlled rotations were used for the implementation.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle
    .. math::
        \mathcal{load\_angle}(\theta, |i\rangle)|\Psi\rangle \
        =\sum_{j=0, j\ne i}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle+ \
        \alpha_i|i\rangle\otimes\big(\cos(\theta)|0\rangle+\sin(\theta) \
        |1\rangle\big)


    Parameters
    ----------
    number_qubits : int
        Number of qubits for the control register. The arity of the gate is number_qubits+1.
    index : int
        Index of the state that we control.
    angle : float
        Angle that we load.
    """

    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)

    routine.apply(mask(number_qubits - 1, index), quantum_register[: number_qubits - 1])
    routine.apply(
        qlm.RY(angle).ctrl(number_qubits - 1),
        quantum_register[: number_qubits - 1],
        quantum_register[number_qubits - 1],
    )
    routine.apply(mask(number_qubits - 1, index), quantum_register[: number_qubits - 1])

    return routine


def load_angles_brute_force(angles: np.array):
    r"""
    Given a list of angles this function creates a QLM routine that applies
    rotations of each angle of the list, over an auxiliary qubit, controlled
    by the different states of the measurement basis.
    Direct QLM multi controlled rotations were used for the implementation.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle
    .. math::
        \mathcal{load\_angles\_brute\_force} \
        ([\theta_j]_{j=0,1,2...2^n-1}) |\Psi\rangle=\sum_{j=0}^{2^n-1} \
        \alpha_j|j\rangle\otimes\big(\cos(\theta_j)|0\rangle+ \
        \sin(\theta_j)|1\rangle\big)


    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
        int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size)) + 1
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(angles.size):
        routine.apply(load_angle(number_qubits, i, angles[i]), quantum_register)
    return routine


def multiplexor_ry(angles: np.array, ordering: str = "sequency"):
    r"""
    Given a list of angles this functions creates a QLM routine that applies
    rotations of each angle of the list, over an auxiliary qubit, controlled
    by the different states of the measurement basis.
    The multi-controlled rotations were implemented using Quantum Multiplexors.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle
    .. math::
        \mathcal{multiplexor\_RY} \
        ([\theta_j]_{j=0,1,2...2^n-1})|\Psi\rangle = \sum_{j=0}^{2^n-1} \
        \alpha_j|j\rangle\otimes\big(\cos(\theta_j)|0\rangle+\sin(\theta_j)|1\rangle\big)

    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
            int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size))
    angles = fwht(angles, ordering=ordering)
    angles = angles / 2**number_qubits
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits + 1)
    control = np.zeros(2**number_qubits, dtype=int)
    for i in range(number_qubits):
        for j in range(2**i - 1, 2**number_qubits, 2**i):
            control[j] = number_qubits - i - 1
    for i in range(2**number_qubits):
        routine.apply(qlm.RY(angles[i]), quantum_register[number_qubits])
        routine.apply(
            qlm.CNOT, quantum_register[control[i]], quantum_register[number_qubits]
        )
    return routine


def load_angles(angles: np.array, method: str = "multiplexor"):
    r"""
    This function serves as an interface for the two different implementations
    of multi controlled rotations: load_angles_brute_force and multiplexor_RY.

    Notes
    -----
    .. math::
        |\Psi\rangle = \sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes|0\rangle

    .. math::
        \mathcal{load\_angles}([\theta_j]_{j=0,1,2...2^n-1})|\Psi\rangle \
        =\sum_{j=0}^{2^n-1}\alpha_j|j\rangle\otimes \
        \big(\cos(\theta_j)|0\rangle+\sin(\theta_j)|1\rangle\big)

    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
        int(np.log2(len(angle)))+1.
    method : string
        Method used in the loading. Default method.
    """
    number_qubits = int(np.log2(angles.size)) + 1
    if np.max(angles) > 2 * np.pi:
        raise ValueError("ERROR: function f not properly normalised")
    if angles.size != 2 ** (number_qubits - 1):
        print("ERROR: size of function f is not a factor of 2")
    if method == "brute_force":
        routine = load_angles_brute_force(angles)
    else:
        routine = multiplexor_ry(angles)
    return routine


def load_array(
    function_array: np.array,
    method: str = "multiplexor",
    id_name: str = None,
):
    """
    Creates a QLM AbstractGate for loading a normalised array into a quantum
    state.

    Parameters
    ----------
    function_array : numpy array
        Numpy array with the normalised array to load. The arity of
        of the gate is int(np.log2(len(probability_array)))+1.
    method : str
        type of loading method used:
            multiplexor : with quantum Multiplexors
            brute_force : using multicontrolled rotations by state
    id_name : str
        name for the Abstract Gate

    Return
    ----------

    f_gate: AbstractGate
        AbstractGate customized for loading a normalised array
    """
    number_qubits = int(np.log2(function_array.size)) + 1

    if id_name is None:
        id_name = str(time.time_ns())

    @qlm.build_gate("F_{" + id_name + "}", [], arity=number_qubits)
    def load_array_gate():
        """
        QLM Routine generation.
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        angles = 2 * np.arccos(function_array)
        routine.apply(load_angles(angles, method=method), register)
        return routine

    return load_array_gate()


def load_probability(
    probability_array: np.array,
    method: str = "multiplexor",
    id_name: str = None,
    #id_name: str = str(time.time_ns())
):
    """
    Creates a QLM Abstract gate for loading a given discretized probability
    distribution using Quantum Multiplexors.

    Parameters
    ----------
    probability_array : numpy array
        Numpy array with the discretized probability to load. The arity of
        of the gate is int(np.log2(len(probability_array))).
    method : str
        type of loading method used:
            multiplexor : with quantum Multiplexors
            brute_force : using multicontrolled rotations by state
    id_name : str
        name for the Abstract Gate

    Returns
    ----------

    P_Gate :  AbstractGate
        Customized Abstract Gate for Loading Probability array using
        Quantum Multiplexors
    """
    number_qubits = int(np.log2(probability_array.size))
    if id_name is None:
        id_name = str(time.time_ns())

    @qlm.build_gate("P_{" + id_name + "}", [], arity=number_qubits)
    def load_probability_gate():
        """
        QLM Routine generation.
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        # Now go iteratively trough each qubit computing the
        # probabilities and adding the corresponding multiplexor
        for m_qbit in range(number_qubits):
            # print(m)
            # Calculates Conditional Probability
            conditional_probability = left_conditional_probability(
                m_qbit, probability_array
            )
            # Rotation angles: length: 2^(i-1)-1 and i the number of
            # qbits of the step
            thetas = 2.0 * (np.arccos(np.sqrt(conditional_probability)))
            if m_qbit == 0:
                # In the first iteration it is only needed a RY gate
                routine.apply(qlm.RY(thetas[0]), register[number_qubits - 1])
            else:
                # In the following iterations we have to apply
                # multiplexors controlled by m_qbit qubits
                # We call a function to construct the multiplexor,
                # whose action is a block diagonal matrix of Ry gates
                # with angles theta
                routine.apply(
                    #multiplexor_ry(thetas),
                    load_angles(thetas, method),
                    register[number_qubits - m_qbit : number_qubits],
                    register[number_qubits - m_qbit - 1],
                )
        return routine

    return load_probability_gate()


def step_array(index: int, size: int):
    """
    Creates are routine which loads an array of size "size".
    This array has ones up to but not included
    the index position. The rest of the values are zero.
    This is why it is called step_array.

    Parameters
    ----------
    index : int
       position where the step is produced
    size : int
        size of the array. It has to be a power of 2

    Returns
    -------
    step_function_gate : Abstract Gate
        gate which loads the corresponding array. Note that
        the arity is: np.log2(size)+1
    """
    (power, remainder) = expmod(size, 2)
    assert remainder == 0, "ERROR: size must be a power of 2"

    arity = power + 1

    @qlm.build_gate("Step[" + str(index) + "," + str(size) + "]", [], arity=arity)
    def step_function_gate():
        ones = np.ones(index)
        zeros = np.zeros(size - index)
        array = np.concatenate((ones, zeros))
        routine = qlm.QRoutine()
        register = routine.new_wires(arity)
        routine.apply(load_array(array), register)
        return routine

    return step_function_gate()


def load_pf(p_gate, f_gate):
    """
    Create a QLM AbstractGate for applying two given operators consecutively.
    The operator to implement is: p_gate*f_gate

    Parameters
    ----------
    p_gate : QLM AbstractGate
        Customized AbstractGate for loading probability distribution.
    f_gate : QLM AbstractGate
        Customized AbstractGatel for loading integral of a function f(x)
    Returns
    ----------
    pf_gate : AbstractGate
    """
    nqbits = f_gate.arity

    @qlm.build_gate("PF", [], arity=nqbits)
    def load_pf_gate():
        """
        QLM Routine generation.
        """
        q_rout = qlm.QRoutine()
        qbits = q_rout.new_wires(nqbits)
        q_rout.apply(p_gate, qbits[:-1])
        q_rout.apply(f_gate, qbits)
        return q_rout

    return load_pf_gate()
