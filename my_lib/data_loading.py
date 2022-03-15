"""
Copyright 2022 CESGA
License:

This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

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

Authors: Alberto Pedro Manzano Herrero

"""

import numpy as np
import qat.lang.AQASM as qlm
from utils import mask, fwht, test_bins, left_conditional_probability

# Loading uniform distribution
@qlm.build_gate("UD", [int], arity=lambda x: x)
def uniform_distribution(number_qubits: int):
    """
    Function to load a uniform distribution in a quantum circuit.
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
    """
    Auxiliary function that transforms the state |0>|index> into
    cos(angle)|0>|index>+sin(angle)|1>|index>.
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

    routine.apply(
        mask(number_qubits-1, index),
        quantum_register[:number_qubits-1]
    )
    routine.apply(
        qlm.RY(angle).ctrl(number_qubits-1),
        quantum_register[:number_qubits-1],
        quantum_register[number_qubits-1]
    )
    routine.apply(
        mask(number_qubits-1, index),
        quantum_register[:number_qubits-1]
    )

    return routine

def load_angles_brute_force(angles: np.array):
    """
    Creates an Abstract gate using multicontrolled rotations that
    transforms the state:
    |0>|0>+ |0>|1>+ |0>|2>+...+ |0>|len(angle)-1>,
    into:
    cos(angle)|0>|0>+cos(angle)|0>|1>+cos(angle)|0>|2>+...
        +cos(angle)|0>|len(angle)-1>
    +sin(angle)|0>|0>+sin(angle)|0>|1>+sin(angle)|0>|2>+...
        +sin(angle)|0>|len(angle)-1>.
    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
        int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size))+1
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(angles.size):
        routine.apply(load_angle(number_qubits, i, angles[i]), quantum_register)
    return routine

def multiplexor_RY(angles: np.array, ordering: str = "sequency"):
    """
    Creates an Abstract gate using Quantum Multiplexors that transforms
    the state:
    |0>|0>+ |0>|1>+ |0>|2>+...+ |0>|len(angle)-1>,
    into:
    cos(angle)|0>|0>+cos(angle)|0>|1>+cos(angle)|0>|2>+...
        +cos(angle)|0>|len(angle)-1>
    +sin(angle)|0>|0>+sin(angle)|0>|1>+sin(angle)|0>|2>+...
        +sin(angle)|0>|len(angle)-1>.
    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
            int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size))
    angles = fwht(angles, ordering=ordering)
    angles = angles/2**number_qubits
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits+1)
    control = np.zeros(2**number_qubits, dtype=int)
    for i in range(number_qubits):
        for j in range(2**i-1, 2**number_qubits, 2**i):
            control[j] = number_qubits-i-1
    for i in range(2**number_qubits):
        routine.apply(qlm.RY(angles[i]), quantum_register[number_qubits])
        routine.apply(
            qlm.CNOT,
            quantum_register[control[i]],
            quantum_register[number_qubits]
        )
    return routine

def load_angles(angles: np.array, method: str = "multiplexor"):
    """
    Auxiliary function that transforms the state:
    |0>|0>+ |0>|1>+ |0>|2>+...+ |0>|len(angle)-1>,
    into:
    cos(angle)|0>|0>+cos(angle)|0>|1>+cos(angle)|0>|2>+...
        +cos(angle)|0>|len(angle)-1>
    +sin(angle)|0>|0>+sin(angle)|0>|1>+sin(angle)|0>|2>+...
        +sin(angle)|0>|len(angle)-1>.
    It serves as an interface for the two methods for loading the angles.
    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is:
        int(np.log2(len(angle)))+1.
    method : string
        Method used in the loading. Default method.
    """
    number_qubits = int(np.log2(angles.size))+1
    if (np.max(angles) > np.pi):
        print("ERROR: function f not properly normalised")
        return
    if (angles.size != 2**(number_qubits-1)):
        print("ERROR: size of function f is not a factor of 2")
    if (method == "brute_force"):
        routine = load_angles_brute_force(angles)
    else:
        routine = multiplexor_RY(angles)
    return routine

def load_array(function_array: np.array, method: str = "multiplexor",\
id_name: str = '1'):
    """
    Creates an Abstract gate for loading a normalised array.

    Parameters
    ----------
    function_array : numpy array
        Numpy array with the normalised array to load. The arity of
        of the gate is int(np.log2(len(probability_array)))+1.
    method : str
        type of loading method used:
            multiplexor : with quantum Multiplexors
            brute_force : using multicontrolled rotations by state

    Return
    ----------

    f_gate: AbstractGate
        AbstractGate customized for loading a normalised array
    """
    number_qubits = int(np.log2(function_array.size))+1
    @qlm.build_gate("F_{"+id_name+"}", [], arity=number_qubits)
    def load_array_gate():
        """
        QLM Routine generation.
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        angles = 2*np.arccos(function_array)
        routine.apply(load_angles(angles, method=method), register)
        return routine
    return load_array_gate()


def load_probability(probability_array: np.array):
    """
    Creates an Abstract gate for loading an input discretized
    Probability Distribution using Quantum Multiplexors.

    Parameters
    ----------
    probability_array : numpy array
        Numpy array with the discretized probability to load. The arity of
        of the gate is int(np.log2(len(probability_array))).

    Returns
    ----------

    P_Gate :  AbstractGate
        Customized Abstract Gate for Loading Probability array using
        Quantum Multiplexors
    """
    number_qubits = int(np.log2(probability_array.size))

    @qlm.build_gate("P", [], arity=number_qubits)
    def load_probability_gate():
        """
        QLM Routine generation.
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        # Now go iteratively trough each qubit computing the
        #probabilities and adding the corresponding multiplexor
        for m in range(number_qubits):
            #print(m)
            #Calculates Conditional Probability
            conditional_probability = left_conditional_probability(m,\
            probability_array)
            #Rotation angles: length: 2^(i-1)-1 and i the number of
            #qbits of the step
            thetas = 2.0*(np.arccos(np.sqrt(conditional_probability)))
            if m == 0:
                # In the first iteration it is only needed a RY gate
                routine.apply(qlm.RY(thetas[0]), register[number_qubits-1])
            else:
                # In the following iterations we have to apply
                # multiplexors controlled by m qubits
                # We call a function to construct the multiplexor,
                # whose action is a block diagonal matrix of Ry gates
                # with angles theta
                routine.apply(
                    multiplexor_RY(thetas),
                    register[number_qubits-m:number_qubits],
                    register[number_qubits-m-1]
                )
        return routine
    return load_probability_gate()

def load_pf(p_gate, f_gate):
    """
    Create complete AbstractGate for applying Operators P and R
    The operator to implement is:
        p_gate*r_gate

    Parameters
    ----------
    p_gate : QLM AbstractGate
        Customized AbstractGate for loading probability distribution.
    f_gate : QLM AbstractGate
        Customized AbstractGatel for loading integral of a function f(x)
    Returns
    ----------
    pr_gate : AbstractGate
        Customized AbstractGate for loading the P and R operators
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
