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


Version: Initial version

MyQLM version:

"""

import time
from copy import deepcopy
import numpy as np
import qat.lang.AQASM as qlm


def reflection(lista: np.ndarray):
    """This function returns a QLM abstract gate
        that does the following transformation
        |lista>-->-|lista>

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
    @qlm.build_gate("R_{"+str(lista)+"}", [], arity=number_qubits)
    def reflection_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)

        for i in range(number_qubits):
            if lista[i] == 0:
                routine.apply(qlm.X, register[-i-1])
        routine.apply(qlm.Z.ctrl(len(lista)-1), register)
        for i in range(number_qubits):
            if lista[i] == 0:
                routine.apply(qlm.X, register[-i-1])
        return routine
    return reflection_gate()



def U0(oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray):
    """Given an oracle O|0> = a|target>+...
    this function returns a QLM gate that does
    a|target>--->-a|target>

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
    U0_gate : QLM gate
    """

    number_qubits = oracle.arity
    @qlm.build_gate("U_0_"+str(time.time_ns()), [], arity=number_qubits)
    def U0_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(reflection(target), [register[i] for i in index])
        return routine
    return U0_gate()


def U(oracle: qlm.QRoutine):
    """This function returns a QLM abstract gate that, given
    O|0> = |Psi> does the transformation:
    |Psi>--->-|Psi>

    Parameters
    ----------
    oracle: QLM routine/gate
        operator O

    Returns
    ----------
    U_gate : QLM gate
    """
    oracle_cp = deepcopy(oracle)
    number_qubits = oracle.arity
    @qlm.build_gate("U_"+str(time.time_ns()), [], arity=number_qubits)
    def U_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(oracle.dag(), register)
        routine.apply(reflection(np.zeros(number_qubits, dtype=int)), register)
        routine.apply(oracle, register)
        return routine
    return U_gate()


def grover(oracle: qlm.QRoutine, target: np.ndarray, index: np.ndarray):
    """This function returns a QLM abstract gate
    that returns the grover associated grover to oracle for a
    given target and index.

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
    @qlm.build_gate("G_"+str(time.time_ns()), [], arity=number_qubits)
    def grover_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(U0(oracle_cp, target, index), register)
        routine.apply(U(oracle_cp), register)
        return routine
    return grover_gate()

