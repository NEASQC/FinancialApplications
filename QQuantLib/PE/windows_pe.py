"""
This module contains different window functions. Based on:
    Effects of cosine tapering window on quantum phase estimation.
    Rendon, Gumaro and Izubuchi, Taku and Kikuchi, Yuta
    Phys. Rev. D, 106. 2022
Author: Gonzalo Ferro Costas
"""

import numpy as np
import pandas as pd
from scipy.special import i0
import qat.lang.AQASM as qlm
from QQuantLib.DL.data_loading import load_probability

@qlm.build_gate("cosinewindow", [int], arity=lambda x: x)
def cosine_window(number_qubits: int):
    """
    Creates a QLM AbstractGate for loading a Cosine Window Function
    into a quantum state.

    Parameters
    ----------
    number_qubits : int
        Number of qubits for the quantum AbstractGate

    Return
    ----------

    window_state: AbstractGate
        AbstractGate for loading a cosine
    """

    window_state = qlm.QRoutine()
    q_bits = window_state.new_wires(number_qubits)
    window_state.apply(qlm.H, q_bits[-1])
    window_state.apply(
        qlm.qftarith.QFT(number_qubits),
        q_bits
    )
    for i, qb in enumerate(q_bits[:-1]):
        window_state.apply(qlm.PH(-np.pi * 2 ** i / 2**number_qubits), qb)
    window_state.apply(
        qlm.PH(np.pi * (2 ** (number_qubits -1)) / 2 ** number_qubits),
        q_bits[-1]
    )
    #window_state.apply(qlm.X, q_bits[0])
    return window_state

@qlm.build_gate("sinewindow", [int], arity=lambda x: x)
def sine_window(number_qubits: int):
    """
    Creates a QLM AbstractGate for loading a Sine Window Function
    into a quantum state.

    Parameters
    ----------
    number_qubits : int
        Number of qubits for the quantum AbstractGate

    Return
    ----------

    window_state: AbstractGate
        AbstractGate for loading a sine
    """
    window_state = qlm.QRoutine()
    q_bits = window_state.new_wires(number_qubits)
    window_state.apply(qlm.H, q_bits[-1])
    window_state.apply(
        qlm.qftarith.QFT(number_qubits),
        q_bits
    )
    for i, qb in enumerate(q_bits[:-1]):
        window_state.apply(qlm.PH(-np.pi * 2 ** i / 2**number_qubits), qb)
    window_state.apply(
        qlm.PH(np.pi * (2 ** (number_qubits -1)) / 2 ** number_qubits),
        q_bits[-1]
    )
    window_state.apply(qlm.X, q_bits[-1])
    return window_state

def kaiser_array(number_qubits, alpha=1.0e-5):
    """
    Creates the probability discretization of a Kaiser window function
    for a given input of number of qubits and a alpha
    Parameters
    ----------
    number_qubits : int
        Number of qubits for building the Kaiser window function
    alpha : float
        Parameter for modified Bessel function or order 0.

    Return
    ----------

    pdf: pandas DataFrame
        pandas DF with the probability discretization of the Kaiser
        window function
    """
    # Integer domain:
    domain_int = np.array(range(-2**(number_qubits-1), 2**(number_qubits-1)))
    x_ = domain_int / 2 ** (number_qubits-1)
    x_ = np.sqrt(1 - x_ ** 2)
    y_ = i0(np.pi * alpha * x_) / i0(np.pi * alpha)
    y_ = y_ / 2 ** number_qubits
    y_ = y_ ** 2
    # Final Probability to load
    y_final = y_ / np.sum(y_)
    pdf = pd.DataFrame([domain_int, y_final]).T
    pdf.rename(columns={0: "Int_neg", 1: "Prob"}, inplace=True)
    # Change to positive integers
    pdf["Int"] = np.where(
        pdf["Int_neg"] < 0,
        2 ** number_qubits + pdf["Int_neg"],
        pdf["Int_neg"]
    )
    # Sort by positive integers
    pdf.sort_values(["Int"], inplace=True)
    pdf.reset_index(drop=True, inplace=True)
    return pdf

def kaiser_window(number_qubits, alpha=1.0e-5):
    """
    Creates a QLM AbstractGate for loading a Kaiser Window Function
    into a quantum state. Uses load_probability function for loading
    the discretization of the probability of the Kaiser window function.

    Parameters
    ----------
    number_qubits : int
        Number of qubits for the quantum AbstractGate
    alpha : float
        Parameter for modified Bessel function or order 0.

    Return
    ----------

    kaiser_state: AbstractGate
        AbstractGate for loading a Kaiser Window
    """
    pdf = kaiser_array(number_qubits, alpha=alpha)
    kaiser_state = load_probability(pdf["Prob"], id_name="KaiserWindow")
    return kaiser_state

def window_selector(window_type, **kwargs):
    """
    Selector funcion for window functions

    Parameters
    ----------
    window_type : str
        String with the desired Window function
    kwargs : keyword arguments
        Keyword arguments for configuring window functions. Mandatory:
        auxiliar_qbits_number. For Kaiser window it is mandatory to
        provide kaiser_alpha

    Return
    ----------

    window gate: AbstractGate
        AbstractGate with the desired window function
    last_control_change : Bool
        last_control_change value
    """
    number_qubits = kwargs.get("auxiliar_qbits_number", None)
    if number_qubits is None:
        raise ValueError("auxiliar_qbits_number is None")

    if window_type in ["Cosine", "cosine", "cos"]:
        return cosine_window(number_qubits), True
    elif window_type in ["Sine", "sine", "sin"]:
        return sine_window(number_qubits), False
    elif window_type in ["Kaiser", "kaiser", "kais"]:
        kaiser_alpha = kwargs.get("kaiser_alpha", None)
        if kaiser_alpha is None:
            raise ValueError("kaiser_alpha not provided")
        return kaiser_window(number_qubits, kaiser_alpha), True
    else:
        raise ValueError(
            "Incorrect window_type provided. Only valid \
            [Cosine, cosine, cos] for cosine window, \
            [Sine,sine, sin] for sine window \
            [Kaiser, kaiser, kais] for Kaiser window"
        )
