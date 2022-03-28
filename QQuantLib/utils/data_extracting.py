"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains auxiliar functions for executing QLM programs based
on QLM QRoutines or QLM gates and for postproccessing results from QLM
qpu executions

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.core import Result
from libraries.utils.utils import check_list_type

pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(suppress=True)

def get_results(quantum_object, linalg_qpu, shots: int = 0, qubits: list = None):
    """
    Function for testing an input gate. This fucntion creates the
    quantum program for an input gate, the correspondent circuit
    and job. Execute the job and gets the results

    Parameters
    ----------
    quantum_object : QLM Gate, Routine or Program
    linalg_qpu : QLM solver
    shots : int
        number of shots for the generated job.
        if 0 True probabilities will be computed
    qubits : list
        list with the qbits for doing the measurement when simulating
        if None measuremnt over all allocated qbits will be provided

    Returns
    ----------
    pdf : pandas DataFrame
        DataFrame with the results of the simulation
    circuit : QLM circuit
    q_prog : QLM Program.
    job : QLM job

    """
    arity = quantum_object.arity

    if qubits is None:
        qubits = np.arange(arity, dtype=int)
    else:
        qubits = check_list_type(qubits, int)

    if type(quantum_object) == qlm.Program:
        q_prog = deepcopy(quantum_object)
    else:
        q_prog = qlm.Program()
        qbits = q_prog.qalloc(arity)
        q_prog.apply(quantum_object, qbits)

    circuit = q_prog.to_circ(submatrices_only=True)
    job = circuit.to_job(nbshots=shots, qubits=qubits)

    result = linalg_qpu.submit(job)
    if not isinstance(result, Result):
        result = result.join()

    # Process the results
    states = []
    list_int = []
    list_int_lsb = []
    for i in range(2**qubits.size):
        reversed_i = int('{:0{width}b}'.format(i, width=qubits.size)[::-1], 2)
        list_int.append(reversed_i)
        list_int_lsb.append(i)
        states.append("|"+ bin(i)[2:].zfill(qubits.size)+">")

    probability = np.zeros(2**qubits.size)
    amplitude = np.zeros(2**qubits.size, dtype=np.complex_)
    for samples in result:
        probability[samples.state.lsb_int] = samples.probability
        amplitude[samples.state.lsb_int] = samples.amplitude

    pdf = pd.DataFrame({
        'States': states,
        'Int_lsb': list_int_lsb,
        'Probability': probability,
        'Amplitude': amplitude,
        'Int': list_int
    })

    return pdf, circuit, q_prog, job

