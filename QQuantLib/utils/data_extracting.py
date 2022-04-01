"""
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
from QQuantLib.utils.utils import check_list_type

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

    #if type(quantum_object) == qlm.Program:
    if isinstance(quantum_object, qlm.Program):
        q_prog = deepcopy(quantum_object)
    else:
        q_prog = create_qprogram(quantum_object)
        #q_prog = qlm.Program()
        #qbits = q_prog.qalloc(arity)
        #q_prog.apply(quantum_object, qbits)

        arity = quantum_object.arity

    if qubits is None:
        qubits = np.arange(arity, dtype=int)
    else:
        qubits = check_list_type(qubits, int)
    #circuit = q_prog.to_circ(submatrices_only=True)
    circuit = create_qcircuit(q_prog)
    #job = circuit.to_job(nbshots=shots, qubits=qubits)
    job = create_qjob(circuit, shots=shots, qubits=qubits)

    result = linalg_qpu.submit(job)
    if not isinstance(result, Result):
        result = result.join()
    # Process the results
    pdf = proccess_qresults(result, qubits)

    #states = []
    #list_int = []
    #list_int_lsb = []
    #for i in range(2**qubits.size):
    #    reversed_i = int('{:0{width}b}'.format(i, width=qubits.size)[::-1], 2)
    #    list_int.append(reversed_i)
    #    list_int_lsb.append(i)
    #    states.append("|"+ bin(i)[2:].zfill(qubits.size)+">")

    #probability = np.zeros(2**qubits.size)
    #amplitude = np.zeros(2**qubits.size, dtype=np.complex_)
    #for samples in result:
    #    probability[samples.state.lsb_int] = samples.probability
    #    amplitude[samples.state.lsb_int] = samples.amplitude

    #pdf = pd.DataFrame({
    #    'States': states,
    #    'Int_lsb': list_int_lsb,
    #    'Probability': probability,
    #    'Amplitude': amplitude,
    #    'Int': list_int
    #})

    return pdf, circuit, q_prog, job

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

def create_qcircuit(prog_q):
    """
    Given a QLM program creates a QLM circuit
    """
    q_prog = deepcopy(prog_q)
    circuit = q_prog.to_circ(submatrices_only=True)
    return circuit

def create_qjob(circuit, shots=0, qubits=None):
    """
    Given a QLM circuit creates a QLM job
    """
    dict_job = {
        'amp_threshold': 0.0
    }
    if qubits is None:
        job = circuit.to_job(nbshots=shots, **dict_job)
    else:
        if isinstance(qubits, (np.ndarray, list)):
            job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
        else:
            raise ValueError('qbits: sould be a list!!!')
    return job

def proccess_qresults(result, qubits):
    """
    PostProcces a QLM results for creating a pandas DataFrame
    """

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
    return pdf
