"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains auxiliar functions for executing QLM programs based
on QLM QRoutines or QLM gates and for postproccessing results from QLM
qpu executions

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

import numpy as np
import qat.lang.AQASM as qlm
from qat.core import Result
from copy import deepcopy
from utils import run_job, postprocess_results

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

def create_circuit(prog_q):
    """
    Given a QLM program creates a QLM circuit
    """
    q_prog = deepcopy(prog_q)
    circuit = q_prog.to_circ(submatrices_only=True)
    return circuit

def create_job(circuit, shots=0, qubits=None):
    """
    Given a QLM circuit creates a QLM job
    """
    dict_job = {
        'amp_threshold': 0.0
    }
    if qubits is None:
        job = circuit.to_job(nbshots=shots, **dict_job)
    else:
        if isinstance(qubits, (list)):
            job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
        else:
            raise ValueError('qbits: sould be a list!!!')
    return job

def get_results(quantum_object, linalg_qpu, shots=0, qubits=None):
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
    pdf_ : pandas DataFrame
        DataFrame with the results of the simulation
    circuit : QLM circuit
    q_prog : QLM Program.
    job : QLM job

    """
    if type(quantum_object) == qlm.Program:
        q_prog = deepcopy(quantum_object)
    else:
        q_prog = qlm.Program()
        qbits = q_prog.qalloc(quantum_object.arity)
        q_prog.apply(quantum_object, qbits)
    #circuit = q_prog.to_circ(submatrices_only=True)
    circuit = create_circuit(q_prog)
    #dict_job = {
    #    'amp_threshold': 0.0
    #}
    #if qubits is None:
    #    job = circuit.to_job(nbshots=shots, **dict_job)
    #if type(qubits) == list:
    #    job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
    job = create_job(circuit, shots=shots, qubits=qubits)
    result = run_job(linalg_qpu.submit(job))
    pdf_ = postprocess_results(result)
    #pdf_.sort_values('Int_lsb', inplace=True)
    return pdf_, circuit, q_prog, job








