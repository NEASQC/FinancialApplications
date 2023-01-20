"""
This module contains auxiliary functions for executing QLM programs based
on QLM Routines or QLM gates and for post processing results from QLM
qpu executions

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

import time
from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.core import Result
from QQuantLib.utils.utils import check_list_type

pd.options.display.float_format = "{:.6f}".format
np.set_printoptions(suppress=True)


def get_results(
    quantum_object,
    linalg_qpu,
    shots: int = 0,
    qubits: list = None,
    complete: bool = False
):
    """
    Function for testing an input gate. This function creates the
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
        list with the qubits for doing the measurement when simulating
        if None measurement over all allocated qubits will be provided
    complete : bool
        for return the complete basis state. Useful when shots is not 0
        and all the posible basis states are necessary.

    Returns
    ----------
    pdf : pandas DataFrame
        DataFrame with the results of the simulation
    circuit : QLM circuit
    q_prog : QLM Program.
    job : QLM job
    """
    #print("BE AWARE!! linalg_qpu : {}".format(linalg_qpu))
    # if type(quantum_object) == qlm.Program:
    if isinstance(quantum_object, qlm.Program):
        q_prog = deepcopy(quantum_object)
        arity = q_prog.qbit_count
    else:
        q_prog = create_qprogram(quantum_object)
        arity = quantum_object.arity

    if qubits is None:
        qubits = np.arange(arity, dtype=int)
    else:
        qubits = check_list_type(qubits, int)
    # circuit = q_prog.to_circ(submatrices_only=True)
    start = time.time()
    circuit = create_qcircuit(q_prog)
    end = time.time()
    time_q_circuit = end - start

    start = time.time()
    # job = circuit.to_job(nbshots=shots, qubits=qubits)
    job = create_qjob(circuit, shots=shots, qubits=qubits)
    end = time.time()
    time_q_job = end - start

    start = time.time()
    result = linalg_qpu.submit(job)
    if not isinstance(result, Result):
        result = result.join()
        # time_q_run = float(result.meta_data["simulation_time"])
        qpu_type = "QLM_QPU"
    else:
        qpu_type = "No QLM_QPU"
    end = time.time()
    time_q_run = end - start
    # Process the results
    start = time.time()
    pdf = proccess_qresults(result, qubits, complete=complete)
    end = time.time()
    time_post_proccess = end - start

    time_dict = {
        "time_q_circuit": time_q_circuit,
        "time_q_job": time_q_job,
        "time_q_run": time_q_run,
        "time_post_proccess": time_post_proccess,
    }
    pdf_time = pd.DataFrame([time_dict])
    pdf_time["time_total"] = pdf_time.sum(axis=1)
    pdf_time["qpu_type"] = qpu_type

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

    Parameters
    ----------

    prog_q : QLM QProgram

    Returns
    ----------

    circuit : QLM circuit
    """
    q_prog = deepcopy(prog_q)
    circuit = q_prog.to_circ(submatrices_only=True)#, inline=True)
    return circuit


def create_qjob(circuit, shots=0, qubits=None):
    """
    Given a QLM circuit creates a QLM job

    Parameters
    ----------

    circuit : QLM circuit
    shots : int
        number of measurmentes
    qubits : list
        with the qubits to be measured

    Returns
    ----------

    job : QLM job
        job for submit to QLM QPU
    """
    dict_job = {"amp_threshold": 0.0}
    if qubits is None:
        job = circuit.to_job(nbshots=shots, **dict_job)
    else:
        if isinstance(qubits, (np.ndarray, list)):
            job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
        else:
            raise ValueError("qbits: sould be a list!!!")
    return job


def proccess_qresults(result, qubits, complete=False):
    """
    Post Process a QLM results for creating a pandas DataFrame

    Parameters
    ----------

    result : QLM results from a QLM qpu.
        returned object from a qpu submit
    qubits : int
        number of qubits
    complete : bool
        for return the complete basis state.
    """

    # Process the results
    if complete:
        states = []
        list_int = []
        list_int_lsb = []
        for i in range(2**qubits.size):
            reversed_i = int("{:0{width}b}".format(i, width=qubits.size)[::-1], 2)
            list_int.append(reversed_i)
            list_int_lsb.append(i)
            states.append("|" + bin(i)[2:].zfill(qubits.size) + ">")

        probability = np.zeros(2**qubits.size)
        amplitude = np.zeros(2**qubits.size, dtype=np.complex_)
        for samples in result:
            probability[samples.state.lsb_int] = samples.probability
            amplitude[samples.state.lsb_int] = samples.amplitude

        pdf = pd.DataFrame(
            {
                "States": states,
                "Int_lsb": list_int_lsb,
                "Probability": probability,
                "Amplitude": amplitude,
                "Int": list_int,
            }
        )
    else:
        list_for_results = []
        for sample in result:
            list_for_results.append([
                sample.state, sample.state.lsb_int, sample.probability,
                sample.amplitude, sample.state.int,
            ])

        pdf = pd.DataFrame(
            list_for_results,
            columns=['States', "Int_lsb", "Probability", "Amplitude", "Int"]
        )
        pdf.sort_values(["Int_lsb"], inplace=True)
    return pdf
