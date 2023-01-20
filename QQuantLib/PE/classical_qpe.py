"""
This module contains necessary functions and classes to implement
the classical Quantum Phase Estimation with inverse of the
Quantum Fourier Transform. Following references were used:

    Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
    Quantum amplitude amplification and estimation.
    AMS Contemporary Mathematics Series, 305.
    https://arxiv.org/abs/quant-ph/0005055v1

    NEASQC deliverable: D5.1: Review of state-of-the-art for Pricing
    and Computation of VaR

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
import numpy as np
import qat.lang.AQASM as qlm
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.utils.utils import load_qn_gate
from QQuantLib.utils.data_extracting import get_results

class CQPE:
    """
    Class for using classical Quantum Phase Estimation, with inverse of
    Quantum Fourier Transformation.

    Parameters
    ----------

    kwars : dictionary
        dictionary that allows the configuration of the CQPE algorithm: \\
        Implemented keys:

        initial_state : QLM Program
            QLM Program with the initial Psi state over the
            Grover-like operator will be applied
            Only used if oracle is None
        unitary_operator : QLM gate or routine
            Grover-like operator which autovalues want to be calculated
            Only used if oracle is None
        cbits_number : int
            number of classical bits for phase estimation
        qpu : QLM solver
            solver for simulating the resulting circuits
        shots : int
            number of shots for quantum job. If 0 exact probabilities
            will be computed.
    """

    def __init__(self, **kwargs):
        """
        Method for initializing the class
        """

        # Setting attributes
        # In this case we load directly the initial state
        # and the grover operator
        self.initial_state = kwargs.get("initial_state", None)
        self.q_gate = kwargs.get("unitary_operator", None)
        if (self.initial_state is None) or (self.q_gate is None):
            text = "initial_state and grover keys should be provided"
            raise KeyError(text)

        # Number Of classical bits for estimating phase
        self.auxiliar_qbits_number = kwargs.get("auxiliar_qbits_number", 8)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            print("Not QPU was provide. PyLinalg will be used")
            self.linalg_qpu = get_qpu("python")

        self.shots = kwargs.get("shots", 10)
        self.complete = kwargs.get("complete", False)

        #Quantum Routine for QPE
        #Auxiliar qbits
        self.q_aux = None
        #Qubits rtegisters
        self.registers = None
        #List of ints with the position of the qbits for measuring
        self.meas_qbits = None
        #For storing results
        self.result = None
        #For storing qunatum times
        self.quantum_times = []
        #For storing the QPE routine
        self.circuit = None

    def run(self):
        """
        Creates the quantum phase estimation routine
        """
        qpe_routine = qlm.QRoutine()
        #Creates the qbits foe applying the operations
        self.registers = qpe_routine.new_wires(self.initial_state.arity)
        #Initializate the registers
        qpe_routine.apply(self.initial_state, self.registers)
        #Creates the auxiliary qbits for phase estimation
        self.q_aux = qpe_routine.new_wires(self.auxiliar_qbits_number)
        #Apply controlled Operator an increasing number of times
        for i, aux in enumerate(self.q_aux):
            #Apply Haddamard to all auxiliary qbits
            qpe_routine.apply(qlm.H, aux)
            #Power of the unitary operator depending of the position
            #of the auxiliary qbit.
            step_q_gate = load_qn_gate(self.q_gate, 2**i)
            #Controlled application of power of unitary operator
            qpe_routine.apply(step_q_gate.ctrl(), aux, self.registers)
        #Apply the QFT
        qpe_routine.apply(qlm.qftarith.QFT(len(self.q_aux)).dag(), self.q_aux)
        self.circuit = qpe_routine

        start = time.time()
        #Getting the result
        self.meas_qbits = [
            len(self.registers) + i for i, aux in enumerate(self.q_aux)]
        self.result, _, _, _ = get_results(
            self.circuit,
            linalg_qpu=self.linalg_qpu,
            shots=self.shots,
            qubits=self.meas_qbits,
            complete=self.complete
        )
        end = time.time()
        self.quantum_times.append(end-start)
        del self.result["Amplitude"]
        self.result["lambda"] = self.result["Int"] / (2**len(self.q_aux))


    #@staticmethod
    #def post_proccess(input_pdf):
    #    """
    #    This function uses the results property and add it additional
    #    columns that are useful for Amplitude Amplification procedure

    #    Returns
    #    ----------

    #    final_result : pandas DataFrame
    #        DataFrame with the final results
    #    circuit : QLM circuit
    #    """
    #    final_results = input_pdf.copy(deep=True)
    #    # Eigenvalue of the Grover-like operator
    #    final_results["2*theta"] = 2 * np.pi * final_results["lambda"]
    #    # Rotation angle for Grover-like operator.
    #    final_results["theta"] = np.pi * final_results["lambda"]
    #    # Only angles between 0 an pi
    #    final_results["theta_90"] = final_results["theta"]
    #    final_results["theta_90"].where(
    #        final_results["theta_90"] < 0.5 * np.pi,
    #        np.pi - final_results["theta_90"],
    #        inplace=True,
    #    )
    #    # Expected value of the function f(x) when x follows a p(x)
    #    # distribution probability
    #    # final_results['E_p(f)'] = np.sin(final_results['Theta'])**2
    #    return final_results
