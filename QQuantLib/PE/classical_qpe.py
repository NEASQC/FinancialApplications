"""
This module contains the necessary functions and classes to implement
the classical Quantum Phase Estimation with the inverse of the
Quantum Fourier Transform. The following references were used:

    *Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
    Quantum amplitude amplification and estimation.
    AMS Contemporary Mathematics Series, 305.
    https://arxiv.org/abs/quant-ph/0005055v1*

    *NEASQC deliverable: D5.1: Review of state-of-the-art for Pricing
    and Computation of VaR*

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
import numpy as np
import qat.lang.AQASM as qlm
from QQuantLib.qpu.get_qpu import get_qpu
from QQuantLib.utils.utils import load_qn_gate
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.DL.data_loading import uniform_distribution
from QQuantLib.PE.windows_pe import window_selector
from qat.lang.AQASM.gates import ParamGate
from qat.lang.AQASM.routines import QRoutine


class CQPE:
    """
    Class for using classical Quantum Phase Estimation, with inverse of
    Quantum Fourier Transformation.

    Parameters
    ----------

    kwars : dictionary
        dictionary that allows the configuration of the CQPE algorithm: \\
        Implemented keys:

    initial_state : kwargs, QLM Program
        QLM Program with the initial Psi state over the
        Grover-like operator will be applied
        Only used if oracle is None
    unitary_operator : kwargs, QLM gate or routine
        Grover-like operator which autovalues want to be calculated
        Only used if oracle is None
    cbits_number : kwargs, int
        number of classical bits for phase estimation
    qpu : kwargs, QLM solver
        solver for simulating the resulting circuits
    shots : kwargs, int
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
        self.kwargs = kwargs
        self.initial_state = self.kwargs.get("initial_state", None)
        self.q_gate = self.kwargs.get("unitary_operator", None)
        if (self.initial_state is None) or (self.q_gate is None):
            text = "initial_state and grover keys should be provided"
            raise KeyError(text)

        # Number Of classical bits for estimating phase
        self.auxiliar_qbits_number = self.kwargs.get("auxiliar_qbits_number", None)
        if self.auxiliar_qbits_number is None:
            raise ValueError("Auxiliary number of qubits not provided")

        # Set the QPU to use
        self.linalg_qpu = self.kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            raise ValueError("Not QPU was provided")

        self.shots = self.kwargs.get("shots", None)
        if self.shots is None:
            print(
                "Be Aware: Not shots povided! \
                Exact simulation (shots=0) will be used"
            )
        self.complete = self.kwargs.get("complete", False)

        # Set the window function to use
        self.window = self.kwargs.get("window", None)
        # Change the sign in the last control
        if self.window is None:
            self.window_gate = uniform_distribution(self.auxiliar_qbits_number)
            self.last_control_change = False
        else:
            if type(self.window) in [ParamGate, QRoutine]:
                self.window_gate = self.window
                self.last_control_change = self.kwargs.get(
                    "last_control_change", None)
                if self.last_control_change is None:
                    raise ValueError(
                        "If you provide a window AbstractGate \
                        last_control_change key CAN NOT BE NONE"
                    )
            elif type(self.window) is str:
                self.window_gate, self.last_control_change = window_selector(
                    self.window, **self.kwargs
                )
            else:
                raise ValueError("Window kwarg not ParamGate neither QRoutine")


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
        # Apply the window function to auxiliary qubits
        # BE AWARE: the probability of the window function
        # Should be loaded taking as a domain the Int_lsb!!!
        qpe_routine.apply(self.window_gate, self.q_aux)
        #Apply controlled Operator an increasing number of times
        for i, aux in enumerate(self.q_aux[:-1]):
            #Apply Haddamard to all auxiliary qbits
            #qpe_routine.apply(qlm.H, aux)
            #Power of the unitary operator depending of the position
            #of the auxiliary qbit.
            step_q_gate = load_qn_gate(self.q_gate, 2**i)
            #Controlled application of power of unitary operator
            qpe_routine.apply(step_q_gate.ctrl(), aux, self.registers)
        # Las Control depends on the type of Window function applied
        if self.last_control_change:
            step_q_gate = load_qn_gate(
                self.q_gate.dag(),
                2**(self.auxiliar_qbits_number - 1)
            )
        else:
            step_q_gate = load_qn_gate(
                self.q_gate,
                2**(self.auxiliar_qbits_number - 1)
            )
        qpe_routine.apply(step_q_gate.ctrl(), self.q_aux[-1], self.registers)
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
        # Transform to lambda. BE AWARE we need to use Int column
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
