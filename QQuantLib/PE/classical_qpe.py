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
from copy import deepcopy
import numpy as np
import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from QQuantLib.utils.data_extracting import create_qprogram
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
        self.linalg_qpu = kwargs.get("qpu", None)  # , get_qpu())
        if self.linalg_qpu is None:
            print("Not QPU was provide. Default QPU will be used")
            self.linalg_qpu = get_default_qpu()
        self.shots = kwargs.get("shots", 10)
        self.complete = kwargs.get("complete", False)

        # Attributes not given as input
        self.q_prog = None
        self.q_aux = None
        self.final_results = None
        self.sumary = None

        self.circuit = None
        self.results = None
        self.time_pdf = None
        self.time_qpe_post_procces = None

    def restart(self):
        """
        Reinitialize several properties for restart purposes
        """
        self.q_prog = None
        self.q_aux = None
        self.final_results = None
        self.circuit = None
        self.results = None

    def init_pe(self):
        """
        Initialize several properties
        """
        self.restart()
        # Create quantum program based on initial state
        self.q_prog = create_qprogram(deepcopy(self.initial_state))
        self.q_aux = self.q_prog.qalloc(self.auxiliar_qbits_number)

    def pe_qft(self):
        """
        This method apply a workflow for executing a complete PE with QFT
        algorithm

        """
        # Initialize program
        self.init_pe()
        # Create algorithm
        self.q_prog = self.apply_pe_wqft(self.q_prog, self.q_gate, self.q_aux)
        # Execute algorithm
        self.results, self.circuit = self.run_qprogram(
            self.q_prog, self.q_aux, self.shots, self.linalg_qpu,
            self.complete
        )
        # Post-Process results
        start = time.time()
        self.final_results = self.post_proccess(self.results)
        end = time.time()
        self.time_qpe_post_procces = end - start

    @staticmethod
    def apply_controlled_operations(q_prog_, q_gate, q_aux):
        """
        This functions creates the first part of the Phase Estimation
        algorithm with QFT. Given a initial state, a unitary operator
        and group of auxiliary bits following steps are done:
        1. Applies a Haddamard Gate to each auxiliary qubit.
        2. Each auxiliary qubit controlls an exponential application
        of the given operator on the principal quantum state.

        Parameters
        ----------

        q_prog_ : QLM program
            QLM Program where the unitary operator will be applied
        q_gate : QLM AbstractGate
            QLM implementation of the unitary operator. We want estimate
            the autovalue theta of this operator
        q_aux : QLM qbit
            auxiliary bits for PE. Each qubit will be the control for
            application of the unitary operator (powers of it in fact)
            to the initial state

        Returns
        ----------

        q_prog : QLM program

        """

        # Getting the principal bits
        q_prog = deepcopy(q_prog_)
        q_bits = q_prog.registers[0]
        for i, aux in enumerate(q_aux):
            q_prog.apply(qlm.H, aux)
            step_q_gate = load_qn_gate(q_gate, 2**i)
            q_prog.apply(step_q_gate.ctrl(), aux, q_bits)

        return q_prog

    @staticmethod
    def apply_inv_qft(q_prog_, q_aux):
        """
        Apply an inverse of Quantum Fourier Transformation to the
        desired bits of a QLM program

        Parameters
        ----------

        q_prog : QLM program
            QLM Program where the unitary operator will be applied
        q_aux : QLM qubit
            bits where the inverse of the QFT will be applied

        Returns
        ----------

        q_prog : QLM program

        """
        q_prog = deepcopy(q_prog_)
        q_prog.apply(qlm.qftarith.QFT(len(q_aux)).dag(), q_aux)
        return q_prog

    @staticmethod
    def apply_pe_wqft(q_prog_, q_gate, q_aux):
        """
        This function apply a complete Phase Estimation with QFT algorithm

        Parameters
        ----------

        q_prog_ : QLM program
            QLM Program where the unitary operator will be applied
        q_gate : QLM AbstractGate
            QLM implementation of the unitary operator. We want estimate
            the autovalue theta of this operator
        q_aux : QLM qbit
            auxiliary bits for PE.

        Returns
        ----------

        q_prog : QLM program

        """
        q_prog = deepcopy(q_prog_)
        q_prog = CQPE.apply_controlled_operations(q_prog, q_gate, q_aux)
        q_prog = CQPE.apply_inv_qft(q_prog, q_aux)
        return q_prog

    @staticmethod
    def run_qprogram(q_prog, q_aux, shots, linalg_qpu, complete=False):
        """
        Executes a complete simulation

        Parameters
        ----------

        q_prog : QLM Program
        q_aux : QLM qbit
            auxiliary qubit for measuring during all ipe steps
        shots : int
            number of shots for simulation
        linalg_qpu : QLM solver
        complete : bool
            For returning the complete state space in the results file

        Returns
        ----------

        result : pandas DataFrame
            DataFrame with the results
        circuit : QLM circuit

        """
        start = q_aux.start
        lenght = q_aux.length
        result, circuit, q_prog, job = get_results(
            q_prog,
            linalg_qpu=linalg_qpu,
            shots=shots,
            qubits=list(range(start, start + lenght, 1)),
            complete=complete
        )
        del result["Amplitude"]
        result["Phi"] = result["Int"] / (2**lenght)
        return result, circuit

    @staticmethod
    def post_proccess(input_pdf):
        """
        This function uses the results property and add it additional
        columns that are useful for Amplitude Amplification procedure

        Returns
        ----------

        final_result : pandas DataFrame
            DataFrame with the final results
        circuit : QLM circuit
        """
        final_results = input_pdf.copy(deep=True)
        # Eigenvalue of the Grover-like operator
        final_results["2*theta"] = 2 * np.pi * final_results["Phi"]
        # Rotation angle for Grover-like operator.
        final_results["theta"] = np.pi * final_results["Phi"]
        # Only angles between 0 an pi
        final_results["theta_90"] = final_results["theta"]
        final_results["theta_90"].where(
            final_results["theta_90"] < 0.5 * np.pi,
            np.pi - final_results["theta_90"],
            inplace=True,
        )
        # Expected value of the function f(x) when x follows a p(x)
        # distribution probability
        # final_results['E_p(f)'] = np.sin(final_results['Theta'])**2
        return final_results
