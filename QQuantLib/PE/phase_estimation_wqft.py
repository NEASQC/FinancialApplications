"""

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.core import Result
from QQuantLib.utils.data_extracting import create_qprogram, create_qjob,\
create_qcircuit, proccess_qresults
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.utils.utils import load_qn_gate
from QQuantLib.utils.data_extracting import get_results


class PhaseEstimationwQFT:
    """
    Class for using Iterative Quantum Phase Estimation (IQPE) algorithm
    """

    def __init__(self, **kwargs):
        """

        Method for initializing the class

        Parameters
        ----------

        kwars : dictionary
            dictionary that allows the configuration of the ML-QPE algorithm:
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
                solver for simulating the resulting circutis
            shots : int
                number of shots for quantum job. If 0 exact probabilities
                will be computed.
        """

        #Setting attributes
        #In this case we load directly the initial state
        #and the grover operator
        self.initial_state = kwargs.get('initial_state', None)
        self.q_gate = kwargs.get('unitary_operator', None)
        if (self.initial_state is None) or (self.q_gate is None):
            text = "initial_state and grover keys should be provided"
            raise KeyError(text)

        #Number Of classical bits for estimating phase
        self.auxiliar_qbits_number_ = kwargs.get('auxiliar_qbits_number', 8)
        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu', None)#, get_qpu())
        if self.linalg_qpu is None:
            self.linalg_qpu = get_qpu()
        self.shots = kwargs.get('shots', 10)
        #self.zalo = kwargs.get('zalo', False)

        #Attributes not given as input
        self.q_prog = None
        self.q_aux = None
        self.c_bits = None
        self.classical_bits = None
        self.final_results = None
        self.sumary = None

        self.circuit = None
        self.job = None

    def restart(self):
        """
        Reinitialize several properties for restart purpouses
        """
        self.q_prog = None
        self.q_aux = None
        self.c_bits = None
        self.classical_bits = None
        self.final_results = None
        self.circuit = None
        self.results = None


        self.job = None

    @property
    def auxiliar_qbits_number(self):
        return self.auxiliar_qbits_number_

    @auxiliar_qbits_number.setter
    def auxiliar_qbits_number(self, value):
        print('The number of auxiliar qbits for phase estimation will be:'\
            '{}'.format(value))
        self.auxiliar_qbits_number_ = value
        #We update the allocate classical bits each time we change cbits_number


    def init_pe(self):
        """
        Initialize several properties
        """
        self.restart()
        #Create quantum program based on initial state
        self.q_prog = create_qprogram(deepcopy(self.initial_state))
        self.q_aux = self.q_prog.qalloc(self.auxiliar_qbits_number)

    def pe_wqft(self, number_of_cbits=None, shots=None):
        """
        This method apply a workflow for executing a complete IQPE
        algorithm

        Parameters
        ----------

        number_of_cbits : int (overwrite correspondient property)
            Number of classical bits for storing the phase estimation
        shots : int (overwrite correspondient property)
            Number of shots for executing the QLM job
        """

        if number_of_cbits is not None:
            self.cbits_number = number_of_cbits
        if shots is not None:
            self.shots = shots

        self.init_pe()
        self.q_prog = self.apply_pe_wqft(self.q_prog, self.q_gate, self.q_aux)
        self.results, self.circuit = self.run(
            self.q_prog,
            self.q_aux,
            self.shots,
            self.linalg_qpu
        )
        self.final_results = self.post_proccess(self.results)


    @staticmethod
    def apply_controlled_operations(q_prog_, q_gate, q_aux):
        """
        This functions creates the first part of the Phase Estimation
        algorithm with QFT. Given a initial state, a unitary operator
        and group of auxiliar qbits following steps are done:
        1. Applies a Haddamard Gate to each auxiliar qbit.
        2. Each auxiliar qbit controlles an exponential application
        of the given operator on the principal quantum state.

        Parameters
        ----------

        q_prog_ : QLM program
            QLM Program where the unitary operator will be applied
        q_gate : QLM AbstractGate
            QLM implementation of the unitary operator. We want estimate
            the autovalue theta of this operator
        q_aux : QLM qbit
            auxiliar qbits for PE. Each qbit will be the control for
            application of the unitary operator (powers of it in fact)
            to the initial state

        Returns
        ----------

        q_prog : QLM program

        """

        #Getting the principal qbits
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
        desired qbits of a QLM program

        Parameters
        ----------

        q_prog : QLM program
            QLM Program where the unitary operator will be applied
        q_aux : QLM qbit
            qbits where the inverse of the QFT will be applied

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
            auxiliar qbits for PE.

        Returns
        ----------

        q_prog : QLM program

        """
        q_prog = deepcopy(q_prog_)
        q_prog = PhaseEstimationwQFT.apply_controlled_operations(q_prog, q_gate, q_aux)
        q_prog = PhaseEstimationwQFT.apply_inv_qft(q_prog, q_aux)
        return q_prog

    @staticmethod
    def run(q_prog, q_aux, shots, linalg_qpu):
        """
        Executes a complete simulation

        Parameters
        ----------

        q_prog : QLM Program
        q_aux : QLM qbit
            auxiliar qbit for measuring during all ipe steps
        shots : int
            number of shots for simulation
        linalg_qpu : QLM solver

        Returns
        ----------

        result : QLM results
        circuit : QLM circuit

        """
        start = q_aux.start
        lenght = q_aux.length
        result, circuit, q_prog, job = get_results(
            q_prog, linalg_qpu=linalg_qpu, shots=shots,
            qubits=list(range(start, start+lenght, 1))
        )
        del result['Amplitude']
        result['Phi'] = result['Int']/(2**lenght)
        #circuit = create_qcircuit(q_prog)
        #if shots == 0:
        #    shots = 10
        #    print('Number of shots can not be 0. It will be used: ',shots)

        #job = create_qjob(
        #    circuit,
        #    shots=shots,
        #    qubits=[q_aux]
        #)
        #result = linalg_qpu.submit(job)
        #if not isinstance(result, Result):
        #    result = result.join()
        return result, circuit


    #@staticmethod
    #def meas_classical_bits(result):
    #    """
    #    Given a QLM aggregated result generate a DataFrame with the
    #    information of the inputs
    #    """
    #    fake_qbits = np.array([i for i in range(result.qregs[0].length)])
    #    pdf=proccess_qresults(result, fake_qbits)
    #    pdf['Phi'] = pdf['Int']/(2**len(fake_qbits))
    #    del pdf['Amplitude']
    #    return pdf

    @staticmethod
    def post_proccess(InputPDF):
        """
        This function uses the results property and add it additional
        columns that are useful for Amplitude Amplification procedure
        """
        final_results = InputPDF.copy(deep=True)
        #Eigenvalue of the Grover-like operator
        final_results['2*theta'] = 2*np.pi*final_results['Phi']
        #Rotation angle for Grover-like operator.
        final_results['theta'] = np.pi*final_results['Phi']
        #Only angles between 0 an pi
        final_results['theta_90'] = final_results['theta']
        final_results['theta_90'].where(
            final_results['theta_90'] < 0.5*np.pi,
            np.pi-final_results['theta_90'],
            inplace=True)
        #Expected value of the function f(x) when x follows a p(x)
        #distribution probability
        #final_results['E_p(f)'] = np.sin(final_results['Theta'])**2
        return final_results

