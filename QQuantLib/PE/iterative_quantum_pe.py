"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains necesary functions and classes to implement
Iterative Quantum Phase Estimation (IQPE). The implementation is based on
following paper:

    Dobšíček, Miroslav and Johansson, Göran and Shumeiko, Vitaly and
    Wendin, Göran*.
    Arbitrary accuracy iterative quantum phase estimation algorithm
    using a single ancillary qubit: A two-qubit benchmark.
    Physical Review A 3(76), 2007.
    https://arxiv.org/abs/quant-ph/0610214

Author:Gonzalo Ferro Costas

"""

from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.core import Result
from libraries.utils.utils import create_qprogram, create_job, load_qn_gate,\
create_circuit
from libraries.utils.qlm_solver import get_qpu


class IterativeQuantumPE:
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
                QLM Program withe the initial Psi state over the
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
            easy : bool
                If True step_iqpe_easy will be used for each step of the
                algorithm
                If False step_iqpe will be used for each step.
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
        self.cbits_number_ = kwargs.get('cbits_number', 8)
        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu', None)#, get_qpu())
        if self.linalg_qpu is None:
            self.linalg_qpu = get_qpu()
        self.shots = kwargs.get('shots', 10)
        self.zalo = kwargs.get('zalo', False)

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
        self.sumary = None


        self.circuit = None
        self.job = None

    @property
    def cbits_number(self):
        return self.cbits_number_

    @cbits_number.setter
    def cbits_number(self, value):
        print('The number of classical bits for phase estimation will be:'\
            '{}'.format(value))
        self.cbits_number_ = value
        #We update the allocate classical bits each time we change cbits_number


    def init_iqpe(self):
        """
        Initialize several properties
        """
        self.restart()
        #Create quantum program based on initial state
        self.q_prog = create_qprogram(deepcopy(self.initial_state))
        self.q_aux = self.q_prog.qalloc(1)
        self.c_bits = self.q_prog.calloc(self.cbits_number)

    def apply_iqpe(self):
        """
        Apply a complete IQPE algorithm
        """
        for l in range(len(self.c_bits)):
            if self.zalo:
                self.q_prog = self.step_iqpe_zalo(
                    self.q_prog,
                    self.q_gate,
                    self.q_aux,
                    self.c_bits,
                    l
                )
            else:
                self.q_prog = self.step_iqpe(
                    self.q_prog,
                    self.q_gate,
                    self.q_aux,
                    self.c_bits,
                    l
                )

    def iqpe(self, number_of_cbits=None, shots=None):
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

        self.init_iqpe()
        self.apply_iqpe()
        results = self.run(
            self.q_prog,
            self.q_aux,
            self.shots,
            self.linalg_qpu
        )
        self.classical_bits = self.meas_classical_bits(results)
        self.final_results = self.post_proccess(self.classical_bits)
        self.sumary = self.sumarize(self.final_results)

    @staticmethod
    def step_iqpe_zalo(q_prog, q_gate, q_aux, c_bits, l):
        """
        Implements a iterative step of the Iterative Phase Estimation (IPE)
        algorithm.

        Parameters
        ----------

        q_prog : QLM program
            QLM Program where the unitary operator will be applied
        q_gate : QLM AbstractGate
            QLM implementation of the unitary operator. We want estimate
            the autovalue theta of this operator
        q_aux : QLM qbit
            auxiliar qbit for IPE. This qbit will be the control
            for application of the unitary operator to the principal qbits
            of the program. Aditionally will be the target qbit for the
            classical bit controlled rotation. This qbit will be reset at
            the end of the step.
        c_bits : list
            list with the classical bits allocated for phase estimation
        l : int
            iteration step of the IPE algorithm
        """
        print('VERSION GONZALO!!')
        q_prog.reset(q_aux)
        #Getting the principal qbits
        q_bits = q_prog.registers[0]
        #First apply a Haddamard Gate to auxiliar qbit
        q_prog.apply(qlm.H, q_aux)
        #number of bits for codify phase
        m = len(c_bits)

        #Number of controlled application of the unitary operator by auxiliar
        #qbit over the principal qbits
        unitary_applications = int(2**(m-l-1))
        #print('unitary_applications: {}'.format(unitary_applications))
        step_q_gate = load_qn_gate(q_gate, unitary_applications)
        q_prog.apply(step_q_gate.ctrl(), q_aux, q_bits)

        for j in range(m-l+1, m+1, 1):
            theta = 2**(m-l-j+1)
            #print('\t j: {}. theta: {}'.format(j-1, theta))
            q_prog.cc_apply(c_bits[j-1], qlm.PH(-(np.pi/2.0)*theta), q_aux)
        #print('m: {}. l: {}'.format(m, l))
        q_prog.apply(qlm.H, q_aux)
        #print(m-l-1)
        q_prog.measure(q_aux, c_bits[m-l-1])
        return q_prog

    @staticmethod
    def step_iqpe(q_prog, q_gate, q_aux, c_bits, l):
        """
        Implements a iterative step of the Iterative Phase Estimation (IPE)
        algorithm.

        Parameters
        ----------

        q_prog : QLM program
            QLM Program where the unitary operator will be applied
        q_gate : QLM AbstractGate
            QLM implementation of the unitary operator. We want estimate
            the autovalue theta of this operator
        q_aux : QLM qbit
            auxiliar qbit for IPE. This qbit will be the control
            for application of the unitary operator to the principal qbits
            of the program. Aditionally will be the target qbit for the
            classical bit controlled rotation. This qbit will be reset at
            the end of the step.
        c_bits : list
            list with the classical bits allocated for phase estimation
        l : int
            iteration step of the IPE algorithm

        """

        #print('VERSION EASY!!')
        q_prog.reset(q_aux)
        #Getting the principal qbits
        q_bits = q_prog.registers[0]
        #First apply a Haddamard Gate to auxiliar qbit
        q_prog.apply(qlm.H, q_aux)
        #number of bits for codify phase
        m = len(c_bits)

        #Number of controlled application of the unitary operator by auxiliar
        #qbit over the principal qbits
        unitary_applications = int(2**(m-l-1))
        step_q_gate = load_qn_gate(q_gate, unitary_applications)
        q_prog.apply(step_q_gate.ctrl(), q_aux, q_bits)
        for j in range(l):
            theta = 1.0/(2**(l-j-1))
            #print('\t j: {}. theta: {}'.format(j, theta))
            q_prog.cc_apply(c_bits[j], qlm.PH(-(np.pi/2.0)*theta), q_aux)
            #q_prog.cc_apply(c_bits[j], qlm.PH((np.pi/2.0)*theta), q_aux)

        #print('m: {}. l: {}'.format(m, l))
        q_prog.apply(qlm.H, q_aux)
        #print(m-l-1)
        q_prog.measure(q_aux, c_bits[l])
        return q_prog

    @staticmethod
    def run(q_prog, q_aux, shots, linalg_qpu):
        circuit = create_circuit(q_prog)
        job = create_job(
            circuit,
            shots=shots,
            qubits=[q_aux]
        )
        job.aggregate_data = False
        result = linalg_qpu.submit(job)
        if not isinstance(result, Result):
            result = result.join()
        return result

    @staticmethod
    def meas_classical_bits(result):
        """
        Post Proccess intermediate measurements from a qlm result.

        Parameters
        ----------

        result : list
            list with qlm results

        Returns
        ----------
        pdf : pandas DataFrame
            contains extracted information from intermediate_measurements
            from a qlm result. Columns:
            BitString : str. String with the bits of the measurements done
                during simulation of the circuit
            BitInt : int. Integer representation of the BitString
            Phi : float. Angle representation of the BitString between [0,1].
            Probability : float. Probability of the measurement of the
                classsical bits.
        """
        list_of_results = []

        for r in result:
            bit_list = []
            for i, im in enumerate(r.intermediate_measurements):
                if i%2 == 1:
                    bit_list.append(str(int(im.cbits[0])))

            #Needed order the bits
            bit_list.reverse()

            bit_string = ''.join(bit_list)
            bit_int = int(bit_string, 2)
            phi = bit_int/(2**len(bit_list))

            pdf = pd.DataFrame({
                #'Probs': [prob_list],
                'BitString': [bit_string],
                'BitInt': [bit_int],
                'Phi': [phi],
            })
            list_of_results.append(pdf)
        pdf_results = pd.concat(list_of_results)
        pdf_results.reset_index(drop=True, inplace=True)
        return pdf_results

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

    @staticmethod
    def sumarize(InputPDF, column=['theta_90']):
        pdf = InputPDF.copy(deep=True)
        pds = pdf.value_counts(column)
        pds.name = 'Frequency'
        return pd.DataFrame(pds).reset_index()
