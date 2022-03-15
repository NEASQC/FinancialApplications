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
import copy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.comm.datamodel.ttypes import OpType
from utils import run_job, postprocess_results
from data_extracting import create_qprogram, create_circuit, create_job
from amplitude_amplification import load_qn_gate, load_q_gate

def get_qpu(qlmass=True):
    """
    Function for selecting solver. User can choose between QLM QPU in CESGA
    or using QLM simulator PyLinalg

    Parameters
    ----------
    
    qlmass : Bool
        If True function will try to use QLM as a Service.
        If False fucntion will invoque PyLinalg QLM simulator
        

    Returns
    ----------
    linalg_qpu : simulator used for solvinf QLM circuits

    """
    if qlmass:
        try:
            from qat.qlmaas import QLMaaSConnection
            connection = QLMaaSConnection()
            LinAlg = connection.get_qpu("qat.qpus:LinAlg")
            linalg_qpu = LinAlg()
        except (ImportError, OSError) as e:
            print('Problem: usin PyLinalg')
            from qat.qpus import PyLinalg
            linalg_qpu = PyLinalg()
    else:
        print('User Forces: PyLinalg')
        from qat.qpus import PyLinalg
        linalg_qpu = PyLinalg()
    return linalg_qpu

def im_postprocess(result):
    """
    Post Proccess intermediate measurements from a qlm result.

    Parameters
    ----------

    result : qlm result
        string from a qlm simulation

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
    bit_list = []
    prob_list = []
    for i, im in enumerate(result.intermediate_measurements):
        if i%2 == 1:
            bit_list.append(str(int(im.cbits[0])))
            prob_list.append(im.probability)

    #Needed order the bits
    bit_list.reverse()
    prob_list.reverse()

    bit_string = ''.join(bit_list)
    bit_int = int(bit_string, 2)
    phi = bit_int/(2**len(bit_list))

    pdf = pd.DataFrame({
        #'Probs': [prob_list],
        'BitString': [bit_string],
        'BitInt': [bit_int],
        'Phi': [phi],
        'Probability': [np.prod(prob_list)],

    })
    return pdf

def get_probability(bit, clasical_bits):
    """
    Calculates the probability of a string of bits based on the
    probabilities for each individual bit

    Parameters
    ----------

    bit : str
        strign of bits that represent an integer number
    clasical_bits : list
        it contains for each position a bolean value and the probability for it
        len(clasical_bits) == len(bit)
        classica_bits[i][0] : bolean value
        classica_bits[i][1] : probability of correspondient bolean value

    Returns
    ----------

    total_probability : float
        Probability of getting input bit having the
        probability configuration of clasical_bits

    """
    p_ = []
    for i, b_ in enumerate(bit):
        #print(i, b_)

        if clasical_bits[i][0] == bool(int(b_)):
            #print('cierto')
            p_.append(clasical_bits[i][1])
        else:
            #print('false')
            p_.append(1.0-clasical_bits[i][1])
        #print(p)
    total_probability = np.prod(p_)
    return total_probability


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
    #print('VERSION GONZALO!!')

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

    #for i in range(unitary_applications):
    #    q_prog.apply(q_gate.ctrl(), q_aux, q_bits)

    for j in range(m-l+1, m+1, 1):
        theta = 2**(m-l-j+1)
        #print('\t j: {}. theta: {}'.format(j-1, theta))
        q_prog.cc_apply(c_bits[j-1], qlm.PH(-(np.pi/2.0)*theta), q_aux)
    #print('m: {}. l: {}'.format(m, l))
    q_prog.apply(qlm.H, q_aux)
    #print(m-l-1)
    q_prog.measure(q_aux, c_bits[m-l-1])
    return q_prog

def step_iqpe_easy(q_prog, q_gate, q_aux, c_bits, l):
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
    #print('unitary_applications: {}'.format(unitary_applications))
    step_q_gate = load_qn_gate(q_gate, unitary_applications)
    q_prog.apply(step_q_gate.ctrl(), q_aux, q_bits)

    #for i in range(unitary_applications):
    #    q_prog.apply(q_gate.ctrl(), q_aux, q_bits)

    for j in range(l):
        theta = 1.0/(2**(l-j-1))
        #print('\t j: {}. theta: {}'.format(j, theta))
        q_prog.cc_apply(c_bits[j], qlm.PH(-(np.pi/2.0)*theta), q_aux)

    #print('m: {}. l: {}'.format(m, l))
    q_prog.apply(qlm.H, q_aux)
    #print(m-l-1)
    q_prog.measure(q_aux, c_bits[l])
    return q_prog


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
            oracle: QLM gate
                QLM gate with the Oracle for implementing the
                Groover-like operator:
                init_q_prog and q_gate will be interpreted as None
            initial_state : QLM Program
                QLM Program withe the initial Psi state over the
                Grover-like operator will be applied
                Only used if oracle is None
            grover : QLM gate or routine
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
        self.oracle = kwargs.get('oracle', None)
        if self.oracle is not None:
            #Creates QLM program from base gate
            self.init_q_prog = create_qprogram(self.oracle)
            #Creates the Grover-like operator from oracle
            self.q_gate = load_q_gate(self.oracle)
        else:
            #In this case we load directly the initial state
            #and the grover operator
            self.init_q_prog = kwargs.get('initial_state', None)
            self.q_gate = kwargs.get('grover', None)
            if (self.init_q_prog is None) or (self.q_gate is None):
                text = """If oracle was not provided initial_state and grover
                keys should be provided"""
                raise KeyError(text)

        #Number Of classical bits for estimating phase
        self.cbits_number_ = kwargs.get('cbits_number', 8)
        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu', get_qpu())
        self.shots = kwargs.get('shots', 0)
        self.restart()
        self.easy = kwargs.get('easy', False)


    def restart(self):
        """
        Reinitialize several properties for restart purpouses
        """
        self.q_prog = None
        self.q_aux = None
        self.c_bits = None
        self.circuit = None
        self.meas_gates = None
        self.job = None
        self.job_result = None
        self.results = None
        self.final_results = None

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
        self.q_prog = copy.deepcopy(self.init_q_prog)
        self.q_aux = self.q_prog.qalloc(1)
        self.c_bits = self.q_prog.calloc(self.cbits_number)

    def apply_iqpe(self):
        """
        Apply a complete IQPE algorithm
        """
        for l in range(len(self.c_bits)):
            if self.easy:
                step_iqpe_easy(self.q_prog, self.q_gate, self.q_aux, self.c_bits, l)
            else:
                step_iqpe(self.q_prog, self.q_gate, self.q_aux, self.c_bits, l)

    def get_circuit(self):
        """
        Creates the quantum circuit from the q_prog property
        """
        #self.circuit = self.q_prog.to_circ(submatrices_only=True)
        self.circuit = create_circuit(self.q_prog)
        self.meas_gates = [i for i, o in enumerate(self.circuit.ops)\
            if o.type == OpType.MEASURE]

    def get_job(self):
        """
        Creates the job from the circuit property
        """
        #self.job = self.circuit.to_job(
        #    qubits=self.q_aux,
        #    nbshots=self.shots,
        #    aggregate_data=False
        #)
        self.job = create_job(
            self.circuit,
            shots=self.shots,
            qubits=[self.q_aux]
        )
        self.job.aggregate_data = False

    def get_job_result(self, qpu=None):
        """
        Submit a job property to QPU and get the results
        """
        if qpu is not None:
            self.linalg_qpu = qpu
        self.job_result = run_job(self.linalg_qpu.submit(self.job))
    
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
        self.get_circuit()
        self.get_job()
        self.get_job_result()
        self.get_classicalbits()
        self.proccess_output()


    def get_classicalbits(self):
        """
        This method gets the classical bits and their probabilities from
        the job_result property
        """

        list_of_results = [im_postprocess(r) for r in self.job_result]
        self.results = pd.concat(list_of_results)
        self.results.reset_index(drop=True, inplace=True)

    def proccess_output(self):
        """
        This function uses the results property and add it additional
        columns that are useful for Amplitude Amplification procedure
        """
        self.final_results = self.results.copy(deep=True)
        #Eigenvalue of the Grover-like operator
        self.final_results['Theta_Unitary'] = 2*np.pi*self.final_results['Phi']
        #Rotation angle for Grover-like operator.
        self.final_results['Theta'] = np.pi*self.final_results['Phi']
        #Only angles between 0 an pi
        self.final_results['theta_90'] = self.final_results['Theta']
        self.final_results['theta_90'].where(
            self.final_results['theta_90'] < 0.5*np.pi,
            np.pi-self.final_results['theta_90'],
            inplace=True)
        #Expected value of the function f(x) when x follows a p(x)
        #distribution probability
        self.final_results['E_p(f)'] = np.cos(self.final_results['Theta'])**2


