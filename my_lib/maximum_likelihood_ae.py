"""
Copyright 2022 CESGA
License:

This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains necesary functions and classes to implement
Maximum Likelihood Amplitude Estimation based on the paper:

    Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N.
    Amplitude estimation without phase estimation
    Quantum Information Processing, 19(2), 2020
    arXiv: quant-ph/1904.10246v2

Author:Gonzalo Ferro Costas

MyQLM version:

"""
import copy
import numpy as np
import pandas as pd
import scipy.optimize as so
from qat.core import Batch

from utils import run_job, postprocess_results
from data_extracting import create_qprogram, create_circuit, create_job
from amplitude_amplification import load_q_gate, load_qn_gate

def get_qpu(qlmass=True):
    """
    Create the lineal solver for quantum jobs

    Parameters
    ----------

    qlmass : bool
        If True  try to use QLM as a Service connection to CESGA QLM
        If False PyLinalg simulator will be used

    Returns
    ----------
    
    linalg_qpu : solver for quantum jobs
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

def apply_gate(q_prog, q_gate, m_k, nbshots=0):
    """
    Apply the self.q_gate to the circuit a input number of times
    This method creates a quantum program that applies the
    Q operator n_ops times on the circuit where the probability
    and the function were loaded

    Parameters
    ----------
    q_gate : QLM gate
        QLM gate with the Groover-like operator to be applied on the q_prog
    m_k : int
        number of times to apply the q_gate to the q_prog
    nbshots : int
        number of shots to perform by the QLM solver

    Returns
    ----------

    pdf : pandas DataFrame
        results of the measurement of the last qbit
    circuit : QLM circuit object
        circuit object generated for the quantum program
    job : QLM job object
        job object generated for the quantum circuit
    """
    prog_q = copy.deepcopy(q_prog)
    q_bits = prog_q.registers[0]
    #for _ in range(m_k):
    #    prog_q.apply(q_gate, q_bits)
    step_q_gate = load_qn_gate(q_gate, m_k)
    prog_q.apply(step_q_gate, q_bits)
    #circuit = prog_q.to_circ(submatrices_only=True)
    circuit = create_circuit(prog_q)
    #job = circuit.to_job(qubits=[len(q_bits)-1], nbshots=nbshots)
    job = create_job(circuit, shots=nbshots, qubits=[len(q_bits)-1])
    return circuit, job


def get_probabilities(InputPDF):
    """
    Auxiliar function for changing the presentation of the results for an
    input pandas DataFrame

    Parameters
    ----------

    InputPDF : pandas DataFrame.
        DataFrame with the info of the measurments. Should have following
        columns:
        States : states for the qbit measurement
        Probability : probability for each state

    Returns
    ----------

    output_pdf : pandas DataFrame
        Changes the presentation of the results of a mesurement.
        Columns are now the  probability of the different states.
    """

    pdf = InputPDF.copy(deep=True)
    columns = ['Probability_{}'.format(i) for i in pdf['States']]
    output_pdf = pd.DataFrame(
        pdf['Probability'].values.reshape(1, len(pdf)),
        columns=columns
    )
    return output_pdf

def likelihood(theta, m_k, h_k, n_k):
    """
    Calculates Likelihood from Suzuki papper. For h_k positive events
    of n_k total events this function calculates the probability of
    this taking into account that the probability of a positive
    event is given by theta and by m_k
    The idea is use this function to minimize it for this reason it gives
    minus Likelihood

    Parameters
    ----------

    theta : float
        Angle (radians) for calculating the probability of measure a
        positive event.
    m_k : pandas Series
        For MLQAE this a pandas Series where each row is the number of
        times the operator Q was applied.
        We needed for calculating the probability of a positive event
        for eack posible m_k value: sin((2*m_k+1)theta)**2.
    h_k : pandas Series
        Pandas Series where each row is the number of positive events
        measured for each m_k
    n_k : pandas Series
        Pandas Series where each row is the number of total events
        measured for each m_k

    Returns
    ----------

    float
        Gives the -Likelihood of the inputs

    """
    theta_ = (2*m_k+1)*theta
    first_term = 2*h_k*np.log(np.abs(np.sin(theta_)))
    second_term = 2*(n_k-h_k)*np.log(np.abs(np.cos(theta_)))
    l_k = first_term + second_term
    return -np.sum(l_k)

class MLAE:
    """
    Class for using Maximum Likelihood Quantum Amplitude Estimation (ML-AE)
    algorithm
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
            list_of_mks : list
                python list with the different m_ks for executing the algortihm
            nbshots : int
                number of shots for quantum job. If 0 exact probabilities
                will be computed.
            qpu : QLM solver
                solver for simulating the resulting circutis
            delta : float
                For avoiding problems when calculating the domain for theta
            default_nbshots : int
                default number of measurements for computing freqcuencies
                when nbshots for quantum job is 0
            iterations : int
                number of iterations of the optimizer
            display : bool
                for displaying additional information in the optimization step
            initial_state : QLM Program
                QLM Program withe the initial Psi state over the
                Grover-like operator will be applied
                Only used if oracle is None
            grover : QLM gate or routine
                Grover-like operator which autovalues want to be calculated
                Only used if oracle is None
        """
        #Setting attributes
        self.oracle = kwargs.get('oracle', None)
        if self.oracle is not None:
            #Creates QLM program from base gate
            self.q_prog = create_qprogram(self.oracle)
            #Creates the Grover-like operator from oracle
            self.q_gate = load_q_gate(self.oracle)
        else:
            #In this case we load directly the initial state
            #and the grover operator
            self.q_prog = kwargs.get('initial_state', None)
            self.q_gate = kwargs.get('grover', None)
            if (self.q_prog is None) or (self.q_gate is None):
                text = """If oracle was not provided initial_state and grover
                keys should be provided"""
                raise KeyError(text)

        #A complete list of m_k
        self.list_of_mks_ = kwargs.get('list_of_mks', 10)
        self.list_of_mks = self.list_of_mks_

        #If 0 we compute the exact probabilities
        self.nbshots = kwargs.get('nbshots', 0)
        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu', get_qpu())
        ##delta for avoid problems in 0 and pi/2 theta limits
        self.delta = kwargs.get('delta', 1.0e-5)
        #This is the default number of shots used for computing
        #the freqcuencies of the results when the computed probabilities
        #instead of freqcuencies are provided (nbshots = 0 when qlm job
        #is created)
        self.default_nbshots = kwargs.get('default_nbshots', 100)
        #number of iterations for optimization of Likelihood
        self.iterations = kwargs.get('iterations', 100)
        #For displaying extra info for optimization proccess
        self.disp = kwargs.get('disp', True)
        #Setting attributes
        self.restart()

    def restart(self):
        self.pdf_mks = None
        self.list_of_circuits = []
        self.list_of_jobs = []
        self.theta = None

    @property
    def list_of_mks(self):
        return self.list_of_mks_

    @list_of_mks.setter
    def list_of_mks(self, value):
        if type(value) in [list, int]:
            if type(value) in [int]:
                self.list_of_mks_ = list(range(value))
                self.list_of_mks_.reverse()
            else:
                self.list_of_mks_ = value
            print('list_of_mks: {}'.format(self.list_of_mks))

        else:
            raise ValueError('For m_k only ints and list are aceptable types')
        #We update the allocate classical bits each time we change cbits_number

    def apply_gate(self, m_k):
        """
        This method apply the self.q_gate to the self.q_prog an input
        number of times, creates the correspondient circuit and job,
        submit the job an get the results for an input number of shots.

        Parameters
        ----------
        m_k : int
            number of times to apply the self.q_gate to the quantum circuit

        Returns
        ----------

        circuit : QLM circuit object
            circuit object generated for the quantum program
        job : QLM job object
        """
        circuit, job = apply_gate(
            self.q_prog,
            self.q_gate,
            m_k,
            nbshots=self.nbshots
        )
        return circuit, job

    def result_processing(self, result_, mk):
        """
        This method receives a QLM Result object and proccess it in a
        propper way for posterior applying of maximum likelihood algorithm

        Parameters
        ----------

        result : QLM result object
        mk : int
            number of times Grover-like operator was applied for the
            input result
        
        Returns
        ----------

        pdf : pandas dataframe
            DataFrame with the results properly formated for maximum
            likelihood algorithm

        """
        #result_ = run_job(result)
        pdf_ = postprocess_results(result_)
        #Change the result presentation
        pdf = get_probabilities(pdf_)
        if self.nbshots == 0:
            #In this case QLM give simulated probabilities so we fixed
            #to self.default_nbshots
            n_k = self.default_nbshots
        else:
            #In this case we use the proper number of total measurements
            n_k = self.nbshots
        pdf['h_k'] = round(
            pdf['Probability_|1>']*n_k, 0
        ).astype(int)
        pdf['n_k'] = n_k
        pdf['m_k'] = mk
        return pdf

    def run_jobs(self, list_of_jobs, list_of_mks):
        """
        This functions submits a list of QLM jobs and get the result of
        the simulation in the propper way for maximum likelihood
        calculations

        Parameters
        ----------
        
        list_of_jobs : list
            list of jobs for executing
        list_of_mks : list
            list with the number of applications (m_k) of the Grover-
            like operator self.q_gate on the Quantum Program self.q_prog
            Should be consistent with the list of jobs

        Returns
        ----------

        pdf_mks : pandas dataframe
            results of the measurement of the last qbit
        """
        #Complete Job Submision

        batch_result = self.linalg_qpu.submit(Batch(list_of_jobs))
        results = run_job(batch_result)

        pdf_list = []
        for result, m_k in zip(results.results, list_of_mks):
            pdf_list.append(self.result_processing(result, m_k))

        pdf_mks = pd.concat(pdf_list)
        pdf_mks.reset_index(drop=True, inplace=True)
            
        return pdf_mks
    
    def run_step(self, m_k):
        """
        This method applies the Grover-like operator self.q_gate to the
        Quantum Program self.q_prog a given number of times m_k.

        Parameters
        ----------
        m_k : int
            number of times to apply the self.q_gate to the quantum circuit

        Returns
        ----------

        pdf_mks : pandas dataframe
            results of the measurement of the last qbit
        circuit : QLM circuit object
            circuit object generated for the quantum program
        job : QLM job object
        """

        #First create the complete circuit and job
        circuit, job = self.apply_gate(m_k)
        #Then execute the job and get the results
        pdf_mks = self.run_jobs([job], [m_k])
        return pdf_mks, circuit, job


    def run_mlae(self):#, list_of_mks=None, nbshots=None):
        """
        This method is the core of the Maximum Likelihood Amplitude
        Estimation. It runs several quantum circuits each one increasing
        the number of self.q_gate applied to the the initial self.q_prog
    
        Parameters
        ----------
        
        list_of_mks : list (corresponding property will be overwrite)
            python list with the different m_ks for executing the algortihm
        nbshots : int (corresponding property will be overwrite)
            number of shots for quantum job. If 0 exact probabilities
            will be computed

        """

        self.restart()
        #Clean the list in each run
        pdf_list = []
        for m_k in self.list_of_mks:
            step_circuit, step_job = self.apply_gate(m_k)
            self.list_of_circuits.append(step_circuit)
            self.list_of_jobs.append(step_job)
        #Then execute the job and get the results
        self.pdf_mks = self.run_jobs(
            self.list_of_jobs,
            self.list_of_mks
        )

        self.theta = self.launch_optimizer(self.pdf_mks)


    def launch_likelihood(self, pdf_input, N=100):
        """
        This method calculates the Likelihood for theta between [0, pi/2]
        for an input pandas DataFrame.

        Parameters
        ----------

        pdf_input: pandas DataFrame
            The DataFrame should have following columns:
            m_k: number of times q_gate was applied
            h_k: number of times the state |1> was measured
            n_k: number of total measuremnts

        N : int
            number of division for the theta interval

        Returns
        ----------

        y : pandas DataFrame
            Dataframe with the likelihood for the p_mks atribute. It has
            2 columns:
            theta : posible valores of the angle
            l_k : likelihood for each theta

        """
        pdf = pdf_input.copy(deep=True)
        if pdf is None:
            print(
                """
                Can not calculate Likelihood because pdf_input is empty.
                Please provide a valida DataFrame.
                """)
            return None
        theta = np.linspace(0+self.delta, 0.5*np.pi-self.delta, N)
        m_k = pdf['m_k']
        h_k = pdf['h_k']
        n_k = pdf['n_k']
        l_k = np.array([likelihood(t, m_k, h_k, n_k) for t in theta])
        y_ = pd.DataFrame({'theta': theta, 'l_k': l_k})
        return y_

    def launch_optimizer(self, results):
        """
        This functions execute a brute force optimization of the
        likelihood function for an input results pdf.

        Parameters
        ----------

        results : pandas DataFrame
            DataFrame with the results from ml-qpe procedure.
            Mandatory columns:
            m_k : number of times Groover like operator was applied
            h_k : number of measures of the state |1>
            n_k : number of measurements done


        Returns
        ----------

        optimum_theta : float
            theta  that minimize likelihood
        """

        theta_domain = (0+self.delta, 0.5*np.pi-self.delta)
        optimizer = so.brute(
            likelihood,
            [theta_domain],
            (results['m_k'], results['h_k'], results['n_k']),
            self.iterations,
            disp=self.disp
        )
        optimum_theta = optimizer[0]
        return optimum_theta

