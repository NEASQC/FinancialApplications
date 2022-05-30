"""
This module contains necesary functions and classes to implement
Maximum Likelihood Amplitude Estimation based on the paper:

    Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N.
    Amplitude estimation without phase estimation
    Quantum Information Processing, 19(2), 2020
    arXiv: quant-ph/1904.10246v2

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

from copy import deepcopy
from functools import partial
import numpy as np
import scipy.optimize as so
import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import bitfield_to_int, check_list_type, load_qn_gate
from QQuantLib.utils.utils import load_qn_gate, check_list_type



class MLAE:
    """
    Class for using Maximum Likelihood Quantum Amplitude Estimation (ML-AE)
    algorithm
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class

        Parameters
        ----------
        oracle: QLM gate
            QLM gate with the Oracle for implementing the
            Grover operator:
            init_q_prog and q_gate will be interpreted as None
        target : list of ints
            python list with the target for the amplitude estimation
        index : list of ints
            qubits which mark the register to do the amplitude
            estimation

        kwars : dictionary
            dictionary that allows the configuration of the MLAE algorithm:
            Implemented keys:
                qpu : QLM solver
                    solver for simulating the resulting circuits
                schedule : list of two lists
                    the schedule for the algorithm
                optimizer :
                    an optimizer with just one possible entry
                delta : float
                    tolerance to avoid division by zero warnings
                ns : int
                    number of grid points for brute scipy optimizer
        """
        #Setting attributes
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)
        self._grover_oracle = grover(self._oracle, self.target, self.index)

        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu', None)
        if self.linalg_qpu is None:
            print('Not QPU was provide. Default QPU will be used')
            self.linalg_qpu = get_default_qpu()
        ##delta for avoid problems in 0 and pi/2 theta limits
        self.delta = kwargs.get('delta', 1.0e-5)
        #ns for the brute force optimizer
        self.ns = kwargs.get('ns', 1000)

        # The schedule of the method
        self.m_k = None
        self.n_k = None
        schedule = kwargs.get('schedule', None)
        if schedule is None:
            self.set_linear_schedule(10, 100)
        else:
            self.schedule = schedule

        # Optimization
        #For avoiding problem with 0 and 0.5*pi
        self.theta_domain = [(0+self.delta, 0.5*np.pi-self.delta)]
        self.brute_force = lambda x: so.brute(func=x, ranges=self.theta_domain, Ns=self.ns)
        self.optimizer = kwargs.get(
            'optimizer',
            self.brute_force
        )
        #For storing results
        self.h_k = None
        self.partial_cost_function = None
        self.theta = None
        self.a = None

    #####################################################################
    @property
    def oracle(self):
        return self._oracle

    @oracle.setter
    def oracle(self, value):
        self._oracle = deepcopy(value)
        self._grover_oracle = grover(self.oracle, self.target, self.index)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = check_list_type(value, int)
        self._grover_oracle = grover(self.oracle, self.target, self.index)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = check_list_type(value, int)
        self._grover_oracle = grover(self.oracle, self.target, self.index)

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, value):
        x_ = check_list_type(value, int)
        if x_.shape[0] != 2:
            raise Exception("The shape of the schedule must be (2,n)")
        self._schedule = x_
        self.m_k = self.schedule[0]
        self.n_k = self.schedule[1]
    #####################################################################

    def set_linear_schedule(self, n: int, n_k: int):
        x_ = np.arange(n)
        y_ = [n_k]*n
        self.schedule = [x_, y_]

    def set_exponential_schedule(self, n: int, n_k: int):
        x_ = 2**np.arange(n)
        y_ = [n_k]*n
        self.schedule = [x_, y_]

    def run_step(self, m_k: int, n_k: int) -> int:
        """
        This method executes on step of the MLAE algorithm

        Parameters
        ----------
        m_k : int
            number of times to apply the self.q_gate to the quantum circuit
        n_k : int
            number of shots

        Returns
        ----------
        h_k: int
            number of positive events
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(self.oracle.arity)
        routine.apply(self.oracle, register)
        routine.apply(load_qn_gate(self._grover_oracle, m_k), register)
        #for i in range(m_k):
        #    routine.apply(self._grover_oracle, register)
        result, circuit, _, job = get_results(
            routine,
            linalg_qpu=self.linalg_qpu,
            shots=n_k,
            qubits=self.index
        )
        h_k = int(result["Probability"].iloc[bitfield_to_int(self.target)]*n_k)

        return h_k, circuit


    @staticmethod
    def likelihood(theta: float, m_k: int, n_k: int, h_k: int)->float:
        r"""
        Calculates Likelihood from Suzuki papper. For h_k positive events
        of n_k total events, this function calculates the probability of
        this taking into account that the probability of a positive
        event is given by theta and by m_k
        The idea is use this function to minimize it for this reason it gives
        minus Likelihood

        Notes
        -----
        .. math::
            l_k(\theta|h_k) = \sin^2\left((2m_k+1)\theta\right)^{h_k}\cos^2
            \left((2m_k+1)\theta\right)^{n_k-h_k}

        Parameters
        ----------

        theta : float
            Angle (radians) for calculating the probability of measure a
            positive event.
        m_k : int
            number of times the grover operator was applied.
        n_k : int
            number of total events measured for the specific  m_k
        h_k : int
            number of positive events measured for each m_k

        Returns
        ----------

        float
            Gives the Likelihood p(h_k with m_k amplifications|theta)

        """
        theta_ = (2*m_k+1)*theta
        p_0 = np.sin(theta_)**2
        p_1 = np.cos(theta_)**2
        l_k = (p_0**h_k)*(p_1**(n_k-h_k))
        return l_k

    @staticmethod
    def log_likelihood(theta: float, m_k: int, n_k: int, h_k: int)->float:
        r"""
        Calculates log of the likelihood from Suzuki papper.

        Notes
        -----
        .. math::
            \log{l_k(\theta|h_k)} = 2h_k\log\big[\sin\left((2m_k+1)\theta\right)\big]
            +2(n_k-h_k)\log\big[\cos\left((2m_k+1)\theta\right)\big]

        Parameters
        ----------

        theta : float
            Angle (radians) for calculating the probability of measure a
            positive event.
        m_k : int
            number of times the grover operator was applied.
        n_k : int
            number of total events measured for the specific  m_k
        h_k : int
            number of positive events measured for each m_k

        Returns
        ----------

        float
            Gives the log Likelihood p(h_k with m_k amplifications|theta)

        """
        theta_ = (2*m_k+1)*theta
        p_0 = np.sin(theta_)**2
        p_1 = np.cos(theta_)**2
        l_k = h_k*np.log(p_0) + (n_k-h_k)*np.log(p_1)
        return l_k


    @staticmethod
    def cost_function(angle: float, m_k: list, n_k: list, h_k: list)->float:
        r"""
        This method calculates the -Likelihood of angle theta
        for a given schedule m_k,n_k

        Notes
        -----
        .. math::
            L(\theta,\mathbf{h}) = -\sum_{k = 0}^M\log{l_k(\theta|h_k)}

        Parameters
        ----------

        angle: float
            Angle (radians) for calculating the probability of measure a
            positive event.
        m_k : list of ints
            number of times the grover operator was applied.
        n_k : list of ints
            number of total events measured for the specific  m_k
        h_k : list of ints
            number of positive events measured for each m_k

        Returns
        ----------

        cost : float
            the aggregation of the individual likelihoods
        """
        log_cost = 0
        for i in range(len(m_k)):
            log_l_k = MLAE.log_likelihood(
                angle,
                m_k[i],
                n_k[i],
                h_k[i]
            )
            log_cost = log_cost+log_l_k
        return -log_cost

    def run_schedule(self, schedule):
        """
        This method execute the run_step method for each pair of values
        of a given schedule.

        Parameters
        ----------

        schedule : list of two lists
            the schedule for the algorithm

        Returns
        ----------
        h_k : list
            list with the h_k result of each pair of the input schedule

        """
        x_ = check_list_type(schedule, int)
        if x_.shape[0] != 2:
            raise Exception("The shape of the schedule must be (2,n)")
        schedule_ = x_
        m_k = schedule_[0]
        n_k = schedule_[1]
        h_k = np.zeros(len(m_k), dtype=int)
        for i in range(len(m_k)):
            h_k[i], _ = self.run_step(m_k[i], n_k[i])
        return h_k

    def mlae(self, schedule, optimizer):
        """
        This method executes a complete Maximum Likelihood Algorithm,
        including executing schedule, defining the correspondient cost
        function and optimizing it.

        Parameters
        ----------

        schedule : list of two lists
            the schedule for the algorithm
        optimizer : optimization routine.
            the optimizer should receive a function of one variable
            the angle to be optimized. Using lambda functions is the
            recomended way.

        Returns
        ----------

        result : optimizer results
            the type of the result is the type of the result of the optimizer
        h_k : list
            list with number of positive outcomes from quantum circuit
            for each pair element of the input schedule
        cost_function_partial : function
            partial cost function with the m_k, n_k and h_k fixed to the
            obtained values of the different experiments.
        """
        h_k = self.run_schedule(schedule)
        m_k = schedule[0]
        n_k = schedule[1]
        cost_function_partial = partial(
            self.cost_function,
            m_k = m_k, n_k=n_k, h_k=h_k
        )
        result = optimizer(cost_function_partial)
        return result, h_k, cost_function_partial

    def run(self)->float:
        r"""
        run method for the class.

        Parameters
        ----------

        Returns
        ----------

        result :
            list with the estimation of a

        Notes
        -----
        .. math::
            a^*  = \sin^2(\theta^*)
            \; where \; \theta^* = \arg \min_{\theta} L(\theta,\mathbf{h})


        """

        #overwrite of the different propeties of the class
        self.theta, self.h_k, self.partial_cost_function = self.mlae(
            self.schedule, self.brute_force
        )
        self.theta = self.theta[0]
        self.a = np.sin(self.theta)**2
        result = self.a
        return result
