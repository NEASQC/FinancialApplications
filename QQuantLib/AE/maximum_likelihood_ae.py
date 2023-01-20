"""
This module contains necessary functions and classes to implement
Maximum Likelihood Amplitude Estimation based on the paper:

    Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N.
    Amplitude estimation without phase estimation
    Quantum Information Processing, 19(2), 2020
    arXiv: quant-ph/1904.10246v2

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
from copy import deepcopy
from functools import partial
import numpy as np
import pandas as pd
import scipy.optimize as so
import qat.lang.AQASM as qlm
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import measure_state_probability, check_list_type, load_qn_gate
from QQuantLib.AE.mlae_utils import likelihood, log_likelihood, cost_function

class MLAE:
    """
    Class for using Maximum Likelihood Quantum Amplitude Estimation
    (MLAE) algorithm

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
        dictionary that allows the configuration of the MLAE algorithm: \\
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
        mcz_qlm : bool
            for using or not QLM implementation of the multi controlled Z
            gate
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class

        """
        # Setting attributes
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            print("Not QPU was provide. PyLinalg will be used")
            self.linalg_qpu = get_qpu("python")
        ##delta for avoid problems in 0 and pi/2 theta limits
        self.delta = kwargs.get("delta", 1.0e-6)
        # ns for the brute force optimizer
        self.ns = kwargs.get("ns", 1000)

        # The schedule of the method
        self.m_k = None
        self.n_k = None
        schedule = kwargs.get("schedule", None)
        if schedule is None:
            self.set_linear_schedule(10, 100)
        else:
            self.schedule = schedule

        self.mcz_qlm = kwargs.get("mcz_qlm", True)

        # Creating the grover operator
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )
        # Optimization
        # For avoiding problem with 0 and 0.5*pi
        self.theta_domain = [(0 + self.delta, 0.5 * np.pi - self.delta)]
        self.brute_force = lambda x: so.brute(
            func=x, ranges=self.theta_domain, Ns=self.ns
        )
        self.optimizer = kwargs.get("optimizer", self.brute_force)
        # For storing results
        self.h_k = None
        self.partial_cost_function = None
        self.theta = None
        self.ae = None
        self.circuit_statistics = None
        self.time_pdf = None
        self.optimizer_time = None
        self.run_time = None
        self.ae_l = None
        self.ae_u = None
        self.schedule_pdf = None
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.quantum_times = []
        self.quantum_time = None

    #####################################################################
    @property
    def oracle(self):
        """
        creating oracle property
        """
        return self._oracle

    @oracle.setter
    def oracle(self, value):
        """
        setter of the oracle property
        """
        self._oracle = deepcopy(value)
        self._grover_oracle = grover(
            self.oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    @property
    def target(self):
        """
        creating target property
        """
        return self._target

    @target.setter
    def target(self, value):
        """
        setter of the target property
        """
        self._target = check_list_type(value, int)
        self._grover_oracle = grover(
            self.oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    @property
    def index(self):
        """
        creating index property
        """
        return self._index

    @index.setter
    def index(self, value):
        """
        setter of the index property
        """
        self._index = check_list_type(value, int)
        self._grover_oracle = grover(
            self.oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    @property
    def schedule(self):
        """
        creating schedule property
        """
        return self._schedule

    @schedule.setter
    def schedule(self, value):
        """
        setter of the schedule property
        """
        x_ = check_list_type(value, int)
        if x_.shape[0] != 2:
            raise Exception("The shape of the schedule must be (2,n)")
        self._schedule = x_
        self.m_k = self.schedule[0]
        self.n_k = self.schedule[1]

    #####################################################################

    def set_linear_schedule(self, n_t: int, n_s: int):
        """
        Creates a scheduler of linear increasing of m_ks.

        Parameters
        ----------
        n_t : int
            number of maximum applications of the grover operator
        n_s : int
            number of shots for each m_k grover applications
        """
        x_ = np.arange(n_t)
        y_ = [n_s] * n_t
        self.schedule = [x_, y_]

    def set_exponential_schedule(self, n_t: int, n_s: int):
        """
        Creates a scheduler of exponential increasing of m_ks.

        Parameters
        ----------
        n_t : int
            number of maximum applications of the grover operator
        n_s : int
            number of shots for each m_k grover applications
        """
        x_ = 2 ** np.arange(n_t)
        y_ = [n_s] * n_t
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
        routine : QLM Routine object
        """

        routine = qlm.QRoutine()
        register = routine.new_wires(self.oracle.arity)
        routine.apply(self.oracle, register)
        routine.apply(load_qn_gate(self._grover_oracle, m_k), register)
        # for i in range(m_k):
        #    routine.apply(self._grover_oracle, register)
        start = time.time()
        result, _, _, job = get_results(
            routine, linalg_qpu=self.linalg_qpu, shots=n_k, qubits=self.index
        )
        end = time.time()
        self.quantum_times.append(end-start)
        h_k = int(measure_state_probability(result, self.target) * n_k)
        #h_k = int(result["Probability"].iloc[bitfield_to_int(self.target)] * n_k)
        return h_k, routine

    @staticmethod
    def likelihood(theta: float, m_k: int, n_k: int, h_k: int) -> float:
        r"""
        Calculates Likelihood from Suzuki paper. For h_k positive events
        of n_k total events, this function calculates the probability of
        this taking into account that the probability of a positive
        event is given by theta and by m_k
        The idea is use this function to minimize it for this reason it gives
        minus Likelihood

        Notes
        -----
        .. math::
            l_k(\theta|h_k) = \sin^2\left((2m_k+1)\theta\right)^{h_k} \
            \cos^2 \left((2m_k+1)\theta\right)^{n_k-h_k}

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
        #theta_ = (2 * m_k + 1) * theta
        #p_0 = np.sin(theta_) ** 2
        #p_1 = np.cos(theta_) ** 2
        #l_k = (p_0**h_k) * (p_1 ** (n_k - h_k))
        l_k = likelihood(theta, m_k, n_k, h_k)
        return l_k

    @staticmethod
    def log_likelihood(theta: float, m_k: int, n_k: int, h_k: int) -> float:
        r"""
        Calculates log of the likelihood from Suzuki paper.

        Notes
        -----
        .. math::
            \log{l_k(\theta|h_k)} = 2h_k\log\big[\sin\left((2m_k+1) \
            \theta\right)\big] +2(n_k-h_k)\log\big[\cos\left((2m_k+1) \
            \theta\right)\big]

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
        #theta_ = (2 * m_k + 1) * theta
        #p_0 = np.sin(theta_) ** 2
        #p_1 = np.cos(theta_) ** 2
        #l_k = h_k * np.log(p_0) + (n_k - h_k) * np.log(p_1)
        l_k = log_likelihood(theta, m_k, n_k, h_k)
        return l_k

    @staticmethod
    def cost_function(angle: float, m_k: list, n_k: list, h_k: list) -> float:
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
        #log_cost = 0
        ## for i in range(len(m_k)):
        #for i, _ in enumerate(m_k):
        #    log_l_k = MLAE.log_likelihood(angle, m_k[i], n_k[i], h_k[i])
        #    log_cost = log_cost + log_l_k
        log_cost = cost_function(angle, m_k, n_k, h_k)
        return log_cost

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
        self.circuit_statistics = {}
        x_ = check_list_type(schedule, int)
        if x_.shape[0] != 2:
            raise Exception("The shape of the schedule must be (2,n)")
        schedule_ = x_
        m_k = schedule_[0]
        n_k = schedule_[1]
        h_k = np.zeros(len(m_k), dtype=int)
        # for i in range(len(m_k)):
        for i, _ in enumerate(m_k):
            h_k[i], circuit = self.run_step(m_k[i], n_k[i])
            step_circuit_stats = circuit.to_circ().statistics()
            step_circuit_stats.update({"n_shots": n_k[i]})
            step_circuit_stats.update({"h_k": h_k[i]})
            self.circuit_statistics.update({m_k[i]: step_circuit_stats})
        return h_k

    def mlae(self, schedule, optimizer):
        """
        This method executes a complete Maximum Likelihood Algorithm,
        including executing schedule, defining the correspondent cost
        function and optimizing it.

        Parameters
        ----------

        schedule : list of two lists
            the schedule for the algorithm
        optimizer : optimization routine.
            the optimizer should receive a function of one variable
            the angle to be optimized. Using lambda functions is the
            recommended way.

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
        cost_function_partial = partial(self.cost_function, m_k=m_k, n_k=n_k, h_k=h_k)
        start = time.time()
        result = optimizer(cost_function_partial)
        end = time.time()
        self.optimizer_time = end - start
        return result, h_k, cost_function_partial

    def run(self) -> float:
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
            a^*  = \sin^2(\theta^*) \; where \; \theta^* = \arg \
            \min_{\theta} L(\theta,\mathbf{h})


        """

        # overwrite of the different properties of the class
        start = time.time()
        self.theta, self.h_k, self.partial_cost_function = self.mlae(
            self.schedule, self.brute_force
        )
        self.theta = self.theta[0]
        self.ae = np.sin(self.theta) ** 2
        result = self.ae
        end = time.time()
        self.run_time = end - start
        self.quantum_time = sum(self.quantum_times)
        
        #Number of oracle call calculation
        self.schedule_pdf = pd.DataFrame(
            [self.m_k, self.n_k, self.h_k],
            index=['m_k', 'n_k', 'h_k']
        ).T
        self.oracle_calls = np.sum(
            self.schedule_pdf['n_k'] * (2 * self.schedule_pdf['m_k'] + 1))
        self.max_oracle_depth = np.max(2 *  self.schedule_pdf['m_k']+ 1)
        return result
