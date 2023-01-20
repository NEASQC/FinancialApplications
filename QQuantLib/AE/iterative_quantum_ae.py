"""
This module contains necessary functions and classes to implement
Iterative Quantum Amplitude Estimation based on the paper:

    Grinko, D., Gacon, J., Zoufal, C. et al.
    Iterative Quantum Amplitude Estimation
    npj Quantum Inf 7, 52 (2021).
    https://doi.org/10.1038/s41534-021-00379-1

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import check_list_type, measure_state_probability


class IQAE:
    """
    Class for Iterative Quantum Amplitude Estimation (IQAE)
    algorithm

    Parameters
    ----------
    oracle: QLM gate
        QLM gate with the Oracle for implementing the
        Grover operator
    target : list of ints
        python list with the target for the amplitude estimation
    index : list of ints
        qubits which mark the register to do the amplitude
        estimation

    kwars : dictionary
        dictionary that allows the configuration of the IQAE algorithm: \\
        Implemented keys:

        qpu : QLM solver
            solver for simulating the resulting circuits
        epsilon : float
            precision
        alpha : float
            accuracy
        shots : int
            number of measurements on each iteration
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
        self.epsilon = kwargs.get("epsilon", 0.01)
        self.alpha = kwargs.get("alpha", 0.05)
        self.shots = int(kwargs.get("shots", 100))
        self.mcz_qlm = kwargs.get("mcz_qlm", True)

        # Creating the grover operator
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

        self.ae_l = None
        self.ae_u = None
        self.theta_l = None
        self.theta_u = None
        self.theta = None
        self.ae = None
        self.circuit_statistics = None
        self.time_pdf = None
        self.run_time = None
        self.schedule = {}
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.schedule_pdf = None
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
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
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
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
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
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    #####################################################################

    @staticmethod
    def find_next_k(
        k: int, theta_lower: float, theta_upper: float, flag: bool, ratio: float = 2
    ):
        """
        This is an implementation of Algorithm 2 from the IQAE paper.
        This function computes the next suitable k.

        Parameters
        ----------
        k : int
            number of times to apply the grover operator to the quantum circuit
        theta_lower : float
            lower bound for the estimation of the angle
        theta_upper : float
            upper bound for the estimation of the angle
        flag : bool
            flag to keep track of weather we are in the
            upper or lower half pane
        ratio : float
            ratio of amplifications between consecutive iterations

        Returns
        ----------
        k : int
            number of times to apply the grover operator to the quantum circuit
        flag : bool
            flag to keep track of weather we are in the
            upper or lower half pane

        """
        # This is K_i in the paper
        bigk_i = 4 * k + 2
        theta_min = bigk_i * theta_lower
        theta_max = bigk_i * theta_upper
        # This K_max in the paper
        bigk_max = np.floor(np.pi / (theta_upper - theta_lower))
        # This is K in the paper
        big_k = bigk_max - np.mod(bigk_max - 2, 4)
        while big_k > ratio * bigk_i:
            q_ = big_k / bigk_i
            if (np.mod(q_ * theta_max, 2 * np.pi) <= np.pi) and (
                np.mod(q_ * theta_min, 2 * np.pi) <= np.pi
            ):
                # This K_next in the paper
                bigk_next = big_k
                flag = True
                k_next = (bigk_next - 2) / 4
                return [int(k_next), flag]
            if (np.mod(q_ * theta_max, 2 * np.pi) >= np.pi) and (
                np.mod(q_ * theta_min, 2 * np.pi) >= np.pi
            ):
                # This K_next in the paper
                bigk_next = big_k
                flag = False
                k_next = (bigk_next - 2) / 4
                return [int(k_next), flag]

            big_k = big_k - 4
        return [int(k), flag]

    @staticmethod
    def invert_sector(a_min: float, a_max: float, flag: bool = True):
        r"""
        This function inverts the expression:

        .. math::
            a = \dfrac{1-\cos(\theta)}{2}

        for a pair of bounds (a_min,a_max). The result
        belongs to the domain (0,2\pi)

        Parameters
        ----------
        a_min : float
            lower bound
        a_max : float
            upper bound
        flag : bool
            flag to keep track of weather we are in the
            upper or lower half pane

        Returns
        ----------
        theta_min : float
           lower bound for the associated angle
        theta_max : float
           upper bound for the associated angle

        """
        theta_1 = np.minimum(np.arccos(1 - 2 * a_min), np.arccos(1 - 2 * a_max))
        theta_2 = np.maximum(np.arccos(1 - 2 * a_min), np.arccos(1 - 2 * a_max))
        if flag:
            theta_min = theta_1
            theta_max = theta_2
        else:
            theta_min = 2 * np.pi - theta_2
            theta_max = 2 * np.pi - theta_1

        return [theta_min, theta_max]

    @staticmethod
    def display_information(
        epsilon: float = 0.01, shots: int = 100, alpha: float = 0.05
    ):
        """
        This function displays information of the properties of the
        method for a given set of parameters

        Parameters
        ----------
        epsilon : float
            precision
        alpha : float
            accuracy
        shots : int
            number of measurements on each iteration

        """

        print("-------------------------------------------------------------")
        print("epsilon: ", epsilon)
        print("alpha: ", alpha)
        print("N: ", shots)
        print("-------------------------------------------------------------")

        # This is T, number of rounds, in the papper
        big_t = np.ceil(np.log2(np.pi / (8 * epsilon)))
        n_max = (
            32
            / (1 - 2 * np.sin(np.pi / 14)) ** 2
            * np.log(2 / alpha * np.log2(np.pi / (4 * epsilon)))
        )
        n_oracle = 50 / epsilon * np.log(2 / alpha * np.log2(np.pi / (4 * epsilon)))
        # This is L in the papper
        big_l = (np.arcsin(2 / shots * np.log(2 * big_t / epsilon))) ** 0.25
        k_max = big_l / epsilon / 2

        print("-------------------------------------------------------------")
        print("Maximum number of rounds: ", int(big_t))
        print("Maximum number of shots per round needed: ", int(n_max))
        print("Maximum number of amplifications: ", int(k_max))
        print("Maximum number of calls to the oracle: ", int(n_oracle))
        print("-------------------------------------------------------------")

    @staticmethod
    def chebysev_bound(n_samples: int, gamma: float):
        r"""
        Computes the length of the confidence interval for a given
        number of samples n_samples and an accuracy gamma:

        .. math::
            \epsilon = \dfrac{1}{\sqrt{2N}}\log\left(\dfrac{2}{\gamma} \
            \right)

        Parameters
        ----------
        n_samples : int
            number of samples
        gamma : float
            accuracy

        Returns
        ----------
        length of the confidence interval : float
        """
        return np.sqrt(1 / (2 * n_samples) * np.log(2 / gamma))

    def iqae(self, epsilon: float = 0.01, shots: int = 100, alpha: float = 0.05):
        """
        This function implements Algorithm 1 from the IQAE paper. The result
        is an estimation of the desired probability with precision at least
        epsilon and accuracy at least alpha.

        Parameters
        ----------
        epsilon : float
            precision
        alpha : float
            accuracy
        shots : int
            number of measurements on each iteration

        Returns
        ----------
        a_l : float
           lower bound for the probability to be estimated
        a_u : float
           upper bound for the probability to be estimated

        """

        self.circuit_statistics = {}
        #####################################################
        i = 0
        k = int(0)
        flag = True
        [theta_l, theta_u] = [0.0, np.pi / 2]
        # This is T the number of rounds in the paper
        big_t = int(np.ceil(np.log2(np.pi / (8 * epsilon))) + 1)
        # This is L in the paper
        big_l = (np.arcsin(2 / shots * np.log(2 * big_t / epsilon))) ** 0.25
        #####################################################
        h_k = 0
        n_effective = 0
        # time_list = []
        j = 0 # pure counter

        while theta_u - theta_l > 2 * epsilon:
            #start = time.time()
            i = i + 1
            k_old = k
            [k, flag] = self.find_next_k(k_old, theta_l, theta_u, flag)
            big_k = 4 * k + 2
            #end = time.time()
            #finding_time = end - start

            #####################################################
            routine = self.quantum_step(k)
            start = time.time()
            results, circuit, _, _ = get_results(
                routine, linalg_qpu=self.linalg_qpu, shots=shots, qubits=self.index
            )
            end = time.time()
            self.quantum_times.append(end-start)
            # time_pdf["m_k"] = k
            a_ = measure_state_probability(results, self.target)
            #a_ = results["Probability"].iloc[bitfield_to_int(self.target)]
            #####################################################
            # Aggregate results from different iterations
            if j == 0:
                # In the first step we need to store the circuit statistics
                step_circuit_stats = circuit.statistics()
                step_circuit_stats.update({"n_shots": shots})
                self.circuit_statistics.update({k: step_circuit_stats})
                self.schedule.update({k:shots})

            if k == k_old:
                h_k = h_k + int(a_ * shots)
                n_effective = n_effective + shots
                a_ = h_k / n_effective
                i = i - 1
                # Only update shots for the k application
                step_circuit_stats = self.circuit_statistics[k]
                step_circuit_stats.update({"n_shots": n_effective})
                self.schedule.update({k:n_effective})

            else:
                h_k = int(a_ * shots)
                n_effective = shots
                # Store the circuit statistics for new k
                step_circuit_stats = circuit.statistics()
                step_circuit_stats.update({"n_shots": shots})
                self.schedule.update({k:shots})
            self.circuit_statistics.update({k: step_circuit_stats})

            # Compute the rest
            epsilon_a = IQAE.chebysev_bound(n_effective, alpha / big_t)
            a_max = np.minimum(a_ + epsilon_a, 1.0)
            a_min = np.maximum(a_ - epsilon_a, 0.0)
            [theta_min, theta_max] = self.invert_sector(a_min, a_max, flag)

            theta_l_ = (
                2 * np.pi * np.floor(big_k * theta_l / (2 * np.pi)) + theta_min
            ) / big_k
            theta_u_ = (
                2 * np.pi * np.floor(big_k * theta_u / (2 * np.pi)) + theta_max
            ) / big_k
            # If bounded limits are worse than step before limits use these ones
            theta_l = np.maximum(theta_l, theta_l_)
            theta_u = np.minimum(theta_u, theta_u_)
            j = j + 1
        [a_l, a_u] = [np.sin(theta_l) ** 2, np.sin(theta_u) ** 2]
        return [a_l, a_u]

    def quantum_step(self, k):
        r"""
        Create the quantum routine needed for the iqae step

        Parameters
        ----------
        k : int
            number of Grover operator applications

        Returns
        ----------
        routine : qlm routine
            qlm routine for the iqae step
        """

        routine = qlm.QRoutine()
        wires = routine.new_wires(self.oracle.arity)
        routine.apply(self.oracle, wires)
        for j in range(k):
            routine.apply(self._grover_oracle, wires)
        return routine

    def run(self):
        r"""
        run method for the class.

        Returns
        ----------

        self.ae :
            amplitude estimation parameter

        """
        start = time.time()
        [self.ae_l, self.ae_u] = self.iqae(
            epsilon=self.epsilon, shots=self.shots, alpha=self.alpha
        )
        self.theta_l = np.arcsin(np.sqrt(self.ae_l))
        self.theta_u = np.arcsin(np.sqrt(self.ae_u))
        self.theta = (self.theta_u + self.theta_l) / 2.0
        self.ae = (self.ae_u + self.ae_l) / 2.0
        end = time.time()
        self.run_time = end - start
        self.schedule_pdf = pd.DataFrame.from_dict(
            self.schedule,
            columns=['shots'],
            orient='index'
        )
        self.schedule_pdf.reset_index(inplace=True)
        self.schedule_pdf.rename(columns={'index': 'm_k'}, inplace=True)
        self.oracle_calls = np.sum(
            self.schedule_pdf['shots'] * (2 * self.schedule_pdf['m_k'] + 1))
        self.max_oracle_depth = np.max(2 *  self.schedule_pdf['m_k']+ 1)
        self.quantum_time = sum(self.quantum_times)
        return self.ae
