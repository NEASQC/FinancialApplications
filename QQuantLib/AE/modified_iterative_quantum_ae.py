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
#from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from QQuantLib.utils.get_qpu import get_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import check_list_type, measure_state_probability


class mIQAE:
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
        self._oracle = oracle
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
        self._oracle = value
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
    def find_next_k( k: int, theta_lower: float, theta_upper: float):
        """
        This is an implementation of Algorithm 2 from the mIQAE paper.
        This function computes the next suitable k.

        Parameters
        ----------
        k : int
            number of times to apply the grover operator to the quantum circuit
        theta_lower : float
            lower bound for the estimation of the angle
        theta_upper : float
            upper bound for the estimation of the angle

        Returns
        ----------
        k : int
            number of times to apply the grover operator to the quantum circuit
        """
        # This is K_i in the paper
        bigk_i = 2 * k + 1
        # This is K in Algorithm 2
        big_k = np.floor(0.5 * np.pi / (theta_upper - theta_lower))

        if big_k % 2 == 0:
            #if K is even
            big_k = big_k -1

        while big_k >= 3.0 * bigk_i:
            amplified_lower = np.floor(big_k * theta_lower / (0.5 * np.pi))
            amplified_upper = np.ceil(big_k * theta_upper / (0.5 * np.pi))
            if amplified_lower == (amplified_upper -1):
                k_i = (big_k - 1) / 2.0
                return int(k_i)
            big_k = big_k -2
        return int(k)

    @staticmethod
    def compute_info(
        epsilon: float = 0.01, shots: int = 100, alpha: float = 0.05
    ):
        """
        This function computes theoretical values of the IQAE algorithm.

        Parameters
        ----------
        epsilon : float
            precision
        alpha : float
            accuracy
        shots : int
            number of measurements on each iteration

        Return
        ------
        info : dict
            python dictionary with the computed information

        """


        # Upper bound for amplification: Kmax in paper
        bigk_max = int(np.pi / (4.0 * epsilon))
        # Maximum number of rounds: T in paper
        big_t = int(np.ceil(np.log2(np.pi / (4 * epsilon)) / np.log2(3)))
        # constant C in the paper
        big_c = 1.0 / ((np.sin(np.pi / 21.0) * np.sin(8.0 * np.pi /21.0)) ** 2)
        # Total number of Grover operator calls
        n_grover = int(1.5 * bigk_max * big_c * np.log(np.sqrt(27) / alpha))
        # Total number of oracle operator calls
        c1 = np.log(3 * bigk_max / alpha)
        c2 = 0.5 * (big_t + 1) * big_t * np.log(3)
        c3 = big_t * np.log(alpha)
        n_oracle = 2.0 * n_grover + int(big_c * (c1 + c2 - c3))


        info = {
            "bigk_max": bigk_max, "big_t": big_t, "n_grover": n_grover,
            "n_oracle": n_oracle
        }

        return info

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

        info_dict = mIQAE.compute_info(epsilon, shots, alpha)

        print("-------------------------------------------------------------")
        print("Maximum amplification (Kmax)", info_dict["bigk_max"])
        print("Maximum number of rounds: ", info_dict["big_t"])
        print("Maximum number of Grover operator calls: ", info_dict["n_grover"])
        print("Maximum number of Oracle operator calls: ", info_dict["n_oracle"])
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

    @staticmethod
    def confidence_intervals(a_min: float, a_max: float, ri: int):
        r"""
        Computes the confidence intervals

        Parameters
        ----------
        a_min : float
            minimum amplitude measured in the round
        a_max : float
            maximum amplitude measured in the round
        ri : int
            number of quadrants passed to get the current angle

        """
        if ri % 2 == 0:
            gamma_min = np.arcsin(np.sqrt(a_min))
            gamma_max = np.arcsin(np.sqrt(a_max))
        else:
            gamma_min = -np.arcsin(np.sqrt(a_max)) + 0.5 * np.pi
            gamma_max = -np.arcsin(np.sqrt(a_min)) + 0.5 * np.pi
        return [gamma_min, gamma_max]


    def miqae(self, epsilon: float = 0.01, shots: int = 100, alpha: float = 0.05):
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
        [theta_l, theta_u] = [0.0, np.pi / 2]

        # Kmax in paper
        bigk_max = np.pi / (4.0 * epsilon)

        while theta_u - theta_l > 2 * epsilon:
            n_ = 0
            i = i + 1
            k_i = k
            # Ki in paper
            big_k_i = 2 * k_i + 1
            # alpha_i in paper
            alpha_i = 2.0 * alpha * big_k_i / (3.0 * bigk_max)
            cte_ = (np.sin(np.pi / 21.0) * np.sin(8.0 * np.pi /21.0)) ** 2
            n_i_max = int(np.floor(2.0 * np.log(2.0 / alpha_i) / cte_))
            #number of quadrants passed
            ri = np.floor(big_k_i * theta_l / (0.5 * np.pi))

            #print("alpha_i: ", alpha_i, "n_i_max: ", n_i_max)
            while k_i == k:
                #####################################################
                shots_ = min(shots, n_i_max - n_)
                routine = self.quantum_step(k_i)
                start = time.time()
                results, circuit, _, _ = get_results(
                    routine,
                    linalg_qpu=self.linalg_qpu,
                    shots=shots_,
                    qubits=self.index
                )
                end = time.time()
                #print("k: ", k_i, "shots: ", shots_)
                self.quantum_times.append(end-start)

                if k_i not in self.schedule:
                    self.schedule.update({k_i:shots_})
                else:
                    self.schedule.update({k_i:self.schedule[k_i] + shots_})

                n_ = n_ + shots_
                a_ = measure_state_probability(results, self.target)
                epsilon_a = mIQAE.chebysev_bound(n_, alpha_i)
                #print("n_: ", n_, "alpha_i: ", alpha_i, "epsilon_a: ", epsilon_a)
                a_max = np.minimum(a_ + epsilon_a, 1.0)
                a_min = np.maximum(a_ - epsilon_a, 0.0)

                gamma_min, gamma_max = mIQAE.confidence_intervals(a_min, a_max, ri)

                theta_l = (0.5 * np.pi * ri + gamma_min) / big_k_i
                theta_u = (0.5 * np.pi * ri + gamma_max) / big_k_i

                k = mIQAE.find_next_k(k_i, theta_l, theta_u)

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
        [self.ae_l, self.ae_u] = self.miqae(
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
