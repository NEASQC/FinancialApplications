"""
This module contains necesary functions and classes to implement
Real Quantum Amplitude Estimation based on the paper:

    Manzano, A., Musso, D., Leitao, A. et al.
    Real Quantum Amplitude Estimation
    Preprint

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import bitfield_to_int, check_list_type, mask


class RQAE:
    """
    Class for Real Quantum Amplitude Estimation (RQAE)
    algorithm
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class

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
            dictionary that allows the configuration of the IQAE algorithm:
            Implemented keys:
            qpu : QLM solver
                solver for simulating the resulting circutis
            q : int
                amplification ratio
            epsilon : int
                precision
            gamma : float
                accuracy
        """
        ###########################################
        # Setting attributes
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu")
        if self.linalg_qpu is None:
            print("Not QPU was provide. Default QPU will be used")
            self.linalg_qpu = get_default_qpu()
        self.epsilon = kwargs.get("epsilon", 0.01)
        self.gamma = kwargs.get("gamma", 0.05)
        # Amplification Ratio: q in the papper
        self.ratio = kwargs.get("q", 2)
        self.mcz_qlm = kwargs.get("mcz_qlm", True)

        # Creating the grover operator
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

        self.ae_l = None
        self.ae_u = None
        self.ae = None
        self.circuit_statistics = None
        self.time_pdf = None
        self.run_time = None

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

    @property
    def shifted_oracle(self):
        """
        creating shifted_oracle property
        """
        return self._shifted_oracle

    @shifted_oracle.setter
    def shifted_oracle(self, shift):
        """
        setter of the shifted_oracle property

        Parameters
        ----------
        shift : float
            shift for the oracle
        """
        self._shifted_oracle = qlm.QRoutine()
        wires = self._shifted_oracle.new_wires(self.oracle.arity + 1)
        self._shifted_oracle.apply(qlm.H, wires[-1])
        self._shifted_oracle.apply(
            qlm.RY(2 * np.arccos(shift)).ctrl(), wires[-1], wires[0]
        )
        self._shifted_oracle.apply(
            mask(
                self.oracle.arity,
                2**self.oracle.arity - 1 - bitfield_to_int(self.target),
            ).ctrl(),
            wires[-1],
            wires[: self.oracle.arity],
        )
        self._shifted_oracle.apply(qlm.X, wires[-1])
        self._shifted_oracle.apply(
            self._oracle.ctrl(), wires[-1], wires[: self._oracle.arity]
        )
        self._shifted_oracle.apply(qlm.X, wires[-1])
        self._shifted_oracle.apply(qlm.H, wires[-1])

    #####################################################################

    def first_step(self, shift: float, shots: int, gamma: float):
        """
        This function implements the first step of the RQAE paper. The result
        is a first estimation of the desired amplitude.

        Parameters
        ----------
        shift : float
            shift for the first iteration
        shots : int
            number of measurements
        gamma : float
            accuracy

        Returns
        ----------
        amplitude_min : float
           lower bound for the amplitude to be estimated
        amplitude_max : float
           upper bound for the amplitude to be estimated
        time_pdf : pandas DataFrame
            DataFrame with time information of the proccess

        """

        self.shifted_oracle = 2 * shift
        results, circuit, _, _ = get_results(
            self._shifted_oracle, self.linalg_qpu, shots=shots
        )
        start = time.time()
        step_circuit_stats = circuit.statistics()
        step_circuit_stats.update({"n_shots": shots})
        self.circuit_statistics.update({0: step_circuit_stats})
        probability_sum = results["Probability"].iloc[
            bitfield_to_int([0] + list(self.target))
        ]
        probability_diff = results["Probability"].iloc[
            bitfield_to_int([1] + list(self.target))
        ]
        epsilon_probability = RQAE.chebysev_bound(shots, gamma)

        amplitude_max = np.minimum(
            (probability_sum - probability_diff) / (4 * shift)
            + epsilon_probability / (2 * np.abs(shift)),
            1.0,
        )
        amplitude_min = np.maximum(
            (probability_sum - probability_diff) / (4 * shift)
            - epsilon_probability / (2 * np.abs(shift)),
            -1.0,
        )
        end = time.time()
        first_step_time = end - start
        # time_pdf["m_k"] = 0
        # time_pdf["rqae_overheating"] = first_step_time

        return [amplitude_min, amplitude_max]

    def run_step(self, shift: float, shots: int, gamma: float, k: int):
        """
        This function implements a step of the RQAE paper. The result
        is a refined estimation of the desired amplitude.

        Parameters
        ----------
        shift : float
            shift for the first iteration
        shots : int
            number of measurements
        gamma : float
            accuracy
        k : int
            number of amplifications

        Returns
        ----------
        amplitude_min : float
           lower bound for the amplitude to be estimated
        amplitude_max : float
           upper bound for the amplitude to be estimated
        time_pdf : pandas DataFrame
            DataFrame with time information of the proccess

        """
        self.shifted_oracle = 2 * shift

        grover_oracle = grover(
            self.shifted_oracle,
            [0] + list(self.target),
            np.arange(len(self.index) + 1),
            mcz_qlm=self.mcz_qlm,
        )
        routine = qlm.QRoutine()
        wires = routine.new_wires(self.shifted_oracle.arity)
        routine.apply(self.shifted_oracle, wires)
        for i in range(k):
            routine.apply(grover_oracle, wires)
        results, circuit, _, _ = get_results(routine, self.linalg_qpu, shots=shots)
        start = time.time()
        step_circuit_stats = circuit.statistics()
        step_circuit_stats.update({"n_shots": shots})
        self.circuit_statistics.update({k: step_circuit_stats})
        probability_sum = results["Probability"].iloc[
            bitfield_to_int([0] + list(self.target))
        ]

        epsilon_probability = RQAE.chebysev_bound(shots, gamma)
        probability_max = min(probability_sum + epsilon_probability, 1)
        probability_min = max(probability_sum - epsilon_probability, 0)
        angle_max = np.arcsin(np.sqrt(probability_max)) / (2 * k + 1)
        angle_min = np.arcsin(np.sqrt(probability_min)) / (2 * k + 1)
        amplitude_max = np.sin(angle_max) - shift
        amplitude_min = np.sin(angle_min) - shift
        end = time.time()
        first_step_time = end - start
        # time_pdf["m_k"] = k
        # time_pdf["rqae_overheating"] = first_step_time

        return [amplitude_min, amplitude_max]

    @staticmethod
    def display_information(
        ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05
    ):
        """
        This function displays information of the propoerties of the method for a given
        set of parameters

        Parameters
        ----------
        ratio: float
            amplification ratio/policy
        epsilon : float
            precision
        gamma : float
            accuracy

        Returns
        ----------
        """
        theoretical_epsilon = 0.5 * np.sin(np.pi / (2 * (ratio + 2))) ** 2
        k_max = int(
            np.ceil(
                np.arcsin(np.sqrt(2 * theoretical_epsilon))
                / np.arcsin(2 * epsilon)
                * 0.5
                - 0.5
            )
        )
        bigk_max = 2 * k_max + 1
        big_t = np.log(
            ratio
            * ratio
            * (np.arcsin(np.sqrt(2 * theoretical_epsilon)))
            / (np.arcsin(2 * epsilon))
        ) / np.log(ratio)
        gamma_i = gamma / big_t
        # This is shots for each iteration: Ni in the papper
        n_i = int(
            np.ceil(1 / (2 * theoretical_epsilon**2) * np.log(2 * big_t / gamma))
        )
        n_oracle = int(n_i / 2 * bigk_max * (1 + ratio / (ratio - 1)))
        print("-------------------------------------------------------------")
        print("Maximum number of amplifications: ", k_max)
        print("Maximum number of rounds: ", int(big_t))
        print("Number of shots per round: ", n_i)
        print("Maximum number of calls to the oracle: ", n_oracle)
        print("-------------------------------------------------------------")

    @staticmethod
    def chebysev_bound(n_samples: int, gamma: float):
        """
        Computes the length of the confidence interval for a given number of samples
        n_samples and an accuracy gamma.

        Parameters
        ----------
        n_samples : int
            number of samples
        gamma : float
            accuracy

        Returns
        ----------
        length of the confidence interval
        """
        return np.sqrt(1 / (2 * n_samples) * np.log(2 / gamma))

    def rqae(self, ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05):
        """
        This function implements the first step of the RQAE paper. The result
        is an estimation of the desired amplitude with precision epsilon
        and accuracy gamma.

        Parameters
        ----------
        ratio : int
            amplification ratio
        epsilon : int
            precision
        gamma : float
            accuracy

        Returns
        ----------
        amplitude_min : float
           lower bound for the amplitude to be estimated
        amplitude_max : float
           upper bound for the amplitude to be estimated

        """
        ######################################

        epsilon = 0.5 * epsilon
        # Always need to clean the cirucit statistics property
        self.circuit_statistics = {}
        # time_list = []
        theoretical_epsilon = 0.5 * np.sin(np.pi / (2 * (ratio + 2))) ** 2
        k_max = int(
            np.ceil(
                np.arcsin(np.sqrt(2 * theoretical_epsilon))
                / np.arcsin(2 * epsilon)
                * 0.5
                - 0.5
            )
        )
        bigk_max = 2 * k_max + 1
        big_t = np.log(
            ratio
            * ratio
            * (np.arcsin(np.sqrt(2 * theoretical_epsilon)))
            / (np.arcsin(2 * epsilon))
        ) / np.log(ratio)
        gamma_i = gamma / big_t
        # This is shots for each iteration: Ni in the papper
        n_i = int(
            np.ceil(1 / (2 * theoretical_epsilon**2) * np.log(2 * big_t / gamma))
        )
        epsilon_probability = np.sqrt(1 / (2 * n_i) * np.log(2 / gamma_i))
        shift = theoretical_epsilon / np.sin(np.pi / (2 * (ratio + 2)))
        #####################################
        # First step
        [amplitude_min, amplitude_max] = self.first_step(
            shift=shift, shots=n_i, gamma=gamma_i
        )
        epsilon_amplitude = (amplitude_max - amplitude_min) / 2
        # time_list.append(time_pdf)
        # Consecutive steps
        while epsilon_amplitude > epsilon:
            k = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            k = min(k, k_max)
            shift = -amplitude_min
            [amplitude_min, amplitude_max] = self.run_step(
                shift=shift, shots=n_i, gamma=gamma_i, k=k
            )
            # time_list.append(time_pdf)
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2

        # self.time_pdf = pd.concat(time_list)
        # self.time_pdf.reset_index(drop=True, inplace=True)
        return [2 * amplitude_min, 2 * amplitude_max]

    def run(self):
        r"""
        run method for the class.

        Returns
        ----------

        self.ae :
            amplitude estimation parameter

        """
        start = time.time()
        [self.ae_l, self.ae_u] = self.rqae(
            ratio=self.ratio, epsilon=self.epsilon, gamma=self.gamma
        )
        self.ae = (self.ae_u + self.ae_l) / 2.0
        end = time.time()
        self.run_time = end - start
        return self.ae
