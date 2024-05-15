"""
This module contains the mRQAE class. Given a quantum oracle operator,
this class estimates the **amplitude** of a given target state using
a modification of the RQAE that has a better theoretical performance
than the original one.

Original RQAE papper:

    *Manzano, A., Musso, D., Leitao, A.
    Real Quantum Amplitude Estimation
    EPJ Quantum Technol. 10, 2 (2023)
    https://doi.org/10.1140/epjqt/s40507-023-00159-0*


Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
#from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from QQuantLib.qpu.get_qpu import get_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import measure_state_probability, bitfield_to_int, check_list_type, mask


class mRQAE:
    """
    Class for Real Quantum Amplitude Estimation (RQAE)
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
        q : int
            amplification ratio
        epsilon : int
            precision
        gamma : float
            accuracy
        mcz_qlm : bool
            for using or not QLM implementation of the multi controlled Z
            gate
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class

        """
        ###########################################
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
        self.gamma = kwargs.get("gamma", 0.05)
        # Amplification Ratio: q in the paper
        self.ratio = kwargs.get("q", 2)
        self.mcz_qlm = kwargs.get("mcz_qlm", True)
        self.save_circuits = kwargs.get("save_circuits", False)

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
        self.schedule = {}
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.schedule_pdf = None
        self.quantum_times = []
        self.quantum_time = None
        self.circuit_dict = {}
        self.info = None

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

        """

        self.shifted_oracle = 2 * shift
        start = time.time()
        results, circuit, _, _ = get_results(
            self._shifted_oracle, self.linalg_qpu, shots=shots
        )
        if self.save_circuits:
            self.circuit_dict.update({"first_step": self._shifted_oracle})

        end = time.time()
        self.quantum_times.append(end-start)
        start = time.time()
        #probability_sum = results["Probability"].iloc[
        #    bitfield_to_int([0] + list(self.target))
        #]
        probability_sum = measure_state_probability(
            results, [0] + list(self.target)
        )

        #probability_diff = results["Probability"].iloc[
        #    bitfield_to_int([1] + list(self.target))
        #]
        probability_diff = measure_state_probability(
            results, [1] + list(self.target)
        )
        epsilon_probability = mRQAE.chebysev_bound(shots, gamma)

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

        """
        #print(shift)
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
        start = time.time()
        results, circuit, _, _ = get_results(routine, self.linalg_qpu, shots=shots)
        if self.save_circuits:
            self.circuit_dict.update(
                {"step_{}".format(k): routine}
            )
        end = time.time()
        self.quantum_times.append(end-start)
        #probability_sum = results["Probability"].iloc[
        #    bitfield_to_int([0] + list(self.target))
        #]
        probability_sum = measure_state_probability(
            results, [0] + list(self.target)
        )

        epsilon_probability = mRQAE.chebysev_bound(shots, gamma)
        probability_max = min(probability_sum + epsilon_probability, 1)
        probability_min = max(probability_sum - epsilon_probability, 0)
        angle_max = np.arcsin(np.sqrt(probability_max)) / (2 * k + 1)
        angle_min = np.arcsin(np.sqrt(probability_min)) / (2 * k + 1)
        amplitude_max = np.sin(angle_max) - shift
        amplitude_min = np.sin(angle_min) - shift
        first_step_time = end - start

        return [amplitude_min, amplitude_max]

    @staticmethod
    def compute_info(
        ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05, **kwargs
    ):
        """
        This function computes theoretical values of the IQAE algorithm.

        Parameters
        ----------
        ratio: float
            amplification ratio/policy
        epsilon : float
            precision
        gamma : float
            accuracy

        Return
        ------
        info : dict
            python dictionary with the computed information

        """
        epsilon = 0.5 * epsilon
        # Maximum amplification
        epsilon_infinity = 0.5 * np.sin(np.pi / (4 * ratio)) ** 2
        k_max = int(
            np.ceil(
                np.arcsin(np.sqrt(2 * epsilon_infinity))
                / np.arcsin(2 * epsilon)
                - 0.5
            )
        )
        bigk_max = 2 * k_max + 1
        # Maximum number of grover calls
        epsilon_p = 0.5 * np.sin(np.pi / (4 * (ratio + 2))) ** 2
        coef_0 = bigk_max * (ratio / (ratio - 1)) / (4 * epsilon_p ** 2)
        inside_log = ratio ** (ratio / (ratio - 1)) / (gamma * (ratio - 1))
        n_grover = coef_0 * np.log(4 * np.sqrt(np.e) * inside_log)
        # Shots for first iteration
        epsilon_0 = 0.5 * np.sin(np.pi/(2 * (ratio + 2)))
        gamma_0 = 0.5 * gamma * (ratio - 1) / (ratio * (2 * k_max + 1))
        shots_first_step = np.ceil(np.log(2 / gamma_0) / (2 * epsilon_0 ** 2))
        # Maximum number of oracle calls
        n_oracle = 2 * n_grover + shots_first_step
        # Maximum number of iterations
        big_t = np.log(
            2.0 * ratio * ratio
            * np.arcsin(np.sqrt(2.0 * epsilon_infinity))
            / np.arcsin(2.0 * epsilon)
        ) / np.log(ratio)

        info = {
            "epsilon_infinity": epsilon_infinity,
            "epsilon_p": epsilon_p, "epsilon_0": epsilon_0,
            "k_max": k_max, "n_0": shots_first_step,
            "n_grover": n_grover, "n_oracle": n_oracle,
            "big_t": big_t
        }

        return info
    @staticmethod
    def display_information(
        ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05, **kwargs
    ):
        """
        This function displays information of the properties of the
        method for a given set of parameters

        Parameters
        ----------
        ratio: float
            amplification ratio/policy
        epsilon : float
            precision
        gamma : float
            accuracy

        """


        info_dict = mRQAE.compute_info(
            ratio = ratio, epsilon = epsilon, gamma=gamma)
        print("-------------------------------------------------------------")
        print("Maximum number of amplifications: ", info_dict["k_max"])
        print("Maximum number of rounds: ", info_dict["big_t"])
        print("Maximum number of Grover operator calls: ", info_dict["n_grover"])
        print("Maximum number of Oracle operator calls: ", info_dict["n_oracle"])
        print("epsilon infinity: ", info_dict["epsilon_infinity"])
        print("Maximum number of shots for first step: ", info_dict["n_0"])
        print("-------------------------------------------------------------")

    @staticmethod
    def chebysev_bound(n_samples: int, gamma: float):
        """
        Computes the length of the confidence interval for a given number
        of samples n_samples and an accuracy gamma.

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

    def mrqae(self, ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05):
        """
        This function implements the first step of the RQAE paper. The
        result is an estimation of the desired amplitude with precision
        epsilon and accuracy gamma.

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

        # General parameters
        epsilon_infinity = 0.5 * np.sin(np.pi / (4 * ratio)) ** 2
        k_max = int(
            np.ceil(
                np.arcsin(np.sqrt(2 * epsilon_infinity))
                / np.arcsin(2 * epsilon)
                - 0.5
            )
        )

        ############### First Step #######################
        # First step shift
        shift_0 = 0.5
        # Bounded error first step
        epsilon_0 = np.abs(shift_0) * np.sin(np.pi / (2 * (ratio + 2)))
        if epsilon_0 < 2 * epsilon * np.abs(shift_0):
            epsilon_0 = 2 * epsilon * np.abs(shift_0)
            gamma_0 = gamma
        else:
            # Maximum circuit depth
            # gamma for first step
            gamma_0 = 0.5 * gamma * (ratio - 1) / (ratio * (2 * k_max + 1))
        # shots for first step
        n_0 = int(np.ceil(np.log(2.0 / gamma_0) / (2 * epsilon_0 ** 2)))
        # first step execution
        [amplitude_min, amplitude_max] = self.first_step(
            shift=shift_0, shots=n_0, gamma=gamma_0
        )
        self.schedule.update({0 : n_0})
        epsilon_amplitude = (amplitude_max - amplitude_min) / 2
        # print("first step. Shift ", shift_0 , "shots: ", n_0, "gamma_0: ", gamma_0)

        ############### Consecutive Steps #######################

        while epsilon_amplitude > epsilon:
            # amplification of the step
            k = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            # maximum k for step
            k = min(k, k_max)
            # new epsilon for step
            epsilon_p = 0.5 * np.sin(
                np.pi / (4 * (ratio + 2/(2 * k + 1)))
            ) ** 2
            # shift of the step
            shift = -amplitude_min
            if shift > 0:
                shift = min(shift, 0.5)
            if shift < 0:
                shift = max(shift, -0.5)

            # gamma of the step
            gamma_i = 0.5 * gamma * (ratio - 1) * (2 * k + 1) \
                / (ratio * (2 * k_max + 1))

            # number of shots of the step
            n_i = int(np.ceil(np.log(2.0 / gamma_i) / (2 * epsilon_p ** 2)))
            # Execute step
            [amplitude_min, amplitude_max] = self.run_step(
                shift=shift, shots=n_i, gamma=gamma_i, k=k
            )
            # Added the shots for the k
            if k not in self.schedule:
                self.schedule.update({k:n_i})
            else:
                # If k exists sum the shots with the before number of shots
                self.schedule.update({k:self.schedule[k] + n_i})
            # print("Step k: ", k, "Shift ", shift , "shots: ", n_i, "gamma_i: ", gamma_i)
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2

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
        [self.ae_l, self.ae_u] = self.mrqae(
            ratio=self.ratio, epsilon=self.epsilon, gamma=self.gamma
        )
        # Here we write the bounds of the method
        self.info = mRQAE.compute_info(
            ratio=self.ratio, epsilon=self.epsilon, gamma=self.gamma
        )
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
