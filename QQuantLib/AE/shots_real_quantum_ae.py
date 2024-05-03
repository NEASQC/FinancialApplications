"""
This module contains necessary functions and classes to implemeneal Quantum Amplitude Estimation based on the paper:

    Manzano, A., Musso, D., Leitao, A. et al.
    Real Quantum Amplitude Estimation
    Preprint

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


class sRQAE:
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
        self.ratio = kwargs.get("q", 3)
        self.shots = int(kwargs.get("shots", 100))
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
        self.circuit_statistics = {}
        self.time_pdf = None
        self.run_time = None
        self.schedule = {}
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.schedule_pdf = None
        self.quantum_times = []
        self.quantum_time = None
        self.info = None
        self.circuit_dict = {}

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

        probability_sum = measure_state_probability(
            results, [0] + list(self.target)
        )

        probability_diff = measure_state_probability(
            results, [1] + list(self.target)
        )
        return probability_sum, probability_diff

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
        return probability_sum


    @staticmethod
    def compute_info(
        ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05, shots: int = 100
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
        # Bounded for the error at each step
        #theoretical_epsilon = 0.5 * np.sin(np.pi / (2 * (ratio + 2))) ** 2
        theoretical_epsilon = None
        k_max = None
        bigk_max = None
        big_t = None
        # Maximum probability failure at each step
        gamma_i = None
        # This is shots for each iteration: Ni in the paper
        shots_max = None
        # Total number of Grover operator calls
        n_grover = None
        # This is the number of calls to the oracle operator (A)
        n_oracle = None

        info = {
            "theoretical_epsilon": theoretical_epsilon, "k_max": k_max,
            "big_t": big_t, "gamma_i": gamma_i, "shots_max": shots_max,
            "n_grover": n_grover, "n_oracle": n_oracle,
        }

        return info

    @staticmethod
    def display_information(
        ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05, shots: int = 100
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

        info_dict = sRQAE.compute_info(
            ratio = ratio, epsilon=epsilon, gamma=gamma)

        print("-------------------------------------------------------------")
        print("BE AWARE: In RQAE with shots the bounds depend on the shots")
        print("Here Not info is provided.")
        print("Maximum number of amplifications: ", info_dict["k_max"])
        print("Maximum number of rounds: ", info_dict["big_t"])
        print("Maximum number of Grover operator calls: ", info_dict["n_grover"])
        print("Maximum number of Oracle operator calls: ", info_dict["n_oracle"])
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

    def srqae(self, epsilon: float = 0.01, gamma: float = 0.05, user_shots: int = 100, ratio: float = 2.0):
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
        shots : int
            shots

        Returns
        ----------
        amplitude_min : float
           lower bound for the amplitude to be estimated
        amplitude_max : float
           upper bound for the amplitude to be estimated

        """
        ######################################
        
        epsilon = 0.5 * epsilon
        # time_list = []
        theoretical_epsilon = 0.5 * np.sin(np.pi / (4.0 * (ratio + 2))) ** 2
        k_max = int(
            np.ceil(
                #np.arcsin(np.sqrt(2 * theoretical_epsilon))
                0.5 * np.pi
                / np.arcsin(2 * epsilon)
                - 0.5
            )
        )
        bigk_max = 2 * k_max + 1
        big_t = np.ceil(np.log(
            ratio
            * ratio
            #* (np.arcsin(np.sqrt(2 * theoretical_epsilon)))
            * np.pi
            / (np.arcsin(2 * epsilon))
        ) / np.log(ratio))
        gamma_i = gamma / big_t
        shots_max = int(np.ceil(1 / (2 * theoretical_epsilon ** 2) * np.log(2 * big_t / gamma)))

        # Compute shift for first iteration
        shift = theoretical_epsilon / np.sin(np.pi / (2 * (ratio + 2)))
        # Compute shots for first iteration
        shots = min(shots_max, user_shots)

        ################ FIRST STEP ###################
        h_plus = 0 
        h_minus = 0 
        n_effective = 0 
        k = 0
        big_k = 2 * k + 1
        epsilon_amplitude = 0.5
        while ((big_k < ratio) and (epsilon_amplitude > epsilon)):
            # First step: return probability of sum and rest
            p_plus, p_minus =  self.first_step(
                shift=shift, shots=shots, gamma=gamma_i
            )

            # combine iterations
            h_plus = h_plus + int(p_plus * shots)
            h_minus = h_minus + int(p_minus * shots)
            n_effective = n_effective + shots
            p_plus = h_plus / n_effective
            p_minus = h_minus / n_effective
            epsilon_probability = sRQAE.chebysev_bound(n_effective, gamma_i)
            amplitude_max = np.minimum(
                (p_plus - p_minus) / (4 * shift)
                + epsilon_probability / (2 * np.abs(shift)),
                0.5,
            )
            amplitude_min = np.maximum(
                (p_plus - p_minus) / (4 * shift)
                - epsilon_probability / (2 * np.abs(shift)),
                -0.5,
            )
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2
            self.schedule.update({k:n_effective})
            k = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            big_k = 2 * k + 1
            # print(
            #     "first step. Shift ", shift , "shots: ", shots, "n_effective: ", n_effective,
            #     "amplitude_max", amplitude_max, "amplitude_min", amplitude_min
            # )
            shots = min(shots_max - n_effective, user_shots)

        ################ NEXT STEPS ###################

        k_old = 0
        big_k_old = 2 * k_old + 1
        h_k = 0
        n_effective = 0
        shots = min(shots_max, user_shots)
        while epsilon_amplitude > epsilon:
            k = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            k = min(k, k_max)
            big_k = 2 *k + 1
            
            if big_k < big_k_old * ratio:
                k = k_old
                #print("DENTRO!")
                shots = min(shots_max - n_effective, user_shots)
            else:
                h_k = 0
                n_effective = 0
                shots = min(shots_max, user_shots)
            shift = -amplitude_min
            if shift > 0:
                shift = min(shift, 0.5)
            if shift < 0:
                shift = max(shift, -0.5)
            p_sum = self.run_step(
                shift=shift, shots=shots, gamma=gamma_i, k=k
            )
            h_k = h_k + int(p_sum * shots)
            n_effective = n_effective + shots
            p_sum = h_k / n_effective
            self.schedule.update({k:n_effective})

            epsilon_probability = sRQAE.chebysev_bound(n_effective, gamma_i)
            probability_max = min(p_sum + epsilon_probability, 1)
            probability_min = max(p_sum - epsilon_probability, 0)
            angle_max = np.arcsin(np.sqrt(probability_max)) / (2 * k + 1)
            angle_min = np.arcsin(np.sqrt(probability_min)) / (2 * k + 1)
            amplitude_max = np.sin(angle_max) - shift
            amplitude_min = np.sin(angle_min) - shift
            # print(
            #     "Step k: ", k, "Shift ", shift , "shots: ", shots, "n_effective: ", n_effective,
            #     "amplitude_max", amplitude_max, "amplitude_min", amplitude_min
            # )
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2
            k_old = k 
            big_k_old = 2 * k_old + 1

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
        # print("RUN")
        # print(self.epsilon, self.gamma, self.shots, self.ratio)
        # print("RUN")
        [self.ae_l, self.ae_u] = self.srqae(
            epsilon=self.epsilon, gamma=self.gamma, user_shots=self.shots, ratio=self.ratio
        )
        self.info = self.compute_info(
            epsilon=self.epsilon, gamma=self.gamma, ratio=self.ratio, shots=self.shots
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
