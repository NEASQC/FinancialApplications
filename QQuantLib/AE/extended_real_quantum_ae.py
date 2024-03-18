
"""
This module contains necessary functions and classes to implement
extended Real Quantum Amplitude Estimation. This algorithm is an
extension of the RQAE paper:

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
from QQuantLib.utils.get_qpu import get_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import measure_state_probability, bitfield_to_int, check_list_type, mask


def schedule_exponential_constant(epsilon=1.0e-2, gamma=0.05, ratio=2):
    """
    Schedule for an exponential amplification schedule
    and a constant gamma schedule

    Parameters
    ----------
    epsilon: float
        Desired error for the estimation
    gamma : float
        confidence level: it will be expected that the probability
        of getting an error higher than epsilon will be lower than alpha
    ratio : float
        amplification ratio

    Returns
    ----------

    k_list : list
        Schedule for the amplification at each step of the extended RQAE
        (Grover operator applications)
    gamma_list : list
        Schedule for the confidence at each step of the extended RQAE

    """

    # This is the maximum amplification of the algorithm
    # it depends of the desired epsilon
    bigk_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0
    k_max_next = np.ceil((bigk_max_next - 1) / 2)

    # The first step allways must be 0 because in the first step
    # we do not amplify. We want the sign
    k_ = 0
    big_k = 2 * k_ + 1
    # We want a exponential schedule for the amplification
    k_next = ratio
    bigk_next = 2 * k_next + 1
    k_list = [k_]
    while bigk_next < bigk_max_next:
        k_ = k_next
        bigk = 2 * k_ + 1
        k_next = k_ * ratio
        bigk_next = 2 * k_next + 1
        k_list.append(int(np.ceil(k_)))
    # We add the maximum amplification to the schedule
    k_list.append(k_max_next)
    # For the gamma we want a constant schedule
    gamma_i = gamma / len(k_list)
    gamma_list = [gamma_i] * len(k_list)
    return k_list, gamma_list

def schedule_exponential_exponential(epsilon, gamma, ratio_epsilon, ratio_gamma):
    """
    Schedule for an exponential amplification schedule
    and a exponential gamma schedule

    Parameters
    ----------
    epsilon: float
        Desired error for the estimation
    gamma : float
        confidence level: it will be expected that the probability
        of getting an error higher than epsilon will be lower than alpha
    ratio_epsilon : float
        amplification ratio (ratio for setting the k at each step).
        Only positive ratios.
    ratio_gamma : float
        ratio for selecting the gamma at each step of the extended RQAE
        ratios can be positive or negative

    Returns
    ----------

    k_list : list
        Schedule for the amplification at each step of the extended RQAE
        (Grover operator applications)
    gamma_list : list
        Schedule for the confidence at each step of the extended RQAE

    """
    # This is the maximum amplification of the algorithm
    # it depends of the desired epsilon
    bigk_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0
    k_max_next = np.ceil((bigk_max_next - 1) / 2)
    # The first step allways must be 0 because in the first step
    # we do not amplify. We want the sign
    k_ = 0
    big_k = 2 * k_ + 1
    # We want a exponential schedule for the amplification
    k_next = np.abs(ratio_epsilon)
    bigk_next = 2 * k_next + 1
    k_list = [k_]
    while bigk_next < bigk_max_next:
        k_ = k_next
        bigk = 2 * k_ + 1
        k_next = k_ * np.abs(ratio_epsilon)
        bigk_next = 2 * k_next + 1
        k_list.append(k_)
    #Schedule for gamma: we want exponential schedule
    gamma_i = [np.abs(ratio_gamma) ** i for i in range(len(k_list))]
    cte = np.sum(gamma_i) / gamma
    gamma_list = [gm / cte for gm in gamma_i]
    if ratio_gamma < 0:
        gamma_list.reverse()
    return k_list, gamma_list

def schedule_linear_linear(epsilon, gamma, slope_epsilon, slope_gamma):
    """
    Schedule for a lineal amplification schedule
    and a lineal gamma schedule

    Parameters
    ----------
    epsilon: float
        Desired error for the estimation
    gamma : float
        confidence level: it will be expected that the probability
        of getting an error higher than epsilon will be lower than alpha
    slope_epsilon : float
        amplification slope (slope for setting the k at each step).
        Only positive slope.
    slope_gamma : float
        slope for selecting the gamma at each step of the extended RQAE
        slope can be positive or negative

    Returns
    ----------

    k_list : list
        Schedule for the amplification at each step of the extended RQAE
        (Grover operator applications)
    gamma_list : list
        Schedule for the confidence at each step of the extended RQAE
    """
    # This is the maximum amplification of the algorithm
    # it depends of the desired epsilon
    bigk_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0
    k_max_next = np.ceil((bigk_max_next - 1) / 2)

    # The first step allways must be 0 because in the first step
    # we do not amplify. We want the sign
    k_ = 0
    big_k = 2 * k_ + 1
    # We want a exponential schedule for the amplification
    k_next = k_ + slope_epsilon
    bigk_next = 2 * k_next + 1
    k_list = [k_]

    while bigk_next < bigk_max_next:
        k_ = k_next
        bigk = 2 * k_ + 1
        k_next = k_ + slope_epsilon
        bigk_next = 2 * k_next + 1
        k_list.append(k_)

    k_list.append(k_max_next)
    #Schedule for gamma: we want a linear schedule
    gamma_i = [np.abs(slope_gamma) * i + 1  for i in range(len(k_list))]
    cte = np.sum(gamma_i) / gamma
    gamma_list = [gm / cte for gm in gamma_i]
    if slope_gamma < 0:
        gamma_list.reverse()
    return k_list, gamma_list

def schedule_linear_constant(epsilon, gamma, slope_epsilon):
    """
    Schedule for a lineal amplification schedule
    and a lineal gamma schedule

    Parameters
    ----------
    epsilon: float
        Desired error for the estimation
    gamma : float
        confidence level: it will be expected that the probability
        of getting an error higher than epsilon will be lower than alpha
    slope_epsilon : float
        amplification slope (slope for setting the k at each step).
        Only positive slope.
    slope_gamma : float
        slope for selecting the gamma at each step of the extended RQAE
        slope can be positive or negative

    Returns
    ----------

    k_list : list
        Schedule for the amplification at each step of the extended RQAE
        (Grover operator applications)
    gamma_list : list
        Schedule for the confidence at each step of the extended RQAE
    """
    # This is the maximum amplification of the algorithm
    # it depends of the desired epsilon
    bigk_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0
    k_max_next = np.ceil((bigk_max_next - 1) / 2)

    # The first step allways must be 0 because in the first step
    # we do not amplify. We want the sign
    k_ = 0
    big_k = 2 * k_ + 1
    # We want a exponential schedule for the amplification
    k_next = k_ + slope_epsilon
    bigk_next = 2 * k_next + 1
    k_list = [k_]

    while bigk_next < bigk_max_next:
        k_ = k_next
        bigk = 2 * k_ + 1
        k_next = k_ + slope_epsilon
        bigk_next = 2 * k_next + 1
        k_list.append(k_)

    k_list.append(k_max_next)
    # For the gamma we want a constant schedule
    gamma_i = gamma / len(k_list)
    gamma_list = [gamma_i] * len(k_list)
    return k_list, gamma_list

def select_schedule(erqae_schedule, epsilon, gamma):
    """
    Scheduler selector.
    Parameters
    ----------
    erqae_schedule : dict
        Dictionary with configuration for scheduler
    epsilon: float
        Desired error for the estimation
    gamma : float
        confidence level: it will be expected that the probability
        of getting an error higher than epsilon will be lower than alpha
    Returns
    ----------

    k_list : list
        Schedule for the amplification at each step of the extended RQAE
        (Grover operator applications)
    gamma_list : list
        Schedule for the confidence at each step of the extended RQAE
    """
    if type(erqae_schedule) != dict:
        raise TypeError("erqae_schedule MUST BE a dictionary")

    schedule_type = erqae_schedule["type"]
    ratio_slope_k = erqae_schedule["ratio_slope_k"]
    ratio_slope_gamma = erqae_schedule["ratio_slope_gamma"]

    if schedule_type == "exp_const":
        k_list, gamma_list = schedule_exponential_constant(
            epsilon, gamma, ratio_slope_k)
    elif schedule_type == "exp_exp":
        k_list, gamma_list = schedule_exponential_exponential(
            epsilon, gamma, ratio_slope_k, ratio_slope_gamma)
    elif schedule_type == "linear_linear":
        k_list, gamma_list = schedule_linear_linear(
            epsilon, gamma, ratio_slope_k, ratio_slope_gamma)
    elif schedule_type == "linear_const":
        k_list, gamma_list = schedule_linear_constant(
            epsilon, gamma, ratio_slope_k)
    else:
        raise ValueError("Not valid schedule_type provided")
    # k_list.append(k_list[-1])
    # gamma_list.append(gamma_list[-1])
    return k_list, gamma_list



class eRQAE:
    """
    Class for extended Real Quantum Amplitude Estimation (RQAE)
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
        self.mcz_qlm = kwargs.get("mcz_qlm", True)
        self.save_circuits = kwargs.get("save_circuits", False)

        # Schedule
        self.erqae_schedule = kwargs.get("erqae_schedule", None)
        if self.erqae_schedule is None:
            raise ValueError("erqae_schedule kwargs CAN NOT BE NONE")

        self.schedule_k, self.schedule_gamma = select_schedule(
            self.erqae_schedule, self.epsilon, self.gamma)
        print(self.schedule_k)
        print(self.schedule_gamma)

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
        step_circuit_stats = circuit.statistics()
        step_circuit_stats.update({"n_shots": shots})
        self.circuit_statistics.update({0: step_circuit_stats})
        self.schedule.update({0 : shots})

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
        epsilon_probability = eRQAE.chebysev_bound(shots, gamma)

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

        return [amplitude_min, amplitude_max], circuit

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
        step_circuit_stats = circuit.statistics()
        step_circuit_stats.update({"n_shots": shots})
        self.circuit_statistics.update({k: step_circuit_stats})
        self.schedule.update({k : shots})
        #probability_sum = results["Probability"].iloc[
        #    bitfield_to_int([0] + list(self.target))
        #]
        probability_sum = measure_state_probability(
            results, [0] + list(self.target)
        )

        epsilon_probability = eRQAE.chebysev_bound(shots, gamma)
        probability_max = min(probability_sum + epsilon_probability, 1)
        probability_min = max(probability_sum - epsilon_probability, 0)
        angle_max = np.arcsin(np.sqrt(probability_max)) / (2 * k + 1)
        angle_min = np.arcsin(np.sqrt(probability_min)) / (2 * k + 1)
        amplitude_max = np.sin(angle_max) - shift
        amplitude_min = np.sin(angle_min) - shift
        first_step_time = end - start

        return [amplitude_min, amplitude_max], circuit

    def erqae( self, epsilon: float = 0.01, gamma: float = 0.05, schedule_k: list = [], schedule_gamma: list = []):
        """
        Implments the extended RQAE algorithm. The result is an estimation
        of the desired amplitude with precision epsilon and accuracy gamma.

        Parameters
        ----------
        epsilon : int
            precision
        gamma : float
            accuracy
        schedule_k : list
            list with the amplification schedule
        schedule_gamma : list
            list with the confidence schedule

        Returns
        ----------
        amplitude_min : float
           lower bound for the amplitude to be estimated
        amplitude_max : float
           upper bound for the amplitude to be estimated

        """
        ######################################
        if len(schedule_k) == 0:
            raise ValueError("The amplification schedule is empty!!")
        if len(schedule_gamma) == 0:
            raise ValueError("The gamma schedule is empty!!")

        epsilon = 0.5 * epsilon
        # Always need to clean the circuit statistics property
        self.circuit_statistics = {}
        # time_list = []
        ############### First Step #######################
        shift = 0.5 # shift [-0.5, 0.5]
        # For computing the desired epsilon to achieve we need the
        # next step scheduled amplification
        k_next = schedule_k[1]
        bigk_next = 2 * k_next + 1
        # Probability epsilon for first step based on the next step

        # scheduled amplification
        epsilon_first_p = np.abs(shift) * np.sin(0.5 * np.pi / (bigk_next + 2))

        # Maximum probability epsilon for first step
        epsilon_first_p_min = 0.5 * np.sin(0.5 * np.arcsin(2 * epsilon)) ** 2
        #print("epsilon: ", epsilon_first_p, "epsilon min: ", epsilon_first_p_min)

        #print(epsilon_first_p, epsilon_first_p_min, k_next)
        # Real probabiliy epsilon to use for first step
        epsilon_first_p = max(epsilon_first_p, epsilon_first_p_min)
        # Gamma used for first step
        gamma_first = schedule_gamma[0]
        # This is shots for each iteration: Ni in the paper
        n_first = int(
            np.ceil(np.log(2 / gamma_first) / (2 * epsilon_first_p**2))
        )
        #print("first step: epsilon_first_p: ", epsilon_first_p, "gamma_first: ", gamma_first, "n_first: ", n_first)
        # Quantum routine for first step
        [amplitude_min, amplitude_max], _ = self.first_step(
            shift=shift, shots=n_first, gamma=gamma_first
        )
        # Real amplitude epsilon
        epsilon_amplitude = (amplitude_max - amplitude_min) / 2

        ############### Consecutive Steps #######################
        print("i: ", 0, "epsilon_amplitude: ", epsilon_amplitude, "epsilon: ", epsilon)
        i = 1
        while (epsilon_amplitude > epsilon) and (i < len(schedule_k) - 1):

            # This is the amplification for the current step
            k_exp = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            bigk_exp = 2 * k_exp + 1
            # Computation of the number of shots for the current iterative step

            # We need the schedule amplification of the current step and
            # the schedule amplification of the following step
            k_next = schedule_k[i+1]
            bigk_next = 2 * k_next + 1
            k_current = schedule_k[i]
            bigk_current = 2 * k_current + 1
            # We compute the desired epsilon for the current step
            epsilon_step_p = 0.5 * np.sin(
                0.25 * np.pi * bigk_current / (bigk_next + 2)) ** 2
            # We compute the maximum epsilon achievable in the step
            epsilon_step_p_min = 0.5 * np.sin(
                0.5 * bigk_exp * np.arcsin(2 * epsilon)) ** 2
            #print("epsilon: ", epsilon_step_p, "epsilon min: ", epsilon_step_p_min)
            # Real probabiliy epsilon that should be achieved in the step
            epsilon_step_p = max(epsilon_step_p, epsilon_step_p_min)
            # gamma used for first step
            gamma_step = schedule_gamma[i]
            # This is the mandatory number of shots of the step for
            # achieving the gamma_step and the epsilon_step_pT
            n_step = int(
                np.ceil(1 / (2 * epsilon_step_p**2) * np.log(2 / gamma_step))
            )
            if bigk_exp < bigk_current:
                print("Albeto fails!")

            # Quantum routine for current step
            shift = -amplitude_min
            if shift > 0:
                shift = min(shift, 0.5)
            if shift < 0:
                shift = max(shift, -0.5)
            #print("step: epsilon_step_p: ", epsilon_step_p, "gamma_step: ",
            #    gamma_step, "n_step: ", n_step, "k_exp: ", k_exp
            #)
            [amplitude_min, amplitude_max], _ = self.run_step(
                shift=shift, shots=n_step, gamma=gamma_step, k=k_exp
            )
            # time_list.append(time_pdf)
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2
            print("i: ", i, "epsilon_amplitude: ", epsilon_amplitude, "epsilon: ", epsilon)
            i = i + 1

        if epsilon_amplitude > epsilon:
            print("Entrnado")
            # This is the amplification for the current step
            k_exp = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            k_current = schedule_k[i]
            bigk_current = 2 * k_current + 1
            bigk_exp = 2 * k_exp + 1
            # Computation of the number of shots for the current iterative step

            # We compute the maximum epsilon achievable in the step
            epsilon_step_p = 0.5 * np.sin(
                0.5 * bigk_exp * np.arcsin(2 * epsilon)) ** 2
            #print("epsilon: ", epsilon_step_p, "epsilon min: ", epsilon_step_p_min)
            # gamma used for first step
            gamma_step = schedule_gamma[i]
            # This is the mandatory number of shots of the step for
            # achieving the gamma_step and the epsilon_step_pT
            n_step = int(
                np.ceil(1 / (2 * epsilon_step_p**2) * np.log(2 / gamma_step))
            )
            if bigk_exp < bigk_current:
                print("Albeto fails!")

            # Quantum routine for current step
            shift = -amplitude_min
            if shift > 0:
                shift = min(shift, 0.5)
            if shift < 0:
                shift = max(shift, -0.5)
            #print("step: epsilon_step_p: ", epsilon_step_p, "gamma_step: ",
            #    gamma_step, "n_step: ", n_step, "k_exp: ", k_exp
            #)
            [amplitude_min, amplitude_max], _ = self.run_step(
                shift=shift, shots=n_step, gamma=gamma_step, k=k_exp
            )
            # time_list.append(time_pdf)
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2
            print("i: ", i, "epsilon_amplitude: ", epsilon_amplitude, "epsilon: ", epsilon)
            i = i + 1

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
        if len(self.schedule_k) == 0:
            raise ValueError("The amplification schedule is empty!!")
        if len(self.schedule_gamma) == 0:
            raise ValueError("The gamma schedule is empty!!")
        [self.ae_l, self.ae_u] = self.erqae(
            epsilon=self.epsilon, gamma=self.gamma,
            schedule_k=self.schedule_k, schedule_gamma=self.schedule_gamma
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
