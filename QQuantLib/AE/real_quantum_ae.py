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

from copy import deepcopy
import numpy as np
import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from qat.core.console import display
from libraries.AA.amplitude_amplification import grover
from libraries.utils.data_extracting import get_results
from libraries.utils.utils import bitfield_to_int, check_list_type, mask



class RQAE:
    """
    Class for using Maximum Likelihood Quantum Amplitude Estimation (ML-AE)
    algorithm
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class

        Parameters
        ----------

        kwars : dictionary
            dictionary that allows the configuration of the MLAE algorithm:
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
        """
        ###########################################
        self.q = 2
        self.epsilon = 0.0005/2
        self.gamma = 0.05

        self.theoretical_epsilon = 0.5*np.sin(np.pi/(2*(self.q+2)))**2
        self.k_max = int(np.ceil(np.arcsin(np.sqrt(2*self.theoretical_epsilon))/np.arcsin(2*self.epsilon)*0.5-0.5))
        self.K_max = 2*self.k_max+1
        self.T = np.log(self.q*self.q*(np.arcsin(np.sqrt(2*self.theoretical_epsilon)))/(np.arcsin(2*self.epsilon)))/np.log(self.q)
        self.gamma_i = self.gamma/self.T
        self.N = int(np.ceil(1/(2*self.theoretical_epsilon**2)*np.log(2*self.T/self.gamma)))
        self.epsilon_probability = np.sqrt(1/(2*self.N)*np.log(2/self.gamma_i))
        ###########################################
        #Setting attributes
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)

        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu')
        if self.linalg_qpu is None:
            self.linalg_qpu = get_default_qpu()

        # Optimization
    #####################################################################
    @property
    def oracle(self):
        return self._oracle

    @oracle.setter
    def oracle(self, value):
        self._oracle = deepcopy(value)

    @property
    def shifted_oracle(self):
        return self._shifted_oracle

    @shifted_oracle.setter
    def shifted_oracle(self, shift):
        self._shifted_oracle = qlm.QRoutine()
        wires = self._shifted_oracle.new_wires(self.oracle.arity+1)
        self._shifted_oracle.apply(qlm.H, wires[-1])
        self._shifted_oracle.apply(
            qlm.RY(2*np.arccos(shift)).ctrl(),
            wires[-1],
            wires[0]
        )
        self._shifted_oracle.apply(
            mask(
                self.oracle.arity,
                2**self.oracle.arity-1-bitfield_to_int(self.target)
            ).ctrl(),
            wires[-1],
            wires[:self.oracle.arity]
        )
        self._shifted_oracle.apply(qlm.X, wires[-1])
        self._shifted_oracle.apply(
            self._oracle.ctrl(),
            wires[-1],
            wires[:self._oracle.arity]
        )
        self._shifted_oracle.apply(qlm.X, wires[-1])
        self._shifted_oracle.apply(qlm.H, wires[-1])

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
    #####################################################################

    def first_step(self):
        shift = self.theoretical_epsilon/np.sin(np.pi/(2*(self.q+2)))

        self.shifted_oracle = 2*shift
        results, _, _, _ = get_results(
            self._shifted_oracle,
            self.linalg_qpu,
            shots=self.N
        )
        probability_sum = results["Probability"].iloc[bitfield_to_int([0]+list(self.target))]
        probability_diff = results["Probability"].iloc[bitfield_to_int([1]+list(self.target))]


        amplitude_max = np.minimum(
            (probability_sum-probability_diff)/(4*shift)+self.epsilon_probability/(2*np.abs(shift)),
            1.
        )
        amplitude_min = np.maximum(
            (probability_sum-probability_diff)/(4*shift)-self.epsilon_probability/(2*np.abs(shift)),
            -1.
        )

        return [amplitude_min, amplitude_max]

    def run_step(self, amplitude_min, amplitude_max):
        epsilon_amplitude = (amplitude_max-amplitude_min)/2


        k = int(np.floor(np.pi/(4*np.arcsin(2*epsilon_amplitude))-0.5))
        if k > self.k_max:
            k = self.k_max
        shift = -amplitude_min
        self.shifted_oracle = 2*shift

        grover_oracle = grover(
            self.shifted_oracle,
            [0]+list(self.target),
            np.arange(len(self.index)+1)
        )
        routine = qlm.QRoutine()
        wires = routine.new_wires(self.shifted_oracle.arity)
        routine.apply(self.shifted_oracle, wires)
        for i in range(k):
            routine.apply(grover_oracle, wires)
        results, _, _, _ = get_results(routine, self.linalg_qpu, shots=self.N)
        probability_sum = results["Probability"].iloc[bitfield_to_int(
            [0]+list(self.target)
        )]


        program = qlm.Program()
        register = program.qalloc(routine.arity)
        program.apply(routine, register)
        circuit = program.to_circ()
        #display(circuit,2)

        probability_max = min(probability_sum+self.epsilon_probability, 1)
        probability_min = max(probability_sum-self.epsilon_probability, 0)
        angle_max = np.arcsin(np.sqrt(probability_max))/(2*k+1)
        angle_min = np.arcsin(np.sqrt(probability_min))/(2*k+1)
        amplitude_max = np.sin(angle_max)-shift
        amplitude_min = np.sin(angle_min)-shift
        epsilon_amplitude = (amplitude_max-amplitude_min)/2
        amplitude_estimation = (amplitude_max+amplitude_min)/2

        print(
            "k: ", k, " a: ",
            2*amplitude_estimation,
            "+/-",
            2*epsilon_amplitude
        )

        return [amplitude_min, amplitude_max]


    def run(self):
        [amplitude_min, amplitude_max] = self.first_step()
        amplitude_estimation = (amplitude_max+amplitude_min)/2
        epsilon_amplitude = (amplitude_max-amplitude_min)/2

        while epsilon_amplitude > self.epsilon:
            [amplitude_min, amplitude_max] = self.run_step(amplitude_min, amplitude_max)
            epsilon_amplitude = (amplitude_max-amplitude_min)/2

        return [amplitude_min, amplitude_max]

