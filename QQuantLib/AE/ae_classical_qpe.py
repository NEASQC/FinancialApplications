"""
This module contains a wrapper class of the PE_QFT class from
QQuantLib/PE/phase_estimation_wqft module for adapting classical
phase estimation algorithm to solve amplitude estimation problems.
Following references were used:

    Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
    Quantum amplitude amplification and estimation.
    AMS Contemporary Mathematics Series, 305.
    https://arxiv.org/abs/quant-ph/0005055v1

    NEASQC deliverable: D5.1: Review of state-of-the-art for Pricing
    and Computation of VaR

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

import time
from copy import deepcopy
import numpy as np
import qat.lang.AQASM as qlm
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.PE.classical_qpe import CQPE
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.utils import check_list_type


class CQPEAE:
    """
    Class for doing Amplitude Estimation (AE) using classical Quantum
    Amplitude Estimation (with QFT) algorithm

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
        dictionary that allows the configuration of the CQPEAE algorithm: \\
        Implemented keys:

        qpu : QLM solver
            solver for simulating the resulting circuits
        shots : int
            number of measurements
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
        self.auxiliar_qbits_number = kwargs.get("auxiliar_qbits_number", 8)
        self.shots = int(kwargs.get("shots", 100))

        self.mcz_qlm = kwargs.get("mcz_qlm", True)
        # First thing is create the grover operator from the oracle
        self._grover_oracle = grover(
            self.oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

        # For storing results
        self.theta = None
        self.ae = None
        self.ae_l = None
        self.ae_u = None
        self.cqpe = None
        self.final_results = None
        self.circuit_statistics = None
        self.run_time = None
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

    #####################################################################

    def run(self):
        r"""
        run method for the class.

        Parameters
        ----------

        Returns
        ----------

        result :
            the estimation of a

        Notes
        -----
        .. math::
            a = \cos^2(\theta)

        Where :math:`\theta`  is:

        .. math::
            \mathcal{Q}|\Psi\rangle = e^{2i\theta}|\Psi\rangle

        And :math:`\mathcal{Q}` the Grover Operator


        """
        start = time.time()
        self.circuit_statistics = {}
        dict_pe_qft = {
            "initial_state": self.oracle,
            "unitary_operator": self._grover_oracle,
            "auxiliar_qbits_number": self.auxiliar_qbits_number,
            "shots": self.shots,
            "qpu": self.linalg_qpu,
        }

        self.cqpe = CQPE(**dict_pe_qft)
        self.cqpe.run()
        step_circuit_stats = self.cqpe.circuit.to_circ().statistics()
        step_circuit_stats.update({"n_shots": self.shots})
        self.circuit_statistics = {"CQPEAE": step_circuit_stats}

        self.final_results = self.cqpe.result
        self.final_results["theta"] = np.pi * self.final_results["lambda"]
        self.final_results["theta_90"] = self.final_results["theta"]
        self.final_results['theta_90'].where(
            self.final_results["theta_90"] < 0.5 * np.pi,
            np.pi - self.final_results["theta_90"],
            inplace=True,
        )

        self.final_results.sort_values(
            "Probability",
            ascending=False,
            inplace=True,
        )
        self.theta = self.final_results["theta_90"].iloc[0]
        self.ae = np.cos(self.theta) ** 2
        end = time.time()
        self.run_time = end - start
        #Total number of oracle calls
        self.oracle_calls = self.shots * np.sum(
            [2 * (2 ** i) + 1 for i in range(self.auxiliar_qbits_number)]
        )
        #Maximum number of oracle applications
        self.max_oracle_depth = 2 ** (int(self.auxiliar_qbits_number)-1) + 1
        self.quantum_times = self.cqpe.quantum_times
        self.quantum_time = sum(self.cqpe.quantum_times)

        return self.ae
