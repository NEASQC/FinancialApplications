"""
This module contains a wraper class of the PE_QFT class from
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

from copy import deepcopy
import numpy as np
import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from QQuantLib.PE.classical_qpe import CQPE
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.utils import check_list_type


class CQPEAE:
    """
    Class for doing Amplitude Estimation (AE) using classical Quantum
    Amplitude Estimation (with QFT) algorithm
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
        """
        # Setting attributes
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)
        # First thing is create the grover operator from the oracle
        self._grover_oracle = grover(self.oracle, self.target, self.index)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)  # , get_qpu())
        if self.linalg_qpu is None:
            print("Not QPU was provide. Default QPU will be used")
            self.linalg_qpu = get_default_qpu()
        self.auxiliar_qbits_number = kwargs.get("auxiliar_qbits_number", 8)
        self.shots = kwargs.get("shots", 100)

        # For storing results
        self.theta = None
        self.ae = None
        self.cqpe = None
        self.final_results = None

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
        self._grover_oracle = grover(self.oracle, self.target, self.index)

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
        self._grover_oracle = grover(self.oracle, self.target, self.index)

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
            \; where \; \theta \; is \;
            \mathcal{Q}|\Psi\rangle = e^{2i\theta}|\Psi\rangle
            \; and \; \mathcal{Q} \; the \; Grover \; Operator


        """
        dict_pe_qft = {
            "initial_state": self.oracle,
            "unitary_operator": self._grover_oracle,
            "auxiliar_qbits_number": self.auxiliar_qbits_number,
            "shots": self.shots,
            "qpu": self.linalg_qpu,
        }

        self.cqpe = CQPE(**dict_pe_qft)
        self.cqpe.pe_qft()
        self.final_results = self.cqpe.final_results
        self.final_results.sort_values(
            "Probability",
            ascending=False,
            inplace=True,
        )
        self.theta = self.final_results["theta_90"].iloc[0]
        self.ae = np.cos(self.theta) ** 2
        return self.ae
