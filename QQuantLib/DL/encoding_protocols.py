"""
This module contains a class for selecting data encoding protocols

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import warnings
import numpy as np
import qat.lang.AQASM as qlm
import QQuantLib.DL.data_loading as dl
from QQuantLib.utils.utils import test_bins

class Encoding:

    """
    Class for data encoding into the quantum circuit.

    """
    def __init__(
        self, array_function, array_probability=None, encoding=None, **kwargs
    ):
        """
        Initialize class for data encoding into the quantum circuit.

        Parameters
        ----------

        array_function : numpy array
            numpy array with the desired function for encoding into the
            Quantum Circuit:
                * MANDATORY of length = 2^n
                * MANDATORY: max(array_function) <= 1.0.
        array_probability : numpy array
            numpy array with the desired probability for encoding into
            the Quantum Circuit:
                * Is None is provided uniform distribution will be used
                * MANDATORY of length = 2^n
                * MANDATORY: sum(array_probability) <= 1.0.
                * MANDATORY: length(array_function) == length(array_probability)
        encoding : int
            Selecting the encode protocol
                * 0 : standard encoding procedure (load density as density)
                * 1 : first encoding procedure (load density as function)
                * 2 : second encoding procedure (double loading of a
                density as a density)
        kwargs : dictionary
        """
        #Inputs arrays MUST be of length 2^n
        self.n_qbits = test_bins(array_function)
        self.function = array_function
        if np.max(np.abs(self.function)) > 1.00:
            error_string = (
                "array_function not properly normalised"
                "Please divide by the max(array_function)"
            )
            raise ValueError(error_string)
        if array_probability is not None:
            if np.any(array_probability < 0):
                error_string = ("There are negative values in the probability")
                raise ValueError(error_string)
            qbits_prob = test_bins(array_probability)
            if self.n_qbits != qbits_prob:
                error_string = (
                    "Lengths of array_function"
                    "and array_probability"
                    "MUST BE equal"
                )
                raise ValueError(error_string)
            if np.sum(array_probability) > 1.00:
                error_string = (
                    "array_probability not properly normalised."
                    "Please divide by the sum(array_probability)"
                )
                raise ValueError(error_string)
        self.probability = array_probability
        self._encoding = encoding

        self.kwargs = kwargs
        self.multiplexor_bool = self.kwargs.get("multiplexor", True)
        if self.multiplexor_bool:
            self.multiplexor = "multiplexor"
        else:
            self.multiplexor = "brute_force"


        self.oracle = None
        self.p_gate = None
        self.function_gate = None
        self.target = None
        self.index = None
        self.registers = None
        self.encoding_normalization = 1.0

    @property
    def encoding(self):
        """
        creating the encoding property
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        """
        setter of the encoding property
        """
        self._encoding = value
        #Every time encoding is changed all staff will be reset
        self.reset()

    def reset(self):
        """
        Method for resetting attributes
        """
        self.oracle = None
        self.p_gate = None
        self.function_gate = None
        self.target = None
        self.index = None
        self.registers = None
        self.encoding_normalization = 1.0

    def oracle_encoding_0(self):
        r"""
        Method for creating the oracle. The probability density will be
        loaded as a probability density using the DL.load_probability
        function and the function array will be loaded with DL.load_array
        function. The SQUARE ROOT of the function array will be loaded!.

        Notes
        -----
        The encoding procedure is summarised as:

        .. math::
            |\Psi\rangle = \mathbf{U}_f\left(I\otimes \mathbf{U}_p \
            \right)|0\rangle\otimes|0\rangle_{n}

        Where :math:`\mathbf{U}_f` will encode :math:`f(x)` as a function
        (using DL.load_array function) and :math:`\mathbf{U}_p` will
        encode probability density :math:`p(x)` as a probability density
        (using DL.load_probability)

        After this protocol the quantum state is in the form:

        .. math::
            |\Psi\rangle = \sum_{i=0}^{2^{n}-1}|i\rangle_{n}\otimes \
            \sqrt{p(x_i)f(x_i)}|0\rangle \; + \; ...

        """

        self.reset()
        self.oracle = qlm.QRoutine()
        # Creation of probability loading gate
        if self.probability is not None:
            self.p_gate = dl.load_probability(
                self.probability,
                method=self.multiplexor
            )
        else:
            self.p_gate = dl.uniform_distribution(self.n_qbits)
            self.encoding_normalization = 2 ** self.p_gate.arity

        if ~np.all(self.function >= 0):
            warnings.warn('Some elements of the input array_function are negative')
        # Creation of function loading gate
        self.function_gate = dl.load_array(
            np.sqrt(np.abs(self.function)), id_name="Function", method=self.multiplexor)
        self.registers = self.oracle.new_wires(self.function_gate.arity)
        # Step 1 of Procedure: apply loading probability gate
        self.oracle.apply(self.p_gate, self.registers[: self.p_gate.arity])
        # Step 2 of Procedure: apply loading function gate
        self.oracle.apply(self.function_gate, self.registers)
        self.target = [0]
        self.index = [self.oracle.arity - 1]

    def oracle_encoding_1(self):
        r"""
        Method for creating the oracle. The probability density and the
        payoff functions will be loaded with the DL.load_array function.
        In this method a uniform distribution is used for creating the
        initial superposition of basis states.

        Notes
        -----
        The encoding procedure is summarised as:

        .. math::
            |\Psi\rangle = \big(I \otimes I \otimes H^{\otimes n}\big) \
            \left(\mathbf{U}_f \otimes I  \right) \
            \left( I \otimes \mathbf{U}_p \right)  \big(I \otimes I \
            \otimes H^{\otimes n}\big) \
            \big(|0\rangle \otimes |0\rangle \otimes|0\rangle_{n}\big)

        Where :math:`\mathbf{U}_f` encodes function :math:`f(x)` and
        :math:`\mathbf{U}_p` encodes probability density :math:`p(x)`.
        Both will be encoded as functions using the
        DL.load_array function.

        After this protocol the quantum state is in the form:

        .. math::
            |\Psi\rangle = \frac{1}{2^n} \sum_{i=0}^{2^{n}-1} \
            p(x_i)f(x_i) |0\rangle \otimes |0\rangle \otimes \
            |0\rangle_n \; + \; ...

        """
        self.reset()
        self.oracle = qlm.QRoutine()
        # For new data loading procedure we need n+2 qubits
        if self.probability is None:
            error_string = (
            "For type encoding 1 array_probability CAN NOT BE NONE"
            )
            raise ValueError(error_string)
        self.registers = self.oracle.new_wires(self.n_qbits + 2)
        # Step 2 of Procedure: apply Uniform distribution
        self.oracle.apply(
            dl.uniform_distribution(self.n_qbits), self.registers[: self.n_qbits]
        )
        # Step 3 of Procedure: apply loading function operator for loading p(x)
        self.p_gate = dl.load_array(
            self.probability,
            id_name="Probability",
            method=self.multiplexor
        )
        self.oracle.apply(
            self.p_gate, [self.registers[: self.n_qbits], self.registers[self.n_qbits]]
        )
        # Step 5 of Procedure: apply loading function operator for loading f(x)
        self.function_gate = dl.load_array(
            self.function,
            id_name="Function",
            method=self.multiplexor
        )
        self.oracle.apply(
            self.function_gate,
            [self.registers[: self.n_qbits], self.registers[self.n_qbits + 1]],
        )
        # Step 7 of Procedure: apply Uniform distribution
        self.oracle.apply(
            dl.uniform_distribution(self.n_qbits), self.registers[: self.n_qbits]
        )
        self.target = [0 for i in range(self.oracle.arity)]
        self.index = [i for i in range(self.oracle.arity)]
        self.encoding_normalization = 2 ** self.n_qbits

    def oracle_encoding_2(self):
        r"""
        Method for encoding where the probability density will be encoding
        as probability density using DL.load_probability (or a uniform
        distribution) and the function with the DL.load_array.

        Notes
        -----
        The encoding procedure is summarised as:

        .. math::
            |\Psi \rangle = \left(I\otimes \mathbf{U}_p \dagger \right) \
            \mathbf{U}_f \left(I\otimes \mathbf{U}_p \right) |0\rangle \
            \otimes|0\rangle_{n}

        Where :math:`\mathbf{U}_f` encodes function :math:`f(x)` as a \
        function (DL.load_array will be used) and :math:`\mathbf{U}_p`
        encodes probability density :math:`p(x)` as a probability
        density (DL.load_probability will be used)

        After this protocol the quantum state is in the form:

        .. math::
            |\Psi \rangle = \sum_{i=0}^{2^{n}-1} p(x_i) f(x_i) \
            |0\rangle \otimes |0\rangle_{n} \; + \; ...

        """
        self.reset()
        self.oracle = qlm.QRoutine()
        # Creation of probability loading gate
        if self.probability is not None:
            self.p_gate = dl.load_probability(
                self.probability,
                id_name="Probability",
                method=self.multiplexor
            )
        else:
            self.p_gate = dl.uniform_distribution(self.n_qbits)
            self.encoding_normalization = 2 ** self.p_gate.arity
        # Creation of function loading gate
        self.function_gate = dl.load_array(
            self.function,
            id_name="Function",
            method=self.multiplexor
            )
        self.registers = self.oracle.new_wires(self.function_gate.arity)
        # Step 1 of Procedure: apply loading probability gate
        self.oracle.apply(self.p_gate, self.registers[: self.p_gate.arity])
        # Step 2 of Procedure: apply loading function gate
        self.oracle.apply(self.function_gate, self.registers)
        # Step 3 of Procedure: apply loading probability gate
        self.oracle.apply(self.p_gate.dag(), self.registers[: self.p_gate.arity])
        self.target = [0 for i in range(self.oracle.arity)]
        self.index = [i for i in range(self.oracle.arity)]


    def run(self):
        if self.encoding is None:
            error_string = (
                "Encoding parameter MUST NOT BE None."
                "Please select 0,1 or 2 for encoding procedure!"
            )
            raise ValueError(error_string)
        if self.encoding == 0:
            self.oracle_encoding_0()
        elif self.encoding == 1:
            self.oracle_encoding_1()
        elif self.encoding == 2:
            self.oracle_encoding_2()
        else:
            raise ValueError("Problem with encoding attribute!")
