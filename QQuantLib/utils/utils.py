"""

This module contains several auxiliary functions needed by other scripts
of the library

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas

Fast Walsh-Hadamard Transform is based on mex function written
by Chengbo Li@Rice Uni for his TVAL3 algorithm:
    https://github.com/dingluo/fwht/blob/master/FWHT.py
"""

import time
import numpy as np
import qat.lang.AQASM as qlm

def bitfield(n_int: int, size: int):
    """Transforms an int n_int to the corresponding bitfield of size size

    Parameters
    ----------
    n_int : int
        integer from which we want to obtain the bitfield
    size : int
        size of the bitfield

    Returns
    ----------
    full : list of ints
        bitfield representation of n_int with size size

    """
    aux = [1 if digit == "1" else 0 for digit in bin(n_int)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size - right.size, 0))
    full = np.concatenate((left, right))
    return full.astype(int)


def bitfield_to_int(lista):
    """Transforms the bitfield list to the corresponding int
    Parameters
    ----------
    lista : ist of ints
        bitfield

    Returns
    ----------
    integer : int
        integer obtained from it's binary representation.
    """

    integer = 0
    for i in range(len(lista)):
        integer += lista[-i - 1] * 2**i
    return int(integer)


def check_list_type(x_input, tipo):
    """Check if a list x_input is of type tipo
    Parameters
    ----------
    x_input : list
    tipo : data type
        it has to be understandable by numpy

    Returns
    ----------
    y_output : np.array
        numpy array of type tipo.
    """
    try:
        y_output = np.array(x_input).astype(tipo, casting="safe")
    except TypeError:
        exception = "Only a list/array of " + str(tipo) + " are aceptable types"
        raise Exception(exception) from TypeError
    return y_output


def expmod(n_input: int, base: int):
    r"""For a pair of integer numbers, performs the decomposition:

    .. math::
        n_input = base^power+remainder

    Parameters
    ----------
    n_input : int
        number to decompose
    base : int
        basis

    Returns
    -------
    power : int
        power
    remainder : int
        remainder
    """
    power = int(np.floor(np.log(n_input) / np.log(base)))
    remainder = int(n_input - base**power)
    return (power, remainder)


@qlm.build_gate("Mask", [int, int], arity=lambda x, y: x)
def mask(number_qubits, index):
    r"""
    Transforms the state :math:`|index\rangle` into the state
    :math:`|11...1\rangle` of size number qubits.

    Parameters
    ----------
    number_qubits : int
    index : int

    Returns
    ----------
    mask : Qlm abstract gate
        the gate that we have to apply in order to transform
        state :math:`|index\rangle`. Note that it affects all states.
    """
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    bits = bitfield(index, number_qubits)
    for k in range(number_qubits):
        if bits[-k - 1] == 0:
            routine.apply(qlm.X, quantum_register[k])

    return routine


def fwht_natural(array: np.array):
    """Fast Walsh-Hadamard Transform of array x in natural ordering
    The result is not normalised
    Parameters
    ----------
    array : numpy array

    Returns
    ----------
    fast_wh_transform : numpy array
        Fast Walsh Hadamard transform of array x.

    """
    fast_wh_transform = array.copy()
    h_ = 1
    while h_ < len(fast_wh_transform):
        for i in range(0, len(fast_wh_transform), h_ * 2):
            for j in range(i, i + h_):
                x_ = fast_wh_transform[j]
                y_ = fast_wh_transform[j + h_]
                fast_wh_transform[j] = x_ + y_
                fast_wh_transform[j + h_] = x_ - y_
        h_ *= 2
    return fast_wh_transform


def fwht_sequency(x_input: np.array):
    """Fast Walsh-Hadamard Transform of array x_input in sequence ordering
    The result is not normalised
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3
    algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications
    of Walsh and Related Functions.
    Parameters
    ----------
    x_input : numpy array

    Returns
    ----------
    x_output : numpy array
        Fast Walsh Hadamard transform of array x_input.

    """
    n_ = x_input.size
    n_groups = int(n_ / 2)  # Number of Groups
    m_in_g = 2  # Number of Members in Each Group

    # First stage
    y_ = np.zeros((int(n_ / 2), 2))
    y_[:, 0] = x_input[0::2] + x_input[1::2]
    y_[:, 1] = x_input[0::2] - x_input[1::2]
    x_output = y_.copy()
    # Second and further stage
    for n_stage in range(2, int(np.log2(n_)) + 1):
        y_ = np.zeros((int(n_groups / 2), m_in_g * 2))
        y_[0 : int(n_groups / 2), 0 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] + x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 1 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] - x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 2 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] - x_output[1:n_groups:2, 1:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 3 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] + x_output[1:n_groups:2, 1:m_in_g:2]
        )
        x_output = y_.copy()
        n_groups = int(n_groups / 2)
        m_in_g = m_in_g * 2
    x_output = y_[0, :]
    return x_output


def fwht_dyadic(x_input: np.array):
    """Fast Walsh-Hadamard Transform of array x_input in dyadic ordering
    The result is not normalised
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3
    algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications
    of Walsh and Related Functions.
    Parameters
    ----------
    array : numpy array

    Returns
    ----------
    x_output : numpy array
        Fast Walsh Hadamard transform of array x_input.

    """
    n_ = x_input.size
    n_groups = int(n_ / 2)  # Number of Groups
    m_in_g = 2  # Number of Members in Each Group

    # First stage
    y_ = np.zeros((int(n_ / 2), 2))
    y_[:, 0] = x_input[0::2] + x_input[1::2]
    y_[:, 1] = x_input[0::2] - x_input[1::2]
    x_output = y_.copy()
    # Second and further stage
    for n_stage in range(2, int(np.log2(n_)) + 1):
        y_ = np.zeros((int(n_groups / 2), m_in_g * 2))
        y_[0 : int(n_groups / 2), 0 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] + x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 1 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 0:m_in_g:2] - x_output[1:n_groups:2, 0:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 2 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] + x_output[1:n_groups:2, 1:m_in_g:2]
        )
        y_[0 : int(n_groups / 2), 3 : m_in_g * 2 : 4] = (
            x_output[0:n_groups:2, 1:m_in_g:2] - x_output[1:n_groups:2, 1:m_in_g:2]
        )
        x_output = y_.copy()
        n_groups = int(n_groups / 2)
        m_in_g = m_in_g * 2
    x_output = y_[0, :]
    return x_output


def fwht(x_input: np.array, ordering: str = "sequency"):
    """Fast Walsh Hadamard transform of array x_input
    Works as a wrapper for the different orderings
    of the Walsh-Hadamard transforms.

    Parameters
    ----------
    x_input : numpy array
    ordering: string
        desired ordering of the transform

    Returns
    ----------
    y_output : numpy array
        Fast Walsh Hadamard transform of array x_input
        in the corresponding ordering
    """

    if ordering == "natural":
        y_output = fwht_natural(x_input)
    elif ordering == "dyadic":
        y_output = fwht_dyadic(x_input)
    else:
        y_output = fwht_sequency(x_input)
    return y_output


def test_bins(array, text="probability"):
    """
    Testing condition for numpy arrays. The length of the array must
    be 2^n with n an int.
    Parameters
    ----------

    array : np.ndarray
        Numpy Array whose dimensionality is going to test
    test : str
        String for identification purposes
    Raises
    ----------

    AssertionError
        If lengt of array is not 2^n with n an int.
    Returns
    ----------

    nqbits : int
        Minimum number of qbits mandatory for storing input array in a
        quantum state
    """
    nqbits_ = np.log2(len(array))
    condition = (nqbits_ % 2 == 0) or (nqbits_ % 2 == 1)
    condition_str = f"Length of the {text} array must be of dimension 2^n with \
        n an int. In this case is: {nqbits_}."
    assert condition, condition_str
    nqbits = int(nqbits_)
    return nqbits


def left_conditional_probability(initial_bins, probability):
    """
    This function calculate f(i) according to the Lov Grover and Terry
    Rudolph 2008 papper:
    'Creating superposition that correspond to efficiently integrable
    probability distributions'
    http://arXiv.org/abs/quant-ph/0208112v1

    Given a discretized probability and an initial number of bins
    the function splits each initial region in 2 equally regions and
    calculates the conditional probabilities for x is located in the
    left part of the new regions when x is located in the region that
    contains the corresponding left region

    Parameters
    ----------

    initial_bins : int
        Number of initial bins for splitting the input probabilities
    probability : np.darray.
        Numpy array with the probabilities to be load.
        initial_bins <= len(Probability)

    Returns
    ----------

    left_cond_prob : np.darray
        conditional probabilities of the new initial_bins+1 splits
    """
    # Initial domain division
    domain_divisions = 2 ** (initial_bins)
    if domain_divisions >= len(probability):
        raise ValueError(
            "The number of Initial Regions (2**initial_bins)\
        must be lower than len(probability)"
        )
    # Original number of bins of the probability distribution
    nbins = len(probability)
    # Number of Original bins in each one of the bins of Initial
    # domain division
    bins_by_dd = nbins // domain_divisions
    # probability for x located in each one of the bins of Initial
    # domain division
    prob4dd = [
        np.sum(probability[j * bins_by_dd : j * bins_by_dd + bins_by_dd])
        for j in range(domain_divisions)
    ]
    # Each bin of Initial domain division is splatted in 2 equal parts
    bins4_left_dd = nbins // (2 ** (initial_bins + 1))
    # probability for x located in the left bin of the new splits
    left_probabilities = [
        np.sum(probability[j * bins_by_dd : j * bins_by_dd + bins4_left_dd])
        for j in range(domain_divisions)
    ]
    # Conditional probability of x located in the left bin when x is located
    # in the bin of the initial domain division that contains the split
    # Basically this is the f(j) function of the article with
    # j=0,1,2,...2^(i-1)-1 and i the number of qubits of the initial
    # domain division
    with np.errstate(divide="ignore", invalid="ignore"):
        left_cond_prob = np.array(left_probabilities) / np.array(prob4dd)
    left_cond_prob[np.isnan(left_cond_prob)] = 0
    return left_cond_prob


def get_histogram(probability, low_limit, high_limit, nbin):
    """
    Given a function probability, convert it into a histogram. The
    function must be positive, the normalization is automatic. Note
    that instead of having an analytical expression, probability could
    just create an arbitrary vector of the right dimensions and positive
    amplitudes.  This procedure could be used to initialize any quantum
    state with real amplitudes

    Parameters
    ----------

    low_limit : float
        lower limit of the interval
    high_limit : float
        upper limit of the interval
    probability : function
        function that we want to convert to a probability mass function
        It does not have to be normalized but must be positive
        in the interval
    nbin : int
        number of bins in the interval

    Returns
    ----------

    centers : np.darray
        numpy array with the centers of the bins of the histogram
    probs : np.darray
        numpy array with the probability at the centers of the bins
        of the histogram
    """
    step = (high_limit - low_limit) / nbin
    # Center of the bin calculation
    centers = np.array([low_limit + step * (i + 1 / 2) for i in range(nbin)])
    prob_n = probability(centers)
    assert np.all(
        prob_n >= 0.0
    ), "Probabilities must be positive, so probability must be \
         a positive function"
    probs = prob_n / np.sum(prob_n)
    assert np.isclose(
        np.sum(probs), 1.0
    ), "probability is not getting \
        normalized properly"
    return centers, probs


def load_qn_gate(qlm_gate, n_times):
    """
    Create an AbstractGate by applying an input gate n times

    Parameters
    ----------

    qlm_gate : QLM gate
        QLM gate that will be applied n times
    n_times : int
        number of times the qlm_gate will be applied

    """

    @qlm.build_gate(f"Q^{n_times}_{time.time_ns()}", [], arity=qlm_gate.arity)
    def q_n_gate():
        """
        Function generator for creating an AbstractGate for apply
        an input gate n times
        Returns
        ----------
        q_rout : quantum routine
            Routine for applying n times an input gate
        """
        q_rout = qlm.QRoutine()
        q_bits = q_rout.new_wires(qlm_gate.arity)
        for _ in range(n_times):
            q_rout.apply(qlm_gate, q_bits)
        return q_rout

    return q_n_gate()

def text_is_none(variable, variable_name, variable_type=float):
    """
    Raise an exception if variable is None
    """
    if variable is None:
        message = (
            variable_name
            + " argument is None. Some "
            + str(variable_type)
            + " should be  provided"
        )
        raise ValueError(message)

def oracle_shots_calculation(m_k, n_k):
    """
    Function for computing the total number of oracle shots.

    Parameters
    ----------
    m_k : list
        list with integers. Applications of the Grover-like
        amplification operator.
    n_k : list
        list with integers. Number of shots for each value of m_k.

    Returns
    ----------
    oracle_shots : int
        Number of total oracle calls for the input schedule
    """

    oracle_shots = 0.0
    for step_k, step_n in zip(m_k, n_k):
        oracle_shots = oracle_shots + (2 * step_k + 1) * step_n
    return oracle_shots

def measure_state_probability(input_result, target):
    """
    From an input result DataFrame gets the probability of target state

    Parameters
    ----------
    input_result : Pandas DataFrame
        DataFrame with measurement results like obtained in the
        get_results function (from QQuantLib.utils.data_extracting)
    target : list
        python list with the state we want to extract

    Returns
    ----------
    output_probability : float
        Probability of the desired target state. If the state it is not
        found then 0.0 is returned.
    """
    probability = input_result[
        input_result['Int_lsb'] == bitfield_to_int(target)
    ]['Probability']
    if len(probability) == 0:
        output_probability = 0.0
    else:
        output_probability = probability.values[0]
    return output_probability
