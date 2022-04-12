"""
This module contains several auxiliar functions needed by other scripts
of the library

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas

"""

import time
import numpy as np
from scipy.stats import norm
import qat.lang.AQASM as qlm

# Convierte un entero n en un array de bits de longitud size
def bitfield(n: int, size: int):
    """ Transforms an int n to the corresponding bitfield of size size

    Parameters
    ----------
    n : int
        integer from which we want to obtain the bitfield
    size : int
        size of the bitfield

    Returns
    ----------
    full : list of ints
        bitfield representation of n with size size

    """
    aux = [1 if digit == '1' else 0 for digit in bin(n)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size-right.size, 0))
    full = np.concatenate((left, right))
    return full.astype(int)

def bitfield_to_int(lista):
    """ Transforms the bitfield lista to the corresponding int
    Parameters
    ----------
    lista : ist of ints
        bitfield

    Returns
    ----------
    integer : int
        integer obtained from it's binay representation.
    """

    integer = 0
    for i in range(len(lista)):
        integer += lista[-i-1]*2**i
    return int(integer)

def check_list_type(x, tipo):
    """ Check if a list x is of type tipo
    Parameters
    ----------
    x : list
    tipo : data type
        it has to be understandable by numpy

    Returns
    ----------
    y : np.array
        numpy array of type tipo.
    """
    try:
        y = np.array(x).astype(tipo, casting="safe")
    except TypeError:
        exception = "Only a list/array of "+str(tipo)+" are aceptable types"
        raise Exception(exception)
    return y

def expmod(n: int, b: int):
    r""" For a pair of integer numbers, performs the decomposition:
    .. math::
        n = b^p+r
    Parameters
    ----------
    n : int
        number to decompose
    b : int
        basis

    Returns
    -------
    p : int
        power
    r : int
        remainder
    """
    p = int(np.floor(np.log(n)/np.log(b)))
    r = int(n-b**p)
    return (p,r)


@qlm.build_gate("Mask", [int, int], arity=lambda x, y: x)
def mask(number_qubits, index):
    """ Transforms the state |index> into the state
    |11...1> of size number qubits.
    Parameters
    ----------
    number_qubits : int
    index : int

    Returns
    ----------
    mask : Qlm abstract gate
        the gate that we have to apply in order to transform
        state |index>. Note that it affects all states.
    """
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    bits = bitfield(index, number_qubits)
    for k in range(number_qubits):
        if bits[-k-1] == 0:
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
    a : numpy array
        Fast Walsh Hadamard transform of array x.

    """
    a = array.copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def fwht_sequency(x: np.array):
    """ Fast Walsh-Hadamard Transform of array x in sequency ordering
    The result is not normalised
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3
    algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications
    of Walsh and Related Functions.
    Parameters
    ----------
    x : numpy array

    Returns
    ----------
    x : numpy array
        Fast Walsh Hadamard transform of array x.

    """
    N = x.size
    G = int(N/2) # Number of Groups
    M = 2 # Number of Members in Each Group

    # First stage
    y = np.zeros((int(N/2), 2))
    y[:, 0] = x[0::2] + x[1::2]
    y[:, 1] = x[0::2] - x[1::2]
    x = y.copy()
    # Second and further stage
    for nStage in  range(2, int(np.log2(N))+1):
        y = np.zeros((int(G/2), M*2))
        y[0:int(G/2), 0:M*2:4] = x[0:G: 2, 0:M:2] + x[1:G:2, 0:M:2]
        y[0:int(G/2), 1:M*2:4] = x[0:G: 2, 0:M:2] - x[1:G:2, 0:M:2]
        y[0:int(G/2), 2:M*2:4] = x[0:G: 2, 1:M:2] - x[1:G:2, 1:M:2]
        y[0:int(G/2), 3:M*2:4] = x[0:G: 2, 1:M:2] + x[1:G:2, 1:M:2]
        x = y.copy()
        G = int(G/2)
        M = M*2
    x = y[0, :]
    return x

def fwht_dyadic(x: np.array):
    """ Fast Walsh-Hadamard Transform of array x in dyadic ordering
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
    x : numpy array
        Fast Walsh Hadamard transform of array x.

    """
    N = x.size
    G = int(N/2) # Number of Groups
    M = 2 # Number of Members in Each Group

    # First stage
    y = np.zeros((int(N/2), 2))
    y[:, 0] = x[0::2] + x[1::2]
    y[:, 1] = x[0::2] - x[1::2]
    x = y.copy()
    # Second and further stage
    for nStage in  range(2, int(np.log2(N))+1):
        y = np.zeros((int(G/2), M*2))
        y[0:int(G/2), 0:M*2:4] = x[0:G:2, 0:M:2] + x[1:G:2, 0:M:2]
        y[0:int(G/2), 1:M*2:4] = x[0:G:2, 0:M:2] - x[1:G:2, 0:M:2]
        y[0:int(G/2), 2:M*2:4] = x[0:G:2, 1:M:2] + x[1:G:2, 1:M:2]
        y[0:int(G/2), 3:M*2:4] = x[0:G:2, 1:M:2] - x[1:G:2, 1:M:2]
        x = y.copy()
        G = int(G/2)
        M = M*2
    x = y[0,:]
    return x

def fwht(x: np.array, ordering: str = "sequency"):
    """ Fast Walsh Hadamard transform of array x
    Works as a wrapper for the different orderings
    of the Walsh-Hadamard transforms.

    Parameters
    ----------
    x : numpy array
    ordering: string
        desired ordering of the transform

    Returns
    ----------
    y : numpy array
        Fast Walsh Hadamard transform of array x
        in the corresponding ordering
    """

    if ordering == "natural":
        y = fwht_natural(x)
    elif ordering == "dyadic":
        y = fwht_dyadic(x)
    else:
        y = fwht_sequency(x)
    return y

def test_bins(array, text='probability'):
    """
    Testing condition for numpy arrays. The length of the array must
    be 2^n with n an int.
    Parameters
    ----------

    array : np.ndarray
        Numpy Array whose dimensionality is going to test
    test : str
        String for identification purpouses
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
    condition = (nqbits_%2 == 0) or (nqbits_%2 == 1)
    condition_str = 'Length of the {} Array must be of dimension 2^n with n \
        an int. In this case is: {}.'.format(text, nqbits_)
    assert condition, condition_str
    nqbits = int(nqbits_)
    return nqbits

def left_conditional_probability(initial_bins, probability):
    """
    This function calculate f(i) according to the Lov Grover and Terry
    Rudolph 2008 papper:
    'Creating superpositions that correspond to efficiently integrable
    probability distributions'
    http://arXiv.org/abs/quant-ph/0208112v1

    Given a discretized probability and an initial number of bins
    the function splits each initial region in 2 equally regions and
    calculates the condicional probabilities for x is located in the
    left part of the new regions when x is located in the region that
    contains the corresponding left region

    Parameters
    ----------

    initial_bins : int
        Number of initial bins for spliting the input probabilities
    probability : np.darray.
        Numpy array with the probabilities to be load.
        initial_bins <= len(Probability)

    Returns
    ----------

    left_cond_prob : np.darray
        conditional probabilities of the new initial_bins+1 splits
    """
    #Initial domain division
    domain_divisions = 2**(initial_bins)
    if domain_divisions >= len(probability):
        raise ValueError('The number of Initial Regions (2**initial_bins)\
        must be lower than len(probability)')
    #Original number of bins of the probability distribution
    nbins = len(probability)
    #Number of Original bins in each one of the bins of Initial
    #domain division
    bins_by_dd = nbins//domain_divisions
    #probability for x located in each one of the bins of Initial
    #domain division
    prob4dd = [
        np.sum(probability[j*bins_by_dd:j*bins_by_dd+bins_by_dd])\
        for j in range(domain_divisions)
    ]
    #Each bin of Initial domain division is splitted in 2 equal parts
    bins4_left_dd = nbins//(2**(initial_bins+1))
    #probability for x located in the left bin of the new splits
    left_probabilities = [
        np.sum(probability[j*bins_by_dd:j*bins_by_dd+bins4_left_dd])\
        for j in range(domain_divisions)
    ]
    #Conditional probability of x located in the left bin when x is located
    #in the bin of the initial domain division that contains the split
    #Basically this is the f(j) function of the article with
    #j=0,1,2,...2^(i-1)-1 and i the number of qbits of the initial
    #domain division
    left_cond_prob = np.array(left_probabilities)/np.array(prob4dd)
    return left_cond_prob


def bs_density(s_t: float,s_0: float,r: float,volatility: float,maturity: float):
    r""" Evaluates the Black-Scholes density function at s_t
    for a given set of parameters. The formula is:

    .. math::
        \dfrac{1}{S_T\sigma\sqrt{2\pi T}}\exp\left(-\dfrac{\left(\log(S_T)-\mu\right)}{2\sigma^2T}\right)

    where :math:`\mu = (r-0.5\sigma^)T+\log(S_0)`.

    Parameters
    ----------
        s_t : float
            point where we do the evaluation
        s_0 : float
            current price
        r : float
            risk free rate
        volatility : float
            the volatility
        maturity: float
            the maturity

    Returns
    -------
        density : float
            value of the Black-Scholes denisty function
            in s_t
    """
    mean = (r-0.5*volatility*volatility)*maturity+np.log(s_0)
    factor = s_t*volatility*np.sqrt(2*np.pi*maturity)
    exponent = -(np.log(s_t)-mean)**2/(2*volatility*volatility*maturity)
    density = np.exp(exponent)/factor
    return density

def bs_probability(s_t: np.array,s_0: float,r: float,volatility: float,maturity: float):
    r""" Computes a discrete probability distribution from the  Black-Scholes 
    density function for a given set of parameters. This is done by evaluating
    the Black-Scholes density function in s_t and the normlising this result.

    Parameters
    ----------
        s_t : numpy array
            points where we define the discrete probability distribution
        s_0 : float
            current price
        r : float
            risk free rate
        volatility : float
            the volatility
        maturity: float
            the maturity

    Returns
    -------
        distribution : numpy array
            discrete probability distribution from Black-Scholes density
    """
    density = bs_density(s_t,s_0,r,volatility,maturity)
    return density/np.sum(density)

def bs_call_price(s_0: float,r: float,volatility: float,maturity: float,strike: float):
    r""" Computes the price for a european call option.
    The formula is:

    .. math::
        C(S,T) = S\Phi(d_1)-Ke^{-rT}\Phi(d_2)


    Parameters
    ----------
        s_0 : float
            current price of the underlying
        r : float
            risk free rate
        volatility : float
            the volatility
        maturity : float
            the maturity
        strike : float
            the strike

    Returns
    -------
        price : float
            price of the european call option
    """
    first = np.log(s_0/strike)
    positive = (r+volatility*volatility/2)*maturity
    negative = (r-volatility*volatility/2)*maturity
    d_1 = (first+positive)/(volatility*np.sqrt(maturity))
    d_2 = (first+negative)/(volatility*np.sqrt(maturity))
    price = s_0*norm.cdf(d_1)-strike*np.exp(-r*maturity)*norm.cdf(d_2)
    return price

def bs_put_price(s_0: float,r: float,volatility: float,maturity: float,strike: float):
    r""" Computes the price for a european put option.
    The formula is:

    .. math::
        C(S,T) = Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1)


    Parameters
    ----------
        s_0 : float
            current price of the underlying
        r : float
            risk free rate
        volatility : float
            the volatility
        maturity : float
            the maturity
        strike : float
            the strike

    Returns
    -------
        price : float
            price of the european put option
    """
    first = np.log(s_0/strike)
    positive = (r+volatility*volatility/2)*maturity
    negative = (r-volatility*volatility/2)*maturity
    d_1 = (first+positive)/(volatility*np.sqrt(maturity))
    d_2 = (first+negative)/(volatility*np.sqrt(maturity))
    price = strike*np.exp(-r*maturity)*norm.cdf(-d_2)-s_0*norm.cdf(-d_1)
    return price

def call_payoff(s_t: float,strike: float):
    r""" Computes the payoff of a european call option.
    
    .. math::
        C(S_T,K) = \left(S_T-K,0\right)^+

    Parameters
    ----------
        s_t : float
            price
        strike : float
            the strike

    Returns
    -------
        payoff : float
            the payoff
    """
    return np.maximum(s_t-strike,0)

def put_payoff(s_t: float,strike: float):
    r""" Computes the payoff of a european put option.
    
    .. math::
        P(S_T,K) = \left(K-S_T,0\right)^+

    Parameters
    ----------
        s_t : float
            price
        strike : float
            the strike

    Returns
    -------
        payoff : float
            the payoff
    """
    return np.maximum(strike-s_t,0)

def futures_payoff(s_t: float,strike: float):
    r""" Computes the payoff of a futures contract.
    
    .. math::
        F(S_T,K) = \left(S_T-K,0\right)

    Parameters
    ----------
        s_t : float
            price
        strike : float
            the strike

    Returns
    -------
        payoff : float
            the payoff
    """
    return s_t-strike



def get_histogram(p, a, b, nbin):
    """
    Given a function p, convert it into a histogram. The function must
    be positive, the normalization is automatic. Note that instead of
    having an analytical expression, p could just create an arbitrary
    vector of the right dimensions and positive amplitudes.
    This procedure could be used to initialize any quantum state
    with real amplitudes

    Parameters
    ----------

    a : float
        lower limit of the interval
    b : float
        upper limit of the interval
    p : function
        function that we want to convert to a probability mass function
        It does not have to be normalized but must be positive
        in the interval
    nbin : int
        number of bins in the interval

    Returns
    ----------

    centers : np.darray
        numpy array with the centers of the bins of the histogtram
    probs : np.darray
        numpy array with the probability at the centers of the bins
        of the histogtram
    """
    step = (b-a)/nbin
    #Center of the bin calculation
    centers = np.array([a+step*(i+1/2) for i in range(nbin)])
    prob_n = p(centers)
    assert np.all(prob_n >= 0.), 'Probabilities must be positive, so p must be \
         a positive function'
    probs = prob_n/np.sum(prob_n)
    assert np.isclose(np.sum(probs), 1.), 'probability is not getting \
        normalized properly'
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
    @qlm.build_gate("Q^{}_{}".format(n_times, time.time_ns()), [],\
    arity=qlm_gate.arity)
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

