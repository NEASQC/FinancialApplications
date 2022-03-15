"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains several auxiliar functions needed by other scripts
of the library

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas

"""

import qat.lang.AQASM as qlm
import numpy as np
import pandas as pd

# Convierte un entero n en un array de bits de longitud size
def bitfield(n, size):
    aux = [1 if digit == '1' else 0 for digit in bin(n)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size-right.size, 0))
    full = np.concatenate((left, right))
    return full.astype(int)

@qlm.build_gate("Mask", [int, int], arity=lambda x, y: x)
def mask(number_qubits, index):
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    bits = bitfield(index, number_qubits)
    for k in range(number_qubits):
        if bits[-k-1] == 0:
            routine.apply(qlm.X, quantum_register[k])

    return routine

def fwht_natural(array: np.array):
    """Fast Walsh-Hadamard Transform of array x in natural ordering
    The result is not normalised"""
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
    if (ordering == "natural"):
        y = fwht_natural(x)
    elif (ordering == "dyadic"):
        y = fwht_dyadic(x)
    else:
        y = fwht_sequency(x)
    return y

#Lo de Zalo
"""
Auxiliary functions
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""


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


def postprocess_results(results):
    """
    Post-processing the results of simulation of a quantum circuit
    Parameters
    ----------

    results : result object from a simulation of a quantum circuit
    Parameters
    ----------

    pdf : pandas datasframe
        results of the simulation. There are 3 different columns:
        States: posible quantum basis states
        Probability: probabilities of the different states
        Amplitude: amplitude of the different states
    """

    list_of_pdfs = []
    for sample in results:
        step_pdf = pd.DataFrame({
            'Probability': [sample.probability],
            'States': [sample.state],
            'Amplitude': [sample.amplitude],
            'Int': [sample.state.int],
            'Int_lsb': [sample.state.lsb_int]
        })
        list_of_pdfs.append(step_pdf)
    pdf = pd.concat(list_of_pdfs)
    pdf.reset_index(drop=True, inplace=True)
    return pdf

def run_job(result):
    try:
        return result.join()
        #State = PostProcessresults(result.join())
    except AttributeError:
        return result
        #State = PostProcessresults(result)

