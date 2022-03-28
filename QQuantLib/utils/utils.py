"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains several auxiliar functions needed by other scripts
of the library

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas

"""

from copy import deepcopy
import numpy as np
import pandas as pd
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

def create_qprogram(quantum_gate):
    """
    Creates a Quantum Program from an input qlm gate or routine

    Parameters
    ----------

    quantum_gate : QLM gate or QLM routine

    Returns
    ----------
    q_prog: QLM Program.
        Quantum Program from input QLM gate or routine
    """
    q_prog = qlm.Program()
    qbits = q_prog.qalloc(quantum_gate.arity)
    q_prog.apply(quantum_gate, qbits)
    return q_prog

def create_circuit(prog_q):
    """
    Given a QLM program creates a QLM circuit
    """
    q_prog = deepcopy(prog_q)
    circuit = q_prog.to_circ(submatrices_only=True)
    return circuit

def create_job(circuit, shots=0, qubits=None):
    """
    Given a QLM circuit creates a QLM job
    """
    dict_job = {
        'amp_threshold': 0.0
    }
    if qubits is None:
        job = circuit.to_job(nbshots=shots, **dict_job)
    else:
        if isinstance(qubits, (list)):
            job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
        else:
            raise ValueError('qbits: sould be a list!!!')
    return job

def get_results(quantum_object, linalg_qpu, shots=0, qubits=None):
    """
    Function for testing an input gate. This fucntion creates the
    quantum program for an input gate, the correspondent circuit
    and job. Execute the job and gets the results

    Parameters
    ----------
    quantum_object : QLM Gate, Routine or Program
    linalg_qpu : QLM solver
    shots : int
        number of shots for the generated job.
        if 0 True probabilities will be computed
    qubits : list
        list with the qbits for doing the measurement when simulating
        if None measuremnt over all allocated qbits will be provided

    Returns
    ----------
    pdf_ : pandas DataFrame
        DataFrame with the results of the simulation
    circuit : QLM circuit
    q_prog : QLM Program.
    job : QLM job

    """
    if type(quantum_object) == qlm.Program:
        q_prog = deepcopy(quantum_object)
    else:
        q_prog = qlm.Program()
        qbits = q_prog.qalloc(quantum_object.arity)
        q_prog.apply(quantum_object, qbits)
    circuit = create_circuit(q_prog)
    job = create_job(circuit, shots=shots, qubits=qubits)
    result = run_job(linalg_qpu.submit(job))
    pdf_ = postprocess_results(result)
    #pdf_.sort_values('Int_lsb', inplace=True)
    return pdf_, circuit, q_prog, job

def postprocess_results(results):
    """
    Post-processing the results of simulation of a quantum circuit
    Parameters
    ----------

    results : result object from a simulation of a quantum circuit

    Returns
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
    """
    This functions receives QLM result object and try to execute
    join method. If fails return input QLM result object

    Parameters
    ----------
    result : QLM result object

    Returns
    ----------
    result : QLM result with join method executed if necesary
    """

    try:
        return result.join()
    except AttributeError:
        return result

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
    @qlm.build_gate("Q^{}".format(n_times), [], arity=qlm_gate.arity)
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

