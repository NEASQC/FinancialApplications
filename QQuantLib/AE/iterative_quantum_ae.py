"""
This module contains necesary functions and classes to implement
Iterative Quantum Amplitude Estimation based on the paper:

    Grinko, D., Gacon, J., Zoufal, C. et al.
    Iterative Quantum Amplitude Estimation
    npj Quantum Inf 7, 52 (2021).
    https://doi.org/10.1038/s41534-021-00379-1

Author: Gonzalo Ferro Costas & Alberto Manzano Herrero

"""

from copy import deepcopy
import sys
import numpy as np
import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import bitfield_to_int, check_list_type, mask
from qat.core.console import display



class IQAE:
    """
    Class for Iterative Quantum Amplitude Estimation (IQAE)
    algorithm
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
        #Setting attributes
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int) 
        self._grover_oracle = grover(self.oracle,self.target,self.index)

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

    @staticmethod
    def find_next_k(k: int,theta_lower: float,theta_upper: float,flag: bool, r: float = 2):
        """
        This is an implementation of Algorithm 2 from the IQAE paper. This function computes
        the next suitable k.

        Parameters
        ----------
        k : int
            number of times to apply the grover operator to the quantum circuit
        theta_lower : float
            lower bound for the estimation of the angle
        theta_upper : float
            upper bound for the estimation of the angle
        flag : bool
            flag to keep track of weather we are in the 
            upper or lower half pane
        r : float
            ratio of amplifications between consecutive iterations

        Returns
        ----------
        k : int
            number of times to apply the grover operator to the quantum circuit
        flag : bool
            flag to keep track of weather we are in the 
            upper or lower half pane
            
        """
        K_i = 4*k+2
        theta_min = K_i*theta_lower
        theta_max = K_i*theta_upper
        K_max = np.floor(np.pi/(theta_upper-theta_lower))
        K = K_max-np.mod(K_max-2,4)
        while (K>r*K_i):
            q = K/K_i
            if (np.mod(q*theta_max,2*np.pi)<=np.pi) and (np.mod(q*theta_min,2*np.pi)<=np.pi): 
                K_next = K
                flag = True
                k_next = (K_next-2)/4
                return [int(k_next),flag]
            if (np.mod(q*theta_max,2*np.pi)>=np.pi) and (np.mod(q*theta_min,2*np.pi)>=np.pi): 
                K_next = K
                flag = False
                k_next = (K_next-2)/4
                return [int(k_next),flag]

            K = K-4
        return [int(k),flag]
    
    @staticmethod
    def invert_sector(a_min: float ,a_max: float,flag: bool = True):
        r"""
        This function inverts the expression:

        Notes
        -----
        .. math::
            a = \dfrac{1-\cos(\theta)}{2}
            
        for a pair of bounds (a_min,a_max). The result
        belongs to the domain (0,2\pi)

        Parameters
        ----------
        a_min : float
            lower bound
        a_max : float
            upper bound
        flag : bool
            flag to keep track of weather we are in the 
            upper or lower half pane

        Returns
        ----------
        theta_min : float
           lower bound for the associated angle
        theta_max : float
           upper bound for the associated angle
            
        """
        theta_1 = np.minimum(np.arccos(1-2*a_min),np.arccos(1-2*a_max))
        theta_2 = np.maximum(np.arccos(1-2*a_min),np.arccos(1-2*a_max))
        if (flag):
            theta_min = theta_1
            theta_max = theta_2
        else:
            theta_min = 2*np.pi-theta_2
            theta_max = 2*np.pi-theta_1

        return [theta_min,theta_max]

    @staticmethod
    def display_information(epsilon: float = 0.01,N: int = 100,alpha: float = 0.05):
        """
        This function displays information of the propoerties of the method for a given
        set of parameters

        Parameters
        ----------
        epsilon : float
            precision
        alpha : float
            accuracy
        N : int
            number of measurements on each iteration

        Returns
        ----------
        """
        T = np.ceil(np.log2(np.pi/(8*epsilon)))
        N_max = 32/(1-2*np.sin(np.pi/14))**2*np.log(2/alpha*np.log2(np.pi/(4*epsilon)))
        N_oracle = 50/epsilon*np.log(2/alpha*np.log2(np.pi/(4*epsilon)))
        L = (np.arcsin(2/N*np.log(2*T/epsilon)))**0.25
        k_max = L/epsilon/2

        print("-------------------------------------------------------------")
        print("Maximum number of rounds: ",int(T))
        print("Maximum number of shots per round needed: ",int(N_max))
        print("Maximum number of amplifications: ",int(k_max))
        print("Maximum number of calls to the oracle: ",int(N_oracle))
        print("-------------------------------------------------------------")
    
    @staticmethod
    def chebysev_bound(N: int,gamma: float):
        """
        Computes the length of the confidence interval for a given number of samples
        N and an accuracy gamma.
        
        Parameters
        ----------
        N : int
            number of samples
        gamma : float
            accuracy 

        Returns
        ----------
        length of the confidence interval
        """
        return np.sqrt(1/(2*N)*np.log(2/gamma))

    
    

    def run(self,epsilon: float = 0.01,N: int = 100,alpha: float = 0.05):
        """
        This function implements Algorithm 1 from the IQAE paper. The result
        is an estimation of the desired probability with precision at least
        epsilon and accuracy at least alpha.

        Parameters
        ----------
        epsilon : float
            precision
        alpha : float
            accuracy
        N : int
            number of measurements on each iteration

        Returns
        ----------
        a_l : float
           lower bound for the probability to be estimated
        a_u : float
           upper bound for the probability to be estimated
            
        """
        #####################################################
        i = 0
        k = int(0)
        flag = True
        [theta_l,theta_u] = [0.0,np.pi/2]
        T = int(np.ceil(np.log2(np.pi/(8*epsilon)))+1)
        L = (np.arcsin(2/N*np.log(2*T/epsilon)))**0.25
        #####################################################
        h_k = 0
        N_effective = 0

        while (theta_u-theta_l>2*epsilon):
            i = i+1
            k_old = k
            [k,flag] = self.find_next_k(k_old,theta_l,theta_u,flag)
            K = 4*k+2

            #####################################################
            routine = qlm.QRoutine()
            wires = routine.new_wires(self.oracle.arity)
            routine.apply(self.oracle,wires)
            for j in range(k):
                routine.apply(self._grover_oracle,wires)
            results,_,_,_ = get_results(routine,linalg_qpu = self.linalg_qpu,shots = N,qubits = self.index)
            a = results["Probability"].iloc[bitfield_to_int(self.target)]
            #####################################################
            # Agregate results from different iterations
            if (k == k_old):
                h_k = h_k+int(a*N)
                N_effective = N_effective+N
                a = h_k/N_effective 
                i = i-1
            else:
                h_k = int(a*N)
                N_effective = N
            

            # Compute the rest
            epsilon_a = IQAE.chebysev_bound(N_effective,alpha/T)
            a_max = np.minimum(a+epsilon_a,1.0)
            a_min = np.maximum(a-epsilon_a,0.0)
            [theta_min,theta_max] = self.invert_sector(a_min,a_max,flag)
            theta_l = (2*np.pi*np.floor(K*theta_l/(2*np.pi))+theta_min)/K
            theta_u = (2*np.pi*np.floor(K*theta_u/(2*np.pi))+theta_max)/K
            
        
        
        [a_l,a_u] = [np.sin(theta_l)**2,np.sin(theta_u)**2]
        return [a_l,a_u]








