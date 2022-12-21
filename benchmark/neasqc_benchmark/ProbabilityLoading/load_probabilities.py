"""
Script for differetn ways of loading probabilities in a quantum state
"""

import sys
import time
import random
import numpy as np
import pandas as pd
from scipy.stats import norm, entropy, chisquare, chi2
from qat.lang.models import KPTree
from qat.qpus import get_default_qpu
sys.path.append("../../../")
from QQuantLib.DL.data_loading import load_probability
from QQuantLib.utils.data_extracting import get_results



def get_probabilities(n_qbits: int) -> (np.ndarray, np.ndarray, float, float, float, int):
    """
    Get the discretization of the PDF for N qubits
    """
    mean = random.uniform(-2., 2.)
    sigma = random.uniform(0.1, 2.)

    intervals = 2 ** n_qbits

    ppf_min = 0.005
    ppf_max = 0.995
    norma = norm(loc=mean, scale=sigma)
    x_ = np.linspace(norma.ppf(ppf_min), norma.ppf(ppf_max), num=intervals)
    step = x_[1] - x_[0]

    data = norma.pdf(x_)
    data = data/np.sum(data)
    mindata = np.min(data)
    shots = min(1000000, max(10000, round(100/mindata)))
    #data = np.sqrt(data)
    return x_, data, mean, sigma, float(step), shots, norma

def loading_probability(data, load_method, shots, qpu):
    """
    executing quantum stuff
    """
    if load_method == "multiplexor":
        p_gate = load_probability(data, method="multiplexor")
    elif load_method == "brute_force":
        p_gate = load_probability(data, method="brute_force")
    elif load_method == "KPTree":
        p_gate = KPTree(np.sqrt(data)).get_routine()
    else:
        error_text = "Not valid load_method argument."\
            "Select between: multiplexor, brute_force or KPTree"
        raise ValueError(error_text)
    tick = time.time()
    result, circuit, _, _ = get_results(
        p_gate,
        linalg_qpu=qpu,
        shots=shots
    )
    tack = time.time()
    quantum_time = tack - tick

    if load_method == "KPTree":
        #Use different order convention
        result.sort_values(by="Int", inplace=True)
    return result, circuit, quantum_time


class LoadProbabilityDensity:
    """
    Probability Loading
    """


    def __init__(self, **kwargs):
        """

        Method for initializing the class

        """

        self.n_qbits = kwargs.get("number_of_qbits", None)
        if self.n_qbits is None:
            error_text = "The number_of_qbits argument CAN NOT BE NONE."
            raise ValueError(error_text)
        self.load_method = kwargs.get("load_method", None)
        if self.load_method is None:
            error_text = "The load_method argument CAN NOT BE NONE."\
                "Select between: multiplexor, brute_force or KPTree"
            raise ValueError(error_text)
        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            print("Not QPU was provide. Default QPU will be used")
            self.linalg_qpu = get_default_qpu()

        self.data = None
        self.p_gate = None
        self.result = None
        self.circuit = None
        self.quantum_time = None
        self.elapsed_time = None
        #Distribution related attributes
        self.x_ = None
        self.data = None
        self.mean = None
        self.sigma = None
        self.step = None
        self.shots = None
        self.dist = None
        #Metric stuff
        self.ks = None
        self.kl = None
        self.chi2 = None
        self.pvalue = None
        self.pdf = None
        self.observed_frecuency = None
        self.expeted_frecuency = None

    def loading_probability(self):
        """
        executing quantum stuff
        """
        self.result, self.circuit, self.quantum_time = loading_probability(
            self.data, self.load_method, self.shots, self.linalg_qpu)

    def get_probabilities(self):
        """
        Computing probability densitiy array
        """
        self.x_, self.data, self.mean, self.sigma, \
            self.step, self.shots, self.dist = get_probabilities(self.n_qbits)

    def get_metrics(self):
        """
        Computing Metrics
        """
        #Kolmogorov-Smirnov
        self.ks = np.abs(
            self.result["Probability"].cumsum() - self.data.cumsum()
        ).max()
        #Kullback-Leibler divergence
        epsilon = self.data.min() * 1.0e-5
        self.kl = entropy(
            self.data,
            np.maximum(epsilon, self.result["Probability"])
        )

        #Chi square
        self.observed_frecuency = np.round(
            self.result["Probability"] * self.shots, decimals=0)
        self.expeted_frecuency = np.round(
            self.data * self.shots, decimals=0)
        try:
            self.chi2, self.pvalue = chisquare(
                f_obs=self.observed_frecuency,
                f_exp=self.expeted_frecuency
            )
        except ValueError:
            self.chi2 = np.sum(
                (self.observed_frecuency - self.expeted_frecuency) **2 / \
                    self.expeted_frecuency
            )
            count = len(self.observed_frecuency)
            self.pvalue = chi2.sf(self.chi2, count -1)

    def exe(self):
        """
        Execution of workflow
        """
        #Create the distribution for loading
        tick = time.time()
        self.get_probabilities()
        #Execute the quantum program
        self.loading_probability()
        self.get_metrics()
        tack = time.time()
        self.elapsed_time = tack - tick
        self.summary()

    def summary(self):
        """
        Pandas summary
        """
        self.pdf = pd.DataFrame()
        self.pdf["n_qbits"] = [self.n_qbits]
        self.pdf["load_method"] = [self.load_method]
        self.pdf["mean"] = [self.mean]
        self.pdf["sigma"] = [self.sigma]
        self.pdf["step"] = [self.step]
        self.pdf["shots"] = [self.shots]
        self.pdf["KS"] = [self.ks]
        self.pdf["KL"] = [self.kl]
        self.pdf["chi2"] = [self.chi2]
        self.pdf["p_value"] = [self.pvalue]
        self.pdf["elapsed_time"] = [self.elapsed_time]
        self.pdf["quantum_time"] = [self.quantum_time]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n_qbits",
        dest="n_qbits",
        type=int,
        help="Number of qbits for interval discretization.",
        default=None,
    )
    parser.add_argument(
        "-method",
        dest="method",
        type=str,
        help="For selecting the load method: multiplexor, brute_force, KPTree",
        default=None,
    )
    args = parser.parse_args()
    print(args)


    configuration = {
        "load_method" : args.method,
        "number_of_qbits": args.n_qbits
    }
    prob_dens = LoadProbabilityDensity(**configuration)
    prob_dens.exe()
    print(prob_dens.pdf)

