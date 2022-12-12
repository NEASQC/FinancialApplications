"""
Scripts for execute the complete Benchmar of an
Amplitude Estimation Algorithm
"""

import sys
import json
from datetime import datetime
from scipy.stats import norm
import pandas as pd
sys.path.append("../../../")
from QQuantLib.utils.qlm_solver import get_qpu




def run_code(n_qbits, configuration, repetitions):
    """
    This method computes the integral of the sine function using
    a properly configured AE method, in a fixed interval integration
    for a domain discretization in n qubits

    Parameters
    ----------

    n_qbits : int
        number of qubits used for domain discretization
    configuration : dictionary
        dictionary with the complete configuration of the
        benchmarked algorihtm
    repetitions : list
        number of repetitions

    Returns
    _______

    metrics : pandas DataFrame
        DataFrame with the desired metrics obtained for the integral computation

    """
    from load_probabilities import LoadProbabilityDensity

    if n_qbits is None:
        raise ValueError("n_qbits CAN NOT BE None")
    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")

    list_of_metrics = []
    configuration.update({"number_of_qbits": n_qbits})
    for i in range(repetitions[0]):
        prob_dens = LoadProbabilityDensity(**configuration)
        prob_dens.exe()
        list_of_metrics.append(prob_dens.pdf)
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)
    return metrics

def compute_samples(metrics, **kwargs):
    """
    This functions computes the number of executions of the benchmark for
    assure an error r with a confidence of alpha

    Parameters
    ----------

    metrics : pandas DataFrame
        DataFrame with the results of pre-benchmark step
    kwargs : keyword arguments
        For configuring the sampling computation

    Returns
    _______

    samples : pandas DataFrame
        DataFrame with the number of executions for each integration interval

    """
    
    #Configuration for sampling computations

    #Desired Error in the benchmark metrics
    relative_error = kwargs.get("relative_error", 0.1)
    #Desired Confidence level
    alpha = kwargs.get("alpha", 0.05)
    #Minimum and Maximum number of samples
    min_meas = kwargs.get("min_meas", 5)
    max_meas = kwargs.get("max_meas", None)
    #Columns desired for sampling computing
    columns = kwargs.get("columns", None)

    #Compute mean and sd by integration interval
    std_ = metrics.groupby("load_method").std()
    std_.reset_index(inplace=True)
    print(std_)
    mean_ = metrics.groupby("load_method").mean()
    mean_.reset_index(inplace=True)
    print(mean_)
    #Metrics
    zalpha = norm.ppf(1-(alpha/2)) # 95% of confidence level
    samples_ = (zalpha * std_[columns] / (relative_error * mean_[columns]))**2
    print(samples_)
    samples_ = samples_.max(axis=1).astype(int)
    print(samples_)
    samples_.name = "samples"
    #samples_ = pd.concat([mean_["interval"], samples_], axis=1)
    samples_.clip(upper=max_meas, lower=min_meas, inplace=True)
    return list(samples_)

def summarize_resuts(csv_results):
    """
    Create summary with statistics
    """
    pdf = pd.read_csv(csv_results, index_col=0, sep=";")
    pdf["classic_time"] = pdf["elapsed_time"] - pdf["quantum_time"]
    results = pdf.groupby(["load_method", "n_qbits"]).agg(
        ["mean", "std", "count"])
    return results


class AE_BENCHMARK:
    """
    Class for execute an AE benchmark

    """


    def __init__(self, bench_conf=None, **kwargs):
        """

        Method for initializing the class

        """
        #Configurtion of benchmarked algorithm or routine
        self.bench_conf = bench_conf
        self.kwargs = kwargs
        if self.bench_conf is None:
            raise ValueError("bench_conf CAN NOT BE None")

        #Benchmark Configuration

        #Repetitions for pre benchmark step
        self.pre_samples = self.kwargs.get("pre_samples", 10)
        #Saving pre benchmark step results
        self.pre_save = self.kwargs.get("pre_save", True)
        #Name for saving the pre benchmark step results
        self.save_name = self.kwargs.get("save_name", None)
        #NNumber of qbits
        self.list_of_qbits = self.kwargs.get("list_of_qbits", [4])

        #Configure names for CSV files
        self.benchmark_times = self.save_name + "_times_benchmark.csv"
        self.csv_results = self.save_name + "_benchmark.csv"
        self.summary_results = self.save_name + "_SummaryResults.csv"

        #Attributes for metrics
        self.pre_metrics = None
        self.metrics = None


    def save(self, save, save_name, save_stuff, save_mode):
        """
        For saving stuff
        """
        if save:
            with open(save_name, save_mode) as f_pointer:
                save_stuff.to_csv(
                    f_pointer,
                    mode=save_mode,
                    header=f_pointer.tell() == 0,
                    sep=';'
                )

    def exe(self):
        """
        Execute complete Benchmark WorkFlow
        """
        start_time = datetime.now().astimezone().isoformat()
        for n_qbits in self.list_of_qbits:
            print("n_qbits: {}".format(n_qbits))
            print("\t Executing Pre-Benchmark")
            #Pre benchmark step
            pre_metrics = run_code(
                n_qbits, self.bench_conf, self.pre_samples
            )
            print(pre_metrics)
            #Save Pre-benchmark steps
            post_name = "_qubits_{}_pre.csv".format(n_qbits)
            pre_save_name = self.save_name + post_name
            self.save(self.pre_save, pre_save_name, pre_metrics, "w")

            #Compute needed samples for desired
            #statistical significance
            samples_ = compute_samples(pre_metrics, **self.kwargs)
            print("\t step samples: {}".format(samples_))
            metrics = run_code(
                n_qbits, self.bench_conf, samples_
            )
            self.save(self.save, self.csv_results, metrics, "a")
        end_time = datetime.now().astimezone().isoformat()
        pdf_times = pd.DataFrame(
            [start_time, end_time],
            index=["StartTime", "EndTime"]
        ).T
        #Saving Time Info
        pdf_times.to_csv(self.benchmark_times)
        #Summarize Results
        results = summarize_resuts(self.csv_results)
        results.to_csv(self.summary_results)

if __name__ == "__main__":

    benchmark_arguments = {
        "pre_samples": [10],
        "pre_save": True,
        "save_name": "./Results/LP",
        "relative_error": 0.1,
        "alpha": 0.05,
        "min_meas": 10,
        "max_meas": None,
        "list_of_qbits": [4],
        "columns":["elapsed_time"]
    }
    algorithm_configuration = {
       "load_method" : "brute_force"
    }
    ae_bench = AE_BENCHMARK(algorithm_configuration, **benchmark_arguments)
    ae_bench.exe()



