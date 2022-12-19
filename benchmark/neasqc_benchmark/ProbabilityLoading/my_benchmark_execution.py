"""
For executing Probability Loading kernel benchmark
"""

import sys
import json
from datetime import datetime
import pandas as pd

def run_code(n_qbits, repetitions, **kwargs):
    """
    For configuration and execution of the benchmark kernel.

    Parameters
    ----------

    n_qbits : int
        number of qubits used for domain discretization
    repetitions : list
        number of repetitions for the integral
    kwargs : keyword arguments
        for configuration of the benchmark kernel

    Returns
    _______

    metrics : pandas DataFrame
        DataFrame with the desired metrics obtained for the integral computation

    """
    if n_qbits is None:
        raise ValueError("n_qbits CAN NOT BE None")
    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")

    #Here the code for configuring and execute the benchmark kernel

    from load_probabilities import LoadProbabilityDensity
    kernel_configuration = kwargs.get("kernel_configuration", None)
    if kernel_configuration is None:
        raise ValueError("kernel_configuration can not be None")

    list_of_metrics = []
    kernel_configuration.update({"number_of_qbits": n_qbits})
    for i in range(repetitions[0]):
        prob_dens = LoadProbabilityDensity(**kernel_configuration)
        prob_dens.exe()
        list_of_metrics.append(prob_dens.pdf)
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)

    return metrics

def compute_samples(**kwargs):
    """
    This functions computes the number of executions of the benchmark
    for assure an error r with a confidence of alpha

    Parameters
    ----------

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


    #Code for computing the number of samples for getting the desired
    #statististical significance. Depends on benchmark kernel

    from scipy.stats import norm
    #Columns desired for sampling computing
    columns = kwargs.get("columns", None)
    #geting the metrics from pre-benchmark step
    metrics = kwargs.get("pre_metrics", None)

    print("###############################")
    print(metrics)
    print("###############################")
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

    #If user wants limit the number of samples

    #Minimum and Maximum number of samples
    min_meas = kwargs.get("min_meas", 5)
    max_meas = kwargs.get("max_meas", None)
    print(samples_)
    samples_.clip(upper=max_meas, lower=min_meas, inplace=True)
    return list(samples_)

def summarize_resuts(csv_results):
    """
    Create summary with statistics
    """

    #Code for summarize the benchamark results. Depending of the
    #kernel of the benchmark

    pdf = pd.read_csv(csv_results, index_col=0, sep=";")
    pdf["classic_time"] = pdf["elapsed_time"] - pdf["quantum_time"]
    results = pdf.groupby(["load_method", "n_qbits"]).agg(
        ["mean", "std", "count"])

    return results

class KERNEL_BENCHMARK:
    """
    Class for execute a Kernerl benchmark

    """


    def __init__(self, **kwargs):
        """

        Method for initializing the class

        """
        #Configurtion of benchmarked algorithm or routine
        self.kwargs = kwargs

        #Benchmark Configuration

        #Repetitions for pre benchmark step
        self.pre_samples = self.kwargs.get("pre_samples", 10)
        #Saving pre benchmark step results
        self.pre_save = self.kwargs.get("pre_save", True)
        #For executing or not the benchmark step
        self.pre_benchmark = self.kwargs.get("pre_benchmark", True)

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

            if self.pre_benchmark:
                print("\t Executing Pre-Benchmark")
                #Pre benchmark step
                pre_metrics = run_code(
                    n_qbits, self.pre_samples, **self.kwargs
                )
                #Save Pre-benchmark steps
                post_name = "_qubits_{}_pre.csv".format(n_qbits)
                pre_save_name = self.save_name + post_name
                self.save(self.pre_save, pre_save_name, pre_metrics, "w")
                #Using pre benchmark results for computing the number of
                #repetitions
                self.kwargs.update({"pre_metrics": pre_metrics})

            #Compute needed samples for desired
            #statistical significance
            samples_ = compute_samples(**self.kwargs)
            print("\t step samples: {}".format(samples_))
            metrics = run_code(
                n_qbits, samples_, **self.kwargs
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
        "pre_benchmark": True,
        "pre_samples": [10],
        "pre_save": True,
        "save_name": "Results/PL_brute_force",
        "relative_error": 0.1,
        "alpha": 0.05,
        "min_meas": 5,
        "max_meas": None,
        "list_of_qbits": [4, 6],
    }

    #Columns for metrics
    benchmark_arguments.update({
        "columns":["elapsed_time", "KS", "KL", "chi2"]
    })
        

    #Configuration for the benchmark kernel
    kernel_configuration = {"load_method" : "brute_force"}
    benchmark_arguments.update({"kernel_configuration": kernel_configuration})
    ae_bench = KERNEL_BENCHMARK(**benchmark_arguments)
    ae_bench.exe()

