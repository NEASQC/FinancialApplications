"""
Scripts for execute the complete Benchmar of an
Amplitude Estimation Algorithm
"""

import sys
import json
from datetime import datetime
from scipy.stats import norm
import pandas as pd
sys.path.append("../../")
from benchmark.neasqc_benchmark.ae_sine_integral import sine_integral
from benchmark.benchmark_utils import combination_for_list
from QQuantLib.utils.qlm_solver import get_qpu

def run_code(n_qbits, interval, ae_problem, repetitions):
    """
    This method computes the integral of the sine function using
    a properly configured AE method, in a fixed interval integration
    for a domain discretization in n qubits

    Parameters
    ----------

    n_qbits : int
        number of qubits used for domain discretization
    interval : int
        integer for defining the integration interval of the sin function
        For the benchmark valid values will be 0 or 2
    ae_problem : dictionary
        dictionary with the complete configuration of the
        Amplitude Estimation algorihtm
    repetitions : int
        number of repetitions for the integral

    Returns
    _______

    metrics : pandas DataFrame
        DataFrame with the desired metrics obtained for the integral computation

    """
    if n_qbits is None:
        raise ValueError("n_qbits CAN NOT BE None")
    if interval is None:
        raise ValueError("interval CAN NOT BE None")
    if repetitions is None:
        raise ValueError("samples CAN NOT BE None")
    linalg_qpu = get_qpu(False)
    ae_problem.update({"qpu": linalg_qpu})
    ae_problem.update({"save": False})

    list_of_metrics = []
    for i in range(repetitions):
        metrics, pdf = sine_integral(n_qbits, interval, ae_problem)
        #for selecting the desired metrics for the benchmark
        columns = [
            "interval", "n_qbits", "absolute_error_exact", "relative_error_exact",
            "absolute_error_sum", "oracle_calls", "elapsed_time", "run_time", "quantum_time"
        ]
        list_of_metrics.append(metrics[columns])
    metrics = pd.concat(list_of_metrics)
    metrics.reset_index(drop=True, inplace=True)
    return metrics

def select_ae(ae_method):
    """
    Function for selecting the AE algorithm used in the benchmark

    Parameters
    ----------

    ae_method : string
       Amplitude Estimation method used in the benchmark

    Returns
    _______

    final_list : list
        list where each element is a complete dictionary with the
        configuration of the Amplitude Estimation algorithm used in
        the benchmark

    """

    lista_ae = []
    if ae_method == "MLAE":
        lista_ae.append("jsons/integral_mlae_configuration.json")
    elif ae_method == "IQAE":
        lista_ae.append("jsons/integral_iqae_configuration.json")
    elif ae_method == "RQAE":
        lista_ae.append("jsons/integral_rqae_configuration.json")
    elif ae_method == "CQPEAE":
        lista_ae.append("jsons/integral_cqpeae_configuration.json")
    elif ae_method == "IQPEAE":
        lista_ae.append("jsons/integral_iqpeae_configuration.json")
    else:
        raise ValueError("ae_method MUST BE: MLAE, IQAE, RQAE, CQPEAE or IQPEAE")

    ae_list = []
    for ae_json in lista_ae:
        with open(ae_json) as json_file:
            ae_list = ae_list + json.load(json_file)
    #Creates the complete configuration for AE solvers
    final_list = combination_for_list(ae_list)
    if len(final_list) > 1:
        text = "There are more than 1 AE algorithm configuration. "\
            "FOR BENCHMARK only 1 configuration should be given. "\
            "Please change the correspondent json!!"
        raise ValueError(text)
    ae_configuration = final_list[0]
    del ae_configuration["integral"]
    del ae_configuration["number_of_tests"]
    return ae_configuration



class AE_BENCHMARK:
    """
    Class for execute an AE benchmark

    """


    def __init__(self, ae_problem=None, **kwargs):
        """

        Method for initializing the class

        """
        self.ae_problem = ae_problem
        if self.ae_problem is None:
            raise ValueError("ae_problem CAN NOT BE None")
        #AE configuration

        #Repetitions for pre benchmark step
        self.pre_samples = kwargs.get("pre_samples", 10)
        #Saving pre benchmark step results
        self.pre_save = kwargs.get("pre_save", True)
        #Name for saving the pre benchmark step results
        self.save_name = kwargs.get("save_name", None)
        #Desired Error in the benchmark metrics
        self.relative_error = kwargs.get("relative_error", 0.1)
        #Desired Confidence level
        self.alpha = kwargs.get("alpha", 0.05)
        #Minimum and Maximum number of samples
        self.min_meas = kwargs.get("min_meas", 5)
        self.max_meas = kwargs.get("max_meas", None)
        self.list_of_qbits = kwargs.get("list_of_qbits", [4])
        self.benchmark_times = self.save_name + "_times_benchmark.csv"
        self.csv_results = self.save_name + "_benchmark.csv"
        self.summary_results = self.save_name + "_SummaryResults.csv"
        self.pre_metrics = None
        self.metrics = None

    def pre_benchmark(self, n_qbits, interval):#, selected_ae)
        """
        This method executes the pre-benchmark step for computing the proper
        number of executions needed for assure statistical significance to the
        benchmark results

        Parameters
        ----------
        n_qbits : int
            number of qubits used for domain discretization
        integral : int
            integer for defining the integration interval of the sin function
            For the benchmark valid values will be 0 or 2
        selected_ae : str
            string for selecting the AE algorithm used for solving the
            benchmark integral


        """
        #Begin the initial tests for getting the number of Benchmark pre_benchmark_repetitions
        pre_metrics = run_code(
            n_qbits, interval, self.ae_problem, self.pre_samples)
        post_name = "_qubits_{}_interval_{}_pre.csv".format(n_qbits, interval)
        pre_save_name = self.save_name + post_name
        self.save(self.pre_save, pre_save_name, pre_metrics, "w")
        return pre_metrics

    def benchmark(self, n_qbits, interval, samples):#, selected_ae)
        """
        This method executes the benchmark step for computing the proper
        number of executions needed for assure statistical significance to the
        benchmark results

        Parameters
        ----------
        n_qbits : int
            number of qubits used for domain discretization
        integral : int
            integer for defining the integration interval of the sin function
            For the benchmark valid values will be 0 or 2
        selected_ae : str
            string for selecting the AE algorithm used for solving the
            benchmark integral


        """
        #Begin the initial tests for getting the number of Benchmark pre_benchmark_repetitions
        self.metrics = run_code(n_qbits, interval, self.ae_problem, samples)
        self.save(self.save, self.csv_results, self.metrics, "a")


    def compute_samples(self, metrics):
        """
        This functions computes the number of executions of the benchmark for
        assure an error r with a confidence of alpha

        Parameters
        ----------

        metrics_pdf : pandas DataFrame
            DataFrame with the results of pre-benchmark step

        Returns
        _______

        samples : pandas DataFrame
            DataFrame with the number of executions for each integration interval

        """
        #Compute mean and sd by integration interval
        std_ = metrics.groupby("interval").std()
        std_.reset_index(inplace=True)
        mean_ = metrics.groupby("interval").mean()
        mean_.reset_index(inplace=True)
        #Metrics
        columns = [
            "absolute_error_exact", "relative_error_exact", "absolute_error_sum",
            "oracle_calls", "elapsed_time"]
        zalpha = norm.ppf(1-(self.alpha/2)) # 95% of confidence level
        samples_ = (zalpha * std_[columns] / (self.relative_error * mean_[columns]))**2
        samples_ = samples_.max(axis=1).astype(int)
        samples_.name = "samples"
        samples_ = pd.concat([mean_["interval"], samples_], axis=1)
        samples_["samples"].clip(upper=self.max_meas, lower=self.min_meas, inplace=True)
        return samples_

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

    def summarize_resuts(self):
        """
        Create summary with statistics
        """
        pdf = pd.read_csv(self.csv_results, index_col=0, sep=";")
        pdf["classic_time"] = pdf["elapsed_time"] - pdf["quantum_time"]
        results = pdf.groupby(["interval", "n_qbits"]).describe()
        results.to_csv(self.summary_results)

    def exe(self):
        """
        Execute complete Benchmark WorkFlow
        """
        start_time = datetime.now().astimezone().isoformat()
        for n_qbits in self.list_of_qbits:
            print("n_qbits: {}".format(n_qbits))
            for interval in [0, 2]:
                print("\t interval: {}".format(interval))
                print("\t Executing Pre-Benchmark")
                step_pre_metrics = self.pre_benchmark(n_qbits, interval)
                step_samples = self.compute_samples(step_pre_metrics)
                samples_ = step_samples[
                    step_samples["interval"] == interval
                ]["samples"].iloc[0]
                print("\t Executing Benchmark")
                self.benchmark(n_qbits, interval, samples_)
        end_time = datetime.now().astimezone().isoformat()
        pdf_times = pd.DataFrame(
            [start_time, end_time],
            index=["StartTime", "EndTime"]
        ).T
        #Saving Time Info
        pdf_times.to_csv(self.benchmark_times)
        #Summarize Results
        self.summarize_resuts()



if __name__ == "__main__":

    AE = "IQAE"
    print(select_ae(AE))
    benchmark_arguments = {
        "pre_samples": 10,
        "pre_save": True,
        "save_name": "./Results/{}".format(AE),
        "relative_error": 0.1,
        "alpha": 0.05,
        "min_meas": 5,
        "max_meas": None,
        "list_of_qbits": [4, 6, 8],
    }
    ae_bench = AE_BENCHMARK(select_ae(AE), **benchmark_arguments)
    ae_bench.exe()
