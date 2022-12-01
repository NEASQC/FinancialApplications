"""
"""
import psutil
import pandas as pd
from collections import OrderedDict

def summarize_results_old(benchmark_file):


    pdf = pd.read_csv(benchmark_file, index_col=0, sep=";")
    n_qbits = list(set(pdf["n_qbits"]))
    intervals = list(set(pdf["interval"]))
    results = []
    list_of_metrics = [
        "absolute_error_exact", "relative_error_exact",
        "absolute_error_sum", "oracle_calls"
    ]

    for n_ in n_qbits:
        for interval in intervals:
            result = OrderedDict()
            result["NumberOfQubits"] = n_
            result["QubitPlacement"] = list(range(n_))#[i for i in range(n_)]
            result["QPUs"] = [1]
            result["CPUs"] = psutil.Process().cpu_affinity()
            result["Interval"] = interval
            step_pdf = pdf[(pdf["interval"] == interval) & (pdf["n_qbits"] == n_)]
            #print(step_pdf)
            result["TotalTime"] = step_pdf["elapsed_time"].mean()
            result["SigmaTotalTime"] = step_pdf["elapsed_time"].std()
            result["QuantumTime"] = step_pdf["quantum_time"].mean()
            result["SigmaQuantumTime"] = step_pdf["quantum_time"].std()
            result["ClassicalTime"] = (step_pdf["elapsed_time"] - step_pdf["quantum_time"]).mean()
            result["SigmaClassicalTime"] = (step_pdf["elapsed_time"] - step_pdf["quantum_time"]).std()
            metrics = []
            for metric_name in list_of_metrics:
                metric = OrderedDict()
                metric["Metric"] = metric_name
                metric["Value"] = step_pdf[metric_name].mean()
                metric["STD"] = step_pdf[metric_name].std()
                metric["MIN"] = step_pdf[metric_name].min()
                metric["MAX"] = step_pdf[metric_name].max()
                metrics.append(metric)
            result["Metrics"] = metrics
            results.append(result)
    return results

def summarize_results(benchmark_file):


    pdf = pd.read_csv(benchmark_file, header=[0, 1], index_col=[0, 1])
    pdf.reset_index(inplace=True)
    n_qbits = list(set(pdf["n_qbits"]))
    intervals = list(set(pdf["interval"]))
    results = []
    list_of_metrics = [
        "absolute_error_exact", "relative_error_exact",
        "absolute_error_sum", "oracle_calls"
    ]

    for n_ in n_qbits:
        for interval in intervals:
            result = OrderedDict()
            result["NumberOfQubits"] = n_
            result["QubitPlacement"] = list(range(n_))#[i for i in range(n_)]
            result["QPUs"] = [1]
            result["CPUs"] = psutil.Process().cpu_affinity()
            result["Interval"] = interval
            step_pdf = pdf[(pdf["interval"] == interval) & (pdf["n_qbits"] == n_)]
            result["TotalTime"] = step_pdf["elapsed_time"]["mean"].iloc[0]#.mean()
            result["SigmaTotalTime"] = step_pdf["elapsed_time"]["std"].iloc[0]#.std()
            result["QuantumTime"] = step_pdf["quantum_time"]["mean"].iloc[0]#.mean()
            result["SigmaQuantumTime"] = step_pdf["quantum_time"]["std"].iloc[0]#.std()
            result["ClassicalTime"] = step_pdf["classic_time"]["mean"].iloc[0]
            result["SigmaClassicalTime"] = step_pdf["classic_time"]["std"].iloc[0]
            #result["ClassicalTime"] = (step_pdf["elapsed_time"] - step_pdf["quantum_time"]).mean()
            #result["SigmaClassicalTime"] = (step_pdf["elapsed_time"] - step_pdf["quantum_time"]).std()
            metrics = []
            for metric_name in list_of_metrics:
                metric = OrderedDict()
                metric["Metric"] = metric_name
                metric["Value"] = step_pdf[metric_name]["mean"].iloc[0]#.mean()
                metric["STD"] = step_pdf[metric_name]["std"].iloc[0]#.std()
                metric["MIN"] = step_pdf[metric_name]["min"].iloc[0]#.min()
                metric["MAX"] = step_pdf[metric_name]["max"].iloc[0]#.max()
                metric["COUNT"] = step_pdf[metric_name]["count"].iloc[0]#.max()
                metrics.append(metric)
            result["Metrics"] = metrics
            results.append(result)
    return results

if __name__ == "__main__":
    import json
    import jsonschema
    json_file = open("NEASQC.Benchmark.V2.Schema_modified.json")
    schema = json.load(json_file)
    json_file.close()

    schema_bench = schema['properties']['Benchmarks']['items']['properties']
    print("Validate Results")
    #print(summarize_results("Benchmark.csv"))
    try:
        jsonschema.validate(
            instance=summarize_results("./SummaryResults.csv"),
            schema=schema_bench['Results']
        )
        print("\t Results is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)
