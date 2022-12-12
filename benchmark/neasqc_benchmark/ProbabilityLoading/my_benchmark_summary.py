"""
Template for properly formating the results of a benchamark kernel
"""
from collections import OrderedDict
import psutil

def summarize_results(**kwargs):
    """
    Mandatory code for properly present the benchmark results following
    the NEASQC jsonschema
    """

    #n_qbits = [4]
    #Info with the benchmark results like a csv or a DataFrame
    #pdf = None
    #Metrics needed for reporting. Depend on the benchmark kernel
    #list_of_metrics = ["MRSE"]

    import pandas as pd
    benchmark_file = kwargs.get("benchmark_file", None)
    pdf = pd.read_csv(benchmark_file, header=[0, 1], index_col=[0, 1])
    pdf.reset_index(inplace=True)
    n_qbits = list(set(pdf["n_qbits"]))
    load_methods = list(set(pdf["load_method"]))
    list_of_metrics = [
        "KS", "KL",
        "chi2", "p_value"
    ]




    results = []
    #In the Probability Loading benchmark several qubits can be tested
    for n_ in n_qbits:
        #For selecting the different loading method using in the benchmark
        for method in load_methods:
            #Fields for benchmark test of a fixed number of qubits
            result = OrderedDict()
            result["NumberOfQubits"] = n_
            result["QubitPlacement"] = list(range(n_))
            result["QPUs"] = [1]
            result["CPUs"] = psutil.Process().cpu_affinity()

            #Select the proper data
            step_pdf = pdf[(pdf["load_method"] == method) & (pdf["n_qbits"] == n_)]

            #result["TotalTime"] = 10.0
            #result["SigmaTotalTime"] = 1.0
            #result["QuantumTime"] = 9.0
            #result["SigmaQuantumTime"] = 0.5
            #result["ClassicalTime"] = 1.0
            #result["SigmaClassicalTime"] = 0.1

            result["TotalTime"] = step_pdf["elapsed_time"]["mean"].iloc[0]
            result["SigmaTotalTime"] = step_pdf["elapsed_time"]["std"].iloc[0]
            result["QuantumTime"] = step_pdf["quantum_time"]["mean"].iloc[0]
            result["SigmaQuantumTime"] = step_pdf["quantum_time"]["std"].iloc[0]
            result["ClassicalTime"] = step_pdf["classic_time"]["mean"].iloc[0]
            result["SigmaClassicalTime"] = step_pdf["classic_time"]["std"].iloc[0]
            #For identify the loading method used. Not mandaatory but
            #useful for identify results
            result["load_method"] = method

            metrics = []
            #For each fixed number of qbits several metrics can be reported
            for metric_name in list_of_metrics:
                metric = OrderedDict()
                #MANDATORY
                metric["Metric"] = metric_name
                #metric["Value"] = 0.1
                #metric["STD"] = 0.001
                metric["Value"] = step_pdf[metric_name]["mean"].iloc[0]
                metric["STD"] = step_pdf[metric_name]["std"].iloc[0]
                #Depending on the benchmark kernel
                metric["COUNT"] = int(step_pdf[metric_name]["count"].iloc[0])
                metrics.append(metric)
            result["Metrics"] = metrics
            results.append(result)
    return results

if __name__ == "__main__":
    import json
    import jsonschema
    json_file = open("../NEASQC.Benchmark.V2.Schema_modified.json")
    schema = json.load(json_file)
    json_file.close()

    ################## Configuring the files ##########################

    configuration = {"benchmark_file" : "save_SummaryResults.csv"}

    ######## Execute Validations #####################################


    schema_bench = schema['properties']['Benchmarks']['items']['properties']
    print("Validate Results")
    print(summarize_results(**configuration))
    try:
        jsonschema.validate(
            instance=summarize_results(**configuration),
            schema=schema_bench['Results']
        )
        print("\t Results is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)
