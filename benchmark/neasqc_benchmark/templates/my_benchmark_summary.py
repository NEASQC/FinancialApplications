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

    n_qbits = [4]
    #Info with the benchmark results like a csv or a DataFrame
    pdf = None
    #Metrics needed for reporting. Depend on the benchmark kernel
    list_of_metrics = ["MRSE"]

    results = []
    #If several qbits are tested
    for n_ in n_qbits:
        #Fields for benchmark test of a fixed number of qubits
        result = OrderedDict()
        result["NumberOfQubits"] = n_
        result["QubitPlacement"] = list(range(n_))
        result["QPUs"] = [1]
        result["CPUs"] = psutil.Process().cpu_affinity()
        result["TotalTime"] = 10.0
        result["SigmaTotalTime"] = 1.0
        result["QuantumTime"] = 9.0
        result["SigmaQuantumTime"] = 0.5
        result["ClassicalTime"] = 1.0
        result["SigmaClassicalTime"] = 0.1
        metrics = []
        #For each fixed number of qbits several metrics can be reported
        for metric_name in list_of_metrics:
            metric = OrderedDict()
            #MANDATORY
            metric["Metric"] = metric_name
            metric["Value"] = 0.1
            metric["STD"] = 0.001
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

    configuration = {"None" : None}

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
