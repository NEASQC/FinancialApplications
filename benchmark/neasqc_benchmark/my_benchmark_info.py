import sys
import platform
import psutil
import pandas as pd
from collections import OrderedDict


def my_benchmark_kernel():
    return "AmplitudeEstimation"

def my_starttime(times_filename="times_benchmark.csv"):
    pdf = pd.read_csv(times_filename, index_col=0)
    return pdf["StartTime"][0]
    
def my_endtime(times_filename="times_benchmark.csv"):
    pdf = pd.read_csv(times_filename, index_col=0)
    return pdf["EndTime"][0]

def my_timemethod():
    return "time.time"

def my_programlanguage():
    return platform.python_implementation()

def my_programlanguage_version():
    return platform.python_version()

def my_programlanguage_vendor():
    return "Unknow"

def my_api():
    """
    Collect the information about the used APIs
    """
    modules = []
    list_of_apis = []
    for module in list(sys.modules):
        api = OrderedDict()
        module = module.split('.')[0]
        if module not in modules:
            modules.append(module)
            api["Name"] = module
            try:
                version = sys.modules[module].__version__
            except AttributeError:
                #print("NO VERSION: "+str(sys.modules[module]))
                try:
                    if  isinstance(sys.modules[module].version, str):
                        version = sys.modules[module].version
                        #print("\t Attribute Version"+version)
                    else:
                        version = sys.modules[module].version()
                        #print("\t Methdod Version"+version)
                except (AttributeError, TypeError) as error:
                    #print('\t NO VERSION: '+str(sys.modules[module]))
                    try:
                        version = sys.modules[module].VERSION
                    except AttributeError:
                        #print('\t\t NO VERSION: '+str(sys.modules[module]))
                        version = "Unknown"
            api["Version"] = str(version)
            list_of_apis.append(api)
    return list_of_apis

def my_quantum_compilation():

    q_compilation = OrderedDict()
    q_compilation["Step"] = "None"
    q_compilation["Version"] = "None"
    q_compilation["Flags"] = "None"
    return [q_compilation]

def my_classical_compilation():
    c_compilation = OrderedDict()
    c_compilation["Step"] = "None"
    c_compilation["Version"] = "None"
    c_compilation["Flags"] = "None"
    return [c_compilation]

def summarize_results(benchmark_file):


    pdf = pd.read_csv(benchmark_file, index_col=0, sep=";")
    n_qbits = list(set(pdf["n_qbits"]))
    intervals = list(set(pdf["interval"]))
    results = []
    metrics = [
        "absolute_error_exact", "relative_error_exact"
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
            for metric_name in metrics:
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


def my_other_info():
    import json
    sys.path.append("../../")
    from benchmark.benchmark_utils import combination_for_list
    lista_ae = [
        "jsons/integral_mlae_configuration.json",
        "jsons/integral_iqae_configuration.json",
        "jsons/integral_rqae_configuration.json",
        "jsons/integral_cqpeae_configuration.json",
        "jsons/integral_iqpeae_configuration.json",
    ]
    info_json = lista_ae[1]
    other_info = OrderedDict()
    with open(info_json) as json_file:
        ae = json.load(json_file)
    final_list = combination_for_list(ae)[0]
    
    new_dict = {}
    for key, value in final_list.items():
        #print(key, value)
        if value is not None:
            other_info[key] = str(value)
            #new_dict.update({key: str(value)})
    return other_info


def my_benchmark_info(
        file_results="./SummaryResults.csv", times_filename="./times_benchmark.csv"
    ):
    from my_benchmark_summary import summarize_results
    benchmark = OrderedDict()
    benchmark["BenchmarkKernel"] = my_benchmark_kernel()
    benchmark["StartTime"] = my_starttime(times_filename)
    benchmark["EndTime"] = my_endtime(times_filename)
    benchmark["ProgramLanguage"] = my_programlanguage()
    benchmark["ProgramLanguageVersion"] = my_programlanguage_version()
    benchmark["ProgramLanguageVendor"] = my_programlanguage_vendor()
    benchmark["API"] = my_api()
    benchmark["QuantumCompililation"] = my_quantum_compilation()
    benchmark["ClassicalCompiler"] = my_classical_compilation()
    benchmark["TimeMethod"] = my_timemethod()
    benchmark["Results"] = summarize_results(file_results)
    benchmark["OtherInfo"] = my_other_info()
    return benchmark

if __name__ == "__main__":
    import json
    import jsonschema
    json_file = open("NEASQC.Benchmark.V2.Schema_modified.json")
    schema = json.load(json_file)
    json_file.close()

    schema_bench = schema['properties']['Benchmarks']['items']['properties']
    #print(schema_bench)

    print("Validate BenchmarkKernel")
    print(my_benchmark_kernel())
    try:
        jsonschema.validate(
            instance=my_benchmark_kernel(),
            schema=schema_bench['BenchmarkKernel']
        )
        print("\t BenchmarkKernel is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate StartTime")
    print(my_starttime())
    try:
        jsonschema.validate(
            instance=my_starttime(),
            schema=schema_bench['StartTime']
        )
        print("\t StartTime is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate EndTime")
    print(my_endtime())
    try:
        jsonschema.validate(
            instance=my_endtime(),
            schema=schema_bench['EndTime']
        )
        print("\t EndTime is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ProgramLanguage")
    print(my_programlanguage())
    try:
        jsonschema.validate(
            instance=my_programlanguage(),
            schema=schema_bench['ProgramLanguage']
        )
        print("\t ProgramLanguage is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ProgramLanguageVersion")
    print(my_programlanguage_version())
    try:
        jsonschema.validate(
            instance=my_programlanguage_version(),
            schema=schema_bench['ProgramLanguageVersion']
        )
        print("\t ProgramLanguageVersion is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ProgramLanguageVendor")
    print(my_programlanguage_vendor())
    try:
        jsonschema.validate(
            instance=my_programlanguage_vendor(),
            schema=schema_bench['ProgramLanguageVendor']
        )
        print("\t ProgramLanguageVendor is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate API")
    #print(my_api())
    try:
        jsonschema.validate(
            instance=my_api(),
            schema=schema_bench['API']
        )
        print("\t API is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate QuantumCompililation")
    print(my_quantum_compilation())
    #print(my_api())
    try:
        jsonschema.validate(
            instance=my_quantum_compilation(),
            schema=schema_bench['QuantumCompililation']#['items']
        )
        print("\t QuantumCompililation is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ClassicalCompiler")
    print(my_classical_compilation())
    try:
        jsonschema.validate(
            instance=my_classical_compilation(),
            schema=schema_bench['ClassicalCompiler']#['items']
        )
        print("\t ClassicalCompiler is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate TimeMethod")
    #print(summarize_results("Benchmark.csv"))
    try:
        jsonschema.validate(
            instance=my_timemethod(),
            schema=schema_bench['TimeMethod']
        )
        print("\t TimeMethod is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate Benchmark INFO")
    schema_bench = schema['properties']['Benchmarks']
    try:
        jsonschema.validate(
            instance=my_benchmark_info(
                file_results="SummaryResults.csv",
                times_filename="times_benchmark.csv"
            ),
            schema=schema_bench['items']
        )
        print("\t Benchmark is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

