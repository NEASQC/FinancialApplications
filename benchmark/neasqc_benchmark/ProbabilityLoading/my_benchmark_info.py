import sys
import platform
import psutil
import pandas as pd
from collections import OrderedDict


def my_benchmark_kernel():
    """
    Name for the benchmark Kernel
    """
    return "ProbabilityLoading"

def my_starttime(times_filename="times_benchmark.csv"):
    """
    Providing the start time of the benchmark
    """
    pdf = pd.read_csv(times_filename, index_col=0)
    return pdf["StartTime"][0]

def my_endtime(times_filename="times_benchmark.csv"):
    """
    Providing the end time of the benchmark
    """
    pdf = pd.read_csv(times_filename, index_col=0)
    return pdf["EndTime"][0]

def my_timemethod():
    """
    Providing the method for getting the times
    """
    return "time.time"

def my_programlanguage():
    """
    Getting the programing language used for benchmark
    """
    return platform.python_implementation()

def my_programlanguage_version():
    """
    Getting the version of the programing language used for benchmark
    """
    return platform.python_version()

def my_programlanguage_vendor():
    """
    Getting the version of the programing language used for benchmark
    """
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
    """
    Information about the quantum compilation part of the benchmark
    """
    q_compilation = OrderedDict()
    q_compilation["Step"] = "None"
    q_compilation["Version"] = "None"
    q_compilation["Flags"] = "None"
    return [q_compilation]

def my_classical_compilation():
    """
    Information about the classical compilation part of the benchmark
    """
    c_compilation = OrderedDict()
    c_compilation["Step"] = "None"
    c_compilation["Version"] = "None"
    c_compilation["Flags"] = "None"
    return [c_compilation]

def my_other_info():
    """
    Other important info user want to store in the final json.
    """
    other_info = OrderedDict()
    other_info["load_method"] = "multiplexor"
    return other_info


def my_benchmark_info(file_results, times_filename):
    """
    Complete WorkFlow for getting all the benchmar informated related info
    """
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
    #benchmark["Results"] = summarize_results(file_results)
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

