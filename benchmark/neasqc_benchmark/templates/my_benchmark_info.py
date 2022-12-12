import sys
import platform
import psutil
import pandas as pd
from collections import OrderedDict

if __package__ is None or __package__ == '':
    from my_benchmark_summary import summarize_results
else:
    from .my_benchmark_summary import summarize_results


def my_benchmark_kernel(**kwargs):
    """
    Name for the benchmark Kernel
    """
    return "AmplitudeEstimation"

def my_starttime(**kwargs):
    """
    Providing the start time of the benchmark
    """
    start_time = "2022-12-12T16:46:57.268509+01:00"
    return start_time

def my_endtime(**kwargs):
    """
    Providing the end time of the benchmark
    """
    end_time = "2022-12-12T16:46:57.268509+01:00"
    return end_time

def my_timemethod(**kwargs):
    """
    Providing the method for getting the times
    """
    time_method = "None"
    return time_method

def my_programlanguage(**kwargs):
    """
    Getting the programing language used for benchmark
    """
    program_language = "None"
    return program_language

def my_programlanguage_version(**kwargs):
    """
    Getting the version of the programing language used for benchmark
    """
    language_version = "None"
    return language_version

def my_programlanguage_vendor(**kwargs):
    """
    Getting the version of the programing language used for benchmark
    """
    language_vendor = "None"
    return language_vendor

def my_api(**kwargs):
    """
    Collect the information about the used APIs
    """
    api = OrderedDict()
    api["Name"] = "None"
    api["Version"] = "None"
    list_of_apis = [api]
    return list_of_apis

def my_quantum_compilation(**kwargs):
    """
    Information about the quantum compilation part of the benchmark
    """
    q_compilation = OrderedDict()
    q_compilation["Step"] = "None"
    q_compilation["Version"] = "None"
    q_compilation["Flags"] = "None"
    return [q_compilation]

def my_classical_compilation(**kwargs):
    """
    Information about the classical compilation part of the benchmark
    """
    c_compilation = OrderedDict()
    c_compilation["Step"] = "None"
    c_compilation["Version"] = "None"
    c_compilation["Flags"] = "None"
    return [c_compilation]

def my_other_info(**kwargs):
    """
    Other important info user want to store in the final json.
    """

    other_info = OrderedDict()
    other_info["None"] = None

    return other_info


def my_benchmark_info(**kwargs):
    """
    Complete WorkFlow for getting all the benchmar informated related info
    """
    benchmark = OrderedDict()
    benchmark["BenchmarkKernel"] = my_benchmark_kernel(**kwargs)
    benchmark["StartTime"] = my_starttime(**kwargs)
    benchmark["EndTime"] = my_endtime(**kwargs)
    benchmark["ProgramLanguage"] = my_programlanguage(**kwargs)
    benchmark["ProgramLanguageVersion"] = my_programlanguage_version(**kwargs)
    benchmark["ProgramLanguageVendor"] = my_programlanguage_vendor(**kwargs)
    benchmark["API"] = my_api(**kwargs)
    benchmark["QuantumCompililation"] = my_quantum_compilation(**kwargs)
    benchmark["ClassicalCompiler"] = my_classical_compilation(**kwargs)
    benchmark["TimeMethod"] = my_timemethod(**kwargs)
    benchmark["Results"] = summarize_results(**kwargs)
    benchmark["OtherInfo"] = my_other_info(**kwargs)
    return benchmark

if __name__ == "__main__":
    import json
    import jsonschema
    json_file = open("../NEASQC.Benchmark.V2.Schema_modified.json")
    schema = json.load(json_file)
    json_file.close()

    ################## Configuration ##########################

    configuration = {"None": None}

    ######## Execute Validations #####################################

    schema_bench = schema['properties']['Benchmarks']['items']['properties']
    #print(schema_bench)

    print("Validate BenchmarkKernel")
    print(my_benchmark_kernel(**configuration))
    try:
        jsonschema.validate(
            instance=my_benchmark_kernel(**configuration),
            schema=schema_bench['BenchmarkKernel']
        )
        print("\t BenchmarkKernel is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate StartTime")
    print(my_starttime(**configuration))
    try:
        jsonschema.validate(
            instance=my_starttime(**configuration),
            schema=schema_bench['StartTime']
        )
        print("\t StartTime is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate EndTime")
    print(my_endtime(**configuration))
    try:
        jsonschema.validate(
            instance=my_endtime(**configuration),
            schema=schema_bench['EndTime']
        )
        print("\t EndTime is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ProgramLanguage")
    print(my_programlanguage(**configuration))
    try:
        jsonschema.validate(
            instance=my_programlanguage(**configuration),
            schema=schema_bench['ProgramLanguage']
        )
        print("\t ProgramLanguage is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ProgramLanguageVersion")
    print(my_programlanguage_version(**configuration))
    try:
        jsonschema.validate(
            instance=my_programlanguage_version(**configuration),
            schema=schema_bench['ProgramLanguageVersion']
        )
        print("\t ProgramLanguageVersion is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ProgramLanguageVendor")
    print(my_programlanguage_vendor(**configuration))
    try:
        jsonschema.validate(
            instance=my_programlanguage_vendor(**configuration),
            schema=schema_bench['ProgramLanguageVendor']
        )
        print("\t ProgramLanguageVendor is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate API")
    print(my_api(**configuration))
    try:
        jsonschema.validate(
            instance=my_api(**configuration),
            schema=schema_bench['API']
        )
        print("\t API is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate QuantumCompililation")
    print(my_quantum_compilation(**configuration))
    #print(my_api(**configuration))
    try:
        jsonschema.validate(
            instance=my_quantum_compilation(**configuration),
            schema=schema_bench['QuantumCompililation']#['items']
        )
        print("\t QuantumCompililation is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate ClassicalCompiler")
    print(my_classical_compilation(**configuration))
    try:
        jsonschema.validate(
            instance=my_classical_compilation(**configuration),
            schema=schema_bench['ClassicalCompiler']#['items']
        )
        print("\t ClassicalCompiler is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate TimeMethod")
    #print(summarize_results("Benchmark.csv"))
    try:
        jsonschema.validate(
            instance=my_timemethod(**configuration),
            schema=schema_bench['TimeMethod']
        )
        print("\t TimeMethod is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    print("Validate Benchmark INFO")
    print(my_benchmark_info(**configuration))
    schema_bench = schema['properties']['Benchmarks']
    try:
        jsonschema.validate(
            instance=my_benchmark_info(**configuration),
            schema=schema_bench['items']
        )
        print("\t Benchmark is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

