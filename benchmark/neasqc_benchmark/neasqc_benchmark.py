""" This module contains a class for doing benchmark for NEASQC

Authors: Gonzalo Ferro

"""
import sys
import json
import pandas as pd
import numpy as np
import platform
import psutil
from collections import OrderedDict





class BENCHMARK:
    """
    Generic class for benchmark applications in NEASQC

    """

    def __init__(self):
        """

        Method for initializing the class

        """

        #json_file = open("NEASQC.Benchmark.V2.Schema.json")
        json_file = open("NEASQC.Benchmark.V2.Schema_modified.json")
        self.schema = json.load(json_file)
        json_file.close()

        self.report = OrderedDict()

    def set_organisation(self, organisation):
        self.report["ReportOrganization"] = organisation

    def set_machine_name(self, node_name):
        self.report["MachineName"] = node_name

    def set_qpu_model(self, qpu_model):
        self.report["QPUModel"] = qpu_model

    def set_qpu_description(self, qpu_description):
        self.report["QPUDescription"] = qpu_description

    def set_cpu_model(self, cpu_model):
        self.report["CPUModel"] = cpu_model

    def set_frecuency(self, frecuency):
        self.report["Frequency"] = frecuency

    def set_network(self, network):
        self.report["Network"] = network

    def set_qpu_cpu_connection(self, qpu_cpu_connection):
        self.report["QPUCPUConnection"] = qpu_cpu_connection

    def set_benchmark_info(self, benchmark_info):
        self.report["Benchmarks"] = benchmark_info





if __name__ == "__main__":
    print("OPA")

    #import my_benchmark_info
    import my_environment_info
    import my_benchmark_info
    import jsonschema
    import json

    benchmark = BENCHMARK()
    benchmark.set_organisation(my_environment_info.my_organisation())
    benchmark.set_machine_name(my_environment_info.my_machine_name())
    benchmark.set_qpu_model(my_environment_info.my_qpu_model())
    benchmark.set_qpu_description(my_environment_info.my_qpu())
    benchmark.set_cpu_model(my_environment_info.my_cpu_model())
    benchmark.set_frecuency(my_environment_info.my_frecuency())
    benchmark.set_network(my_environment_info.my_network())
    benchmark.set_qpu_cpu_connection(my_environment_info.my_QPUCPUConnection())
    #Execute Benchmark
    benchmark.set_benchmark_info(
        my_benchmark_info.my_benchmark_info(
            file_results="./Results/IQAE_SummaryResults.csv",
            times_filename="./Results/IQAE_times_benchmark.csv"
        )
    )

    #print(benchmark.report)
    print("Validate REPORT")
    try:
        jsonschema.validate(
            instance=benchmark.report,
            schema=benchmark.schema["properties"]
        )
        print("\t REPORT is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)
    with open("./Results/IQAE_json_benchmark.json", "w") as outfile:
        json.dump(benchmark.report, outfile)
