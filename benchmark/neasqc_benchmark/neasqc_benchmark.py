""" This module contains a class for doing benchmark for NEASQC

Authors: Gonzalo Ferro

"""
import sys
import json
import jsonschema
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

    def exe(self, info):
        self.set_organisation(info["ReportOrganization"])
        self.set_machine_name(info["MachineName"])
        self.set_qpu_model(info["QPUModel"])
        self.set_qpu_description(info["QPUDescription"])
        self.set_cpu_model(info["CPUModel"])
        self.set_frecuency(info["Frequency"])
        self.set_network(info["Network"])
        self.set_qpu_cpu_connection(info["QPUCPUConnection"])
        self.set_benchmark_info(info["Benchmarks"])
        self.validate()
        self.save(info["json_file_name"])

    def validate(self):
        print("Validate REPORT")
        print(self.report)
        try:
            jsonschema.validate(
                instance=self.report,
                schema=self.schema["properties"]
            )
            print("\t REPORT is Valid")
        except jsonschema.exceptions.ValidationError as ex:
            print(ex)

    def save(self, filename):
        with open(filename, "w") as outfile:
            json.dump(benchmark.report, outfile)


if __name__ == "__main__":
    print("OPA")

    from templates import my_environment_info, my_benchmark_info
    #import templates

    ################## Configuration ##########################

    configuration = {"None": None}

    benchmark_stuff = {
        "ReportOrganization": my_environment_info.my_organisation(**configuration),
        "MachineName": my_environment_info.my_machine_name(**configuration),
        "QPUModel": my_environment_info.my_qpu_model(**configuration),
        "QPUDescription": my_environment_info.my_qpu(**configuration),
        "CPUModel": my_environment_info.my_cpu_model(**configuration),
        "Frequency": my_environment_info.my_frecuency(**configuration),
        "Network": my_environment_info.my_network(**configuration),
        "QPUCPUConnection":my_environment_info.my_QPUCPUConnection(**configuration),
        "Benchmarks": my_benchmark_info.my_benchmark_info(**configuration),
        "json_file_name": "./json_stuff.json"
    }

    benchmark = BENCHMARK()
    benchmark.exe(benchmark_stuff)

