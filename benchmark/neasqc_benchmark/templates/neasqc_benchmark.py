""" This module contains a class for doing benchmark for NEASQC

Authors: Gonzalo Ferro

"""
import json
from collections import OrderedDict
import jsonschema

class BENCHMARK:
    """
    Generic class for benchmark applications in NEASQC

    """

    def __init__(self):
        """

        Method for initializing the class

        """

        #json_file = open("NEASQC.Benchmark.V2.Schema.json")
        json_file = open("../NEASQC.Benchmark.V2.Schema_modified.json")
        self.schema = json.load(json_file)
        json_file.close()

        self.report = OrderedDict()

    def set_organisation(self, organisation):
        """
        Method for filling ReportOrganization field.

        Parameters
        ----------

        organisation : string

        """
        self.report["ReportOrganization"] = organisation

    def set_machine_name(self, node_name):
        """
        Method for filling MachineName field.

        Parameters
        ----------

        node_name : string

        """
        self.report["MachineName"] = node_name

    def set_qpu_model(self, qpu_model):
        """
        Method for filling QPUModel field.

        Parameters
        ----------

        qpu_model : str

        """
        self.report["QPUModel"] = qpu_model

    def set_qpu_description(self, qpu_description):
        """
        Method for filling QPUDescription field.

        Parameters
        ----------

        qpu_description : OrderedDict

        """
        self.report["QPUDescription"] = qpu_description

    def set_cpu_model(self, cpu_model):
        """
        Method for filling CPUModel field.

        Parameters
        ----------

        cpu_model : str

        """
        self.report["CPUModel"] = cpu_model

    def set_frecuency(self, frecuency):
        """
        Method for filling Frequency field.

        Parameters
        ----------

        frecuency : int

        """
        self.report["Frequency"] = frecuency

    def set_network(self, network):
        """
        Method for filling Network field.

        Parameters
        ----------

        network : OrderedDict

        """
        self.report["Network"] = network

    def set_qpu_cpu_connection(self, qpu_cpu_connection):
        """
        Method for filling QPUCPUConnection field.

        Parameters
        ----------

        qpu_cpu_connection : OrderedDict

        """
        self.report["QPUCPUConnection"] = qpu_cpu_connection

    def set_benchmark_info(self, benchmark_info):
        """
        Method for filling Benchmarks field.

        Parameters
        ----------

        benchmark_info : OrderedDict

        """
        self.report["Benchmarks"] = benchmark_info

    def exe(self, info):
        """
        Method for filling the report.

        Parameters
        ----------

        info : dictionary

        """
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
        """
        Method for validating the generated benchmark report.

        """
        print("Validate REPORT")
        #print(self.report)
        try:
            jsonschema.validate(
                instance=self.report,
                schema=self.schema["properties"]
            )
            print("\t REPORT is Valid")
        except jsonschema.exceptions.ValidationError as ex:
            print(ex)

    def save(self, filename):
        """
        Method for saving the generated benchmark report

        Parameters
        ----------

        filename : string
            pathf for storing the benchmark report

        """
        with open(filename, "w") as outfile:
            json.dump(self.report, outfile)


if __name__ == "__main__":

    import my_environment_info
    import my_benchmark_info

    ################## Configuration ##########################

    kwargs = {"None": None}

    benchmark_stuff = {
        "ReportOrganization": my_environment_info.my_organisation(
            **kwargs),
        "MachineName": my_environment_info.my_machine_name(**kwargs),
        "QPUModel": my_environment_info.my_qpu_model(**kwargs),
        "QPUDescription": my_environment_info.my_qpu(**kwargs),
        "CPUModel": my_environment_info.my_cpu_model(**kwargs),
        "Frequency": my_environment_info.my_frecuency(**kwargs),
        "Network": my_environment_info.my_network(**kwargs),
        "QPUCPUConnection":my_environment_info.my_QPUCPUConnection(
            **kwargs),
        "Benchmarks": my_benchmark_info.my_benchmark_info(**kwargs),
        "json_file_name": "./benchmark_report.json"
    }

    benchmark = BENCHMARK()
    benchmark.exe(benchmark_stuff)
