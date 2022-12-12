import platform
import psutil
from collections import OrderedDict

def my_organisation():
    """
    Given information about the organisation how uploads the benchmark
    """
    return "CESGA"

def my_machine_name():
    """
    Name of the machine where the benchmark was performed
    """
    return platform.node()

def my_qpu_model():
    """
    Name of the model of the QPU
    """
    return "QLM"

def my_qpu():
    """
    Complete info about the used QPU
    """
    #Basic schema
    #QPUDescription = {
    #    "NumberOfQPUs": 1,
    #    "QPUs": [
    #        {
    #            "BasicGates": ["none", "none1"],
    #            "Qubits": [
    #                {
    #                    "QubitNumber": 0,
    #                    "T1": 1.0,
    #                    "T2": 1.00
    #                }
    #            ],
    #            "Gates": [
    #                {
    #                    "Gate": "none",
    #                    "Type": "Single",
    #                    "Symmetric": False,
    #                    "Qubits": [0],
    #                    "MaxTime": 1.0
    #                }
    #            ],
    #            "Technology": "other"
    #        },
    #    ]
    #}

    #Defining the Qubits of the QPU
    qubits = OrderedDict()
    qubits["QubitNumber"] = 0
    qubits["T1"] = 1.0
    qubits["T2"] = 1.0

    #Defining the Gates of the QPU
    gates = OrderedDict()
    gates["Gate"] = "none"
    gates["Type"] = "Single"
    gates["Symmetric"] = False
    gates["Qubits"] = [0]
    gates["MaxTime"] = 1.0


    #Defining the Basic Gates of the QPU
    qpus = OrderedDict()
    qpus["BasicGates"] = ["none", "none1"]
    qpus["Qubits"] = [qubits]
    qpus["Gates"] = [gates]
    qpus["Technology"] = "other"

    qpu_description = OrderedDict()
    qpu_description['NumberOfQPUs'] = 1
    qpu_description['QPUs'] = [qpus]

    return qpu_description

def my_cpu_model():
    """
    model of the cpu used in the benchmark
    """
    return platform.processor()

def my_frecuency():
    """
    Frcuency of the used CPU
    """
    #Use the nominal frequency. Here, it collects the maximum frequency
    #print(psutil.cpu_freq())
    return psutil.cpu_freq().max/1000

def my_network():
    """
    Network connections if several QPUs are used
    """
    network = OrderedDict()
    network["Model"] = "None"
    network["Version"] = "None"
    network["Topology"] = "None"
    return network

def my_QPUCPUConnection():
    """
    Connection between the QPU and the CPU used in the benchmark
    """
    #
    # Provide the information about how the QPU is connected to the CPU
    #
    qpuccpu_conn = OrderedDict()
    qpuccpu_conn["Type"] = "memory"
    qpuccpu_conn["Version"] = "None"
    return qpuccpu_conn

if __name__ == "__main__":
    """
    For comparing the results of the environment info vs the jsonschema
    """
    import json
    import jsonschema
    json_file = open("NEASQC.Benchmark.V2.Schema_modified.json")
    schema = json.load(json_file)
    json_file.close()

    schema_org = schema['properties']['ReportOrganization']
    print("Validate ReportOrganization")
    print(my_organisation())
    try:
        jsonschema.validate(
            instance=my_organisation(),
            schema=schema_org
        )
        print("\tReportOrganization is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_nodename = schema['properties']['MachineName']
    print("Validate MachineName")
    print(my_machine_name())
    try:
        jsonschema.validate(
            instance=my_machine_name(),
            schema=schema_nodename
        )
        print("\tMachineName is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_model = schema['properties']['QPUModel']
    print("Validate QPUModel")
    print(my_qpu_model())
    try:
        jsonschema.validate(
            instance=my_qpu_model(),
            schema=schema_nodename
        )
        print("\tQPUModel is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)


    schema_qpu = schema['properties']['QPUDescription']['items']
    print("Validate QPUDescription")
    print(my_qpu())
    try:
        jsonschema.validate(
            instance=my_qpu(),
            schema=schema_qpu
        )
        print("\tQPUDescription is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_cpu = schema['properties']['CPUModel']
    print("Validate QCPUModel")
    print(my_cpu_model())
    try:
        jsonschema.validate(
            instance=my_cpu_model(),
            schema=schema_cpu
        )
        print("\tQCPUModel is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_frecuency = schema['properties']['Frequency']
    print("Validate Frequency")
    print(my_frecuency())
    try:
        jsonschema.validate(
            instance=my_frecuency(),
            schema=schema_frecuency
        )
        print("\tFrequency is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_network = schema['properties']['Network']
    print("Validate Network")
    print(my_network())
    try:
        jsonschema.validate(
            instance=my_network(),
            schema=schema_network
        )
        print("\tNetwork is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)

    schema_qpucpu_conn = schema['properties']['QPUCPUConnection']
    print("Validate QPUCPUConnection")
    print(my_QPUCPUConnection())
    try:
        jsonschema.validate(
            instance=my_QPUCPUConnection(),
            schema=schema_qpucpu_conn
        )
        print("\tQPUCPUConnection is Valid")
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)



