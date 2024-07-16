"""
This module implements several functions for configuring a noisy
hardware model for creating the corresponding noisy qpu.
**BE AWARE**
The functions of this module can be only used with the Qaptiva™ Appliance
and when the user is locally in a QLM. The Qaptiva Access library **CAN
NOT BE** used with these functions.
"""

import numpy as np
import functools

# Obtenido de ibm_brisbane: 2024/04/04
# error_gate_1qb = 2.27e-4
# error_gate_2qbs = 7.741e-3
# t_gate_1qb = 35 #ns
# t_gate_2qbs = 660 #ns
# t1 = 231.94e3
# t2 = 132.71e3

def set_gate_times(t_gate_1qb=35, t_gate_2qbs=660, t_readout=4000):
    """
    Set the gate times for a noise model.

    Parameters
    ----------

    t_gate_1qb : int
        time for 1 qubit gate length in ns
    t_gate_2qbs : int
        time for 2 qubits gate length in ns
    t_readout : int
        time for readout gate in ns

    Return
    ------

    gate_time_dict : dict
        dictionary with the default gates and their time length

    """
    from qat.hardware import DefaultHardwareModel
    from qat.hardware.default import _CtrlParametricChannel, _ParametricChannel
    from qat.quops.quantum_channels import QuantumChannelKraus
    hw_m = DefaultHardwareModel()
    gate_time_dict = {}
    for gate, value in hw_m.gates_specification.quantum_channels.items():
        if gate not in ["measure", "reset", "logic"]:
            if isinstance(value, _CtrlParametricChannel):
                gate_time_dict.update({gate: lambda angle: t_gate_2qbs})
            if isinstance(value, _ParametricChannel):
                gate_time_dict.update({gate: lambda angle: t_gate_1qb})
            if isinstance(value, QuantumChannelKraus):
                if value.arity == 1:
                    gate_time_dict.update({gate: t_gate_1qb})
                if value.arity == 2:
                    gate_time_dict.update({gate: t_gate_2qbs})
        else:
            if gate == "measure":
                gate_time_dict.update({gate: t_readout})
    return gate_time_dict

def noisy_hw_model(hw_cfg):
    """
    My noisy hardware model: It is composed by 3 types of noise channels:
    Amplitude Damping and Dephasing channels for idle qubits
    Depolarizing channel applied after any gate.

    Parameters
    ----------

    hw_cfg :  dict
        Python dictionary with parameters for the noisy hardware:
        * error_gate_1qb : Error for 1-qubit gate (for Depolarizing channel)
        * error_gate_2qbs: Error for 2-qubits gates (for Depolarizing channel)
        * t_gate_1qb : duration time in nanoseconds for 1 qubit gates
        * t_gate_2qbs : duration time in nanoseconds for 2 qubit gates
        * t1 : T1 time in nanoseconds (Amplitude Damping and Dephasing channels)
        * t2 : T2 time in nanoseconds (Dephasing channel)

    Return
    ------

    my_hw_model : Qaptiva HardwareModel
        my HardwareModel definition
    """
    t_gate_1qb = hw_cfg.get("t_gate_1qb", 35)
    t_gate_2qbs = hw_cfg.get("t_gate_2qbs", 660)
    t_readout = hw_cfg.get("t_readout", 4000)
    #Gates Specification
    gate_time_dict = set_gate_times(t_gate_1qb, t_gate_2qbs, t_readout)
    depol_channel = hw_cfg.get("depol_channel")
    if depol_channel["active"]:
        from qat.hardware import make_depolarizing_hardware_model
        # Hardware model for depolarizing channel
        error_gate_1qb = depol_channel.get("error_gate_1qb", 2.27e-4)
        error_gate_2qbs = depol_channel.get("error_gate_2qbs", 7.741e-3)
        my_hw_model = make_depolarizing_hardware_model(
            eps1=error_gate_1qb, eps2=error_gate_2qbs
        )
        # BE AWARE: Parametric gates are not included in the
        # the dictionary

        # First: 1 qubit parametric gates will be included
        gate_1_qubit = my_hw_model.gate_noise["H"]
        for gate_ in ["RZ", "RX", "RY", "PH"]:
            my_hw_model.gate_noise.update({
                gate_: functools.partial(
                    gate_1_qubit.func,
                    rb_eps=gate_1_qubit.keywords["rb_eps"],
                    nqbits=gate_1_qubit.keywords["nqbits"],
                    method_2q=gate_1_qubit.keywords["method_2q"],
                    depol_type=gate_1_qubit.keywords["depol_type"]
                )
            })
        # Secong: 2 qubit parametric gates will be included
        gate_2_qubits = my_hw_model.gate_noise["CNOT"]
        for gate_ in ["C-RZ", "C-RX", "C-RY", "C-PH"]:
            my_hw_model.gate_noise.update({
                gate_: functools.partial(
                    gate_2_qubits.func,
                    rb_eps=gate_2_qubits.keywords["rb_eps"],
                    nqbits=gate_2_qubits.keywords["nqbits"],
                    method_2q=gate_2_qubits.keywords["method_2q"],
                    depol_type=gate_2_qubits.keywords["depol_type"]
                )
            })
    else:
        from qat.hardware import DefaultHardwareModel
        my_hw_model = DefaultHardwareModel()
    # Setting time for the gates
    my_hw_model.gates_specification.gate_times.update(gate_time_dict)
    idle = hw_cfg.get("idle")
    idle_noise_list = []
    if idle["amplitude_damping"]:
        from qat.quops import ParametricAmplitudeDamping
        # Setting AmplitudeDamping iddle channel
        t1 = idle.get("t1", 231.94e3)
        idle_noise_list.append(ParametricAmplitudeDamping(T_1=t1))
        if idle["dephasing_channel"]:
            from qat.quops import ParametricPureDephasing
            # Setting Dephasing iddle channel
            t2 = idle.get("t2", 132.71e3)
            tphi = 1/(1/t2 - 1/(2 * t1))
            idle_noise_list.append(ParametricPureDephasing(T_phi=tphi))
        # Setting idle channels
        my_hw_model.idle_noise = idle_noise_list
    meas = hw_cfg.get("meas")
    if meas["active"]:
        # Setting Measurement noise channel
        readout_error = meas["readout_error"]
        meas_prep = np.array([[readout_error, 0.0],[0.0, 1.0-readout_error]])
        my_hw_model.gates_specification.meas = meas_prep
    return my_hw_model

def create_qpu(hw_cfg):
    """
    Create QPU. Using an input hardware configuration this function creates
    a QPU. It could be a noisy or a ideal qpu depending on the value of the key
    qpu of the hw_cfg dictionary. Additionally adds a plugin for rewiting the
    Toffolis using CNOTS and local gates.

    Parameters
    ----------

    hw_cfg :  dict
        Python dictionary with parameters for configuring the QPU
        * qpu : If noisy the function creates a Noisy QPU. Else create the corresponding ideal QPU.
        * error_gate_1qb : Error for 1-qubit gate (for Depolarizing channel)
        * error_gate_2qbs: Error for 2-qubits gates (for Depolarizing channel)
        * t_gate_1qb : duration time in nanoseconds for 1 qubit gates
        * t_gate_2qbs : duration time in nanoseconds for 2 qubit gates
        * t1 : T1 time in nanoseconds (Amplitude Damping and Dephasing channels)
        * t2 : T2 time in nanoseconds (Dephasing channel)
    Return
    ------

    my_qpu : Qaptiva QPU
        generated QPU (can be a noisy one)
    """

    from qat.synthopline.compiler import EXPANSION_COLLECTION
    from qat.pbo import PatternManager
    from qat.qpus import NoisyQProc, LinAlg
    # Rewritter for Toffolis
    toffoli_plugin = PatternManager(collections=[EXPANSION_COLLECTION[:1]])
    if hw_cfg["qpu_type"] == "noisy":
        model_noisy = noisy_hw_model(hw_cfg)
        my_qpu= NoisyQProc(
            hardware_model=model_noisy,
            sim_method="deterministic-vectorized",
            backend_simulator=LinAlg()
        )
    else:
        my_qpu = LinAlg()
    my_qpu = toffoli_plugin | toffoli_plugin | my_qpu
    return my_qpu
