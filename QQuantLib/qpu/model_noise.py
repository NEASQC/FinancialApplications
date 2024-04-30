"""
Noisy hardare model
"""


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
    import numpy as np
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
    if hw_cfg["qpu"] == "noisy":
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

"""
    t1 = hw_cfg.get("t1", 231.94e3)
    t2 = hw_cfg.get("t2", 132.71e3)
    tphi = 1/(1/t2 - 1/(2 * t1))
    # gates_1qb = ["H", "X", "Y", "Z", "I", "S", "T"]
    # gates_1qb_par = ["RZ", "RX", "RY", "PH"]
    # gates_2qb = ["CNOT", "C-H", "C-X", "C-Y", "C-Z", "C-S", "C-T"]
    # gates_2qb_par = ["C-RX", "C-RY", "C-RZ", "C-PH"]
    # gate_time_dict = {gate: t_gate_1qb for gate in gates_1qb}
    # gate_time_dict.update(
    #     {gate: lambda angle: t_gate_1qb for gate in gates_1qb_par})
    # gate_time_dict.update(
    #     {gate: t_gate_2qbs for gate in gates_2qb})
    # gate_time_dict.update(
    #     {gate: lambda angle: t_gate_2qbs for gate in gates_2qb_par})

    # Hardware model for depolarizing channel
    depol_hw = make_depolarizing_hardware_model(
        eps1=error_gate_1qb, eps2=error_gate_2qbs
    )
    # Setting Gate times in hardware model
    depol_hw.gates_specification.gate_times = gate_time_dict
    # Setting AmplitudeDamping iddle channel
    relaxation_noise = ParametricAmplitudeDamping(T_1=t1)
    # Setting Dephasing iddle channel
    dephasing_noise = ParametricPureDephasing(T_phi=tphi)
    depol_hw.idle_noise = [relaxation_noise, dephasing_noise]
    return depol_hw
"""
