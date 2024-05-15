"""
This module implements the select_qpu function that allows to the user
select a complete QPU. The input of this function is a Python dictionary
with the following keys:

    **qpu_type**
        The value is a Python string for selecting the type of QPU. The
        values can be: *c, python, linalg, mps, qlmass_linalg, qlmass_mps*
        for using with pure ideal QPU (the get_qpu function  from
        QQuantLib.qpu.get_qpu module is used for getting the QPU).
        Additionally, the values can be: *ideal* for using an ideal qpu
        with a Toffoli rewritter pluging or *noisy* for configuring and
        using a noisy QPU. In both cases the create_qpu function  from
        QQuantLib.qpu.model_noise module is used for creating the QPU.

    **t_gate_1qb**
        For setting the time for the 1-qubit gates (in ns). Only valid
        if *qpu_type* is noisy.

    **t_gate_2qbs**
        For setting the time for the 2-qubit gates (in ns). Only valid
        if *qpu_type* is noisy.

    **t_readout**
        For setting the time for the measuring operations (in ns). Only
        valid if *qpu_type* is noisy.

    **depol_channel**
        For setting the parameters for a depolarizing channel. The value
        is a complete dictionary with the following keys: **active**
        the boolean key for activating or not the channel. **error_gate_1qb**
        error for 1-qubit gates. **error_gate_2qbs** error for 2-qubits
        gates.

    **idle**
        For setting the parameters for idle qubits. The value is a
        complete dictionary with the following keys: **amplitude_damping**
        the boolean key for activating or not an Amplitude Damping channel.
        **dephasing_channel** boolean key for activating or not a
        Dephasing channel. **t1** time T1 of the qubits (in ns)
        **t2** time T2 of the qubits (in nsa).

    **meas**
        For setting the parameters for a measuring error. The value is a
        complete dictionary with the following keys: **active** boolean
        key for activating or not this error. **readout_error** measuring
        error.
"""

def select_qpu(hw_cfg):
    """
    This function allows to select a QPU (a ideal or a noisy one).

    Parameters
    ----------

    hw_cfg : dict
        Python dictionary for configuring a complete (ideal or noisy)
        QPU. When an "ideal" QPU is selected the get_qpu from get_qpu
        module is used. If a "noisy" QPU is selected then the differents
        keys of the dictionary are used for configruing a noisy hardware
        model using functions from model_noise module.
    """

    if hw_cfg["qpu_type"] in ["noisy", "ideal"]:
        from QQuantLib.qpu.model_noise import create_qpu
        qpu = create_qpu(hw_cfg)
    else:
        from QQuantLib.qpu.get_qpu import get_qpu
        qpu = get_qpu(hw_cfg["qpu_type"])
    return qpu

if __name__ == "__main__":
    import json
    import argparse
    import sys
    sys.path.append("../../")
    from QQuantLib.utils.benchmark_utils import combination_for_list

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        dest="count",
        default=False,
        action="store_true",
        help="For counting elements on the list",
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing "
    )
    parser.add_argument(
        "-id",
        dest="id",
        type=int,
        help="For executing only one element of the list",
        default=None,
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="jsons/qpu.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )
    args = parser.parse_args()
    print(args)
    with open(args.json_qpu) as json_file:
        noisy_cfg = json.load(json_file)
    final_list = combination_for_list(noisy_cfg)
    if args.count:
        print(len(final_list))
    if args.print:
        if args.id is not None:
            print(final_list[args.id])
        else:
            print(final_list)
    if args.execution:
        if args.id is not None:
            print(select_qpu(final_list[args.id]))

