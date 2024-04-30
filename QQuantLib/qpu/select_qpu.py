"""
Selector for QPU.
"""


def select_qpu(hw_cfg):

    if hw_cfg["qpu"] in ["noisy", "ideal"]:
        from QQuantLib.qpu.model_noise import create_qpu 
        qpu = create_qpu(hw_cfg)
    else:
        from QQuantLib.qpu.get_qpu import get_qpu 
        qpu = get_qpu(hw_cfg["qpu"])
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

