"""
Functions for continue a training
"""
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.append("../../")
from QQuantLib.qml4var.data_utils import get_dataset
from QQuantLib.qml4var.architectures import hardware_efficient_ansatz, \
     z_observable, normalize_data, init_weights
from QQuantLib.qml4var.myqlm_workflows import qdml_loss_workflow, mse_workflow
from QQuantLib.qml4var.losses import numeric_gradient
from QQuantLib.qml4var.adam import adam_optimizer_loop

def batch_generator(X, Y, batch_size):
    return [(X[i:i+batch_size] , Y[i:i+batch_size]) for i in range(0, len(X), batch_size)]

def continue_training(**kwargs):
    """
    Continue a previously started training
    """
    # Get the Base folder with the data
    base_folder = kwargs.get("base_folder", None)
    # Get the Base nanme for the datasets
    base_name = kwargs.get("base_name", None)
    data_file = base_folder + base_name
    # Load the data
    x_train, y_train, x_test, y_test = get_dataset(data_file)
    # Get Data INFO
    with open(base_folder + "data.json") as json_file:
        data_info = json.load(json_file)
    # Normalization of the features
    # base_frecuency, shift_feature = normalize_data(
    #     [data_info["minval"]] * data_info["features_number"],
    #     [data_info["maxval"]] * data_info["features_number"],
    #     [-0.5*np.pi] * data_info["features_number"],
    #     [0.5*np.pi] * data_info["features_number"],
    # )
    # Get Architecture Parameters
    train_folder = base_folder + kwargs.get("train_folder", None)
    with open(train_folder + "pqc_dict.json") as json_file:
        pqc_info = json.load(json_file)
    # Re create Architecture
    pqc, weights_names, features_names = hardware_efficient_ansatz(**pqc_info)
    observable = z_observable(**pqc_info)
    # Get the QPU info
    qpu_info = kwargs.get("qpu_info", None)
    nbshots = 0
    # Get Optimizer INFO
    optimizer_info = kwargs.get("optimizer_info", None)
    if optimizer_info is None:
        # Load default optimizer settings
        with open(train_folder + "optimizer_dict.json") as json_file:
            optimizer_info = json.load(json_file)
    # number of shots should be provided into the optimizer_info
    nbshots = optimizer_info["nbshots"]
    # number of discretization points for domain should be provided into
    # the optimizer_info
    points = optimizer_info["points"]
    # Get Dask client if provided
    dask_client = kwargs.get("dask_client", None)
    # Configuration for workflows
    workflow_cfg = {
        "pqc" : pqc,
        "observable" : observable,
        "weights_names" : weights_names,
        "features_names" : features_names,
        "nbshots" : nbshots,
        "minval" : [data_info["minval"]] * data_info["features_number"],
        "maxval" : [data_info["maxval"]] * data_info["features_number"],
        "points" : points,
        "qpu_info" : qpu_info
    }
    # Configure the loss function for gradiente computation
    qdml_loss_workflow_ = lambda w_, x_, y_: qdml_loss_workflow(
        w_, x_, y_, dask_client=dask_client, **workflow_cfg)
    # Configure the numeric gradient function
    numeric_gradient_ = lambda w_, x_, y_: numeric_gradient(
        w_, x_, y_, qdml_loss_workflow_)
    # Configure the loss function for evaluation
    training_loss = lambda w_: qdml_loss_workflow(
        w_, x_train, y_train, dask_client=dask_client, **workflow_cfg)
    # Configure the MSE for evaluation in testing data
    testing_metric = lambda w_: mse_workflow(
        w_, x_test, y_test, dask_client=dask_client, **workflow_cfg)
    # Set the Batch size and th eBatch generator
    batch_size = kwargs.get("batch_size", None)
    if batch_size is None:
        batch_size = len(x_train)
    batch_generator_ = batch_generator(x_train, y_train, batch_size)
    save = True

    # Load Evolution from previous training
    file_to_load = train_folder + "evolution.csv"
    optimizer_info.update({"file_to_save": file_to_load})
    evolution = pd.read_csv(file_to_load, sep=";", index_col=0)
    # Get Weights for continue training
    weights = dict(evolution[weights_names].iloc[-1])
    # Get initial time
    initial_time = evolution["t"].iloc[-1]
    print("initial_time: {}".format(initial_time))
    print("Initial weigths: \n {}".format(weights))


    weights = adam_optimizer_loop(
        weights_dict=weights,
        loss_function=training_loss,
        metric_function=testing_metric,
        gradient_function=numeric_gradient_,
        batch_generator=batch_generator_,
        initial_time=initial_time,
        **optimizer_info
    )
    return weights




if __name__ == "__main__":
    import argparse
    from QQuantLib.utils.benchmark_utils import combination_for_list
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-base_folder",
        dest="base_folder",
        type=str,
        help="Folder where the data is located",
        default="/mnt/netapp1/Store_CESGA/home/cesga/gferro/2024_09_21_cVar/",
    )
    parser.add_argument(
        "-base_name",
        dest="base_name",
        type=str,
        help="Base name for csv with datasets",
        default="base_name",
    )
    parser.add_argument(
        "-train_folder",
        dest="train_folder",
        type=str,
        help="Folder with the training info",
        default="/mnt/netapp1/Store_CESGA/home/cesga/gferro/2024_09_21_cVar/",
    )
    parser.add_argument(
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="./qpu/qpu_ideal.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "-json_optimizer",
        dest="json_optimizer",
        type=str,
        default=None,
        help="JSON with the Optimizer parameters",
    )
    parser.add_argument(
        "-json_dask",
        dest="json_dask",
        type=str,
        default=None,
        help="JSON with the Optimizer parameters",
    )
    parser.add_argument(
        "-qpu_id",
        dest="qpu_id",
        type=int,
        help="Identify which qpu to use",
        default=None,
    )
    parser.add_argument(
        "--print",
        dest="print",
        default=False,
        action="store_true",
        help="For printing "
    )
    parser.add_argument(
        "--exe",
        dest="execution",
        default=False,
        action="store_true",
        help="For executing program",
    )

    args = parser.parse_args()
    # Defining QPU
    with open(args.json_qpu) as json_file:
        qpu_dict = json.load(json_file)
    qpu_list = combination_for_list(qpu_dict)
    # Getting Optimizer Parameters
    if args.json_optimizer is not None:
        with open(args.json_optimizer) as json_file:
            optimizer_dict = json.load(json_file)
    else:
        optimizer_dict = None
    if args.print:
        if args.qpu_id is None:
            print(qpu_list)
        else:
            print("************ QPU INFO ****************************")
            print(qpu_list[args.qpu_id])
            print("***********************************************")
            print("************ OPTIMIZER INFO **********************")
            print(optimizer_dict)
            print("***********************************************")

    if args.execution:
        if args.qpu_id is not None:
            # Get Dask Client
            dask_client = None
            if args.json_dask is not None:
                from distributed import Client
                dask_client = Client(scheduler_file=args.json_dask)
                import dask
                dask.config.set({
                    "distributed.scheduler.worker-ttl": "10min"
                })
            qpu_info = qpu_list[args.qpu_id]
            info = vars(args)
            info.update({"qpu_info": qpu_info})
            info.update({"optimizer_info": optimizer_dict})
            info.update({"dask_client": dask_client})
            continue_training(**info)
        else:
            print("***********************************************")
            print("You should provide qpu_id for selecting a QPU")
            print("***********************************************")
