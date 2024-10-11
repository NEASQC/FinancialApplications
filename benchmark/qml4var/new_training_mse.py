"""
Script for Training a PQC that works as a surrogate model for a complex
and time consuming financial CDF.
This training needs the computation of the CDF and the corresponding PDF
"""
import sys
import os
import uuid
import json
import numpy as np
import pandas as pd
sys.path.append("../../")
from QQuantLib.qml4var.data_utils import get_dataset
from QQuantLib.qml4var.architectures import hardware_efficient_ansatz, \
     z_observable, normalize_data, init_weights
from QQuantLib.qml4var.myqlm_workflows import mse_workflow
from QQuantLib.qml4var.losses import numeric_gradient
from QQuantLib.qml4var.adam import adam_optimizer_loop


def store_info(base_folder, optimizer_dict, pqc_dict):
    """"
    Function for saving Initial Info
    Parameters
    ----------

    base folder : path
        path for storing the data
    optimizer_dict : dict
        dictionary with the info of the Optimizer
    pqc_dict : dict
        dictionary with the info of the PQC
    Returns
    -------

    unique_directory : path
        path with folder for storing the resuls. Unique identificator
    """
    # Create unique folder name for storing results
    unique_directory = base_folder + str(uuid.uuid4())+"/"
    if not os.path.exists(unique_directory):
        os.makedirs(unique_directory)
    # Saving Optimizer info
    with open(unique_directory + "optimizer_dict.json", "w")  as outfile: 
        outfile.write(json.dumps(optimizer_dict))
    # Saving PQC info
    with open(unique_directory + "pqc_dict.json", "w")  as outfile:
        outfile.write(json.dumps(pqc_dict))
    return unique_directory

def batch_generator(X, Y, batch_size):
    return [(X[i:i+batch_size] , Y[i:i+batch_size]) for i in range(0, len(X), batch_size)]


def new_training(**kwargs):
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
    base_frecuency, shift_feature = normalize_data(
        [data_info["minval"]] * data_info["features_number"],
        [data_info["maxval"]] * data_info["features_number"],
        [-0.5*np.pi] * data_info["features_number"],
        [0.5*np.pi] * data_info["features_number"],
    )
    # Get PQC parameter Configuration
    pqc_info = kwargs.get("pqc_info", None)
    pqc_info.update({
        "base_frecuency" : list(base_frecuency),
        "shift_feature" : list(shift_feature)
    })
    # Create PQC and Observable
    pqc, weights_names, features_names = hardware_efficient_ansatz(**pqc_info)
    observable = z_observable(**pqc_info)
    # Get the QPU info
    qpu_info = kwargs.get("qpu_info", None)
    # Get Optimizer INFO
    optimizer_info = kwargs.get("optimizer_info", None)
    # number of shots should be provided into the optimizer_info
    nbshots = optimizer_info["nbshots"]
    # number of discretization points for domain should be provided into
    # the optimizer_info
    points = optimizer_info["points"]
    # Get Dask client if provided
    dask_client = kwargs.get("dask_client", None)
    # Get Optimizer INFO
    optimizer_info = kwargs.get("optimizer_info", None)
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
    mse_loss_ = lambda w_, x_, y_: mse_workflow(
        w_, x_, y_, dask_client=dask_client, **workflow_cfg)
    # Configure the numeric gradient function
    numeric_gradient_ = lambda w_, x_, y_: numeric_gradient(
        w_, x_, y_, mse_loss_)
    # Configure the loss function for evaluation
    training_loss = lambda w_: mse_workflow(
        w_, x_train, y_train, dask_client=dask_client, **workflow_cfg)
    # Configure the MSE for evaluation in testing data
    testing_metric = lambda w_: mse_workflow(
        w_, x_test, y_test, dask_client=dask_client, **workflow_cfg)
    # Set the Batch size and th eBatch generator
    batch_size = kwargs.get("batch_size", None)
    if batch_size is None:
        batch_size = len(x_train)
    batch_generator_ = batch_generator(x_train, y_train, batch_size)

    # Do the stuff
    save = True
    repetitions = kwargs.get("repetitions", None)
    for i in range(repetitions):
        # Initial weights
        initial_weights = init_weights(weights_names)
        optimizer_info.update({"file_to_save": None})
        # Saving staff
        if save:
            # Create the Folder with uuid
            store_folder = store_info(base_folder, optimizer_info, pqc_info)
            # Create the csv to store results
            columns = weights_names + ["t", "loss", "metric"]
            pdf = pd.DataFrame(columns=columns)
            file_to_save = store_folder + "evolution.csv"
            pdf.to_csv(file_to_save, sep=";", index=True)
            optimizer_info.update({"file_to_save": file_to_save})
        # Training Time
        weights = adam_optimizer_loop(
            weights_dict=initial_weights,
            loss_function=training_loss,
            metric_function=testing_metric,
            gradient_function=numeric_gradient_,
            batch_generator=batch_generator_,
            initial_time=0,
            **optimizer_info
        )
        print(weights)



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
        "-json_qpu",
        dest="json_qpu",
        type=str,
        default="./qpu/qpu_ideal.json",
        help="JSON with the qpu configuration",
    )
    parser.add_argument(
        "-json_pqc",
        dest="json_pqc",
        type=str,
        default="./base_pqc.json",
        help="JSON with the PQCs parameters",
    )
    parser.add_argument(
        "-json_optimizer",
        dest="json_optimizer",
        type=str,
        default="./base_optimizer.json",
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
        "-json_dask",
        dest="json_dask",
        type=str,
        default=None,
        help="JSON with the Optimizer parameters",
    )
    parser.add_argument(
        "-repetitions",
        dest="repetitions",
        type=int,
        help="Number of trainings to execute.",
        default=1,
    )
    parser.add_argument(
        "-batch_size",
        dest="repetitions",
        type=int,
        help="Batch Size",
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
    # Getting PQC Parameters
    with open(args.json_pqc) as json_file:
        pqc_dict = json.load(json_file)
    # Getting Optimizer Parameters
    with open(args.json_optimizer) as json_file:
        optimizer_dict = json.load(json_file)
    if args.print:
        if args.qpu_id is None:
            print(qpu_list)
        else:
            print("************ QPU INFO ****************************")
            print(qpu_list[args.qpu_id])
            print("***********************************************")
            print("************ PQC INFO ****************************")
            print(pqc_dict)
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
            qpu_info = qpu_list[args.qpu_id]
            info = vars(args)
            info.update({"qpu_info": qpu_info})
            info.update({"pqc_info": pqc_dict})
            info.update({"optimizer_info": optimizer_dict})
            info.update({"dask_client": dask_client})

            new_training(**info)
        else:
            print("***********************************************")
            print("You should provide qpu_id for selecting a QPU")
            print("***********************************************")

