"""
Functions for creating DataSets
"""
import json
import sys
import numpy as np
from scipy.stats import norm, multivariate_normal
from itertools import product
sys.path.append("../../")
from QQuantLib.qml4var.data_utils import  empirical_cdf, bs_cdf, \
    bs_samples, saving_datasets


def create_random_data(**kwargs):
    """
    Create DataSets

    Parameters
    ----------
    kwargs : keyworg arguments:
        n_points_train : int. Number of points for training
        n_points_test : int. Number of points for testing
        min_val : float, int or list. Minimun value for testing dataset
        max_val : float, int or list. Maximum value for testing dataset
        features_number : number of features
    Returns
    ------

    train_x : numpy array
        Array with the training dataset features:
        shape = (n_points_train, features_number)
    train_y : numpy array
        Array with the training dataset labels
        shape = (n_points_train, 1)
    test_x : numpy array
        Array with the testing dataset features
        shape = (n_points_test, features_number)
    test_y : numpy array
        Array with the testing dataset labels
        shape = (n_points_test, 1)
    """

    n_points_train = kwargs.get("n_points_train", None)
    n_points_test = kwargs.get("n_points_test", None)
    minval = kwargs.get("minval", None)
    maxval = kwargs.get("maxval", None)
    # Create Features
    feature_number = kwargs.get("features_number", 1)
    train_x = np.random.normal(size=(n_points_train, feature_number))
    # Build minval and maxval for properly dimension array generation
    if type(minval) in [list, float, int]:
        if type(maxval) in [list, float, int]:
            if not isinstance(maxval, list) and not isinstance(minval, list):
                minval = [minval] * feature_number
                maxval = [maxval] * feature_number
        else:
            raise ValueError("maxval SHOULD BE: int, float or list")
    else:
        raise ValueError("minval SHOULD BE: int, float or list")

    test_x = np.linspace(minval, maxval, n_points_test)
    test_x = np.array(list(
        product(*[test_x[:, i] for i in range(test_x.shape[1])])
    ))
    # Create Labels
    train_y = empirical_cdf(train_x) - 0.5
    train_y = train_y.reshape((-1, 1))
    if feature_number == 1:
        test_y = norm.cdf(test_x) - 0.5
    elif feature_number > 1:
        means_ = [0] * feature_number
        covs_ = [
            [int(i == j) for j in range(feature_number)] \
            for i in range(feature_number)
        ]
        mnorm = multivariate_normal(mean=means_, cov=covs_)
        test_y = mnorm.cdf(test_x) - 0.5
    test_y = test_y.reshape((-1, 1))
    # Saving datasets and info
    saving_datasets(train_x, train_y, test_y, test_y, **kwargs)
    return train_x, train_y, test_x, test_y

def create_bs_data(**kwargs):
    """
    Create DataSets with Black Scholes. Only for 1 input feature

    Parameters
    ----------
    kwargs : keyworg arguments. In addition to the kwargs provided to
    create_random_data function the following arguments for configuring
    BS CDF can be provided. BE AWARE: features_number MUST BE 1:
        s_0 : initial value of the stock
        risk_free_rate: risk free rate
        volatility: volatility of the stock
        maturity: maturity of the stock

    Returns
    ------

    train_x : numpy array
        Array with the training dataset features
        shape = (n_points_train, 1)
    train_y : numpy array
        Array with the training dataset labels
        shape = (n_points_train, 1)
    test_x : numpy array
        Array with the testing dataset features
        shape = (n_points_test, 1)
    test_y : numpy array
        Array with the testing dataset labels
        shape = (n_points_test, 1)
    """
    n_points_train = kwargs.get("n_points_train", None)
    n_points_test = kwargs.get("n_points_test", None)
    minval = kwargs.get("minval", None)
    maxval = kwargs.get("maxval", None)
    # Create Features
    feature_number = kwargs.get("features_number", 1)
    # Build minval and maxval for properly dimension array generation
    if type(minval) != float:
        raise ValueError("minval SHOULD BE a float")
    if type(maxval) != float:
        raise ValueError("maxval SHOULD BE a float")
    train_x = bs_samples(n_points_train, **kwargs)
    train_x = train_x.reshape((-1, 1))
    test_x = np.linspace(minval, maxval, n_points_test)
    test_x = test_x.reshape((-1, 1))
    # Create Labels
    train_y = empirical_cdf(
        np.reshape(train_x, (n_points_train, feature_number))) - 0.5
    test_y = bs_cdf(test_x, **kwargs) - 0.5

    # Saving datasets and info
    saving_datasets(train_x, train_y, test_x, test_y, **kwargs)
    return train_x, train_y, test_x, test_y

if __name__ == "__main__":


    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json",
        dest="json_arg",
        type=str,
        default="./random.json",
        help="JSON file with the data configuration",
    )
    parser.add_argument(
        "-folder",
        dest="folder_path",
        type=str,
        help="Path for storing data",
        default=None,
    )
    parser.add_argument(
        "-base_name",
        dest="base_name",
        type=str,
        help="Base name for csv with datasets",
        default="base_name",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For saving datasets",
    )
    args = parser.parse_args()
    with open(args.json_arg) as json_file:
        data_cfg = json.load(json_file)

    data_cfg.update({"name_for_saving":None})
    if args.save:
        if not os.path.exists(args.folder_path):
            os.makedirs(args.folder_path)
        name_for_saving = args.folder_path + "/" + args.base_name
        print(name_for_saving)
        data_cfg.update({"folder_path":args.folder_path})
        data_cfg.update({"name_for_saving":name_for_saving})

    print(data_cfg)
    if data_cfg["distribution"] == "bs":
        x_train, y_train, x_test, y_test = create_bs_data(**data_cfg)
    elif data_cfg["distribution"] == "random":
        x_train, y_train, x_test, y_test = create_random_data(**data_cfg)
