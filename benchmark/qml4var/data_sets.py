"""
Functions for creating DataSets
"""
import json
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.special import erf

def get_dataset(name_for_loading):
    # load Datasets
    pdf_training = pd.read_csv(
        name_for_loading+"_training.csv", sep=";", index_col=0
    )
    pdf_testing = pd.read_csv(
        name_for_loading+"_testing.csv", sep=";", index_col=0
    )
    feat = [col for col in pdf_training.columns if "Features" in col]
    x_train = pdf_training[feat].values
    y_train = pdf_training["Labels"].values
    y_train = y_train.reshape((-1, 1))
    x_test = pdf_testing[feat].values
    y_test = pdf_testing["Labels"].values
    y_test = y_test.reshape((-1, 1))
    return x_train, y_train, x_test, y_test

def empirical_distribution_function(data_points: np.array):
    """
    Given an array of data points create the corresponding empirical
    distribution dunction
    Parameters
    ----------

    data_points : numpy array
        numpy array with data sampled

    Returns
    ------

    batch_ : QLM Batch
        QLM Batch with the jobs for computing graidents
    """
    n_sample = data_points.shape[0]
    distribution = np.zeros(n_sample)
    for m_ in range(n_sample):
        count = 0
        for n_ in list(range(0, m_))+list(range(m_+1, n_sample)):
            check = np.all(data_points[m_] >= data_points[n_])
            if check:
                count = count+1

        distribution[m_] = count/(n_sample-1)
    return distribution

def create_random_data(**kwargs):
    """
    Create DataSets

    Parameters
    ----------
    kwargs : keyworg arguments:
        n_points_train : int. Number of points for training
        n_points_test : int. Number of points for testing
        min_val : float. Minimun value for testing dataset
        max_val : float. Maximum value for testing dataset
    Returns
    ------

    x_train : numpy array
        Array with the training dataset features
    y_train : numpy array
        Array with the training dataset labels
    x_test : numpy array
        Array with the testing dataset features
    y_test : numpy array
        Array with the testing dataset labels
    """

    n_points_train = kwargs.get("n_points_train", None)
    n_points_test = kwargs.get("n_points_test", None)
    minval = kwargs.get("minval", None)
    maxval = kwargs.get("maxval", None)
    # Create Features
    feature_number = kwargs.get("feature_number", 1)
    x_train = np.random.randn(n_points_train)
    x_test = np.linspace(minval, maxval, n_points_test)
    # Create Labels
    y_train = empirical_distribution_function(
        np.reshape(x_train, (n_points_train, 1))) - 0.5
    y_train = y_train.reshape(-1, 1)
    y_test = norm.cdf(x_test) - 0.5
    x_train = x_train.reshape((n_points_train, 1))
    x_test = x_test.reshape((n_points_test, 1))
    # Saving datasets and info
    saving_datasets(x_train, y_train, x_test, y_test, **kwargs)

    return x_train, y_train, x_test, y_test

def bs_pdf(
        s_t: float, s_0: float = 1.0, risk_free_rate: float = 0.0,
        volatility: float = 0.5, maturity: float = 0.5, **kwargs):
    """
    Black Scholes PDF
    """

    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    factor = s_t * volatility * np.sqrt(2 * np.pi * maturity)
    exponent = -((np.log(s_t) - mean) ** 2) / (2 * volatility * volatility * maturity)
    density = np.exp(exponent) / factor
    return density

def bs_cdf(
        s_t: float, s_0: float = 1.0, risk_free_rate: float = 0.0,
        volatility: float = 0.5, maturity: float = 0.5, **kwargs):
    """
    Black Scholes PDF
    """
    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    variance = volatility * volatility * maturity
    return 0.5 * (1 + erf((np.log(s_t) - mean) / (np.sqrt(2 * variance))))

def bs_samples(
        number_samples: int, s_0: float = 1.0, risk_free_rate: float = 0.0,
        volatility: float = 0.5, maturity: float = 0.5, **kwargs):
    """
    Black Scholes Samples
    """

    dW = np.random.randn(number_samples)
    s_t = s_0 * np.exp(
        (risk_free_rate - 0.5 * volatility * volatility) * maturity +
        volatility * dW * np.sqrt(maturity))
    return s_t

def create_bs_data(**kwargs):
    """
    Create DataSets with Black Scholes

    Parameters
    ----------

    n_points_train : int
        number of points for training
    n_points_test : int
        number of points for testing
    min_val : float
        minimun value for testing dataset
    max_val : float
        maximum value for testing dataset
    Returns
    ------

    x_train : numpy array
        Array with the training dataset features
    y_train : numpy array
        Array with the training dataset labels
    x_test : numpy array
        Array with the testing dataset features
    y_test : numpy array
        Array with the testing dataset labels
    """
    n_points_train = kwargs.get("n_points_train", None)
    n_points_test = kwargs.get("n_points_test", None)
    minval = kwargs.get("minval", None)
    maxval = kwargs.get("maxval", None)
    # Create Features
    feature_number = kwargs.get("feature_number", 1)
    x_train = bs_samples(n_points_train, **kwargs)
    x_train = x_train.reshape((n_points_train, feature_number))
    x_test = np.linspace(minval, maxval, n_points_test)
    # Create Labels
    y_train = empirical_distribution_function(
        np.reshape(x_train,(n_points_train, feature_number))) - 0.5
    y_test = bs_cdf(x_test, **kwargs) - 0.5
    # Saving datasets and info
    saving_datasets(x_train, y_train, x_test, y_test, **kwargs)
    return x_train, y_train, x_test, y_test

def saving_datasets(x_train, y_train, x_test, y_test, **kwargs):
    """
    Saving Data sets
    """
    name_for_saving = kwargs.get("name_for_saving", None)
    if name_for_saving is not None:
        features = ["Features_{}".format(x_) for x_ in range(x_train.shape[1])]
        pdf_training = pd.DataFrame(x_train, columns=features)
        pdf_training["Labels"] = y_train
        pdf_testing = pd.DataFrame(x_test, columns=features)
        pdf_testing["Labels"] = y_test
        pdf_training.to_csv(
            name_for_saving+"_training.csv", sep=";", index=True)
        pdf_testing.to_csv(
            name_for_saving+"_testing.csv", sep=";", index=True)
        with open(kwargs.get("folder_path") + "/data.json", "w")  as outfile:
            outfile.write(json.dumps(kwargs))


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
