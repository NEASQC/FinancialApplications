"""
Functions for heliping to build the datasets
"""
import json
import numpy as np
import pandas as pd
from scipy.special import erf

def empirical_distribution_function_old(data_points: np.array):
    """
    Given an array of data points create the corresponding empirical
    distribution dunction
    Parameters
    ----------

    data_points : numpy array
        numpy array with data sampled

    Returns
    -------

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

def empirical_cdf(data_points):
    """
    Given an array of data points create the corresponding empirical
    distribution function
    Parameters
    ----------
    data_points : numpy array
        numpy array with data sampled
    Returns
    -------
    emp_cdf : numpy array
        numpy array with the empirical cdf of the input data
    """
    if len(data_points.shape) == 1:
        data_points = data_points.reshape((data_points.shape[0], 1))


    emp_cdf = np.array(
        [np.sum(np.all(data_points <= x, axis=1)) for x in data_points]
    ) / data_points.shape[0]
    return emp_cdf


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
