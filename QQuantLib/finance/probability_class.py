"""
Definition for DensityProbability Class.

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro
"""
from functools import partial
import QQuantLib.finance.classical_finance as cf
from QQuantLib.utils.utils import text_is_none



class DensityProbability:

    """
    Class for selecting pay off functions
    algorithm

    Parameters
    ----------

    probability_type : string
       type of probability density function to load
    kwargs: dictionary
        Dictionary for configuring the asset and the probability \\
        used for simulating its behaviour.Implemented keys:

        s_0 : float
            initial value of the asset
        risk_free_rate : float
            risk free ratio
        maturity : float
            time where the probability wants to be calculated.
        volatiliy : float
            volatility of the asset.
    """

    def __init__(self, probability_type: str, **kwargs):
        """

        Method for initializing the class

        """

        self.probability_type = probability_type
        text_is_none(self.probability_type, "probability_type", variable_type=str)
        self.probability = None
        self.density_probability = None
        self.probability = self.get_density(
            self.probability_type, **kwargs)
        self.density_probability = self.get_density_prob(
            self.probability_type, **kwargs)

    @staticmethod
    def get_density(probability_type, **kwargs):
        """
        Create the probability function

        Parameters
        ----------

        probability_type : string
           type of probability density function to load
        kwargs: dictionary
            with necessary information for configuring the probability \\
            density

            s_0 : float
                initial value of the asset
            risk_free_rate : float
                risk free ratio
            maturity : float
                time where the probability wants to be calculated
            volatiliy : float
                volatility of the asset

        """

        if probability_type == "Black-Scholes":

            s_0 = kwargs.get("s_0", None)
            text_is_none(s_0, "s_0", variable_type=float)
            risk_free_rate = kwargs.get("risk_free_rate", None)
            text_is_none(risk_free_rate, "risk_free_rate", variable_type=float)
            maturity = kwargs.get("maturity", None)
            text_is_none(maturity, "maturity", variable_type=float)
            volatility = kwargs.get("volatility", None)
            text_is_none(volatility, "volatility", variable_type=float)

        else:
            raise ValueError()

        return partial(
            cf.bs_probability,
            s_0=s_0,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            maturity=maturity,
        )

    @staticmethod
    def get_density_prob(probability_type, **kwargs):
        """
        Configures a probability density

        Parameters
        ----------

        probability_type : string
           type of probability density function to load
        kwargs: dictionary
            with necessary information for configuring the probability \\
            density.

            s_0 : float
                initial value of the asset
            risk_free_rate : float
                risk free ratio
            maturity : float
                time where the probability wants to be calculated
            volatiliy : float
                volatility of the asset
        """

        if probability_type == "Black-Scholes":

            s_0 = kwargs.get("s_0", None)
            text_is_none(s_0, "s_0", variable_type=float)
            risk_free_rate = kwargs.get("risk_free_rate", None)
            text_is_none(risk_free_rate, "risk_free_rate", variable_type=float)
            maturity = kwargs.get("maturity", None)
            text_is_none(maturity, "maturity", variable_type=float)
            volatility = kwargs.get("volatility", None)
            text_is_none(volatility, "volatility", variable_type=float)

        else:
            raise ValueError()
        return partial(
            cf.bs_density,
            s_0=s_0,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            maturity=maturity,
        )
