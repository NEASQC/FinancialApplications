"""
Definition for DensityProbability Class.

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro
"""
from functools import partial
import QQuantLib.finance.classical_finance as cf
from QQuantLib.utils.utils import text_is_none



class DensityProbability:

    """
    Class for selectin pay off functions
    algorithm
    """

    def __init__(self, probability_type: str, **kwargs):
        """

        Method for initializing the class

        Parameters
        ----------

        probability_type : string
           type of probability density function to load
        kwargs: dictionary
            with necesary information for connfiguring the probability
            density.
            * s_0: initial value of the asset
            * risk_free_rate: risk free ratio
            * maturity: time where the probability wants to be calculated.
            * volatiliy: volatility of the asset.
        """

        self.probability_type = probability_type
        text_is_none(self.probability_type, "probability_type", variable_type=str)
        self.probability = None
        self.density_probability = None
        self.get_density(**kwargs)
        self.get_density_prob(**kwargs)

    def get_density(self, **kwargs):
        """
        Create the probability function

        Parameters
        ----------

        kwargs: dictionary
            with necesary information for connfiguring the probability
            density.
            * s_0: initial value of the asset
            * risk_free_rate: risk free ratio
            * maturity: time where the probability wants to be calculated.
            * volatiliy: volatility of the asset.

        """

        if self.probability_type == "Black-Scholes":

            s_0 = kwargs.get("s_0", None)
            text_is_none(s_0, "s_0", variable_type=float)
            risk_free_rate = kwargs.get("risk_free_rate", None)
            text_is_none(risk_free_rate, "risk_free_rate", variable_type=float)
            maturity = kwargs.get("maturity", None)
            text_is_none(maturity, "maturity", variable_type=float)
            volatility = kwargs.get("volatility", None)
            text_is_none(volatility, "volatility", variable_type=float)

            self.probability = partial(
                cf.bs_probability,
                s_0=s_0,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                maturity=maturity,
            )
        else:
            raise ValueError()

    def get_density_prob(self, **kwargs):
        """
        Configures a probabiliy density

        Parameters
        ----------

        kwargs: dictionary
            with necesary information for connfiguring the probability
            density.
            * s_0: initial value of the asset
            * risk_free_rate: risk free ratio
            * maturity: time where the probability wants to be calculated.
            * volatiliy: volatility of the asset.

        """

        if self.probability_type == "Black-Scholes":

            s_0 = kwargs.get("s_0", None)
            text_is_none(s_0, "s_0", variable_type=float)
            risk_free_rate = kwargs.get("risk_free_rate", None)
            text_is_none(risk_free_rate, "risk_free_rate", variable_type=float)
            maturity = kwargs.get("maturity", None)
            text_is_none(maturity, "maturity", variable_type=float)
            volatility = kwargs.get("volatility", None)
            text_is_none(volatility, "volatility", variable_type=float)

            self.density_probability = partial(
                cf.bs_density,
                s_0=s_0,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                maturity=maturity,
            )
        else:
            raise ValueError()
