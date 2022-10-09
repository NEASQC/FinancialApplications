"""
Class definition for the PayOff

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro
"""
from functools import partial
import QQuantLib.finance.classical_finance as cf
from QQuantLib.utils.utils import text_is_none


class PayOff:

    """
    Class for selecting derivative otpions and configuring them.
    """

    def __init__(self, **kwargs):
        """

        Method for initializing the class

        Parameters
        ----------

        pay_off : string
           type of pay_off function to load
        kwargs: dictionary
            dictionary with keys for configuring the option.
            * strike: float
                strike of the option.
            * coupon: float
                only valid for Digital Options
        """

        self.pay_off_type = kwargs.get("pay_off_type", None)
        text_is_none(self.pay_off_type, "pay_off_type", variable_type=str)
        # European Options
        self.strike = kwargs.get("strike", None)
        # Digital Options
        self.coupon = kwargs.get("coupon", None)
        self.pay_off = None
        self.pay_off_bs = None
        self.get_pay_off()

    def get_pay_off(self):
        """
        Select of a PayOff
        """

        if self.pay_off_type == "European_Call_Option":
            text_is_none(self.strike, "strike", variable_type=float)
            # PayOff
            self.pay_off = partial(cf.call_payoff, strike=self.strike)
            # Exact Solution for PayOff under BS
            self.pay_off_bs = partial(cf.bs_call_price, strike=self.strike)
        elif self.pay_off_type == "European_Put_Option":
            text_is_none(self.strike, "strike", variable_type=float)
            # PayOff
            self.pay_off = partial(cf.put_payoff, strike=self.strike)
            # Exact Solution for PayOff under BS
            self.pay_off_bs = partial(cf.bs_put_price, strike=self.strike)
        elif self.pay_off_type == "Digital_Call_Option":
            text_is_none(self.strike, "strike", variable_type=float)
            text_is_none(self.coupon, "coupon", variable_type=float)
            # PayOff
            self.pay_off = partial(
                cf.digital_call_payoff, strike=self.strike, coupon=self.coupon
            )
            # Exact Solution for PayOff under BS
            self.pay_off_bs = partial(
                cf.bs_digital_call_price, strike=self.strike, coupon=self.coupon
            )
        elif self.pay_off_type == "Digital_Put_Option":
            text_is_none(self.strike, "strike", variable_type=float)
            text_is_none(self.coupon, "coupon", variable_type=float)
            # PayOff
            self.pay_off = partial(
                cf.digital_put_payoff, strike=self.strike, coupon=self.coupon
            )
            # Exact Solution for PayOff under BS
            self.pay_off_bs = partial(
                cf.bs_digital_put_price, strike=self.strike, coupon=self.coupon
            )
        elif self.pay_off_type == "Futures":
            text_is_none(self.strike, "strike", variable_type=float)
            # PayOff
            self.pay_off = partial(cf.futures_payoff, strike=self.strike)
            # Exact Solution for PayOff under BS
            self.pay_off_bs = partial(cf.bs_forward_price, strike=self.strike)
        else:
            raise ValueError(
                "Not valid pay off type was provided. Valid types: \
                European_Call_Option,\n \
                European_Put_Option,\n \
                Digital_Call_Option,\n \
                Digital_Put_Option"
            )
