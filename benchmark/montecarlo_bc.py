"""
Price Estimation class for benchmarking
"""
# pylint: disable=wrong-import-position

import sys
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm

sys.path.append("../")
import benchmark.payoff_class as po
import benchmark.probability_class as pc
import QQuantLib.DL.data_loading as dl
from QQuantLib.AE.maximum_likelihood_ae import MLAE
from QQuantLib.AE.ae_classical_qpe import CQPEAE
from QQuantLib.AE.ae_iterative_quantum_pe import IQPEAE
from QQuantLib.AE.iterative_quantum_ae import IQAE
from QQuantLib.AE.real_quantum_ae import RQAE


def text_is_none(variable, variable_name, variable_type=float):
    if variable is None:
        message = (
            variable_name
            + " argument is None. Some "
            + str(variable_type)
            + " should be  provided"
        )
        raise ValueError(message)


class MonteCarlo:

    """
    Class for selectin pay off functions
    algorithm. The loading datsa used is the function loading one.
    """

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.n_qbits = self.kwargs.get("n_qbits", None)
        self.ae_type = self.kwargs.get("ae_type", None)
        text_is_none(self.ae_type, "ae_type", variable_type=str)

        self.x0 = self.kwargs.get("x0", 1.0)
        self.xf = self.kwargs.get("xf", 3.0)

        self.domain = np.linspace(self.x0, self.xf, 2**self.n_qbits)

        self.pc = pc.DensityProbability(**self.kwargs)
        self.probability = self.pc.probability(self.domain)
        if np.sum(self.probability) > 1.0:
            raise ValueError("Probability not properly normalised")

        self.po = po.PayOff(**self.kwargs)
        self.pay_off = self.po.pay_off(self.domain)
        self.payoff_normalisation = 1.0
        if np.max(self.pay_off) > 1.00:
            print("Be aware Pay Off is not properly normalised")
            self.payoff_normalisation = np.max(self.pay_off)
        self.pay_off_normalised = self.pay_off / self.payoff_normalisation
        self.p_gate = None
        self.derivative_oracle = None
        self.registers = None
        self.pay_off_gate = None
        self.solver_ae = None
        self.ae_derivative_price = None
        self.classical_price = None
        self.exact_solution = None
        self.pdf = None
        self.co_target = None
        self.co_index = None
        self.ae_pdf = None
        self.normalised_classical_price = None
        # self.run()

    def run_mc(self):
        """
        run montecarlo
        """

        shots = self.kwargs["shots"]

        mc_domain = (self.xf - self.x0) * np.random.rand(shots) + self.x0
        base = self.xf - self.x0
        mc_integration = base * np.mean(
            self.pc.density_probability(mc_domain) * self.po.pay_off(mc_domain)
        )

        self.classical_price = np.sum(self.pay_off * self.probability)
        # black scholes derivative price
        if self.po.pay_off_bs is not None:
            self.exact_solution = self.po.pay_off_bs(**self.kwargs)

        # Derivative price
        self.ae_derivative_price = mc_integration

        # Configure pandas DataFrame
        pdf = pd.DataFrame([self.kwargs])
        pdf["payoff_normalisation"] = self.payoff_normalisation
        pdf["classical_price"] = self.classical_price
        pdf["derivative_price"] = self.ae_derivative_price
        pdf["classical_price_rfr"] = self.classical_price * np.exp(
            -pdf["risk_free_rate"] * pdf["maturity"]
        )
        pdf["derivative_price_rfr"] = self.ae_derivative_price * np.exp(
            -pdf["risk_free_rate"] * pdf["maturity"]
        )

        pdf["exact_solution"] = self.exact_solution
        pdf["error_classical"] = abs(
            pdf["derivative_price_rfr"] - pdf["classical_price_rfr"]
        )
        pdf["relative_error_classical"] = (
            pdf["error_classical"] / pdf["classical_price_rfr"]
        )

        if self.kwargs["save"]:
            with open(self.kwargs["file_name"], "a") as f_pointer:
                pdf.to_csv(f_pointer, mode="a", header=f_pointer.tell() == 0)
        return pdf

    def run(self):
        """
        For running a complete price estimation.
        """
        lista = []
        for i in range(self.kwargs["number_of_tests"]):
            lista.append(self.run_mc())

        self.pdf = pd.concat(lista)
        # self.pdf.to_csv(self.kwargs["file_name"])
