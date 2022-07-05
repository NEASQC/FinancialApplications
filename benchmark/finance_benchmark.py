"""
Price Estimation class for benchmarking
"""
#pylint: disable=wrong-import-position

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
        message = variable_name+' argument is None. Some '+str(variable_type)+' should be  provided'
        raise ValueError(message)

class PriceEstimation:

    """
    Class for selectin pay off functions
    algorithm. The loading datsa used is the function loading one.
    """

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.n_qbits = self.kwargs.get('n_qbits', None)
        self.ae_type = self.kwargs.get('ae_type', None)
        text_is_none(self.ae_type, 'ae_type', variable_type=str)

        self.x0 = self.kwargs.get('x0', 1.0)
        self.xf = self.kwargs.get('xf', 3.0)

        self.domain = np.linspace(self.x0, self.xf, 2**self.n_qbits)

        self.pc = pc.DensityProbability(**self.kwargs)
        self.probability = self.pc.probability(self.domain)
        if np.sum(self.probability) > 1.0:
            raise ValueError('Probability not properly normalised')

        self.po = po.PayOff(**self.kwargs)
        self.pay_off = self.po.pay_off(self.domain)
        self.payoff_normalisation = 1.0
        if np.max(self.pay_off) > 1.00:
            print('Be aware Pay Off is not properly normalised')
            self.payoff_normalisation = np.max(self.pay_off)
        self.pay_off_normalised = self.pay_off/self.payoff_normalisation
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

    def oracle_loading_m01(self):
        """
        Method for creating the oracle. The probability density will be
        loaded with proability density gate and the
        payoff functions will be loaded as function arrays.
        """

        self.derivative_oracle = qlm.QRoutine()
        # Creation of probability loading gate
        self.p_gate = dl.load_probability(
            self.probability
        )
        # Creation of function loading gate
        self.pay_off_gate = dl.load_array(
            np.sqrt(self.pay_off_normalised),
            id_name='PayOff'
        )
        self.registers = self.derivative_oracle.new_wires(
            self.pay_off_gate.arity
        )
        #Step 1 of Procedure: apply loading probabilty gate
        self.derivative_oracle.apply(
            self.p_gate,
            self.registers[:self.p_gate.arity]
        )
        #Step 2 of Procedure: apply loading function gate
        self.derivative_oracle.apply(
            self.pay_off_gate,
            self.registers
        )
        self.co_target = [0]
        self.co_index = [self.derivative_oracle.arity - 1]

    def oracle_loading_m02(self):
        """
        Method for creating the oracle. The probability density and the
        payoff functions will be loaded as function arrays.
        """

        self.derivative_oracle = qlm.QRoutine()
        #For new data loading procedure we need n+2 qbits
        self.registers = self.derivative_oracle.new_wires(self.n_qbits+2)
        #Step 2 of Procedure: apply Uniform distribution
        self.derivative_oracle.apply(
            dl.uniform_distribution(self.n_qbits),
            self.registers[:self.n_qbits]
        )
        #Step 3 of Procedure: apply loading function operator for loading p(x)
        self.p_gate = dl.load_array(
            self.probability,
            id_name='Probability'
        )
        self.derivative_oracle.apply(
            self.p_gate,
            [self.registers[:self.n_qbits], self.registers[self.n_qbits]]
        )
        #Step 5 of Procedure: apply loading function operator for loading f(x)
        self.pay_off_gate = dl.load_array(
            self.pay_off_normalised,
            id_name='PayOff'
        )
        self.derivative_oracle.apply(
            self.pay_off_gate,
            [self.registers[:self.n_qbits], self.registers[self.n_qbits+1]]
        )
        #Step 7 of Procedure: apply Uniform distribution
        self.derivative_oracle.apply(
            dl.uniform_distribution(self.n_qbits),
            self.registers[:self.n_qbits]
        )
        self.co_target = [0 for i in range(self.derivative_oracle.arity)]
        self.co_index = [i for i in range(self.derivative_oracle.arity)]

    def create_oracle(self):
        """
        For selecting the loading method.
        """
        if self.kwargs["probability_loading"]:
            self.oracle_loading_m01()
        else:
            self.oracle_loading_m02()

    def run_ae(self):
        """
        Method for executing an amplitude estimation problem.
        """
        # The input epsilon will be the error we want in the final
        # derivative price.
        # delta_price = self.kwargs.get("epsilon", None)
        # Only IQAE and RQAE will have the epsilon. Other algorithms
        # do not need it
        # self.kwargs.update({"epsilon": None})
        epsilon = self.kwargs.get("epsilon", None)
        self.kwargs.update({"delta_price": None})

        if self.ae_type == 'MLAE':
            self.solver_ae = MLAE(
                self.derivative_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == 'CQPEAE':
            self.solver_ae = CQPEAE(
                self.derivative_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == 'IQPEAE':
            self.solver_ae = IQPEAE(
                self.derivative_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == 'IQAE':
            # if delta_price is None:
            #    raise ValueError("For IQAE epsilon can not be None")

            # epsilon = delta_price / (self.payoff_normalisation * np.exp(
            #     -self.kwargs["risk_free_rate"] * self.kwargs["maturity"]))

            # delta_price = epsilon * self.payoff_normalisation * np.exp(
            #     -self.kwargs["risk_free_rate"] * self.kwargs["maturity"])


            # if not self.kwargs["probability_loading"]:
            #     # In this case we need to know the value of the estimation.
            #     # We are going to use the classical result:
            #     sqrt_a = np.sum(
            #         self.pay_off_normalised * self.probability) / (2**self.n_qbits)
            #     epsilon = epsilon * 2 * sqrt_a / (2**self.n_qbits)
            # print("IQAE_epsilon: ", epsilon)

            #self.kwargs.update({"delta_price": delta_price})
            self.solver_ae = IQAE(
                self.derivative_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == 'RQAE':
            # if delta_price is None:
            #     raise ValueError("For RQAE epsilon can not be None")

            # epsilon = delta_price/ (self.payoff_normalisation * np.exp(
            #     -self.kwargs["risk_free_rate"]*self.kwargs["maturity"]
            # ) * 2**self.n_qbits)

            # delta_price = epsilon * self.payoff_normalisation * np.exp(
            #     -self.kwargs["risk_free_rate"]*self.kwargs["maturity"]
            # ) * 2**self.n_qbits

            if self.kwargs["probability_loading"]:
                # RQAE method can not be used with pure probability
                # density loading. So probability_loading == False
                string_error = (
                    "RQAE method can not be use with pure"
                    "density probability loading.\n Only can be used with"
                    "density probability array loading.\n"
                    "probability_loading MUST BE False"
                )
                raise ValueError(string_error)

            # self.kwargs.update({"delta_price": delta_price})
            self.solver_ae = RQAE(
                self.derivative_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )

        # run the amplitude estimation algorithm
        self.solver_ae.run()
        # classical derivative price
        self.classical_price = np.sum(self.pay_off*self.probability)
        # black scholes derivative price
        if self.po.pay_off_bs is not None:
            self.exact_solution = self.po.pay_off_bs(**self.kwargs)

        #Recover amplitude estimation from ae_solver
        self.ae_pdf = pd.DataFrame(
            [self.solver_ae.ae, self.solver_ae.ae_l, self.solver_ae.ae_u],
            index=["ae", "ae_l", "ae_u"]).T

        # Using classical price undo the payoff normalisation
        self.normalised_classical_price =  self.classical_price / self.payoff_normalisation

        if self.kwargs["probability_loading"]:
            # For density loaded as a density
            a_estimation = self.ae_pdf
            if self.ae_type == 'IQAE':
                # Epsilon propagation for IQAE
                delta_price = epsilon * self.payoff_normalisation * np.exp(
                    -self.kwargs["risk_free_rate"] * self.kwargs["maturity"])
                self.kwargs.update({"delta_price": delta_price})
        else:
            # For density loaded as a function
            if self.ae_type == 'RQAE':
                # Estimation for RQAE
                a_estimation = 2**self.n_qbits*self.ae_pdf
                # Epsilon propagation for RQAE
                delta_price = epsilon * self.payoff_normalisation * np.exp(
                    -self.kwargs["risk_free_rate"]*self.kwargs["maturity"]
                ) * 2**self.n_qbits
                self.kwargs.update({"delta_price": delta_price})
                # For  getting the amplitude estimation True Result.
                self.normalised_classical_price = self.normalised_classical_price / 2**self.n_qbits
            else:
                # Estimation for other no RQAE
                a_estimation = 2**self.n_qbits * np.sqrt(self.ae_pdf)
                # For  getting the amplitude estimation True Result.
                self.normalised_classical_price = (self.normalised_classical_price / 2**self.n_qbits)**2

            if self.ae_type == 'IQAE':
                delta_price = epsilon * 2**self.n_qbits * self.payoff_normalisation * \
                    np.exp(-self.kwargs["risk_free_rate"] * self.kwargs["maturity"]) / \
                    (2 * np.sqrt(self.solver_ae.ae))

                self.kwargs.update({"delta_price": delta_price})

        # Derivative price
        self.ae_derivative_price = a_estimation * self.payoff_normalisation

        #Configure pandas DataFrame
        pdf = pd.DataFrame([self.kwargs])
        pdf = pd.concat([pdf, self.ae_pdf], axis=1)
        pdf["payoff_normalisation"] = self.payoff_normalisation
        pdf["classical_price"] = self.classical_price
        pdf[["derivative_price_" + col for col in a_estimation.columns]] = \
            self.ae_derivative_price
        pdf["classical_price_rfr"] = self.classical_price*np.exp(
            -pdf["risk_free_rate"]*pdf["maturity"]
        )
        pdf[["derivative_price_rfr_" + col for col in a_estimation.columns]] = \
            self.ae_derivative_price * \
            np.exp(-pdf["risk_free_rate"]*pdf["maturity"]).iloc[0]

        pdf["exact_solution"] = self.exact_solution
        pdf["error_classical"] = abs(pdf["derivative_price_rfr_ae"] - pdf["classical_price_rfr"])
        pdf["relative_error_classical"] = pdf["error_classical"] / pdf["classical_price_rfr"]
        pdf["error_exact"] = abs(pdf["derivative_price_rfr_ae"] - pdf["exact_solution"])
        pdf["relative_error_exact"] = pdf["error_classical"] / pdf["exact_solution"]
        pdf["normalised_classical_price"] = self.normalised_classical_price
        pdf["error_ae"] = abs(pdf["ae"] - pdf["normalised_classical_price"])
        pdf["circuit_stasts"] = [self.solver_ae.circuit_statistics]
        pdf["run_time"] = self.solver_ae.run_time

        if self.kwargs["save"]:
            with open(self.kwargs["file_name"], 'a') as f_pointer:
                pdf.to_csv(f_pointer, mode="a", header=f_pointer.tell() == 0)
        return pdf


    def run(self):
        """
        For running a complete price estimation.
        """
        lista = []
        for i in range(self.kwargs["number_of_tests"]):
            self.create_oracle()
            lista.append(self.run_ae())

        self.pdf = pd.concat(lista)
        #self.pdf.to_csv(self.kwargs["file_name"])
