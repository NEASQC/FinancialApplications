"""
Price Estimation class for benchmarking
"""
# pylint: disable=wrong-import-position

import sys
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm

sys.path.append("../")
import benchmark.probability_class as pc
import QQuantLib.DL.data_loading as dl
from QQuantLib.AE.maximum_likelihood_ae import MLAE
from QQuantLib.AE.ae_classical_qpe import CQPEAE
from QQuantLib.AE.ae_iterative_quantum_pe import IQPEAE
from QQuantLib.AE.iterative_quantum_ae import IQAE
from QQuantLib.AE.real_quantum_ae import RQAE


def text_is_none(variable, variable_name, variable_type=float):
    """
    Raise an exception if variable is None
    """
    if variable is None:
        message = (
            variable_name
            + " argument is None. Some "
            + str(variable_type)
            + " should be  provided"
        )
        raise ValueError(message)


class CumulativeSumVaR:

    """
    Class for selectin pay off functions
    algorithm. The loading datsa used is the function loading one.
    """

    def __init__(self, index=None, **kwargs):
        """
        Parameters
        ----------
        
        index : int
            index until do the cumulative sum
        kwargs : dictionary
            python dictionary for configurating the complete Cumulative
            sum problem
        """

        self.kwargs = kwargs
        self.n_qbits = self.kwargs.get("n_qbits", None)
        self.ae_type = self.kwargs.get("ae_type", None)
        # text_is_none(self.ae_type, "ae_type", variable_type=str)

        self.x0 = self.kwargs.get("x0", 1.0)
        self.xf = self.kwargs.get("xf", 3.0)

        self.domain = np.linspace(self.x0, self.xf, 2**self.n_qbits)

        self.pc = pc.DensityProbability(**self.kwargs)
        self.probability = self.pc.probability(self.domain)
        if np.sum(self.probability) > 1.0:
            raise ValueError("Probability not properly normalised")

        if index is None:
            raise ValueError("index CAN NOT BE None")
        self.index = index

        if index >= len(self.domain):
            raise ValueError("Index is bigger than the leng of the Probability array")

        # QLM routine properties
        self.var_oracle = None
        self.p_gate = None
        self.step_gate = None
        self.registers = None
        self.co_target = None
        self.co_index = None
        # Amplitude Estimation related properties
        self.solver_ae = None
        self.classical_var = None
        self.ae_pdf = None
        self.ae_var = None
        self.pdf = None
        # self.run()

    def oracle_loading_m01(self):
        """
        Method for creating the oracle. The probability density will be
        loaded with proability density gate and the
        payoff functions will be loaded as function arrays.
        """

        self.var_oracle = qlm.QRoutine()
        # Creation of probability loading gate
        self.p_gate = dl.load_probability(self.probability)
        # creation of the gate for loading step function
        self.step_gate = dl.step_array(self.index, 2**self.n_qbits)

        self.registers = self.var_oracle.new_wires(self.step_gate.arity)
        # Step 1 of Procedure: apply loading probabilty gate
        self.var_oracle.apply(self.p_gate, self.registers[: self.p_gate.arity])
        # Step 2 of Procedure: apply loading function gate
        self.var_oracle.apply(self.step_gate, self.registers)
        self.co_target = [0]
        self.co_index = [self.var_oracle.arity - 1]

    def oracle_loading_m02(self):
        """
        Method for creating the oracle. The probability density and the
        payoff functions will be loaded as function arrays.
        """

        self.var_oracle = qlm.QRoutine()
        # For new data loading procedure we need n+2 qbits
        self.registers = self.var_oracle.new_wires(self.n_qbits + 2)
        # Step 2 of Procedure: apply Uniform distribution
        self.var_oracle.apply(
            dl.uniform_distribution(self.n_qbits), self.registers[: self.n_qbits]
        )
        # Step 3 of Procedure: apply loading function operator for loading p(x)
        self.p_gate = dl.load_array(self.probability, id_name="Probability")
        self.var_oracle.apply(
            self.p_gate, [self.registers[: self.n_qbits], self.registers[self.n_qbits]]
        )
        # Step 5 of Procedure: apply loading function operator for loading step function
        self.step_gate = dl.step_array(self.index, 2**self.n_qbits)
        self.var_oracle.apply(
            self.step_gate,
            [self.registers[: self.n_qbits], self.registers[self.n_qbits + 1]],
        )
        # Step 7 of Procedure: apply Uniform distribution
        self.var_oracle.apply(
            dl.uniform_distribution(self.n_qbits), self.registers[: self.n_qbits]
        )
        self.co_target = [0 for i in range(self.var_oracle.arity)]
        self.co_index = [i for i in range(self.var_oracle.arity)]

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

        Returns
        ----------

        pdf : pandas DataFrame
            pandas DataFrame with different information about the cumulative
            sum calculations
        """
        # The input epsilon will be the error we want in the final
        # derivative price.
        # delta_price = self.kwargs.get("epsilon", None)
        # Only IQAE and RQAE will have the epsilon. Other algorithms
        # do not need it
        # self.kwargs.update({"epsilon": None})
        epsilon = self.kwargs.get("epsilon", None)
        self.kwargs.update({"delta_price": None})

        if self.ae_type == "MLAE":
            self.solver_ae = MLAE(
                self.var_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == "CQPEAE":
            self.solver_ae = CQPEAE(
                self.var_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == "IQPEAE":
            self.solver_ae = IQPEAE(
                self.var_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == "IQAE":
            self.solver_ae = IQAE(
                self.var_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )
        elif self.ae_type == "RQAE":

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
                self.var_oracle,
                target=self.co_target,
                index=self.co_index,
                **self.kwargs
            )

        # run the amplitude estimation algorithm
        self.solver_ae.run()
        # classical derivative price
        self.classical_var = np.sum(self.probability[:self.index])
        # Recover amplitude estimation from ae_solver
        self.ae_pdf = pd.DataFrame(
            [self.solver_ae.ae, self.solver_ae.ae_l, self.solver_ae.ae_u],
            index=["ae", "ae_l", "ae_u"],
        ).T


        if self.kwargs["probability_loading"]:
            # For density loaded as a density
            a_estimation = self.ae_pdf
        else:
            # For density loaded as a function
            if self.ae_type == "RQAE":
                # Estimation for RQAE
                a_estimation = 2**self.n_qbits * self.ae_pdf
            else:
                # Estimation for other no RQAE
                a_estimation = 2**self.n_qbits * np.sqrt(self.ae_pdf)


        self.ae_var = pd.DataFrame()
        for i in a_estimation.columns:
            self.ae_var['Var_' + i] = a_estimation[i]
        #self.ae_var = a_estimation
        #self.ae_var.columns = ["Var_" + col for col in a_estimation.columns]

        # Configure pandas DataFrame
        pdf = pd.DataFrame([self.kwargs])
        pdf = pd.concat([pdf, self.ae_pdf, self.ae_var], axis=1)
        pdf["classical_var"] = self.classical_var
        # pdf[
        #     ["Var_" + col for col in a_estimation.columns]
        # ] = self.ae_var

        pdf["error_classical"] = abs(
            pdf["Var_ae"] - pdf["classical_var"]
        )
        pdf["relative_error_classical"] = (
            pdf["error_classical"] / pdf["classical_var"]
        )
        pdf["circuit_stasts"] = [self.solver_ae.circuit_statistics]
        pdf["run_time"] = self.solver_ae.run_time

        if self.kwargs["save"]:
            with open(self.kwargs["file_name"], "a") as f_pointer:
                pdf.to_csv(f_pointer, mode="a", header=f_pointer.tell() == 0)
        return pdf

    def run(self):
        """
        Run a cumulative sum to index
        """
        if self.kwargs['ae_type'] in ["MLAE", "IQAE", "RQAE", "CQPEAE", "IQPEAE"]:
            # Computing using AE
            self.create_oracle()
            self.pdf = self.run_ae()
            cumsum = self.ae_var['Var_ae'].iloc[0]
        else:
            cumsum = np.sum(self.probability[:self.index])
        return cumsum

class VaR:

    """
    VaR computations
    """

    def __init__(self, alpha_var, **kwargs):
        """
        
        Initialize the class for Var Computation
        
        Parameters
        ----------

        alpha_var : float
            Value between 0 and 1.0 used for VaR computations

        kwargs : dictionary
            python dictionary for configurating the complete VaR problem
        """


        self.n_qbits = kwargs.get("n_qbits", None)
        self.kwargs = kwargs
        self.alpha_var = alpha_var
        self.pdf_var = None
        self.cumsum_classes = []
        self.list_cumsum = []
        #start index
        self.var_index = None
        self.var = None
        self.cumsum_var = None

    def run(self):
        """
        Method for execute a VaR computation

        """

        index = 0
        sign = 1
        pdf_list = []
        for i in range(1, self.n_qbits + 1):
            index = index + sign * 2 ** (self.n_qbits - i)

            cumsum_index = CumulativeSumVaR(index, **self.kwargs)
            cumulative_sum = cumsum_index.run()
            self.cumsum_classes.append(cumsum_index)
            self.list_cumsum.append(cumulative_sum)

            if cumulative_sum <= (1.0 - self.alpha_var):
                sign = 1
            else:
                sign = -1

        if sign == 1:
            index = index + sign
        print("Result is: ", index)
        # Index where the VaR is located
        self.var_index = index
        # VaR
        self.var = self.cumsum_classes[-1].domain[self.var_index]
        # Final Cumulative Sum for VaR
        self.cumsum_var = np.sum(
            self.cumsum_classes[-1].probability[:self.var_index]
        )

