"""
Function for automation of different option price estimation using
Amplitude Estimation algorithms.

The function uses the DensityProbability and the PayOff classes for
defining the option price estimation problem. Then q_solve_integral
function is used for computing the expected value integral. The function
deals with all the mandatory normalisations for returned the desired price
estimation.

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

import sys
import numpy as np
import pandas as pd
from QQuantLib.finance.probability_class import DensityProbability
from QQuantLib.finance.payoff_class import PayOff
from QQuantLib.finance.quantum_integration import q_solve_integral


def ae_price_estimation(**kwargs):
    """
    Configures an option price estimation problem and solving it using
    AE integration techniques

    Parameters
    ----------

    kwargs : dictionary.
        Dictionary for configuring the price estimation problem, the
        encoding of the price estimation data into the quantum circuit
        and the AE integration technique for solving it.

    Note
    ____

    The keys for the input kwargs dictionary will be the necessary keys
    for configuring the DensityProbability class \\
    (see QQuantLib.finance.probability_class), the PayOff class \\
    (see QQuantLib.finance.payoff_class) and the q_solve_integral \\
    function (see QQuantLib.finance.quantum_integration).

    Returns
    _______

    pdf : Pandas DataFrame
        DataFrame with the configuration of the AE problem and the solution
    """

    ae_problem = kwargs
    #Building the domain
    n_qbits = ae_problem.get("n_qbits", None)
    x0 = ae_problem.get("x0", 1.0)
    xf = ae_problem.get("xf", 3.0)
    domain = np.linspace(x0, xf, 2**n_qbits)

    #Building the Probability distribution
    pc = DensityProbability(**ae_problem)
    p_x = pc.probability(domain)
    #Normalisation of the probability distribution
    p_x_normalisation = np.sum(p_x) + 1e-8
    norm_p_x = p_x / p_x_normalisation

    #Building the option payoff
    po = PayOff(**ae_problem)
    pay_off = po.pay_off(domain)
    #Normalisation of the pay off
    pay_off_normalisation = np.max(np.abs(pay_off)) + 1e-8
    norm_pay_off = pay_off / pay_off_normalisation

    #Getting the exact price of the option under BS
    exact_solution = None
    if po.pay_off_bs is not None:
        exact_solution = po.pay_off_bs(**ae_problem)

    lista = []
    #For doing several repetitions
    for i in range(ae_problem["number_of_tests"]):
        #Each loop step solves a complete price estimation problem

        #Now we update the input dictionary with the probability and the
        #function arrays
        ae_problem.update({
            "array_function" : norm_pay_off,
            "array_probability" : norm_p_x,
        })

        #EXECUTE COMPUTATION
        solution, solver_object = q_solve_integral(**ae_problem)

        #For generating the output DataFrame we delete the arrays
        del ae_problem["array_function"]
        del ae_problem["array_probability"]

        #Undoing the normalisations
        ae_expectation = solution * pay_off_normalisation * p_x_normalisation

        #Creating the output DataFrame with the complete information

        #The basis will be the input python dictionary for traceability
        pdf = pd.DataFrame([ae_problem])
        #Added normalisation constants
        pdf["payoff_normalisation"] = pay_off_normalisation
        pdf["p_x_normalisation"] = p_x_normalisation

        #Expectation calculation using Riemann sum
        pdf["riemann_expectation"] = np.sum(p_x * pay_off)
        #Expectation calculation using AE integration techniques
        pdf[
            [col + "_expectation" for col in ae_expectation.columns]
        ] = ae_expectation

        #Option price estimation using expectation computed as Riemann sum
        pdf["riemann_price_estimation"] = pdf["riemann_expectation"] * np.exp(
            -pdf["risk_free_rate"] * pdf["maturity"]
        )
        #Exact option price under the Black-Scholes model
        pdf["exact_price"] = exact_solution
        #Option price estimation using expectation computed by AE integration
        pdf[[col + "_price_estimation" for col in ae_expectation.columns]] = (
            ae_expectation
            * np.exp(-pdf["risk_free_rate"] * pdf["maturity"]).iloc[0]
        )
        #Computing Absolute: Rieman vs AE techniques
        pdf["error_riemann"] = np.abs(
            pdf["ae_price_estimation"] - pdf["riemann_price_estimation"]
        )
        #Computing Relative: Rieman vs AE techniques
        pdf["relative_error_riemann"] = (
            pdf["error_riemann"] / pdf["riemann_price_estimation"]
        )
        #Computing Absolute error: Exact BS price vs AE techniques
        pdf["error_exact"] = np.abs(
            pdf["ae_price_estimation"] - pdf["exact_price"])
        #Computing Relative error: Exact BS price vs AE techniques
        pdf["relative_error_exact"] = pdf["error_exact"] / pdf["exact_price"]
        #Other interesting staff
        if solver_object is None:
            #Computation Fails Encoding 0 and RQAE
            pdf["schedule_pdf"] = [None]
            pdf["oracle_calls"] = [None]
            pdf["max_oracle_depth"] = [None]
            pdf["circuit_stasts"] = [None]
            pdf["run_time"] = [None]
        else:
            if solver_object.schedule_pdf is None:
                pdf["schedule_pdf"] = [None]
            else:
                pdf["schedule_pdf"] = [solver_object.schedule_pdf.to_dict()]
            pdf["oracle_calls"] = solver_object.oracle_calls
            pdf["max_oracle_depth"] = solver_object.max_oracle_depth
            pdf["circuit_stasts"] = [solver_object.solver_ae.circuit_statistics]
            pdf["run_time"] = solver_object.solver_ae.run_time

        #Saving pdf
        if ae_problem["save"]:
            with open(ae_problem["file_name"], "a") as f_pointer:
                pdf.to_csv(f_pointer, mode="a", header=f_pointer.tell() == 0, sep = ";")
        lista.append(pdf)
    complete_pdf = pd.concat(lista)
    return complete_pdf
