"""
This module implements the *ae_price_estimation_step_po* function that
allows to the user configure a price estimation problem using financial
parameters, encode the expected value integral to compute in a quantum
state and estimate it using the different **AE** algorithms implemented
in the **QQuantLib.AE** package.

This function uses the DensityProbability and the PayOff classes (from
*finance.probability_class* and *finance.payoff_class* modules
respectively) for defining the option price estimation problem.
Then the q_solve_integral function (from *finance.quantum_integration*
module) is used for computing the expected value integral.

The *ae_price_estimation_step_po* functions load and estimate the
amplitude for the positive and negative parts of the payoff separately
and process the results to get the desired price estimation.


Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

import sys
import numpy as np
import pandas as pd
from QQuantLib.finance.probability_class import DensityProbability
from QQuantLib.finance.payoff_class import PayOff
from QQuantLib.finance.quantum_integration import q_solve_integral

def ae_price_estimation_step_po(**kwargs):
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

    #### Positive Pay Off part execution ####
    npo_positive = np.where(norm_pay_off < 0, 0.0, norm_pay_off)
    ae_problem.update({
        "array_function" : npo_positive,
        "array_probability" : norm_p_x,
    })
    solution_p, solver_object_p = q_solve_integral(**ae_problem)

    #### Negative Pay Off part execution ####
    npo_neagtive = np.abs(np.where(norm_pay_off >= 0, 0.0, norm_pay_off))
    ae_problem.update({
        "array_function" : npo_neagtive,
        "array_probability" : norm_p_x,
    })
    solution_n, solver_object_n = q_solve_integral(**ae_problem)

    ###### Combine Solutions ##########

    # First compute errors of both contributions
    epsilon_p = (solution_p["ae_u"] - solution_p["ae_l"]) / 2.0
    epsilon_n = (solution_n["ae_u"] - solution_n["ae_l"]) / 2.0
    #epsilon_final = np.sqrt(epsilon_p ** 2 + epsilon_n ** 2)
    epsilon_final = epsilon_p + epsilon_n
    # Second compute the expected value
    solution = solution_p["ae"] - solution_n["ae"]
    # Compute the expected value to compute
    ae_expectation = solution * pay_off_normalisation * p_x_normalisation
    # Compute the associated error
    measured_epsilon = epsilon_final * pay_off_normalisation * p_x_normalisation
    ###### Creation of the output ##########
    #The basis will be the input python dictionary for traceability
    pdf = pd.DataFrame([ae_problem])
    pdf.drop(["array_function", "array_probability"], axis=1, inplace=True)
    #Added normalisation constants
    pdf["payoff_normalisation"] = pay_off_normalisation
    pdf["p_x_normalisation"] = p_x_normalisation
    #Expectation calculation using Riemann sum
    pdf["riemann_expectation"] = np.sum(p_x * pay_off)
    # Positive Part estimation
    pdf[[col + "_positive_part" for col in solution_p.columns]] = solution_p
    # Negative part estimation
    pdf[[col + "_negative_part" for col in solution_p.columns]] = solution_n
    #Expectation calculation using AE integration techniques
    pdf["ae_expectation"] = ae_expectation
    # Pure integration Absolute Error
    pdf["absolute_error"] = np.abs(
        pdf["ae_expectation"] - pdf["riemann_expectation"])
    pdf["measured_epsilon"] = measured_epsilon
    # Finance Info
    #Exact option price under the Black-Scholes model
    pdf["finance_exact_price"] = exact_solution
    #Option price estimation using expectation computed as Riemann sum
    pdf["finance_riemann_price"] = pdf["riemann_expectation"] * np.exp(
        -pdf["risk_free_rate"] * pdf["maturity"]
    )
    #Option price estimation using expectation computed by AE integration
    pdf["finance_price_estimation"] = pdf["ae_expectation"] * \
        np.exp(-pdf["risk_free_rate"] * pdf["maturity"]).iloc[0]
    # Associated error of the price estimation
    pdf["finance_price_epsilon"] = pdf["measured_epsilon"] * \
        np.exp(-pdf["risk_free_rate"] * pdf["maturity"]).iloc[0]
    #Computing Absolute with discount: Rieman vs AE techniques
    pdf["finance_error_riemann"] = np.abs(
        pdf["finance_price_estimation"] - pdf["finance_riemann_price"]
    )
    #Computing Absolute error: Exact BS price vs AE techniques
    pdf["finance_error_exact"] = np.abs(
        pdf["finance_price_estimation"] - pdf["finance_exact_price"])

    # We have two objects. It is not interesting have the schedules
    pdf["schedule_pdf"] = [None]
    pdf["oracle_calls"] = solver_object_p.oracle_calls + solver_object_n.oracle_calls
    pdf["max_oracle_depth"] = max(
        solver_object_p.max_oracle_depth, solver_object_n.max_oracle_depth)
    pdf["circuit_stasts"] = [None]
    pdf["run_time"] = solver_object_p.solver_ae.run_time + solver_object_n.solver_ae.run_time
    return pdf
