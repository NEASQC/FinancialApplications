"""
This module implements the *ae_clique_return_estimation* function
that allows to the user configure a cliquet option, encode the expected
value integral to compute in a quantum state and estimate it using the
different **AE** algorithms implemented in the **QQuantLib.AE** package.


The function deals with all the mandatory normalisations for returning
the desired price estimation.

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""
import numpy as np
import pandas as pd
from QQuantLib.finance.classical_finance import bs_tree
from QQuantLib.finance.classical_finance import tree_to_paths
from QQuantLib.finance.classical_finance import cliquet_cashflows
from QQuantLib.utils.utils import text_is_none
from QQuantLib.finance.quantum_integration import q_solve_integral

def ae_cliquet_estimation_step_po(**kwargs):
    """
    Configures a cliquet option return estimation problem and solving it
    using AE integration techniques

    Parameters
    ----------

    kwargs : dictionary
        Dictionary for configuring the price estimation problem, the
        encoding of the price estimation data into the quantum circuit
        and the AE integration technique for solving it.
    n_qbits: kwargs, int
        Number of qubits for domain discretization
    s_0 : kwargs float
        Value of the asset at initial time step
    risk_free_rate : kwargs, float
        Risk free rate for discounting the expected value of the payoff
    volatility : kwargs, float
        Volatility of the asset
    reset_dates : kwargs, list
        List with the reset dates to asset evaluation
    bounds : kwargs, float
        Bound for truncating the probability density
    local_cap : kwargs, float
        For upper truncation of the return at each reset date
    local_floor : kwargs, float
        For lower truncation of the return at each reset date
    global_cap : kwargs, float
        For upper truncation of the final return
    global_floor : kwargs, float
        For lower truncation of the final return

    Note
    ----

    Other kwargs input dictionary keys will be related with the encoding \\
    of the integral into the quantum circuit \\
    (see QQuantLib.DL.encoding_protocols) and for the configuration \\
    of the AE algorithm used (see QQuantLib.AE.ae_class)

    Returns
    _______

    pdf : Pandas DataFrame
        DataFrame with the configuration of the AE problem and the solution
    """

    ae_problem = kwargs

    n_qbits = ae_problem.get("n_qbits", None)
    s_0 = ae_problem.get("s_0", None)
    text_is_none(s_0, "s_0", variable_type=float)
    risk_free_rate = ae_problem.get("risk_free_rate", None)
    text_is_none(risk_free_rate, "risk_free_rate", variable_type=float)
    volatility = ae_problem.get("volatility", None)
    text_is_none(volatility, "volatility", variable_type=float)
    reset_dates = ae_problem.get("reset_dates", None)
    text_is_none(reset_dates, "reset_dates", variable_type=list)
    reset_dates = np.array(reset_dates)
    bounds = ae_problem.get("bounds", None)
    text_is_none(bounds, "bounds", variable_type=float)

    # Built paths and probabilities
    tree_s, bs_path_prob = bs_tree(
        s_0=s_0,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        times=reset_dates,
        discretization=2**n_qbits,
        bounds=bounds
    )
    #probability definition
    p_x = bs_path_prob[-1]
    #probability normalisation
    p_x_normalisation = np.sum(p_x)
    norm_p_x = p_x / p_x_normalisation

    # Build Cliquet PayOffs for paths
    local_cap = ae_problem.get("local_cap", None)
    text_is_none(local_cap, "local_cap", variable_type=float)
    local_floor = ae_problem.get("local_floor", None)
    text_is_none(local_floor, "local_floor", variable_type=float)
    global_cap = ae_problem.get("global_cap", None)
    text_is_none(global_cap, "global_cap", variable_type=float)
    global_floor = ae_problem.get("global_floor", None)
    text_is_none(global_floor, "global_floor", variable_type=float)

    # Table format paths
    paths_s = tree_to_paths(tree_s)
    # Build payoff for each possible path
    cliqet_payoffs = cliquet_cashflows(
        local_cap=local_cap,
        local_floor=local_floor,
        global_cap=global_cap,
        global_floor=global_floor,
        paths=paths_s
    )
    #Function definition
    f_x = cliqet_payoffs
    #Function normalisation
    f_x_normalisation = np.max(np.abs(f_x))
    norm_f_x = f_x / f_x_normalisation

    #### Positive Pay Off part execution ####
    npo_positive = np.where(norm_f_x < 0, 0.0, norm_f_x)
    ae_problem.update({
        "array_function" : npo_positive,
        "array_probability" : norm_p_x,
    })
    solution_p, solver_object_p = q_solve_integral(**ae_problem)
    
    #### Negative Pay Off part execution ####
    npo_neagtive = np.abs(np.where(norm_f_x >= 0, 0.0, norm_f_x))
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
    ae_expectation = solution * f_x_normalisation * p_x_normalisation
    # Compute the associated error
    measured_epsilon = epsilon_final * p_x_normalisation * p_x_normalisation
    ###### Creation of the output ##########
    #The basis will be the input python dictionary for traceability
    pdf = pd.DataFrame([ae_problem])
    pdf.drop(["array_function", "array_probability"], axis=1, inplace=True)
    #Added normalisation constants
    pdf["payoff_normalisation"] = f_x_normalisation
    pdf["p_x_normalisation"] = p_x_normalisation
    #Expectation calculation using Riemann sum
    pdf["riemann_expectation"] = np.sum(p_x * f_x)
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
    pdf["finance_exact_price"] = None
    #Option price estimation using expectation computed as Riemann sum
    pdf["finance_riemann_price"] = pdf["riemann_expectation"] * np.exp(
        -pdf["risk_free_rate"] * reset_dates[-1]
    )
    #Option price estimation using expectation computed by AE integration
    pdf["finance_price_estimation"] = pdf["ae_expectation"] * \
        np.exp(-pdf["risk_free_rate"] * reset_dates[-1]).iloc[0]
    # Associated error of the price estimation
    pdf["finance_price_epsilon"] = pdf["measured_epsilon"] * \
        np.exp(-pdf["risk_free_rate"] * reset_dates[-1]).iloc[0]
    #Computing Absolute with discount: Rieman vs AE techniques
    pdf["finance_error_riemann"] = np.abs(
        pdf["finance_price_estimation"] - pdf["finance_riemann_price"]
    )
    #Computing Absolute error: Exact BS price vs AE techniques
    pdf["finance_error_exact"] = None

    # We have two objects. It is not interesting have the schedules
    pdf["schedule_pdf"] = [None]
    pdf["oracle_calls"] = solver_object_p.oracle_calls + solver_object_n.oracle_calls
    pdf["max_oracle_depth"] = max(
        solver_object_p.max_oracle_depth, solver_object_n.max_oracle_depth)
    pdf["circuit_stasts"] = [None]
    pdf["run_time"] = solver_object_p.solver_ae.run_time + solver_object_n.solver_ae.run_time
    return pdf

