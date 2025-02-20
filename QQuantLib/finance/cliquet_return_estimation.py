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

def ae_cliquet_estimation(**kwargs):
    """
    Configures a cliquet option return estimation problem and solving it
    using AE integration techniques

    Parameters
    ----------

    n_qbits : kwargs, int
        Number of qubits for domain discretization
    s_0 : kwargs, float
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

    #Now we update the input dictionary with the probability and the
    #function arrays
    ae_problem.update({
        "array_function" : norm_f_x,
        "array_probability" : norm_p_x,
    })
    #EXECUTE COMPUTATION
    solution, solver_object = q_solve_integral(**ae_problem)

    #For generating the output DataFrame we delete the arrays
    del ae_problem["array_function"]
    del ae_problem["array_probability"]

    #Undoing the normalisations
    ae_expectation = solution * p_x_normalisation * f_x_normalisation

    #Creating the output DataFrame with the complete information

    #The basis will be the input python dictionary for traceability
    pdf = pd.DataFrame([ae_problem])
    #Added normalisation constants
    pdf["payoff_normalisation"] = f_x_normalisation
    pdf["p_x_normalisation"] = p_x_normalisation

    #Expectation calculation using Riemann sum
    pdf["riemann_expectation"] = np.sum(p_x * f_x)
    #Expectation calculation using AE integration techniques
    pdf[
        [col + "_expectation" for col in ae_expectation.columns]
    ] = ae_expectation
    # Pure integration Absolute Error
    pdf["absolute_error"] = np.abs(
        pdf["ae_expectation"] - pdf["riemann_expectation"])
    pdf["measured_epsilon"] = np.abs(
        pdf["ae_u_expectation"] - pdf["ae_l_expectation"]) / 2.0
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
    #Computing Absolute with discount: Rieman vs AE techniques
    pdf["finance_error_riemann"] = np.abs(
        pdf["finance_price_estimation"] - pdf["finance_riemann_price"]
    )

    #Computing Absolute error: Exact BS price vs AE techniques
    pdf["finance_error_exact"] = None

    #Other interesting staff
    if solver_object is None:
        #Computation Fails Encoding 0 and RQAE
        pdf["schedule_pdf"] = [None]
        pdf["oracle_calls"] = [None]
        pdf["max_oracle_depth"] = [None]
        pdf["run_time"] = [None]
    else:
        if solver_object.schedule_pdf is None:
            pdf["schedule_pdf"] = [None]
        else:
            pdf["schedule_pdf"] = [solver_object.schedule_pdf.to_dict()]
        pdf["oracle_calls"] = solver_object.oracle_calls
        pdf["max_oracle_depth"] = solver_object.max_oracle_depth
        pdf["run_time"] = solver_object.solver_ae.run_time

    return pdf
