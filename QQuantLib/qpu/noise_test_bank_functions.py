import sys
import numpy as np
sys.path.append("../../")
from QQuantLib.DL.encoding_protocols import Encoding
from QQuantLib.finance.probability_class import DensityProbability
from QQuantLib.finance.payoff_class import PayOff
#from QQuantLib.AE.real_quantum_ae import RQAE



def create_arrays(price_problem):
    n_qbits = price_problem.get("n_qbits", None)
    x0 = price_problem.get("x0", 1.0)
    xf = price_problem.get("xf", 3.0)
    domain = np.linspace(x0, xf, 2**n_qbits)
    #Building the Probability distribution
    pc = DensityProbability(**price_problem)
    p_x = pc.probability(domain)
    #Normalisation of the probability distribution
    p_x_normalisation = np.sum(p_x) + 1e-8
    norm_p_x = p_x / p_x_normalisation
    #Building the option payoff
    po = PayOff(**price_problem)
    pay_off = po.pay_off(domain)
    #Normalisation of the pay off
    pay_off_normalisation = np.max(np.abs(pay_off)) + 1e-8
    norm_pay_off = pay_off / pay_off_normalisation
    return domain, norm_pay_off, norm_p_x, pay_off_normalisation, p_x_normalisation


def first_step(epsilon, ratio, gamma):
    epsilon = 0.5 * epsilon
    theoretical_epsilon = 0.5 * np.sin(np.pi / (2 * (ratio + 2))) ** 2
    k_max = int(
        np.ceil(
            np.arcsin(np.sqrt(2 * theoretical_epsilon))
            / np.arcsin(2 * epsilon)
            * 0.5
            - 0.5
        )
    )
    bigk_max = 2 * k_max + 1
    big_t = np.log(
        ratio
        * ratio
        * (np.arcsin(np.sqrt(2 * theoretical_epsilon)))
        / (np.arcsin(2 * epsilon))
    ) / np.log(ratio)
    gamma_i = gamma / big_t
    n_i = int(
        np.ceil(1 / (2 * theoretical_epsilon**2) * np.log(2 * big_t /  gamma))
    )
    epsilon_probability = np.sqrt(1 / (2 * n_i) * np.log(2 / gamma_i))
    shift = theoretical_epsilon / np.sin(np.pi / (2 * (ratio + 2)))
    return shift, n_i, gamma_i, theoretical_epsilon
# 
# 
# def create_rqae_price(price_problem, ae_problem):
# 
#     domain, norm_pay_off, norm_p_x, pay_off_normalisation, p_x_normalisation = create_arrays(price_problem)
# 
#     encode_class_rqae = Encoding(
#         array_function=norm_pay_off,
#         array_probability=norm_p_x,
#         encoding=2,
#         **ae_problem
#     )
#     encode_class_rqae.run()
#     rqae_object = RQAE(
#         oracle=encode_class_rqae.oracle,
#         target=encode_class_rqae.target,
#         index=encode_class_rqae.index,
#         **ae_problem
#     )
# 
#     epsilon = 0.5 * rqae_object.epsilon
#     theoretical_epsilon = 0.5 * np.sin(np.pi / (2 * (rqae_object.ratio + 2))) ** 2
#     k_max = int(
#         np.ceil(
#             np.arcsin(np.sqrt(2 * theoretical_epsilon))
#             / np.arcsin(2 * epsilon)
#             * 0.5
#             - 0.5
#         )
#     )
#     bigk_max = 2 * k_max + 1
#     big_t = np.log(
#         rqae_object.ratio
#         * rqae_object.ratio
#         * (np.arcsin(np.sqrt(2 * theoretical_epsilon)))
#         / (np.arcsin(2 * epsilon))
#     ) / np.log(rqae_object.ratio)
#     gamma_i = rqae_object.gamma / big_t
#     n_i = int(
#         np.ceil(1 / (2 * theoretical_epsilon**2) * np.log(2 * big_t /  rqae_object.gamma))
#     )
#     epsilon_probability = np.sqrt(1 / (2 * n_i) * np.log(2 / gamma_i))
#     shift = theoretical_epsilon / np.sin(np.pi / (2 * (rqae_object.ratio + 2)))
#     rqae_object.shifted_oracle = shift
#     routine = rqae_object._shifted_oracle
#     return routine, rqae_object
