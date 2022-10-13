"""
This module contains several auxiliary functions used in quantitative
classical finances

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

import numpy as np
from scipy.stats import norm


def call_payoff(s_t: float, strike: float, **kwargs):
    r"""Computes the payoff of a European call option.

    Notes
    -----

    .. math::
        C(S_T, K) = \left(S_T-K, 0\right)^+

    Parameters
    ----------
    s_t : float
        price
    strike : float
        the strike

    Returns
    -------
    payoff : float
        the payoff
    """
    return np.maximum(s_t - strike, 0)


def put_payoff(s_t: float, strike: float, **kwargs):
    r"""Computes the payoff of a European put option.

    Notes
    -----

    .. math::
        P(S_T, K) = \left(K-S_T, 0\right)^+

    Parameters
    ----------
    s_t : float
        price
    strike : float
        the strike

    Returns
    -------
    payoff : float
        the payoff
    """
    return np.maximum(strike - s_t, 0)


def futures_payoff(s_t: float, strike: float, **kwargs):
    r"""Computes the payoff of a futures contract.

    Notes
    -----

    .. math::
        F(S_T, K) = \left(S_T-K, 0\right)

    Parameters
    ----------
    s_t : float
        price
    strike : float
        the strike

    Returns
    -------
    payoff : float
        the payoff
    """
    return s_t - strike


def digital_call_payoff(s_t: float, strike: float, coupon: float = 1.0, **kwargs):
    r"""Computes the payoff for a digital(binary) call option.
    The formula is:

    Notes
    -----

    .. math::
        DC(S_T, K, Coupon) = Coupon\mathbb{1}_{\{S_T-K\}}

    Parameters
    ----------
    s_t : float
        current price of the underlying
    strike : float
        the strike
    coupon : float
        the amount received in case
        that the underlying pring exceeds
        the strike

    Returns
    -------
    payoff : float
        payoff of the european digital call option
    """
    return np.where(s_t > strike, coupon, 0.0)


def digital_put_payoff(s_t: float, strike: float, coupon: float = 1.0, **kwargs):
    r"""
    Computes the payoff for a digital(binary) put option.

    Notes
    -----
    The formula is:

    .. math::
        DC(S_T, K, Coupon) = Coupon\mathbb{1}_{\{K-S_T\}}

    Parameters
    ----------
    s_t : float
        current price of the underlying
    strike : float
        the strike
    coupon : float
        the amount received in case
        that the underlying pring exceeds
        the strike

    Returns
    -------
    payoff : float
        payoff of the european digital call option
    """
    return np.where(s_t < strike, coupon, 0.0)


def bs_density(
    s_t: float,
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    **kwargs
):
    r"""
    Evaluates the Black-Scholes density function at s_t for a given set
    of parameters. The formula is:

    Notes
    -----

    .. math::
        \dfrac{1}{S_T\sigma\sqrt{2\pi T}}\exp\left(-\dfrac{ \
        \left(\log(S_T)-\mu\right)}{2\sigma^2T}\right)

    where :math:`\mu = (risk\_free\_rate-0.5\sigma)T+\log(S_0)`.

    Parameters
    ----------
    s_t : float
        point where we do the evaluation
    s_0 : float
        current price
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity: float
        the maturity

    Returns
    -------
    density : float
        value of the Black-Scholes density function in s_t
    """
    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    factor = s_t * volatility * np.sqrt(2 * np.pi * maturity)
    exponent = -((np.log(s_t) - mean) ** 2) / (2 * volatility * volatility * maturity)
    density = np.exp(exponent) / factor
    return density


def bs_probability(
    s_t: np.array,
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    **kwargs
):
    r"""
    Computes a discrete probability distribution from the  Black-Scholes
    density function for a given set of parameters. This is done by evaluating
    the Black-Scholes density function in s_t and the normlising this result.

    Parameters
    ----------
    s_t : numpy array
        points where we define the discrete probability distribution
    s_0 : float
        current price
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity: float
        the maturity

    Returns
    -------
    distribution : numpy array
        discrete probability distribution from Black-Scholes density
    """
    density = bs_density(s_t, s_0, risk_free_rate, volatility, maturity)
    return density / np.sum(density)


def bs_sde_solution(
    x_: np.array,
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    **kwargs
):
    r"""
    For a certain parametrization :math:`x` it returns a value of the
    underlying :math:`S_T(x)` and the probability density of that
    value of the underlying.

    Notes
    -----
    The formulas are:

    .. math::
        S_T = S_0e^{\sigma x+(risk_free_rate-\sigma^2/2)t}
        p(S_T(x)) = N(x;mean = 0, variance = T)

    Parameters
    ----------
    x_ : numpy array
        parametrization
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity

    Returns
    -------
    s_t : numpy array
        value of the underlying corresponding to parameter x
    probability_density : numpy array
        probability density corresponding to s_t
    """
    probability = norm.pdf(x_) * np.sqrt(maturity)
    probability = probability / np.sum(probability)
    s_t = s_0 * np.exp(
        volatility * x_ + (risk_free_rate - volatility * volatility / 2) * maturity
    )
    return s_t, probability


def bs_call_price(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    strike: float,
    **kwargs
):
    r"""
    Computes the price for a European call option. 

    Notes
    -----

    The formula is:

    .. math::
        C(S, T) = S\Phi(d_1)-Ke^{-rT}\Phi(d_2)

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike

    Returns
    -------
    price : float
        price of the European call option
    """
    first = np.log(s_0 / strike)
    positive = (risk_free_rate + volatility * volatility / 2) * maturity
    negative = (risk_free_rate - volatility * volatility / 2) * maturity
    d_1 = (first + positive) / (volatility * np.sqrt(maturity))
    d_2 = (first + negative) / (volatility * np.sqrt(maturity))
    price = s_0 * norm.cdf(d_1) - strike * np.exp(
        -risk_free_rate * maturity
    ) * norm.cdf(d_2)
    return price


def bs_put_price(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    strike: float,
    **kwargs
):
    r"""
    Computes the price for a European put option.

    Notes
    -----
    The formula is:

    .. math::
        C(S, T) = Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1)

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike

    Returns
    -------
    price : float
        price of the European put option
    """
    first = np.log(s_0 / strike)
    positive = (risk_free_rate + volatility * volatility / 2) * maturity
    negative = (risk_free_rate - volatility * volatility / 2) * maturity
    d_1 = (first + positive) / (volatility * np.sqrt(maturity))
    d_2 = (first + negative) / (volatility * np.sqrt(maturity))
    price = strike * np.exp(-risk_free_rate * maturity) * norm.cdf(
        -d_2
    ) - s_0 * norm.cdf(-d_1)
    return price


def bs_digital_call_price(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    strike: float,
    coupon: float,
    **kwargs
):
    r"""
    Computes the price for a digital(binary) call option.

    Notes
    -----
    The formula is:

    .. math::
        DC(S, T) = e^{-rT}Coupon N(d_2)

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike
    coupon : float
        the amount received in case that the underlying pring exceeds
        the strike

    Returns
    -------
    price : float
        price of the digital call option
    """
    first = np.log(s_0 / strike)
    negative = (risk_free_rate - volatility * volatility / 2) * maturity
    d_2 = (first + negative) / (volatility * np.sqrt(maturity))
    price = coupon * np.exp(-risk_free_rate * maturity) * norm.cdf(d_2)
    return price


def bs_digital_put_price(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    strike: float,
    coupon: float,
    **kwargs
):
    r"""
    Computes the price for a digital (binary) put option.

    Notes
    -----
    The formula is:

    .. math::
        DC(S, T) = e^{-rT}Coupon N(-d_2)

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike
    coupon : float
        the amount received in case that the underlying pring exceeds
        the strike

    Returns
    -------
    price : float
        price of the digital call option
    """
    first = np.log(s_0 / strike)
    negative = (risk_free_rate - volatility * volatility / 2) * maturity
    d_2 = (first + negative) / (volatility * np.sqrt(maturity))
    price = coupon * np.exp(-risk_free_rate * maturity) * norm.cdf(-d_2)
    return price


def bs_exact_samples(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    number_samples: int,
    **kwargs
):
    r"""
    Computes samples from the exact solution of the Black-Scholes SDE.

    Notes
    -----
    The formula is:

    .. math::
        S_T = S_0e^{\sigma*W_t+(risk_free_rate-\sigma^2/2)t}

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike
    number_samples : int
        number of samples

    Returns
    -------
    s_t : numpy array of floats
        array of samples from the SDE.
    """
    dw_t = np.random.randn(number_samples) * np.sqrt(maturity)
    s_t = s_0 * np.exp(
        volatility * dw_t + (risk_free_rate - volatility * volatility / 2) * maturity
    )
    return s_t


def bs_em_samples(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    number_samples: int,
    time_steps: int,
    **kwargs
):
    r"""
    Computes samples from the approximated solution of the Black-Scholes
    SDE using the Euler-Maruyama discrimination.

    Notes
    -----
    The formula is:

    .. math::
        S_{t+\Delta t} = S_t+rS_tdt+\sigma S_t N(0, 1)\sqrt{dt}

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike
    number_samples : int
        number of samples
    time steps : int
        number of time steps

    Returns
    -------
    s_t : numpy array of floats
        array of samples from the SDE.
    """
    dt = maturity / time_steps
    s_t = np.ones(number_samples) * s_0
    for i in range(time_steps):
        dw_t = np.random.randn(number_samples) * np.sqrt(dt)
        s_t = s_t + risk_free_rate * s_t * dt + volatility * s_t * dw_t
    return s_t


def bs_tree(
    s_0: float,
    risk_free_rate: float,
    volatility: float,
    maturity: float,
    number_samples: int,
    time_steps: int,
    discretization: int,
    bounds: float,
    **kwargs
):
    r"""
    Computes the probabilities of all possible pahts from the
    approximated solution of the Black-Scholes SDE using the
    Euler-Maruyama discretization for a given discretization of the
    Brownian motion.

    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    volatility : float
        the volatility
    maturity : float
        the maturity
    strike : float
        the strike
    number_samples : int
        number of samples
    time steps : int
        number of time steps
    discretization : float
        number of points to build the discrete version
        of the gaussian density
    bounds : float
        bounds of the Gaussian density

    Returns
    -------
    s_t : numpy array of floats
        array of samples from the SDE.
    """
    dt = maturity / time_steps
    x_ = np.linspace(-bounds, bounds, discretization)
    p_x = norm.pdf(x_)
    p_x = p_x / np.sum(p_x)

    s_t = []
    p_t = []
    s_t.append(np.array([s_0]))
    p_t.append(np.array([1.0]))
    for i in range(time_steps):
        all_possible_paths = np.array(np.zeros(discretization ** (i + 1)))
        all_possible_probabilities = np.array(np.zeros(discretization ** (i + 1)))
        for j in range(len(s_t[i])):
            single_possible_paths = (
                s_t[i][j]
                + risk_free_rate * s_t[i][j] * dt
                + volatility * s_t[i][j] * x_ * np.sqrt(dt)
            )
            single_possible_probabilities = p_t[i][j] * p_x

            index = j * discretization
            all_possible_paths[index : index + discretization] = single_possible_paths
            all_possible_probabilities[
                index : index + discretization
            ] = single_possible_probabilities

        s_t.append(all_possible_paths)
        p_t.append(all_possible_probabilities)
    return s_t, p_t


def geometric_sum(base: float, exponent: int, coeficient: float = 1.0, **kwargs):
    r"""
    Computes a geometric sum

    Notes
    -----

    .. math::
        s = a+a*base+a*base^2+a*base^3+...+a*base^{exponent} = \\
        = \sum_{k=0}^{exponent}a*base^k = \
        a\big(\frac{1-base^{exponent+1}}{1-base}\big)

    Parameters
    ----------

    base : float
        base of the geometric sum
    exponent: int
        maximum exponent for the geometric sum
    coeficient : float
        coefficient for the geometric sum

    Returns
    _______
    return : float
        result of the geometric sum

    """
    return coeficient * (base ** (exponent + 1) - 1) / (base - 1)


def bs_forward_price(
    s_0: float, risk_free_rate: float, maturity: float, strike: float, **kwargs
):
    r"""
    Computes the price for a forward contract.  Note that it doesn't
    assume that we are at the begining of the lifetime of the contract

    Notes
    -----
    The formula is:

    .. math::
        V(S, T) = S-Ke^{-r(T-t)}
    Parameters
    ----------
    s_0 : float
        current price of the underlying
    risk_free_rate : float
        risk free rate
    maturity : float
        time to expiration of the contract
    strike : float
        the strike
    Returns
    -------
    price : float
        price of the forward
    """
    price = s_0 - strike * np.exp(-risk_free_rate * maturity)
    return price
