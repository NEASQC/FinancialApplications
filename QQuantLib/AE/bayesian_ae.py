"""
This module contains the BAYESQAE class. Given a quantum oracle operator,
this class estimates the **probability** of a given target state using
the Bayesian Quantum Amplitude Estimation algorithm based on the paper:

    Alexandra Ramôa and Luis Paulo Santos
    Bayesian Quantum Amplitude Estimation
    https://arxiv.org/abs/2412.04394 (2024)

Author: Gonzalo Ferro Costas & Alexandra Ramôa

"""

import time
#from copy import deepcopy
import numpy as np
import pandas as pd
import qat.lang.AQASM as qlm
from QQuantLib.qpu.get_qpu import get_qpu
from QQuantLib.AA.amplitude_amplification import grover
from QQuantLib.utils.data_extracting import get_results
from QQuantLib.utils.utils import check_list_type, measure_state_probability
from QQuantLib.AE.mlae_utils import likelihood, log_likelihood

class BAYESQAE:
    """
    Class for Bayesian Quantum Amplitude Estimation algorithm (BAYESQAE)

    Parameters
    ----------
    oracle: QLM gate
        QLM gate with the Oracle for implementing the
        Grover operator
    target : list of ints
        python list with the target for the amplitude estimation
    index : list of ints
        qubits which mark the register to do the amplitude
        estimation

    kwars : dictionary
        dictionary that allows the configuration of the BAYESQAE algorithm: \\
        Implemented keys:

        qpu : QLM solver
            solver for simulating the resulting circuits
        shots : int
            number of measurements on each iteration
        mcz_qlm : bool
            for using or not QLM implementation of the multi controlled Z
            gate
    """

    def __init__(self, oracle: qlm.QRoutine, target: list, index: list, **kwargs):
        """

        Method for initializing the class
        """
        # Setting attributes
        self._oracle = oracle
        self._target = check_list_type(target, int)
        self._index = check_list_type(index, int)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            print("Not QPU was provide. PyLinalg will be used")
            self.linalg_qpu = get_qpu("python")
        # Default setting in BAE paper
        self.mcz_qlm = kwargs.get("mcz_qlm", True)

        # Creating the grover operator
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

        # Loading Algorithm configuration
        # Stop Configuration
        self.epsilon = kwargs.get("epsilon", 1.0e-3)
        self.alpha = kwargs.get("alpha", 0.05)
        self.max_iterations = kwargs.get("max_iterations", 500)
        # Utility function
        self.utility_function = kwargs.get(
            "utility_function", variance_function)
        # Control configuration
        self.n_evals = kwargs.get("n_evals", 50)
        self.k_0 = kwargs.get("k_0", 2)
        self.R = kwargs.get("R", 3)
        self.T = kwargs.get("T", 3)
        # Particle Configuration
        self.particles = kwargs.get("particles", 1000)
        # Threshold for resampling
        self.threshold = kwargs.get("threshold", 0.5)
        # Kernel Selection
        self.kernel = kwargs.get("kernel", "Metro") #Metro, LW
        # Kernel LW configuration
        self.alpha_lw = kwargs.get("alpha_lw", 0.9)
        # Kernel Metropoli configuration
        self.c = kwargs.get("c", 2.38)
        # Frecuency for saving SMC probabilities
        self.save_smc_prob = kwargs.get("save_smc_prob", 10)
        self.print_info = kwargs.get("print_info", 10)
        # Shots
        self.shots = kwargs.get("shots", 1)
        self.warm_shots = kwargs.get("warm_shots", 1)
        self.bayes_shots = kwargs.get("bayes_shots", 1)
        self.control_bayes_shots = kwargs.get("control_bayes_shots", 1)
        # Good theta: For fake simulation
        self.fake = kwargs.get("fake", False)
        self.theta_good = kwargs.get("theta_good", None)



        # Estimation result of the algorithm
        self.ae_l = None
        self.ae_u = None
        self.ae = None
        # Theta estimation of the algorithm
        self.theta_l = None
        self.theta_u = None
        self.theta = None

        self.circuit_statistics = None
        self.time_pdf = None
        self.run_time = None
        self.schedule = {}
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.schedule_pdf = None
        self.quantum_times = []
        self.quantum_time = None

        # Store the SMC distributions
        self.pdf_theta = None
        self.pdf_weights = None

        # Store Evolution of algorithm
        self.outcome_list = None
        self.control_list = None
        self.shots_list = None
        # Store Evolution of estimations
        self.mean_a = None
        self.lower_a = None
        self.upper_a = None
        self.pdf_estimation = None

        # Internal variables for optimize control:
            # Min value for control domain interval
        self.m_min = None
            # Max value for control domain interval
        self.m_max = None
            # Boolean for first control domain interval
        self.first_interval = True
            # Counter for changing control domain interval
        self.interval_change_counter = 0

    #####################################################################
    @property
    def oracle(self):
        """
        creating oracle property
        """
        return self._oracle

    @oracle.setter
    def oracle(self, value):
        """
        setter of the oracle property
        """
        self._oracle = value
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    @property
    def target(self):
        """
        creating target property
        """
        return self._target

    @target.setter
    def target(self, value):
        """
        setter of the target property
        """
        self._target = check_list_type(value, int)
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    @property
    def index(self):
        """
        creating index property
        """
        return self._index

    @index.setter
    def index(self, value):
        """
        setter of the index property
        """
        self._index = check_list_type(value, int)
        self._grover_oracle = grover(
            self._oracle, self.target, self.index, mcz_qlm=self.mcz_qlm
        )

    def quantum_measure_step(self, m_k, n_k):
        r"""
        Create the quantum routine and execute the measurement.

        Parameters
        ----------
        m_k : int
            number of Grover operator applications
        n_k : int
            number of shots

        Returns
        -------
        p_k: float
            probability of getting the Good state
        routine : QLM Routine object
            qlm routine for the QAE step
        """

        routine = qlm.QRoutine()
        wires = routine.new_wires(self.oracle.arity)
        routine.apply(self.oracle, wires)
        for j in range(m_k):
            routine.apply(self._grover_oracle, wires)
        results, circuit, _, _ = get_results(
            routine, linalg_qpu=self.linalg_qpu, shots=n_k, qubits=self.index
        )
        # time_pdf["m_k"] = k
        p_k = measure_state_probability(results, self.target)
        return p_k, routine

    def fake_quantum_measure_step(self, m_k, n_k, theta_good=None):
        """
        Simulates a Fake Quantum Amplitude Estimation experiment

        Parameters
        ----------
        m_k : int
            number of Grover operator applications
        n_k : int
            number of shots

        Returns
        -------

        p_k : float
            probability of getting the Good state
        """
        if theta_good is None:
            raise ValueError("theta_good parameter needed for fake simulation")
        # Exact probability
        p_m = np.sin((2 * m_k + 1) * theta_good) ** 2
        # Binomial experiment using the exact probability
        outcome_k = np.random.binomial(n_k, p_m)
        # The experiment should return a measured probability
        p_k = outcome_k / n_k
        return p_k

    def optimize_control(self, thetas, weights, control_bayes_shots, **kwargs):
        """
        Get the optimal control that minimizes an input utility function
        given an SMC prior probability

        Parameters
        ----------

        thetas : numpy array
            SMC Prior distribution of the desired parameter. Parameter Values
        weights : numpy array
            SMC Prior distribution of the desired parameter. Weights.
        control_shots : int
            Shots for simulating virtual QAE experiments mandatory for
            computing the expected values of the utility function

        Returns
        -------

        m_opt : int
            Optimal control based on the minimization of an utility
            function and a domain interval control
        """
        # Number of evaluations for otpimize the control
        n_evals = kwargs.get("n_evals", self.n_evals)
        # Multiplicative coeeficient for first interval
        k_0 = kwargs.get("k_0", self.k_0)
        # Condition for changing the domain interval control:

        # R is the window where we want as condition for changing the
        # domain interval control. If the optimal control is between
        # the R highest posible domain controls then we want, maybe,
        # change the limits of the domain control
        window_r = kwargs.get("R", self.R)

        # T is the maximum number of times we allow that the optimize
        # control is between the R highest posible domain controls.
        # If we reach this T then we are going to increase the limits
        # of the domain control
        window_t = kwargs.get("T", self.T)

        if self.first_interval:
            # first step
            self.m_min = 0
            self.m_max = k_0 * n_evals
        # We chose n_evals controls randomly between the limits
        test_controls = np.random.randint(self.m_min, self.m_max, n_evals)
        # We compute the expected value of the utiltiy function
        # for each possible control
        values_ = [average_expectation(
            thetas, weights, m_, control_bayes_shots, **kwargs
        ) for m_ in test_controls]
        # Get the optimal control that gives the minimum
        # of the utility function
        m_opt = test_controls[np.argmin(values_)]
        # Test if the optimal control is into the R highest posible
        # domain controls used for computing utility function
        if np.sum(test_controls >= m_opt) <= window_r:
            # Increase the counter for interval change
            self.interval_change_counter = self.interval_change_counter + 1
        # Test if we need to change the domain control interval
        if window_t == self.interval_change_counter:
            if self.first_interval:
                self.first_interval = False
            # New interval limits for searching new optimal control
            self.m_min = self.m_max
            self.m_max = 2 * self.m_min
            # Reset the internal counter for changing interval
            self.interval_change_counter = 0
        return m_opt

    def bayesqae(self, **kwargs):
        """
        This function implements BAE algorithm.

        Parameters
        ----------
        kwargs : dictionary
            Configuration of the BAYESQAE algorithm

        Returns
        ----------
        a_m : float
           mean for the probability to be estimated
        a_l : float
           lower bound for the probability to be estimated
        a_u : float
           upper bound for the probability to be estimated

        """
        # Initialization of attributes related with optimize_control:
        self.first_interval = True
        self.interval_change_counter = 0
        # Initialized quantum times
        self.quantum_times = []

        # Number of particles for SMC
        particles = kwargs.get("particles", self.particles)
        # Number of shots for measuring the quantum device
        shots = kwargs.get("shots", self.shots)
        # Number of Shots for Non amplification step. Default BAE paper
        warm_shots = kwargs.get("warm_shots", self.warm_shots)
        # Shots for Bayesian Updating. Default BAE paper
        bayes_shots = kwargs.get("bayes_shots", self.bayes_shots)
        # Shots for Control optimization. Default BAE paper
        control_bayes_shots = kwargs.get(
            "control_bayes_shots", self.control_bayes_shots)

        # Confidence for the desired epsilon
        alpha = kwargs.get("alpha", self.alpha)
        # desired interval estimation epsilon (arround mean)
        epsilon = kwargs.get("epsilon", self.epsilon)
        # Maximum number of iterations if desired epsilon and alpha
        # confidence is not achieved
        max_iterations = kwargs.get("max_iterations", self.max_iterations)

        # Save SMC probability frecuency
        save_smc_prob = kwargs.get("save_smc_prob", self.save_smc_prob)
        print_info = kwargs.get("print_info", self.print_info)
        # Fake simulation
        fake = kwargs.get("fake", self.fake)
        theta_good = kwargs.get("theta_good", self.theta_good)

        # Configuration keywords for bayesian update and control
        conf = {
            "threshold" : kwargs.get("threshold", self.threshold),
            "kernel" : kwargs.get("kernel", self.kernel),
            "c" : kwargs.get("c", self.c),
            "alpha_lw" : kwargs.get("alpha_lw", self.alpha_lw),
            "n_evals" : kwargs.get("n_evals", self.n_evals),
            "k_0" : kwargs.get("k_0", self.k_0),
            "R" : kwargs.get("R", self.R),
            "T" : kwargs.get("T", self.T),
            "utility_function" : kwargs.get(
                "utility_function", self.utility_function),
        }

        # SMC Prior probability distribution
        theta_prior = np.random.uniform(0.0, 0.5 * np.pi, particles)
        weights_prior = np.ones(len(theta_prior)) / len(theta_prior)

        # Saving Data
        self.pdf_theta = pd.DataFrame(theta_prior, columns=["prior"])
        self.pdf_weights = pd.DataFrame(weights_prior, columns=["prior"])

        # Initialize list for storing experiment results
        self.outcome_list = []
        self.control_list = []
        self.shots_list = []

        # Initialized list for storing estimation results
        self.mean_a = []
        self.lower_a = []
        self.upper_a = []

        ################## WARM UP STEP: START ###################
        warm_control = int(0)
        self.control_list.append(warm_control)
        # Gives a probability

        start = time.time()
        if fake:
            warm_outcome = self.fake_quantum_measure_step(
                warm_control, warm_shots, theta_good)
        else:
            warm_outcome, routine = self.quantum_measure_step(
                warm_control, warm_shots)
        end = time.time()
        self.quantum_times.append(end-start)

        # Transform to counts
        warm_outcome = round(warm_outcome * warm_shots)
        self.outcome_list.append(warm_outcome)
        self.shots_list.append(warm_shots)
        # Updated SMC posterior probability
        assert len(self.control_list) == 1
        assert len(self.shots_list) == 1
        assert len(self.outcome_list) == 1
        theta_posterior, weights_posterior = bayesian_update(
            theta_prior, weights_prior,
            self.control_list,
            self.shots_list,
            self.outcome_list,
            **conf
        )
        self.pdf_theta = pd.DataFrame(
            theta_posterior, columns=["WarmPosterior"])
        self.pdf_weights = pd.DataFrame(
            weights_posterior, columns=["WarmPosterior"])

        # Computes lower and higher values for a
        theta_l, theta_u = confidence_intervals(
            theta_posterior, weights_posterior, alpha
        )
        a_l = np.sin(theta_l) ** 2
        self.lower_a.append(a_l)
        a_u = np.sin(theta_u) ** 2
        self.upper_a.append(a_u)
        # Computes mean value for a
        self.mean_a.append(
            (np.sin(theta_posterior) ** 2) @ weights_posterior
        )
        ################## WARM UP STEP: END ###################

        ################## LOOP STEPS: START ########################

        # Amplification Loop
        counter = 0
        failure_test = True

        # print("{}: m: {}. o:{}, meas_ep: {}. estim: {}".format(
        #     counter,
        #     0,
        #     self.outcome_list[-1],
        #     0.5 * (a_u - a_l),
        #     self.mean_a[-1]
        # ))

        while (a_u - a_l > 2 * epsilon) and (failure_test == True):
            #optimize control for next iteration
            optimal_control = self.optimize_control(
                theta_posterior, weights_posterior,
                control_bayes_shots, **conf
            )
            # Quantum execution using optimal control
            start = time.time()
            if fake:
                p_m = self.fake_quantum_measure_step(
                    optimal_control, shots, theta_good)
            else:
                p_m, routine = self.quantum_measure_step(
                    optimal_control, shots)
            end = time.time()
            self.quantum_times.append(end-start)

            self.control_list.append(optimal_control)
            self.shots_list.append(bayes_shots)
            self.outcome_list.append(round(p_m * bayes_shots))

            # Update SMC Posterior probability
            theta_posterior, weights_posterior = bayesian_update(
                theta_posterior, weights_posterior,
                self.control_list, self.shots_list, self.outcome_list, **conf
            )
            # Computes mean of the parameter a using SMC posterior
            self.mean_a.append(weights_posterior @ np.sin(theta_posterior) ** 2)
            # Computes lower and upper bounds for desired confidence alpha
            theta_l, theta_u = confidence_intervals(
                theta_posterior, weights_posterior, alpha)
            a_l = np.sin(theta_l) ** 2
            self.lower_a.append(a_l)
            a_u = np.sin(theta_u) ** 2
            self.upper_a.append(a_u)
            # Saving Posterior SMC probabilities
            if counter % save_smc_prob == 0:
                self.pdf_theta["c_{}".format(counter)] = theta_posterior
                self.pdf_weights["c_{}".format(counter)] = weights_posterior
            if counter % print_info == 0:
                print("{}: optimal_control: {}. output:{}, interval: {}".format(
                    counter,
                    optimal_control,
                    self.outcome_list[-1],
                    0.5 * (a_u - a_l)
                ))
            if counter >= max_iterations:
                failure_test = False
            counter = counter + 1
        # Store the final SMC posterior distribution
        self.pdf_theta["final_posterior"] = theta_posterior
        self.pdf_weights["final_posterior"] = weights_posterior

        ################## LOOP STEPS: END ########################
        a_l = self.lower_a[-1]
        a_u = self.upper_a[-1]
        a_m = self.mean_a[-1]
        return a_m, a_l, a_u

    def run(self):
        r"""
        run method for the class.

        Returns
        ----------

        self.ae :
            amplitude estimation parameter

        """
        bayes_dict = {
            "epsilon" : self.epsilon,
            "alpha" : self.alpha,
            "max_iterations" : self.max_iterations,
            "utility_function" : self.utility_function,
            "n_evals" : self.n_evals,
            "k_0" : self.k_0,
            "R" : self.R,
            "T" : self.T,
            "particles" : self.particles,
            "threshold" : self.threshold,
            "kernel" : self.kernel,
            "alpha_lw" : self.alpha_lw,
            "c" : self.c,
            "save_smc_prob" : self.save_smc_prob,
            "print_info" : self.print_info,
            "shots" : self.shots,
            "warm_shots" : self.warm_shots,
            "bayes_shots" : self.bayes_shots,
            "control_bayes_shots" : self.control_bayes_shots,
            "theta_good" : self.theta_good,
            "fake" : self.fake
        }
        start = time.time()
        self.ae, self.ae_l, self.ae_u = self.bayesqae()
        end = time.time()
        self.run_time = end - start
        self.schedule_pdf = pd.DataFrame(
            [self.control_list, self.shots_list, self.outcome_list],
            index=["m_k", "shots", "h_k"]
        ).T
        # self.schedule_pdf = pd.DataFrame.from_dict(
        #     self.schedule,
        #     columns=['shots'],
        #     orient='index'
        # )
        # self.schedule_pdf.reset_index(inplace=True)
        # self.schedule_pdf.rename(columns={'index': 'm_k'}, inplace=True)
        self.oracle_calls = np.sum(
            self.schedule_pdf['shots'] * (2 * self.schedule_pdf['m_k'] + 1))
        self.max_oracle_depth = np.max(2 *  self.schedule_pdf['m_k']+ 1)
        self.quantum_time = sum(self.quantum_times)
        self.pdf_estimation = pd.DataFrame(
            [self.mean_a, self.lower_a, self.upper_a],
            index = ["mean", "lower", "upper"],
        ).T
        return self.ae



def posterior_weights(thetas, weights, m_k, n_k, o_k):
    """
    Compute posterior probability weights given an input QAE experiment
    and a SMC prior probability distribution.

    Parameters
    ----------

    thetas : numpy array
        SMC Prior distribution of the desired parameter. Parameter Values
    weights : numpy array
        SMC Prior distribution of the desired parameter. Weights.
    m_k : int
        Exponent of the Grover operator of the QAE experiment
    n_k : int
        Number of shots of the QAE experiment
    o_k : int
        Outcome of the QAE experiment

    Returns
    -------

    posterior_weights_ : numpy array
        Posterior distribution given the outcome of the QAE experiment
    """
    like_ = likelihood(thetas, m_k, n_k, o_k)
    posterior_weights_ = like_ * weights
    # Renormalize weights
    posterior_weights_ = posterior_weights_ / (posterior_weights_.sum())
    return posterior_weights_

def ess(weights):
    """
    Compute the Effective Sample Size for an input weights

    Parameters
    ----------
    weights : numpy array
        array with the weights

    Returns
    -------

    ess_ : float
        efective sample size
    """

    #ess_ = np.sum(weights) ** 2 / np.sum(weights**2)
    ess_ = np.sum(weights) ** 2 / np.sum(weights**2)
    return ess_

def Metropolis_kernel(theta, m_k, n_k, o_k, **kwargs):
    """
    Metropolis kernel. Perturbation kernel used for enhancing resampling

    Parameters
    ----------

    theta : numpy array
        Input parameters from a SMC probability distribution.
    m_k : list
        Exponent of the Grover operator of the QAE experiment
    n_k : list
        Number of shots of the QAE experiment
    o_k : list
        Outcome of the QAE experiment
    kwargs: c : float
        coeficient for standard deviation multiplication

    Returns
    -------
    new_theta : numpy array
        Perturbed parameters for SMC probability distribution using
        Metropoli kernel

    """
    c_ = kwargs.get("c", 2.38)
    # First a proposal is needed: for each input theta we sample
    # from a Gaussian distribution with a mean equals to the theta
    new_theta = np.random.normal(theta, c_ * theta.std())
    # Change from second quadrant to first quadrant
    new_theta = np.where(new_theta > np.pi * 0.5, np.pi -new_theta, new_theta)
    # Change from forth quadrant to first quadrant
    new_theta = np.where(
        (new_theta < 0) & (new_theta >= -np.pi * 0.5), -new_theta, new_theta)
    # Change from third quadrant to first quadrant
    new_theta = np.where(new_theta < -np.pi * 0.5, np.pi + new_theta, new_theta)
    # Test: new_theta MUST BE in first quadrant
    condition = (new_theta < 0) | (new_theta > 0.5* np.pi)
    if condition.any():
        raise ValueError("Metropolis Kernel: new thetas outside firs quadrant")

    # Compute likelihood for input value distribution.
    like_old = np.array([
        log_likelihood(theta, m_k_, n_k_, o_k_)
        for m_k_, n_k_, o_k_ in zip(m_k, n_k, o_k)
    ]).sum(axis=0)

    # Compute likelihood for new value distribution.
    like_new = np.array([
        log_likelihood(new_theta, m_k_, n_k_, o_k_)
        for m_k_, n_k_, o_k_ in zip(m_k, n_k, o_k)
    ]).sum(axis=0)
    # Compute probability for using input theta or new_theta.
    # We need to clip probabilities between 0 and 1
    keep_old_probability = np.exp(like_old - like_new).clip(max=1.0, min=0.0)

    #print("Acceptance: {}".format(
    #    round((1.0 -keep_old_probability.mean()) * 100)))

    # We select which theta (new or input) depending on the
    # keep_old_probability
    samples = np.random.binomial(1, keep_old_probability)
    new_theta = np.where(samples == 1, theta, new_theta)
    return new_theta

def LW_kernel(theta, **kwargs):
    """
    Liu-West kernel. Perturbation kernel used for enhancing resampling

    Parameters
    ----------
    theta : numpy array
        Input parameters from a SMC probability distribution.
    kwargs: alpha_lw : float
        coeficient for creating new thetas

    Returns
    -------
    new_theta : numpy array
        Perturbed parameters for SMC probability distribution using
        Liu-West kernel
    """
    alpha_lw = kwargs.get("alpha_lw", 0.2)
    mean_ = np.mean(theta)
    new_mean = alpha_lw * theta + (1.0 - alpha_lw) * mean_
    new_std = np.sqrt(1.0 - alpha_lw ** 2) * theta.std()
    # Only mean and standard deviation is used
    new_theta = np.random.normal(new_mean, new_std)
    return new_theta

def bayesian_update(thetas, weights, m_k, n_k, o_k, resample=True, **kwargs):
    """
    Given a SMC prior probabilty distribution and a outcome of QAE experiment
    returns the SMC posterior probabilty distribution. It can performs resampling

    Parameters
    ----------

    thetas : numpy array
        SMC Prior distribution of the desired parameter. Parameter Values
    weights : numpy array
        SMC Prior distribution of the desired parameter. Weights.
    m_k : list
        Exponent of the Grover operator of the QAE experiment
    n_k : list
        Number of shots of the QAE experiment
    o_k : list
        Outcome of the QAE experiment
    kwargs: threshold : float
        Threshold for resampling the SMC probability distribution

    Returns
    -------

    new_thetas : numpy array
        SMC Posterior distribution of the desired parameter. Parameter Values
    posterior_weights_ : numpy array
        SMC Posterior distribution of the desired parameter. Weights.
    """
    threshold = kwargs.get("threshold", 0.5)
    kernel = kwargs.get("kernel", "Metro")
    # Compute posterior weights: We only use the last experiment result
    posterior_weights_ = posterior_weights(
        thetas, weights, m_k[-1], n_k[-1], o_k[-1])

    # We select when we want resampling. In general resampling in the computations
    # for optimize control are not mandatory
    if resample:
        # Compute effective sample size
        ess_ = ess(np.array(posterior_weights_))
        # Condition for resampling
        if ess_ < threshold * len(posterior_weights_):
            thetas_ = np.random.choice(
                thetas,
                p=posterior_weights_,
                size=len(thetas)
            )
            #print(pd.value_counts(thetas_))
            # uniform weights
            posterior_weights_ = np.ones(len(posterior_weights_))
            posterior_weights_ = posterior_weights_ / len(posterior_weights_)

            if kernel == "Metro":
                # We need all the experiment results!!
                new_thetas = Metropolis_kernel(thetas_, m_k, n_k, o_k, **kwargs)
            elif kernel == "LW":
                new_thetas = LW_kernel(thetas_, **kwargs)
            else:
                raise ValueError(
                    "Plese select a kernel for resamplin: Metro or LW")
            #print(pd.value_counts(new_thetas))
            return new_thetas, posterior_weights_
        else:
            return thetas, posterior_weights_
    else:
        return thetas, posterior_weights_

def weighted_expectaction(thetas, weights, m_k, n_k, o_k, **kwargs):
    """
    Computes the expected value of an input utility function for a given outcome
    of a QAE experiment (m_k, n_k, h_k) and a SMC prior probability
    weighted by the total probability getting the o_k outcome.

    Parameters
    ----------

    thetas : list
        SMC Prior distribution of the desired parameter. Parameter Values
    weights : list
        SMC Prior distribution of the desired parameter. Weights.
    m_k : int
        Exponent of the Grover operator of the QAE experiment
    n_k : int
        Number of shots of the QAE experiment
    o_k : int
        Outcome of the QAE experiment
    kwargs: utility_function : function
        Desired utility function for computing the expected value

    Returns
    -------

    weighted_expectation_ : float
        product of the expected value function for a given QAE experiment
        result and the probability of the given result

    """

    utility_function = kwargs.get("utility_function", None)
    if utility_function is None:
        raise ValueError("Provide a utility_function")
    # Computes the SMC posterior probability for the QAE experiment
    thetas_h_k_posterior, weights_h_k_posterior = bayesian_update(
        thetas, weights, #
        [m_k], [n_k], [o_k], False, **kwargs
    )
    # Compute the conditional expectaction of the utility function given
    # the control (m_k) the number of shots (n_k) and the outcome (o_k)
    # under the SMC posterior probability distribution:
    # E_{P(\theta/D;m)}[U(\theta, D;m)]
    conditional_uf = utility_function(
        thetas_h_k_posterior,
        weights_h_k_posterior
    )
    # Computes the probability of obtaining the outcome o_k for the QAE
    # experiment with control m_k and number of shots n_k given a SMC
    # priori probabilty distribution (input thetas and weights):
    # E_{P(\theta)}[P(D;m)]
    outcome_probability = likelihood(thetas, m_k, n_k, o_k) @ weights
    # We need the weighted expectaction of the utility function
    weighted_expectation_ = conditional_uf * outcome_probability
    return weighted_expectation_

def average_expectation(thetas, weights, m_k, n_k, **kwargs):
    """
    Computes the average expected value of a function for all posible
    outcomes froma QAE experiment with control m_k and number of shots
    n_k given a SMC prior probability function.

    Parameters
    ----------

    thetas : list
        SMC Prior distribution of the desired parameter. Parameter Values
    weights : list
        SMC Prior distribution of the desired parameter. Weights.
    m_k : int
        Exponent of the Grover operator of the QAE experiment
    n_k : int
        Number of shots of the QAE experiment
    kwargs: utility_function : function
        Function for computing its average expectation

    Returns
    -------

    value_: float
        Desired average expectation
    """
    value_ = np.sum([
        weighted_expectaction(thetas, weights, m_k, n_k, o_k, **kwargs)
        for o_k in range(n_k+1)
    ])
    return value_

def variance_function(thetas, weights, **kwargs):
    """
    Computes the exprected value of the variance for a SMC probability
    distribution

    Parameters
    ----------

    thetas : list
        SMC Prior distribution of the desired parameter. Parameter Values
    weights : list
        SMC Prior distribution of the desired parameter. Weights.

    Returns
    -------

    variance: float
        Expected value of the variance under the input SMC probability
        distribution
    """
    # Computes the expected value for the mean
    mean_expectation = thetas @ weights
    # Computes the square error with respect to the expected value of
    # the mean
    square_error = (thetas - mean_expectation) ** 2
    variance = square_error @ weights
    return variance

def confidence_intervals(thetas, weights, delta):
    """
    Computes the confidence intervals for a confidence level delta of a
    given SMC probability distribution

    Parameters
    ----------

    thetas : list
        SMC Prior distribution of the desired parameter. Parameter Values
    weights : list
        SMC Prior distribution of the desired parameter. Weights.
    delta : float
        Desired confidence level

    Returns
    -------

    theta_lower: float
        Lower value for confidence interval
    theta_upper: float
        Upper value for confidence interval
    """
    if (weights == 1.0/ weights.size).all():
        # In this case all the weights has the same probability
        # So we order thetas
        index_ = (weights.cumsum() > 0.5 * delta) \
            & (weights.cumsum() < (1.0 -0.5 * delta))
        covered_thetas = np.sort(thetas)[index_]
    else:
        # Weights has different probability
        # Ordering data by ascending probability
        sort_index = np.argsort(weights)[::-1]
        sort_thetas = thetas[sort_index]
        sort_weights = weights[sort_index]
        # Compute cumulative sum
        cumulative_prob = np.cumsum(sort_weights)
        # Compute thetas that enclose desired alpha
        covered_thetas = sort_thetas[
            (cumulative_prob > 0.5 * delta)
            & (cumulative_prob < (1.0 - 0.5 * delta))
        ]
    theta_l = covered_thetas.min()
    theta_u = covered_thetas.max()
    return theta_l, theta_u
