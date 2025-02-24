
    def rqae(self, ratio: float = 2, epsilon: float = 0.01, gamma: float = 0.05):
        """
        This function implements the first step of the RQAE paper. The
        result is an estimation of the desired amplitude with precision
        epsilon and accuracy gamma.

        Parameters
        ----------
        ratio : int
            amplification ratio
        epsilon : int
            precision
        gamma : float
            accuracy

        Returns
        ----------
        amplitude_min : float
           lower bound for the amplitude to be estimated
        amplitude_max : float
           upper bound for the amplitude to be estimated

        """
        ######################################

        epsilon = 0.5 * epsilon
        # Always need to clean the circuit statistics property
        self.circuit_statistics = {}
        # time_list = []
        
        
        ############### First Step #######################
        shift = 0.5 # shift [-0.5, 0.5]

        knext = schedule_k[1]
        Knext = 2 * knext + 1
        # Probability epsilon
        epsilon_first_p = np.abs(shift) * np.sin(0.5 * np.pi / (Knext + 2))
        # Maximum probability epsilon
        epsilon_first_p_max = 0.5 * np.sin(0.5 * np.arcsin(2 * epsilon)) ** 2
        # Real probabiliy epsilon to use
        epsilon_first_p = min(epsilon_first_p, epsilon_first_p_max)
        gamma_first = schedule_gamma[0]
        # This is shots for each iteration: Ni in the paper
        n_first = int(
            np.ceil(1 / (2 * epsilon_first_p**2) * np.log(2 / gamma_first))
        )
        # Quantum routine for first step
        [amplitude_min, amplitude_max] = self.first_step(
            shift=shift, shots=n_first, gamma=gamma_first
        )
        epsilon_amplitude = (amplitude_max - amplitude_min) / 2


        ############### Consecutive Steps #######################
        i = 1
        while epsilon_amplitude > epsilon:
            k_empirical = int(np.floor(np.pi / (4 * np.arcsin(2 * epsilon_amplitude)) - 0.5))
            K_empirical = 2 * k_empirical + 1

            k_next = schedule_k[i+1]
            K_next = 2 * k_next + 1
            k_current = schedule_k[i]
            K_current = 2 * k_current + 1

            # Probability epsilon
            epsilon_step_p = 0.5 * np.sin(0.25 * np.pi * K_current / (K_next + 2) ) ** 2
            # Maximum probability epsilon
            epsilon_step_p_max = 0.5 * np.sin(0.5 * K_empirical * np.arcsin(2 * epsilon)) ** 2
            # Real probabiliy epsilon to use
            epsilon_step_p = min(epsilon_step_p, epsilon_step_p_max)
            gamma_step = schedule_gamma[i]
            # This is shots for each iteration: Ni in the paper
            n_step = int(
                np.ceil(1 / (2 * epsilon_step_p**2) * np.log(2 / gamma_step))
            )
            
            if K_empirical < K_current:
                print("Albeto fails!")

            # Quantum routine for first step
            shift = -amplitude_min
            shift = min(shift, 0.5)
            [amplitude_min, amplitude_max] = self.run_step(
                shift=shift, shots=n_step, gamma=gamma_step, k=k_empirical
            )
            # time_list.append(time_pdf)
            epsilon_amplitude = (amplitude_max - amplitude_min) / 2
            i = i + 1

        return [2 * amplitude_min, 2 * amplitude_max]



    def schedule_exponential_constant(epsilon, gamma, ratio):
        
        K_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0 

        k_max_next = np.ceil((K_max_next - 1) / 2)

        k = 0
        K = 2 * k + 1
        k_next = k * ratio
        K_next = 2 * k_next + 1
        k_list = [k]
        while K_next < K_max_next:
            k = k_next
            K = 2 * k + 1
            k_next = k * ratio
            K_next = 2 * k_next + 1
            k_list.append(k)

        k_list.append(k_max_next)
        
        gamma_i = gamma / len(k_list)
        gamma_list = [gamma_i] * len(k_list)

    
    def schedule_exponential_exponential(epsilon, gamma, ratio):
        
        K_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0 

        k_max_next = np.ceil((K_max_next - 1) / 2)

        k = 0
        K = 2 * k + 1
        k_next = k * ratio
        K_next = 2 * k_next + 1
        k_list = [k]
        while K_next < K_max_next:
            k = k_next
            K = 2 * k + 1
            k_next = k * ratio
            K_next = 2 * k_next + 1
            k_list.append(k)

        k_list.append(k_max_next)
        
        # Hacerlo exponencia usando la lista de ks
        # Ojo primer ganma
        gamma_i = np.array([i for i in k_list])
        cte = gamma / np.sum(gamma_i)
        gamma_i / cte
        
            
    def schedule_linear_linear(epsilon, gamma, slope):
        K_max_next = 0.5 / np.arcsin(2 * epsilon) - 2.0 

        k_max_next = np.ceil((K_max_next - 1) / 2)

        k = 0
        K = 2 * k + 1
        k_next = k + slope
        K_next = 2 * k_next + 1
        k_list = [k]
        while K_next < K_max_next:
            k = k_next
            K = 2 * k + 1
            k_next = k_next + slope
            K_next = 2 * k_next + 1
            k_list.append(k)

        k_list.append(k_max_next)
        
        # Hacerlo exponencia usando la lista de ks
        # Ojo primer ganma
        gamma_i = np.array([i for i in k_list])
        cte = gamma / np.sum(gamma_i)
        gamma_i / cte
            
            
            
            
























