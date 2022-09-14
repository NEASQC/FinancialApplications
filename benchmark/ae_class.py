
"""
This module contains a class for selecting data encoding protocols

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import sys
import numpy as np
import pandas as pd
sys.path.append("../")
from encoding_protocols import Encoding
from QQuantLib.AE.maximum_likelihood_ae import MLAE
from QQuantLib.AE.ae_classical_qpe import CQPEAE
from QQuantLib.AE.ae_iterative_quantum_pe import IQPEAE
from QQuantLib.AE.iterative_quantum_ae import IQAE
from QQuantLib.AE.real_quantum_ae import RQAE
from QQuantLib.utils.utils import text_is_none


class AE:
    """
    Class for creating and solving an AE problem
    """
    def __init__(self, **kwargs):
        """
        AE class initialization.
        """

        self.kwargs = kwargs
        #Atributes for Encoding protocol
        self.encode_class = None
        self.oracle = None
        self.index = None
        self.target = None
        self.encoding = None
        self.n_qbits = None 
        #Atributes for AE algorithm
        self.ae_type = None
        self.solver_ae = None
        self.solver_dict = None
        self.ae_pdf = None 

    def create_oracle(self):
        """
        Method for creating the oracle and the state to amplify
        """

        #Mandatory kwargs for encoding data
        array_function = self.kwargs.get("array_function", None)
        text_is_none(array_function, "array_function", variable_type=np.ndarray)
        array_probability = self.kwargs.get("array_probability", None)
        self.encoding = self.kwargs.get("encoding", None)
        text_is_none(self.encoding, "encoding", variable_type=int)
        encoding_dict = {"multiplexor": self.kwargs.get("multiplexor", True)}
        self.encode_class = Encoding(
            array_function=array_function,
            array_probability=array_probability,
            encoding=self.encoding,
            **encoding_dict
        )
        #execute run method of the encoding class
        self.encode_class.run()

        #Oracle for Loading Data
        self.oracle = self.encode_class.oracle
        #State info of the oracle
        self.index = self.encode_class.co_index
        self.target = self.encode_class.co_target
        self.n_qbits = self.encode_class.n_qbits

    def create_ae_solver(self):
        """
        Method for instantiate the AE solver
        """

        if self.oracle is None:
            raise ValueError("oracle atribute is None. \
                Please execute create_oracle method!")

        self.ae_type = self.kwargs.get("ae_type", None)
        text_is_none(self.ae_type, "ae_type", variable_type=str)
        self.solver_dict = {
            "mcz_qlm" : self.kwargs.get("mcz_qlm", True),
            "qpu" : self.kwargs.get("qpu", None)
        }

        if self.ae_type == "MLAE":
            self.solver_dict.update({
                "delta" : self.kwargs.get("delta", None),
                "ns" : self.kwargs.get("ns", None),
                "schedule" : self.kwargs.get("schedule", None)
            })
            self.solver_ae = MLAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "CQPEAE":
            self.solver_dict.update({
                "auxiliar_qbits_number" : self.kwargs.get("auxiliar_qbits_number", None),
                "shots" : self.kwargs.get("shots", None)
            })
            self.solver_ae = CQPEAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "IQPEAE":
            self.solver_dict.update({
                "cbits_number" : self.kwargs.get("cbits_number", None),
                "shots" : self.kwargs.get("shots", None)
            })
            self.solver_ae = IQPEAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "IQAE":
            self.solver_dict.update({
                "epsilon" : self.kwargs.get("epsilon", None),
                "alpha" : self.kwargs.get("alpha", None),
                "shots" : self.kwargs.get("shots", None)
            })
            self.solver_ae = IQAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "RQAE":
            self.solver_dict.update({
                "epsilon" : self.kwargs.get("epsilon", None),
                "gamma" : self.kwargs.get("gamma", None),
                "q" : self.kwargs.get("q", None),
                "shots" : self.kwargs.get("shots", None)
            })
            self.solver_ae = RQAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        else:
            raise ValueError("AE algorithm IS NOT PROVIDED in ae_type parameter \
            Please use: MLAE, CQPEAE, IQPEAE, IQAE, RQAE")

    def run(self):
        """
        Method for running an AE problem
        """
        #create Oracle
        self.create_oracle()
        #create AE algorithm object
        self.create_ae_solver()
        if (self.encoding == 0) and (self.ae_type == "RQAE"):
            string_error = (
                "RQAE method CAN NOT BE USED with encoding protocol: "+str(self.encoding)
            )
            raise ValueError(string_error)
        # run the amplitude estimation algorithm
        self.solver_ae.run()

        # Recover amplitude estimation from ae_solver
        self.ae_pdf = pd.DataFrame(
            [self.solver_ae.ae, self.solver_ae.ae_l, self.solver_ae.ae_u],
            index=["ae", "ae_l", "ae_u"],
        ).T
        #Post Procces output
        a_estimation = None
        if self.encoding == 0:
            if self.encode_class.probability is None:
                #In this case the uniform distribution was used.
                #We need to add the probabilities to the state
                a_estimation = self.ae_pdf*2**self.n_qbits
            else:
                #Probability distribution used so is inside of the AE
                a_estimation = self.ae_pdf
        elif (self.encoding == 1):
            if self.ae_type == "RQAE":
                #Amplitude is provided directly by this algorithm
                a_estimation = 2**self.n_qbits * self.ae_pdf
            else:
                #Other algorithms return probability
                a_estimation = 2**self.n_qbits * np.sqrt(self.ae_pdf)
        elif (self.encoding == 2):
            if self.ae_type == "RQAE":
                #Amplitude is provided directly by this algorithm
                a_estimation = 2**self.n_qbits * self.ae_pdf
            else:
                if self.encode_class.probability is None:
                    #In this case the uniform distribution was used.
                    #We need to add the probabilities to the state
                    a_estimation = np.sqrt(self.ae_pdf)*2**self.n_qbits
                else:
                    #Probability distribution used so is inside of the AE
                    a_estimation = np.sqrt(self.ae_pdf)
        return a_estimation

