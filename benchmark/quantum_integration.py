
"""
This module contains a class for selecting data encoding protocols

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import warnings
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
sys.path.append("../")
from encoding_protocols import Encoding
from QQuantLib.AE.ae_class import AE
from QQuantLib.utils.utils import text_is_none
from QQuantLib.utils.qlm_solver import get_qpu

def q_solve_integral(**kwargs):
    """
    Function for solving an integral using quantum amplitude estimationtechniques.

    Parameters
    ----------
    Mandatory keys of the input kwargs will be:

    array_function : numpy array
        numpy array wiht the desired function for encoding into the Quantum Circuit.
    encoding : int
        Selecting the encode protocol
    array_probability : numpy array
        numpy array wiht the desired probability for encoding into the
        Quantum Cirucit. It can be None (uniform distribution will be used)
    ae_type : string
        string with the desired AE algorithm:
        MLAE, CQPEAE, IQPEAE, IQAE, RQAE

    Other keys will be realted with circuit implementations of the encoding procedure
    or/and AE algorithm configuration (see QQQuantLib.AE.ae_class)
    """


    encoding = kwargs.get("encoding", None)
    ae_type = kwargs.get("ae_type", None)
    if (encoding == 0) and (ae_type == "RQAE"):
        string_error = (
            "RQAE method CAN NOT BE USED with encoding protocol: "+str(encoding)
        )

        warnings.warn(string_error)

        ae_estimation = pd.DataFrame(
            [None, None, None],
            index=["ae", "ae_l", "ae_u"],
        ).T
        return ae_estimation, None, None
    else:

        #Mandatory kwargs for encoding data
        array_function = kwargs.get("array_function", None)
        text_is_none(array_function, "array_function", variable_type=np.ndarray)
        array_probability = kwargs.get("array_probability", None)
        text_is_none(encoding, "encoding", variable_type=int)
        encoding_dict = {"multiplexor": kwargs.get("multiplexor", True)}
        #instantiate encoding class
        encode_class = Encoding(
            array_function=array_function,
            array_probability=array_probability,
            encoding=encoding,
            **encoding_dict
        )
        #execute run method of the encoding class
        encode_class.run()


        if encode_class.oracle is None:
            raise ValueError("Oracle was not created!!")

        #Mandatory kwargs for ae solver
        ae_dict = deepcopy(kwargs)
        #Delete keys from encoding
        for step in ["array_function", "array_probability", "encoding", "multiplexor"]:
            ae_dict.pop(step, None)
        ae_dict.pop("ae_type", None)
        #Instantiate AE solver
        solver_ae = AE(
            oracle=encode_class.oracle,
            target=encode_class.target,
            index=encode_class.index,
            ae_type=ae_type,
            **ae_dict)
        # run the amplitude estimation algorithm
        solver_ae.run()
        # Recover amplitude estimation from ae_solver
        if encoding == 0:
            ae_pdf = solver_ae.ae_pdf
        elif encoding == 1:
            if ae_type == "RQAE":
                #Amplitude is provided directly by this algorithm
                ae_pdf = solver_ae.ae_pdf
            else:
                #Other algorithms return probability
                ae_pdf = np.sqrt(solver_ae.ae_pdf)
        elif encoding == 2:
            if ae_type == "RQAE":
                #RQAE provides amplitude directly.
                ae_pdf = solver_ae.ae_pdf
            else:
                #Other algorithms return probability
                ae_pdf = np.sqrt(solver_ae.ae_pdf)
        else:
            raise ValueError("Not valid encoding key was provided!!!")
        #Now we need to deal with encoding normalisation
        ae_estimation = ae_pdf * encode_class.encoding_normalization
        return ae_estimation, solver_ae, encode_class
