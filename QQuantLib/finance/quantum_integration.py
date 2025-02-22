"""
This module contains the *q_solve_integral* function that allows to the
user codify easily an integral in a quantum state, using the different
encoding protocols from the *Encoding* class from
**QQuantLib.DL.encoding_protocols** module, and estimating it using the
different **AE** algorithms implemented in the **QQuantLib.AE** package.


The *q_solve_integral* function deals with all the normalisations needed
for transforming the estimated amplitude into the desired integral.


Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from QQuantLib.DL.encoding_protocols import Encoding
from QQuantLib.AE.ae_class import AE
from QQuantLib.utils.utils import text_is_none

def q_solve_integral(**kwargs):
    """
    Function for solving an integral using quantum amplitude \\
    estimation techniques.

    Parameters
    ----------

    array_function : kwargs, numpy array
        numpy array with the desired function for encoding into the \\
        Quantum Circuit.
    encoding : kwargs, int
        Selecting the encode protocol
    array_probability : kwargs, numpy array
        numpy array with the desired probability for encoding into the \\
        Quantum Circuit. It can be None (uniform distribution will be used)
    ae_type : kwargs, string
        string with the desired AE algorithm: MLAE, CQPEAE, IQPEAE, \\
        IQAE, RQAE

    Note
    ----
    
    Other kwargs input dictionary keys will be related with the encoding \\
    of the integral into the quantum circuit \\
    (see QQuantLib.DL.encoding_protocols) and for the configuration \\
    of the AE algorithm used (see QQuantLib.AE.ae_class)

    Return
    ----------

    ae_estimation: pandas DataFrame
        DataFrame with the desired integral computation.
    solver_ae: objet based on the AE class

    """


    encoding = kwargs.get("encoding", None)
    ae_type = kwargs.get("ae_type", None)
    if (encoding == 0) and (ae_type in ["RQAE", "eRQAE", "sRQAE", "mRQAE"]):
        string_error = (
            "RQAEs methods CAN NOT BE USED with encoding protocol: "+str(encoding)
        )

        warnings.warn(string_error)

        ae_estimation = pd.DataFrame(
            [None, None, None],
            index=["ae", "ae_l", "ae_u"],
        ).T
        solver_ae = None
        encode_class = None
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
        # Set the QPU to use
        linalg_qpu = kwargs.get("qpu", None)
        if linalg_qpu is None:
            raise ValueError("qpu is None. Please provide a valid qpu")
        #del kwargs['qpu']

        # ae_dict = deepcopy(kwargs)
        #ae_dict.update({"qpu": linalg_qpu})
        #Delete keys from encoding
        # for step in ["array_function", "array_probability", "encoding", "multiplexor"]:
        #     ae_dict.pop(step, None)
        # ae_dict.pop("ae_type", None)
        #Instantiate AE solver
        solver_ae = AE(
            oracle=encode_class.oracle,
            target=encode_class.target,
            index=encode_class.index,
            #ae_type=ae_type,
            **kwargs)

        # run the amplitude estimation algorithm
        solver_ae.run()

        # Recover amplitude estimation from ae_solver
        if encoding == 0:
            # In the square encoding the np.sqrt(p_x * f_norm_x) is
            # loaded in the amplitude of the state |x>^n|0>.
            # The probability of measuring a |0> in the additional qubit
            # is just sum_x(|p_x * f_norm_x|)
            ae_pdf = solver_ae.ae_pdf
        elif encoding == 1:
            if ae_type in ["RQAE", "eRQAE", "mRQAE", "sRQAE"]:
                #Amplitude is provided directly by this algorithm
                ae_pdf = solver_ae.ae_pdf
            else:
                #Other algorithms return probability
                ae_pdf = np.sqrt(solver_ae.ae_pdf)
                if ae_type != "MLAE":
                    # We need to provided the mean of the square root bounds
                    # as the measured ae. Not valid for IQAE
                    ae_pdf["ae"] = (ae_pdf["ae_l"] + ae_pdf["ae_u"]) / 2.0
        elif encoding == 2:
            # In the direct encoding we load the sum_x(p_x * f_norm_x)
            # into the amplitude of the |0>^{n+1} state
            if ae_type in ["RQAE", "eRQAE", "mRQAE", "sRQAE"]:
                # RQAE provides amplitude directly.
                ae_pdf = solver_ae.ae_pdf
            else:
                # AE algorithms provide an estimation of the probability
                # so they provide: sum_x(p_x * f_norm_x) ** 2
                # we need to compute sqrt(sum_x(p_x * f_norm_x) ** 2)
                ae_pdf = np.sqrt(solver_ae.ae_pdf)
                # We need to provided the mean of the square root bounds
                # as the measured ae.
                ae_pdf["ae"] = (ae_pdf["ae_l"] + ae_pdf["ae_u"]) / 2.0
        else:
            raise ValueError("Not valid encoding key was provided!!!")
        #Now we need to deal with encoding normalisation
        ae_estimation = ae_pdf * encode_class.encoding_normalization
    return ae_estimation, solver_ae
