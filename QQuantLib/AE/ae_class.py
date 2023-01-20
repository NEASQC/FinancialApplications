"""
This module contains a general class for solving AE problems
using the algorithm classes from QQuantLib.AE library package
Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro

"""

import pandas as pd
from QQuantLib.utils.qlm_solver import get_qpu
from QQuantLib.AE.maximum_likelihood_ae import MLAE
from QQuantLib.AE.ae_classical_qpe import CQPEAE
from QQuantLib.AE.ae_iterative_quantum_pe import IQPEAE
from QQuantLib.AE.iterative_quantum_ae import IQAE
from QQuantLib.AE.real_quantum_ae import RQAE
from QQuantLib.AE.montecarlo_ae import MCAE
from QQuantLib.utils.utils import text_is_none

class AE:
    """
    Class for creating and solving an AE problem

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
    ae_type : string
        string with the desired AE algorithm:
        MLAE, CQPEAE, IQPEAE, IQAE, RQAE
    kwars : dictionary
        dictionary that allows the configuration of the AE algorithm. \\
        The different configration keys of the different AE algorithms \\
        can be provided.
"""
    def __init__(self, oracle=None, target=None, index=None, ae_type=None, **kwargs):
        """

        Method for initializing the class

        """

        # Setting attributes
        text_is_none(oracle, "oracle", variable_type="QLM Routine")
        self.oracle = oracle
        text_is_none(target, "target", variable_type=list)
        self.target = target
        text_is_none(index, "index", variable_type=list)
        self.index = index

        #Processing kwargs
        self.kwargs = kwargs
        self.linalg_qpu = self.kwargs.get("qpu", None)

        # Set the QPU to use
        self.linalg_qpu = kwargs.get("qpu", None)
        if self.linalg_qpu is None:
            print("Not QPU was provide. PyLinalg will be used")
            self.linalg_qpu = get_qpu("python")

        self._ae_type = ae_type
        #attributes created
        self.solver_ae = None
        self.ae_pdf = None
        self.solver_dict = None
        self.oracle_calls = None
        self.max_oracle_depth = None
        self.schedule_pdf = None
        self.quantum_times = []
        self.quantum_time = None
        self.run_time = None

    @property
    def ae_type(self):
        """
        creating ae_type property
        """
        return self._ae_type

    @ae_type.setter
    def ae_type(self, stringvalue):
        """
        setter of the target property
        """
        self._ae_type = stringvalue
        self.solver_ae = None
        self.ae_pdf = None
        self.solver_dict = None
        self.oracle_calls = None

    def create_ae_solver(self):
        """
        Method for instantiate the AE algorithm class.
        """
        text_is_none(self.ae_type, "ae_type attribute", variable_type=str)
        #common ae settings
        self.solver_dict = {
            "mcz_qlm" : self.kwargs.get("mcz_qlm", True),
            "qpu" : self.kwargs.get("qpu", None)
        }


        if self.ae_type == "MLAE":
            for par in ["delta", "ns", "schedule"]:
                val_par = self.kwargs.get(par)
                if val_par is not None:
                    self.solver_dict.update({par : val_par})
            self.solver_ae = MLAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "CQPEAE":
            for par in ["auxiliar_qbits_number", "shots"]:
                val_par = self.kwargs.get(par)
                if val_par is not None:
                    self.solver_dict.update({par : val_par})
            self.solver_ae = CQPEAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "IQPEAE":
            for par in ["cbits_number", "shots"]:
                val_par = self.kwargs.get(par)
                if val_par is not None:
                    self.solver_dict.update({par : val_par})
            self.solver_ae = IQPEAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "IQAE":
            for par in ["epsilon", "alpha", "shots"]:
                val_par = self.kwargs.get(par)
                if val_par is not None:
                    self.solver_dict.update({par : val_par})
            self.solver_ae = IQAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "RQAE":
            for par in ["epsilon", "gamma", "q"]:
                val_par = self.kwargs.get(par)
                if val_par is not None:
                    self.solver_dict.update({par : val_par})
            self.solver_ae = RQAE(
                self.oracle,
                target=self.target,
                index=self.index,
                **self.solver_dict
            )
        elif self.ae_type == "MCAE":
            for par in ["shots"]:
                val_par = self.kwargs.get(par)
                if val_par is not None:
                    self.solver_dict.update({par : val_par})
            self.solver_ae = MCAE(
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
        #create AE algorithm object
        self.create_ae_solver()
        self.solver_ae.run()
        self.oracle_calls = self.solver_ae.oracle_calls
        self.max_oracle_depth = self.solver_ae.max_oracle_depth
        self.schedule_pdf = self.solver_ae.schedule_pdf
        
        # Recover amplitude estimation from ae_solver
        self.ae_pdf = pd.DataFrame(
            [self.solver_ae.ae, self.solver_ae.ae_l, self.solver_ae.ae_u],
            index=["ae", "ae_l", "ae_u"],
        ).T
        self.quantum_times = self.solver_ae.quantum_times
        self.quantum_time = self.solver_ae.quantum_time
        self.run_time = self.solver_ae.run_time
