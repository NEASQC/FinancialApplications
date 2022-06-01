"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains functions for calling QLM solver

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

from qat.qpus import PyLinalg


def get_qpu(qlmass=False):
    """
    Function for selecting solver. User can chose between:
    * LinAlg: for submitting jobs to a QLM server
    * PyLinalg: for simulating jobs using myqlm lineal algebra.

    Parameters
    ----------

    qlmass : bool
        If True  try to use QLM as a Service connection to CESGA QLM
        If False PyLinalg simulator will be used

    Returns
    ----------

    linal_qpu : solver for quantum jobs
    """
    if qlmass:
        try:
            from qlmaas.qpus import LinAlg

            linalg_qpu = LinAlg()
            print("Using: LinAlg")
        except (ImportError, OSError) as exception:
            raise ImportError(
                """Problem Using QLMaaS.
            Please create config file or use mylm solver"""
            ) from exception
    else:
        print("Using PyLinalg")
        linalg_qpu = PyLinalg()
    return linalg_qpu
