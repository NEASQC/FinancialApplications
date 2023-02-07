"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains functions for calling QLM solver

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

from qat.qpus import get_default_qpu


def get_qpu(qpu=None):
    """
    Function for selecting solver.

    Parameters
    ----------

    qpu : str
        * qlmass: for trying to use QLM as a Service connection to CESGA QLM
        * python: for using PyLinalg simulator.
        * c: for using CLinalg simulator

    Returns
    ----------

    linal_qpu : solver for quantum jobs
    """

    if qpu is None:
        raise ValueError(
            "qpu CAN NOT BE NONE. Please select one of the three" +
            " following options: qlmass, python, c")
    elif qpu == "qlmass":
        try:
            from qlmaas.qpus import LinAlg
            linalg_qpu = LinAlg()
        except (ImportError, OSError) as exception:
            raise ImportError(
                "Problem Using QLMaaS. Please create config file" +
                "or use mylm solver") from exception
    elif qpu == "python":
        from qat.qpus import PyLinalg
        linalg_qpu = PyLinalg()
    elif qpu == "c":
        from qat.qpus import CLinalg
        linalg_qpu = CLinalg()
    elif qpu == "default":
        linalg_qpu = get_default_qpu()
    else:
        raise ValueError(
            "Invalid value for qpu. Please select one of the three "+
            "following options: qlmass, python, c")
    print("Following qpu will be used: {}".format(linalg_qpu))
    return linalg_qpu
