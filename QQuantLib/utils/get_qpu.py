"""
Selector for QPU.
"""

def get_qpu(qpu=None):
    """
    Function for selecting solver.

    Parameters
    ----------

    qpu : str
        * qlmass: for trying to use QLM as a Service connection
            to CESGA QLM
        * python: for using PyLinalg simulator.
        * c: for using CLinalg simulator
        * mps: for using mps

    Returns
    ----------

    linal_qpu : solver for quantum jobs
    """

    if qpu is None:
        raise ValueError(
            "qpu CAN NOT BE NONE. Please select one of the three" +
            " following options: qlmass, python, c")
    if qpu == "qlmass_linalg":
        try:
            from qlmaas.qpus import LinAlg
            linalg_qpu = LinAlg()
        except (ImportError, OSError) as exception:
            raise ImportError(
                "Problem Using QLMaaS. Please create config file" +
                "or use mylm solver") from exception
    elif qpu == "qlmass_mps":
        try:
            from qlmaas.qpus import MPS
            #linalg_qpu = MPS(lnnize=True)
            linalg_qpu = MPS()
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
    elif qpu == "linalg":
        from qat.qpus import LinAlg
        linalg_qpu = LinAlg()
    elif qpu == "mps":
        from qat.qpus import MPS
        linalg_qpu = MPS()
    return linalg_qpu
