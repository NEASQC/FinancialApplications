"""
This module implements the get_qpu function that allows to the user
select a **EVIDEN QPU** for ideal simulation. The qpu variable is a
string that should take some of the following values:


    **qlmass_linalg**.
        For selecting the linear algebra simulator LinAlg. This QPU
        can be used only with QaptivaÔäó Appliance when the user sends the computations to a remote QPU. The user must have remote
        access to a remote QLM using the Qaptiva Access (QLM as a Service)
        library.
    **qlmass_mps**.
        For selecting the Matrix Product State (MPS) simulator. This QPU
        can be used only with QaptivaÔäó Appliance when the user sends the computations to a remote QPU. The user must have remote
        access to a remote QLM using the Qaptiva Access (QLM as a Service)
        library.
    **python**.
        For selecting the linear algebra simulator PyLinalg. This a pure
        Python algebra simulator. This QPU is provided by the myQLM
        library. It can not be used with QaptivaÔäó Appliance.
    **c**
        For selecting the linear algebra simulator CLinalg. This a pure
        C algebra simulator. This QPU is provided by the myQLM
        library. It can not be used with QaptivaÔäó Appliance.
    **linalg**
        For selecting the linear algebra simulator LinAlg. This QPU
        can be used only with QaptivaÔäó Appliance when the user is locally
        in a QLM.
    **mps**
        For selecting the Matrix Product State (MPS) simulator This QPU
        can be used only with QaptivaÔäó Appliance when the user is locally
        in a QLM.
"""

def get_qpu(qpu=None):
    """
    Function for selecting solver.

    Parameters
    ----------

    qpu : str
        string with the desired qpu

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
