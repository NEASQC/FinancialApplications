from qat.qpus import PyLinalg
global_qlmaas = True
try:
    from qlmaas.qpus import LinAlg
except (ImportError, OSError) as exception:
    global_qlmaas = False

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
        if global_qlmaas:
            print('Using: LinAlg')
            linalg_qpu = LinAlg()
        else:
            raise ImportError("""Problem Using QLMaaS.
            Please create config file or use mylm solver""")
    else:
        print('Using PyLinalg')
        linalg_qpu = PyLinalg()
    return linalg_qpu
