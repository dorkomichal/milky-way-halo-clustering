import os

MAX_PARALLEL_PROCESSES = 1


def initialize_multiprocessing():
    """Initialise number of OpenMP threads and number of processes to use
    for multiprocess fitting
    """
    # Attempt to initialise with some sensible values
    global MAX_PARALLEL_PROCESSES
    cpu_count = os.cpu_count()
    open_mp_threads = int(cpu_count / 4)
    MAX_PARALLEL_PROCESSES = int(cpu_count / open_mp_threads)
    os.environ["OMP_NUM_THREADS"] = f"{open_mp_threads}"


def get_max_processes():
    global MAX_PARALLEL_PROCESSES
    return MAX_PARALLEL_PROCESSES
