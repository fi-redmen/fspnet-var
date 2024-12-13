"""
Utility functions for multiprocessing
"""
import os
import re
import subprocess
from time import time

#from netloader.utils.utils import progress_bar
def progress_bar(i: int, total: int, text: str = '', **kwargs: any) -> None:
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    text : str, default = ''
        Optional text to place at the end of the progress bar

    **kwargs
        Optional keyword arguments to pass to print
    """
    filled: int
    length: int = 50
    percent: float
    bar_fill: str
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='', **kwargs)

    if i == total:
        print()

def mpi_multiprocessing(cpus: int, total: int, arg: str, python_path: str = 'python3'):
    """
    Creates and tracks multiple workers running a Python module using MPI,
    tracking is done through the worker print statement 'update'

    Can track failures through worker print statement 'fail_num='

    Supports single threading with debugging support (tested with PyCharm)

    Parameters
    ----------
    cpus : str
        Number of threads to use
    total : str
        Total number of tasks
    arg : str
        Python module argument after python3 -m
    python_path : str, default = python3
        Path to the python executable if using virtual environments
    """
    failure_total = count = 0
    initial_time = time()
    text = ''

    # Start workers
    if cpus == 1:
        subprocess.run(f'{python_path} -m {arg}'.split(), check=True)
        return

    print(f'Starting {cpus} workers...')
    with subprocess.Popen(
            f'mpiexec -n {cpus} --use-hwthread-cpus '
            f'{python_path} -m {arg}'.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
    ) as proc:
        # Track progress of workers
        for line in iter(proc.stdout.readline, b''):
            #print("line: ", line)
            line = line.decode('utf-8').strip()
            #print("line: ", line)
            fail_num = re.search(r'fail_num=(\d+)', line)

            if fail_num and count != 0:
                failure_total += int(fail_num.group(1))
                success = count * 100 / (count + failure_total)
                text = f'\tSuccess: {success:.1f} % '
            elif 'update' in line:
                count += 1
                eta = (time() - initial_time) * (total / count - 1)
                text += f'ETA: {eta:.2f} s\tProgress: {count} / {total}'
                progress_bar(count, total + 1, text=text)
                text = ''
            elif 'error' in line.lower():
                raise RuntimeError(f'Multiprocessing error:\n{line}')

    print(f'\nWorkers finished\tTime: {time() - initial_time:.3e} s')


def check_cpus(cpus: int) -> int:
    """
    Checks if cpus is greater than the total number of threads, or if it is less than 1,
    if so, set the number of threads to the maximum number available

    Parameters
    ----------
    cpus : int
        Number of threads to check

    Returns
    -------
    int
        Number of threads
    """
    if cpus < 1 or cpus > os.cpu_count():
        return int(os.cpu_count())

    return cpus
