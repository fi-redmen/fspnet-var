import os
import pickle
from typing import Any, BinaryIO

import numpy as np
from numpy import ndarray
from fspnet.utils.utils import open_config
from fspnet.utils.multiprocessing import check_cpus, mpi_multiprocessing

from fspnetvar.utils.misc_utils import ROOT

def pyxspec_test(
        worker_dir: str,
        names: ndarray,
        params: ndarray,
        cpus: int = 1,
        job_name: str | None = None,
        python_path: str = 'python3') -> None:
    """
    Calculates the PGStat loss using PyXspec
    Done using multiprocessing if > 2 cores available

    Parameters
    ----------
    worker_dir : str
        Directory to where to save worker data
    names : ndarray
        Files names of the FITS spectra corresponding to the parameters
    params : ndarray
        Parameter predictions
    cpus : int, default = 1
        Number of threads to use, 0 will use all available
    job_name : str, default = None
        If not None, file name to save the output to
    python_path : str, default = python3
        Path to the python executable if using virtual environments
    """
    i: int
    data: list[ndarray] = []
    worker_names: list[ndarray]
    worker_params: list[ndarray]
    job: ndarray
    data_: ndarray
    names_batch: ndarray
    params_batch: ndarray

    # Divide work between workers
    cpus = check_cpus(cpus)
    worker_names = np.array_split(names, cpus)
    worker_params = np.array_split(params, cpus)

    # Save data to file for each worker
    for i, (names_batch, params_batch) in enumerate(zip(worker_names, worker_params)):
        job = np.hstack((np.expand_dims(names_batch, axis=1), params_batch))
        np.savetxt(f'{worker_dir}worker_{i}_job.csv', job, delimiter=',', fmt='%s')

    # Run workers to calculate PGStat
    mpi_multiprocessing(
        cpus,
        len(names),
        f'fspnet.utils.pyxspec_worker {worker_dir}',
        python_path=python_path,
    )

    # Retrieve worker outputs
    for i in range(cpus):
        data.append(np.loadtxt(f'{worker_dir}worker_{i}_job.csv', delimiter=',', dtype=str))

    data_ = np.concatenate(data)

    # If job_name is provided, save all worker data to file
    if job_name:
        np.savetxt(f'{worker_dir}{job_name}.csv', data_, delimiter=',', fmt='%s')

    # Median loss
    print(f'Reduced PGStat Loss: {np.median(data_[:, -1].astype(float)):.3e}')

    return data_


def pyxspec_tests(
        data: dict[str, ndarray],
        config: str | dict[str, Any] = './config.yaml') -> None:
    """
    Tests the PGStats of the different fitting methods using PyXspec

    Parameters
    ----------
    data : dict[str, ndarray]
        ids : ndarray
            Files names of the FITS spectra corresponding to the parameters
        latent : ndarray
            Parameter predictions
        targets : ndarray
            Best fit parameters
    config : string | dictionary, default = './config.yaml'
        Configuration dictionary or path to the configuration dictionary
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    # Initialize variables
    cpus: int = config['training']['cpus']
    python: str = config['training']['python-path']
    worker_dir: str = config['output']['worker-directory']
    default_params: list[float] = config['model']['default-parameters']
    worker_data: dict[str, Any] = {
        'optimize': True,
        'dirs': [
            ROOT,
            config['data']['spectra-directory'],
        ],
        'iterations': config['model']['iterations'],
        'step': config['model']['step'],
        'fix_params': config['model']['fixed-parameters'],
        'model': config['model']['model-name'],
        'custom_model': config['model']['custom-model-name'],
        'model_dir': config['model']['model-directory'],
    }
    file: BinaryIO

    # Save worker variables
    with open(os.path.join(ROOT, f'{worker_dir}worker_data.pickle'), 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder validation performance
    print('\nTesting Encoder...')
    data_ = pyxspec_test(
        worker_dir,
        data['ids'],
        data['latent'],
        cpus=cpus,
        job_name='Encoder_output',
        python_path=python,
    )

    # Xspec performance
    # print('\nTesting Xspec...')
    # pyxspec_test(
    #     worker_dir,
    #     data['ids'],
    #     data['targets'],
    #     cpus=cpus,
    #     job_name='Xspec_output',
    #     python_path=python,
    # )

    # Default performance
    # print('\nTesting Defaults...')
    # pyxspec_test(
    #     worker_dir,
    #     data['ids'],
    #     np.repeat([default_params], len(data['ids']), axis=0),
    #     cpus=cpus,
    #     python_path=python,
    # )

    # Allow Xspec optimization
    worker_data['optimize'] = True
    with open(f'{worker_dir}worker_data.pickle', 'wb') as file:
        pickle.dump(worker_data, file)

    # Encoder + Xspec performance
    # print('\nTesting Encoder + Fitting...')
    # pyxspec_test(
    #     worker_dir,
    #     data['ids'],
    #     data['latent'],
    #     cpus=cpus,
    #     job_name='Encoder_Xspec_output',
    #     python_path=python,
    # )

    # Default + Xspec performance
    # print('\nTesting Defaults + Fitting...')
    # pyxspec_test(
    #     worker_dir,
    #     data['ids'],
    #     np.repeat([default_params], len(data['ids']), axis=0),# data['targets'],
    #     cpus=cpus,
    #     job_name='Default_Xspec_output',
    #     python_path=python,
    # )

    return data_

