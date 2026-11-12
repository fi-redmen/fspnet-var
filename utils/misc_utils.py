import numpy as np
from numpy import ndarray
import xspec
import os
from typing import Any

from fspnet.utils.utils import open_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_DIR: str = ROOT+'/data/synth_spectra_clean.pickle'

def sample(
    data: dict,
    num_specs = 1,
    num_samples = 1,
    spec_scroll=0,
    config: str | dict[str, Any] = './config.yaml',
    ):
    '''
    Samples from given distribution (made up of many samples)

    Parameters
    ----------
    data: dict
        Data containing whole distribution to be sampled from
    num_specs:
        Number of spectra to loop over

    Returns
    -------
    all_samples: list
        Samples accross spectra, number of samples we want for that spectrum, and the number of parameter distributions we are sampling from
        - Shape: spectrum number, sample number, parameter number
    '''

    if isinstance(config, str):
        _, config = open_config('synthesize-spectra', os.path.join(ROOT, config))

    param_limits = np.array([list(lim.values()) for lim in config['model']['parameter-limits']])

    all_samples = []
    for spec_num in range(spec_scroll,num_specs+spec_scroll):
        # gets num_samples sets of parameter samples from dist_lats
        dist_lats = data['latent'][spec_num]
        samples=[]
        indexes = np.random.randint(0, dist_lats.shape[0], size=num_samples)
        for i in indexes:
            params = dist_lats[i]
            # if param_limits are given, applies them
            if param_limits is not None:
                margin = np.minimum(1e-6, 1e-6 * (param_limits[:, 1] - param_limits[:, 0]))
                param_min = param_limits[:, 0] + margin
                param_max = param_limits[:, 1] - margin
                params = np.clip(params, a_min=param_min, a_max=param_max)

            samples.append(params)

        all_samples.append(samples)

    return all_samples

def get_energy_widths(

):
    bin_limits = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)

    bins = np.array([], dtype=int)
    for i in range(len(bin_limits[0])-1):
        bins = np.concatenate((bins, np.arange(bin_limits[0][i], bin_limits[0][i+1], bin_limits[1][i])))
    bins=np.concatenate([bins, [1500]])

    energy_bins = ((bins * 10) + 5 ) / 1e3
    cut_off = (0.3, 10)
    cut_indices_min = np.argwhere((energy_bins < cut_off[0]))
    cut_indices_plus = np.argwhere((energy_bins > cut_off[1]))[1:]
    energy_bins = np.delete(energy_bins, np.concatenate((cut_indices_min, cut_indices_plus)))
    energy_width = np.diff(energy_bins)

    return energy_width


def add_state_labels(data: dict,
    feature = ['targets', None],
    save_dir: str = None,
    save_name: str = None):
    """
    Looks at either the targets or latent to determine and label the spectrum with the state of the BHB in given observation

    Parameters
    ----------
    data: dict
        Dictionary containing the data to label
    feature: str
        Whether to use latent (['latent', np.mean], ['latent', np.median] or ['latent', np.mode]) or targets (['targets']) to classify state. Default: targets
    save_dir: str
        Where to save the labelled spectra to (if not provided, will not save). Default: no save

    Returns
    -------
    labelled_data: dict
        Data labelled using our classification
    """

    all_params = feature[1](data[feature[0]], axis=1) if feature[1] else data[feature[0]][:,0,:]
    labelled_data=data.copy()
    labelled_data['state']=[]

    # groups different spectra based on parameters
    for params in all_params:
        gamma = params[1] # Names parameter to make sure we are using the correct values
        fsc = params[2]
        kT = params[3]

        if gamma>2 and fsc<0.1 and kT>0.3:
            labelled_data['state'].append('Thermal')
        elif gamma<2.1 and fsc>0.3 and kT<0.5:
            labelled_data['state'].append('Hard')
        elif 2.5<gamma<3 and 0.1<fsc<0.6 and kT>0.5:
            labelled_data['state'].append('SPL')
        else:
            labelled_data['state'].append('Misc')

    # makes into array of strings of varying length (to allow for appending ' Pegged') to idexes indexed by list
    labelled_data['state'] = np.array(labelled_data['state'], dtype=np.dtypes.StringDType())

    # adds ' Pegged' to state if one of the parameters of the observation is at its maximum or minimum
    pegged_idxs = []
    for params in all_params.swapaxes(0,1): # params shape = 2160
        pegged_idxs.append(list(np.argwhere(
            (params == np.max(params)) |
            (params == np.min(params))).flatten()))

    # flatten list and remove duplicates 
    pegged_idxs = list(set([pegged_idx for pegged_idxss in pegged_idxs for pegged_idx in pegged_idxss]))  

    # renames pegged indexes to pegged
    labelled_data['state'][pegged_idxs] += ' Pegged'


    if save_dir and save_name:
        with open(os.path.join(save_dir, save_name), 'wb') as file:
            pickle.dump(labelled_data, file)
    
    return labelled_data

def order_by_state(data: dict,
    order: list = ['Thermal', 'Hard', 'SPL', 'Misc', 
                   'Thermal Pegged', 'Hard Pegged', 'SPL Pegged', 'Misc Pegged']):
    """
    Orders data by state

    Parameters
    ----------
    data: dict

    """
    # makes sure data is labelled
    labelled_data = data if 'state' in data else add_state_labels(data)
    
    state = labelled_data['state']
    def sort_key(i):
        try:
            return order.index(state[i])
        except ValueError:
            return len(order)  # unknown state go to the end

    perm = sorted(range(len(state)), key=sort_key)

    ordered_data = {}
    for k, v in labelled_data.items():
        if isinstance(v, np.ndarray):
            ordered_data[k] = v[perm]
        elif isinstance(v, list):
            ordered_data[k] = [v[i] for i in perm]
        else:
            ordered_data[k] = v

    return ordered_data
