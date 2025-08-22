import numpy as np
from numpy import ndarray
import xspec
import os
from astropy.io import fits

PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])

def sample(
    data: dict,
    param_limits: ndarray = PARAM_LIMS,
    num_specs = 1,
    num_samples = 1,
    spec_scroll=0
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

    all_samples = []
    for spec_num in range(spec_scroll,num_specs+spec_scroll):
        # gets num_samples sets of parameter samples from dist_lats
        dist_lats = data['latent'][spec_num]
        samples=[]
        indexes = np.random.randint(0, len(dist_lats[1]), size=num_samples)
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

def reduced_PG(
    params,
    spec_name,
    param_limits: ndarray = PARAM_LIMS,
    data_dir = '/Users/astroai/Projects/FSPNet/data/spectra/'
    ):

    # params = [2.0, 2.5, 2.0e-2, 1.0, 1.0]  # Example parameters for the model

    os.chdir(data_dir)

    # with fits.open(spec_name) as file:
    #     spectrum_info = file[1].header

    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    # Load spectrum and model
    xspec.Spectrum(spec_name)
    xspec.AllModels.lmod("simplcutx", "/Users/astroai/Downloads/simplcutx/")

    # settings for xspec
    xspec.AllModels.setEnergies("0.003  300. 1000 log")
    xspec.Plot.xAxis="keV"
    xspec.Plot.background = True
    xspec.AllModels.systematic = 0.
    xspec.Fit.statMethod = 'pgstat'
    xspec.Xset.abund = "wilm"

    # make reconstruction from model
    xspec_model = xspec.Model("tbabs(simplcutx(ezdiskbb))")
    xspec_model.setPars(list(np.concat([params[:3], [0.0, 100.0], params[3:]])) )
    xspec.AllData.ignore("**-0.3 10.0-**")

    value = xspec.Fit.statistic / xspec.Fit.dof

    xspec.AllData.clear()
    xspec.AllModels.clear()

    return value