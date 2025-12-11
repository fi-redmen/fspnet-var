
import xspec
from astropy.io import fits
import numpy as np
import os

def recon(
    spectrum_dir : str,
    spectrum_name : str,
    params : np.ndarray,
    param_limits : np.ndarray = None,
):
    '''
    makes a reconstruction using the xspec model

    Parameters
    ----------
    spectrum_dir : str
        directory containing the spectrum we want to reconstruct
    spectrum_name : str
        name of the spectrum we want to reconstruct
    params : np.ndarray
        parameters to be used for the xspec model
    param_limits : np.ndarray
        limits for the parameters to be used for the xspec model
    '''

    # plotting xspec reconstruction
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    # Load the spectrum information
    with fits.open(os.join(spectrum_dir,spectrum_name)) as file:
        spectrum_info = file[1].header

    fake_base = xspec.FakeitSettings(
    response=spectrum_dir + spectrum_info['RESPFILE'],
    arf=spectrum_dir + spectrum_info['ANCRFILE'],
    background=spectrum_dir + spectrum_info['BACKFILE'],
    exposure=spectrum_info['EXPOSURE'])

    # loads the simplcutx model for xspec reconstruction
    xspec.AllModels.lmod("simplcutx", "/Users/astroai/Downloads/simplcutx/")

    # det_num = int(spectrum_info['RESPFILE'][7:9])
    det_num=48

    if param_limits is not None:
        margin = np.minimum(1e-6, 1e-6 * (param_limits[:, 1] - param_limits[:, 0]))
        param_min = param_limits[:, 0] + margin
        param_max = param_limits[:, 1] - margin
        params = np.clip(params, a_min=param_min, a_max=param_max)

    # plotting xspec reconstruction
    xspec_params = list(np.concat([params[:3], [0.0, 100.0], params[3:]])) 
    xspec.AllModels.setEnergies("0.001 1000. 1500 log")
    xspec.Plot.xAxis="keV"
    xspec.Plot.background = True
    xspec.AllModels.systematic = 0.01
    xspec.Xset.abund = "wilm"
    xspec_model = xspec.Model("tbabs(simplcutx(ezdiskbb))") #, setPars=xspec_params)
    xspec_model.setPars(xspec_params)
    xspec.AllData.fakeit(1, settings=fake_base, applyStats=False, noWrite=True)
    xspec.AllData.ignore("**-0.3 10.0-**")
    xspec.Plot.xLog=True
    xspec.Plot("data")
    energy = xspec.Plot.x()
    recon = np.array(xspec.Plot.y())/det_num

    return energy, recon

# def true_posterior(
#     spectrum_dir : str,
#     spectrum_name : str,
#     params : np.ndarray,
#     ):
#     '''
#     makes the posterior from MCMC sampling after fitting an xspec model
#     '''

