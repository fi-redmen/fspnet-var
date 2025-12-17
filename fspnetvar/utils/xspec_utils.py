
import xspec
from astropy.io import fits
import numpy as np
from numpy import ndarray
import os
import re
import pickle


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])

def reduced_PG(
    params,
    spec_name,
    param_limits: ndarray = PARAM_LIMS,
    data_dir = ROOT+'/data/spectra/'
    ):

    # params = [2.0, 2.5, 2.0e-2, 1.0, 1.0]  # Example parameters for the model

    os.chdir(data_dir)

    # with fits.open(spec_name) as file:
    #     spectrum_info = file[1].header

    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    # Load spectrum and model
    xspec.Spectrum(spec_name)
    try:
        xspec.AllModels.lmod('simplcutx', dirPath=ROOT+'/simplcutx/')
    except Exception as e:
        xspec.AllModels.tclLoad(ROOT+'/simplcutx/libjscutx.dylib')

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

def xspec_reconstruction(
    xspec_params: ndarray,
    spectrum: str | int,
    data_dir: str | None = ROOT+'/data/',
    ):
    
    """
    Reconstructs the spectra from parameters using xspec.
    Parameters
    ----------
    xspec_params : ndarray
        Parameters for xspec
    decoder_params : ndarray
        Parameters in the latent space
    decoder : Module
        Decoder used to reconstruct the spectra
    Returns
    -------
    ndarray
        Reconstructed spectra
    """

    # prevents xspec from printing to console
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    # gets info from provided spectrum file
    if type(spectrum) is str or type(spectrum) is not int: # could maybe change to os.path_exists..??
        data_dir+= 'spectra/'
        with fits.open(data_dir+ spectrum) as file:
            spectrum_info = file[1].header

    else:
        with open(data_dir+'synth_spectra_clean.pickle', 'rb') as file:
            synthetic_data = pickle.load(file)
            spectrum_info = {
                'RESPFILE': synthetic_data['info'][spectrum].response[5:],
                'ANCRFILE': synthetic_data['info'][spectrum].arf[5:],
                'BACKFILE': synthetic_data['info'][spectrum].background[5:],
                'EXPOSURE': synthetic_data['info'][spectrum].exposure
            }

    # det_num = int(spectrum_info['RESPFILE'][7:9])
    # det_num=48
    det_num=int(re.search(r'_d(\d+)', spectrum_info['RESPFILE']).group(1))

    # fakeit settings
    fake_base = xspec.FakeitSettings(
    response=data_dir + spectrum_info['RESPFILE'],
    arf=data_dir + spectrum_info['ANCRFILE'],
    background=data_dir + spectrum_info['BACKFILE'],
    exposure=spectrum_info['EXPOSURE'])

    # creates list of xspec parameters - includes fixed values.
    xspec_params = list(np.concat([xspec_params[:3], [0.0, 100.0], xspec_params[3:]])) 

    # loads the simplcutx model
    try:
        xspec.AllModels.lmod('simplcutx', dirPath=ROOT+'/simplcutx/')
    except Exception as e:
        xspec.AllModels.tclLoad(ROOT+'/simplcutx/libjscutx.dylib')

    # settings for xspec
    xspec.AllModels.setEnergies("0.003  300. 1000 log")
    # xspec.AllModels.setEnergies("0.001 1000. 1500 log")
    xspec.Plot.xAxis="keV"
    xspec.Plot.background = True
    xspec.AllModels.systematic = 0.
    xspec.Fit.statMethod = 'pgstat'
    xspec.Xset.abund = "wilm"

    # make reconstruction from model
    xspec_model = xspec.Model("tbabs(simplcutx(ezdiskbb))")
    xspec_model.setPars(xspec_params)
    xspec.AllData.fakeit(1, settings=fake_base, applyStats=False, noWrite=True)
    xspec.AllData.ignore("**-0.3 10.0-**")

    # plot reconstruction in xspec
    xspec.Plot.xLog=True
    xspec.Plot("data")

    # get data from plot
    xs_energies = xspec.Plot.x()
    xs_recon = np.array(xspec.Plot.y())/det_num

    return xs_energies, xs_recon