from numpy import ndarray
import numpy as np
from astropy.io import fits
import xspec
import pickle
import re

import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase
from matplotlib.axes import Axes


RECTANGLE: tuple[int, int] = (16, 9)
PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])

def _init_subplots(
        subplots: str | tuple[int, int] | list | ndarray,
        fig: FigureBase | None = None,
        fig_size: tuple[int, int] = RECTANGLE,
        **kwargs: any) -> tuple[dict[str, Axes] | ndarray[Axes], FigureBase]:
    """
    Generates subplots within a figure or sub-figure

    Parameters
    ----------
    subplots : str | tuple[int, int] | list | ndarray
        Parameters for subplots or subplot_mosaic
    fig : FigureBase | None, default = None
        FigureBase to add subplots to
    fig_size : tuple[integer, integer]
        Size of the figure, only used if fig is None

    **kwargs
        Optional kwargs to pass to subplots or subplot_mosaic

    Returns
    -------
    tuple[dict[str, Axes] | ndarray[Axes], FigureBase]
        Dictionary or array of subplot axes and figure
    """
    axes: dict[str, Axes] | ndarray[Axes]

    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

    # Plots either subplot or mosaic
    if isinstance(subplots, tuple):
        axes = fig.subplots(*subplots, **kwargs)
    else:
        axes = fig.subplot_mosaic(subplots, **kwargs)

    return axes, fig

def make_margins(
    axis: Axes,
    x: tuple | ndarray | list,
    y: tuple | ndarray | list,
    x_margin: int = 0.05,
    y_margin: int = 0.05,
    log_plot: bool = False
    ):

    """
    Sets x and y lims to give margins irrespective of error bars

    Parameters
    ----------
    targs : dict
        x values on plot - can have more dimensions for each set of data plotted
    lats : dict
        y values on plot - can have more dimensions for each set of data plotted
    x_margin : int
        Percentage of margin space vs. data on x scale
    y_,argin : int
        Percentafe of margin space vs. data on y scale
    """

    min_xy = np.min(np.array([x,y]))
    max_xy = np.max(np.array([x,y]))

    min_x = np.min(x)
    min_y = np.min(y)

    max_x = np.max(x)
    max_y = np.max(y)

    if log_plot:
        # Add margin to data range in linear space
        x_inc = (max_x - min_x) * x_margin
        y_inc = (max_y - min_y) * y_margin

        # Set the limits to be larger than the data range
        low_x = min_x - x_inc
        high_x = max_x + x_inc
        low_y = min_y - y_inc
        high_y = max_y + y_inc

        # Ensure that the limits are all positive for log scaling
        low_x = max(low_x, 1e-10)  # Avoid log(0) or log(negative)
        low_y = max(low_y, 1e-10)

        # Apply log scaling 
        axis.set_xscale('log')
        axis.set_yscale('log')

        # Set the limits in log space
        axis.set_xlim([low_x, high_x])
        axis.set_ylim([low_y, high_y])

    else:
        x_inc = (max_x - min_x) * x_margin
        y_inc = (max_y - min_y) * y_margin

        axis.set_xlim(min_x-x_inc, max_x+x_inc)
        axis.set_ylim(min_xy-y_inc, max_xy+y_inc)

def _create_bin(
        x_data: ndarray,
        clow: float,
        chi: float,
        nchan: int) -> tuple[ndarray, ndarray]:
    """
    Bins x & y data

    Removes data of bad quality defined as 1 [Not yet implemented]

    Parameters
    ----------
    x_data : ndarray
        x data that will be averaged per bin
    y_data : ndarray
        y data that will be summed per bin
    clow : float
        Lower limit of data that will be binned
    chi : float
        Upper limit of datat that will be binned
    nchan : int
        Number of channels per bin

    Returns
    -------
    (ndarray, ndarray)
        Binned (x, y) data
    """
    x_data = x_data[clow:chi]\

    x_data = np.mean(x_data.reshape(-1, int(nchan)), axis=1)

    return x_data

def _binning(x_data: ndarray, bins: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Bins data to match binning performed in Xspec

    Parameters
    ----------
    x_data : ndarray
        x data to be binned
    y_data : ndarray
        y data to be binned
    bins : ndarray
        Array of bin change index & bin size

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Binned x & y data & bin energy per data point
    """
    # Initialize variables
    x_bin = np.array(())

    # Bin data
    for i in range(bins.shape[1] - 1):
        x_new = _create_bin(x_data, bins[0, i], bins[0, i + 1], bins[1, i])
        x_bin = np.append(x_bin, x_new)
    
    return x_bin

def get_energies(
    spectrum: str | None = 'js_ni0100320101_0mpu7_goddard_GTI0.jsgrp',
    data_dir: str | None = '/Users/work/Projects/FSPNet/data/spectra/'

):
    # to get energies for spectra
    with fits.open(data_dir+spectrum) as hdul:
        # extract the data
        loaded_data = hdul[1].data
        channels = loaded_data['CHANNEL']
    
    # binning data as Ethan did
    bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)
    channels=_binning(channels, bins)
    energies = ((channels * 10) + 5 ) / 1e3
    cut_off = (0.3, 10)
    cut_indices = np.argwhere((energies < cut_off[0]) | (energies > cut_off[1]))
    energies = np.delete(energies, cut_indices)

    return energies

def xspec_reconstruction(
    xspec_params: ndarray,
    spectrum: str | int,
    data_dir: str | None = '/Users/work/Projects/FSPNet/data/',
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
        xspec.AllModels.lmod('simplcutx', dirPath='/Users/work/Projects/FSPNet/simplcutx/')
    except Exception as e:
        xspec.AllModels.tclLoad('/Users/work/Projects/FSPNet/simplcutx/libjscutx.dylib')

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

def decoder_reconstruction(
    params: ndarray,
    decoder,
    network,
    spectrum: str = 'js_ni0100320101_0mpu7_goddard_GTI0.jsgrp',
    data_dir: str = '/Users/work/Projects/FSPNet/data/spectra/',
):
    '''
    Obtains decoder reconstructions from given parameters

    Parameters
    ----------
    params: ndarray
        Parameters to create reconstructions with
    decoder:
        Decoder
    network:
        Network
    spectrum: str
        Name of spectrum file
    data_dir: str
        Directory of spectrum file
    '''
    
    # transforms sampled parameters for decoder to take
    sample_transform = network.transforms['targets']
    transformed_param_samples = sample_transform(np.array([params]))

    # makes and transforms decoder reconstruction in normal spectral space
    transform = network.transforms['inputs']
    dec_recon = transform(decoder(transformed_param_samples), back=True)[0]

    return dec_recon
    
def quantile_limits(
    fig,
    min_quant, 
    max_quant, 
    param_names,
    param_lims = PARAM_LIMS,
    NF_data = None,
    xspec_gauss_data = None,
    xspec_MCMC_data = None,
    object_name = None
    ):

    # creating array out of all input dataframes to find minimum value in
    all_data = []
    if NF_data is not None:
        all_data.append(np.array(NF_data))
    if xspec_gauss_data is not None:
        all_data.append(np.array(xspec_gauss_data))
    if xspec_MCMC_data is not None:
        all_data.append(np.array(xspec_MCMC_data))

    # looping over bottom and left plots, setting the limits for each
    for bottom_plot_num in range(0,len(param_names)):
                for left_plot_num in range(0,len(param_names)):
                    if bottom_plot_num!=left_plot_num:

                        y_min = np.max([np.min([np.quantile(all_data[i][:,bottom_plot_num], min_quant) for i in range(len(all_data))]),
                                        param_lims[bottom_plot_num][0]])
                        
                        y_max = np.min([np.max([np.quantile(all_data[i][:,bottom_plot_num], max_quant) for i in range(len(all_data))]),
                                        param_lims[bottom_plot_num][1]])

                        x_min = np.max([np.min([np.quantile(all_data[i][:,left_plot_num], min_quant) for i in range(len(all_data))]),
                                        param_lims[left_plot_num][0]])

                        x_max = np.min([np.max([np.quantile(all_data[i][:,left_plot_num], max_quant) for i in range(len(all_data))]),
                                        param_lims[left_plot_num][1]])

                        # y_min = np.min([np.quantile(all_data[i][:,bottom_plot_num], min_quant) for i in range(len(all_data))])
                        
                        # y_max = np.max([np.quantile(all_data[i][:,bottom_plot_num], max_quant) for i in range(len(all_data))])

                        # x_min = np.min([np.quantile(all_data[i][:,left_plot_num], min_quant) for i in range(len(all_data))])

                        # x_max = np.max([np.quantile(all_data[i][:,left_plot_num], max_quant) for i in range(len(all_data))])

                        # # extra condition I made for MAXI J1535 - could maybe try to generalise this code..?
                        # if left_plot_num == 1 and object_name == 'MAXI J1535-571':
                        #     x_max = 1.45
                        # if bottom_plot_num == 1 and object_name == 'MAXI J1535-571':
                        #     y_max = 1.45

                        fig.axes[bottom_plot_num*5+left_plot_num].set_ylim(y_min, y_max)
                        fig.axes[bottom_plot_num*5+left_plot_num].set_xlim(x_min, x_max)

                    else: 
                        x_min = np.max([np.min([np.quantile(all_data[i][:,bottom_plot_num], min_quant) for i in range(len(all_data))]),
                                        param_lims[bottom_plot_num][0]])
                        x_max = np.min([np.max([np.quantile(all_data[i][:,bottom_plot_num], max_quant) for i in range(len(all_data))]),
                                        param_lims[bottom_plot_num][1]])

                        # x_min = np.min([np.quantile(all_data[i][:,bottom_plot_num], min_quant) for i in range(len(all_data))])
                        
                        # x_max = np.max([np.quantile(all_data[i][:,bottom_plot_num], max_quant) for i in range(len(all_data))])

                        if left_plot_num == 1 and object_name == 'MAXI J1535-571':
                            x_max = 1.45

                        fig.axes[bottom_plot_num*5+left_plot_num].set_xlim(x_min, x_max)
