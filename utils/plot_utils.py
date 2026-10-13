from numpy import ndarray
import numpy as np
from astropy.io import fits
import xspec
import pickle
import re
import os

import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase
from matplotlib.axes import Axes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])

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
    data_dir: str | None = ROOT+'/data/spectra/'

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

def decoder_reconstruction(
    params: ndarray,
    decoder,
    network,
    spectrum: str = 'js_ni0100320101_0mpu7_goddard_GTI0.jsgrp',
    data_dir: str = ROOT+'/data/spectra/',
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
