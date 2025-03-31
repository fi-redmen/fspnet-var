import numpy as np
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase
from matplotlib.axes import Axes

from astropy.io import fits

from fspnet.utils.utils import legend_marker, subplot_grid
from fspnet.utils.plots import _legend
from fspnet.utils.preprocessing import _channel_kev

import sciplots

from xspec import *
import torch
import os

RECTANGLE: tuple[int, int] = (16, 9)
MAJOR: int = 28
MINOR: int = 24
SCATTER_NUM: int = 1000
HI_RES: tuple[int, int] = (32, 18)
ERR_DIST: bool = False  # plot error in the distribution
CAPSIZE: int = 2
PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])

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
    #     x_inc = (10**(max(x)) - 10**(min(x))) * (10**x_margin)
    #     y_inc = (10**(max(y)) - 10**(min(y))) * (10**y_margin)

    #     low_x = np.log10(min(x)-x_inc)
    #     high_x = np.log10(max(x)+x_inc)

    #     low_y = np.log10(min_xy-y_inc)
    #     high_y = np.log10(max_xy+y_inc)

    #     axis.set_xlim(low_x, high_x)
    #     axis.set_ylim(low_y, high_y)
        # Calculate margin in linear space (not log space)

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



def comparison_plot(
    targs: dict,
    lats: dict,
    param_names: str | list[str],
    log_params: int | list[int],
    dir_name: str,
    save_name: str,
    err: str = 'both', # 'none', 'sep', 'both' for no errors, errors on separate plot and errors as errpr bars separately
    cut_err: bool = False,
    lim_valrange = False
    ):

    """
    Plots comparison between input and output parameters

    Parameters
    ----------
    targs : dict
        Target parameters
    lats : dict
        Target uncertainties
    param_names : str | list[str]
        Names of each parameter
    log_params : list[int]
        Defines which parameters are logged
    dir_name : str
        Name of the directory to save plot
    save_name : str
        File name to save plot
    err : dict
        Whether error should be plotted and whether it should be a separate plot
    show : bool
        Whether to show the plot before saving
    cut_err: bool
        Whether to cut out errors
    lim_valrange: bool
        Whether to change limits of plots based on the value range of the inputs
    """

    colours = np.array(['blue'] * SCATTER_NUM, dtype=object)

    axes, fig = _init_subplots(subplot_grid(len(param_names)))

    # add extra dimension if uncertainties arent included
    # if err=='none':
    #     targs = targs[:,None,:SCATTER_NUM]
    # else:

    targs = targs[:,:SCATTER_NUM]

    lats = lats[:,:SCATTER_NUM]

    # Highlight parameters that are maximally or minimally constrained
    for target_param in targs:
        colours[np.argwhere(
            (target_param == np.max(target_param)) |
            (target_param == np.min(target_param))
        )] = 'red'

    # Scatter plot for each parameter
    # For with uncertainties
    for i, (name, axis, targ, lat) in enumerate(zip(
            param_names,
            axes.values(),
            # targs[1,:],
            targs,
            # lats[:,1,:],
            lats)):
    
    #for without unceertainties
    # for i, (name, axis, targ, lat) in enumerate(zip(
    #         param_names,
    #         axes.values(),
    #         targs[:,0,:],
    #         lats[:,0,:])):

        # if i in log_params:     #conversion for log parameters
        #     if no_prop==False:  # if no_prop is false, propagates errors
        #         lat_err = (1/np.log(10)) * (lat_err/lat)
        #         targ_err = (1/np.log(10)) * (targ_err/targ)

        #     lat = np.log10(lat)
        #     targ = np.log10(targ)

        axis.set_xlabel('inputs', fontsize=MINOR)
        axis.set_ylabel('outputs', fontsize=MINOR)
        axis.set_title(name, fontsize=MAJOR)
        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

        # for plotting parameters
        value_range = (np.min(targ), np.max(targ))
        axis.scatter(targ, lat, color=colours, alpha=0.2)

        axis.plot(value_range, value_range, color='k')
        axis.tick_params(labelsize=MINOR)
        plt.xticks(fontsize=MINOR)
        plt.yticks(fontsize=MINOR)
        axis.xaxis.get_offset_text().set_visible(False)
        axis.yaxis.get_offset_text().set_size(MINOR)

        if lim_valrange==True:
            axis.set_xlim(min(value_range), max(value_range)) 
            axis.set_ylim(min(value_range), max(value_range))
        else:
            if cut_err==True:   # makes margins irrespective of errors
                make_margins(axis, x=targ, y=lat)
            else:
                axis.margins(0.05,0.05)

    _legend(legend_marker(['blue', 'red'], ['Free', 'Pegged']), fig)

    plt.savefig(dir_name+save_name, dpi=300)

def comparison_plot_NF(
    data: dict,
    param_names: str | list[str],
    log_params: list[int],
    decoder,
    network,
    dir_name: str,
    save_name: str,
    n_points: int = 100,
    num_specs: int = 3,
    n_recons: int = 3,
    param_limits: list = PARAM_LIMS
    ):
    '''
    Plots:
    1) a random 1000? points (in grey) corresponding to maximum of each parameter distribution
    2) 5? sets of 10? points (in different colours) where each set of points is sampled from different sets of parameter distributions
    3) 1:1 relation
    '''

    axes, fig = _init_subplots(subplot_grid(len(param_names)))

    # grey plot - data for the grey plots has 5 parameters which each have their own 1620 corresponding spectra (limited to SCATTER_NUM)
    grey_targs = data['targets'].swapaxes(0,1).swapaxes(1,2)[0][:,:SCATTER_NUM]
    grey_lats = data['latent'].swapaxes(0,1).swapaxes(1,2)[0][:,:SCATTER_NUM]
    # loop through each set of parameters, plotting (with errorbars?) the outputs vs inputs for SCATTER_NUM points
    for i, (targ, lat, axis) in enumerate(zip(grey_targs, grey_lats, axes.values())):

        # for taking maximum from the distibution rather than frist sample
        # lat = [np.max(data['latent'].swapaxes(1,2).swapaxes(1,0)[i][j]) for j in range(0,SCATTER_NUM)]
        
        axis.plot([np.min(targ), np.max(targ)], [np.min(targ), np.max(targ)], color='k')
        axis.scatter(x=targ, y=lat, linestyle='None', color='grey', alpha=0.3, s=2)
    plt.savefig(dir_name+'NF_comparison_grey.png', dpi=300)

    # multi-color plot
    multi_targs = data['targets'].swapaxes(1,2).swapaxes(0,1)
    multi_lats = data['latent'].swapaxes(1,2).swapaxes(0,1)
    # loop through each set of parameters, plotting, for 5? different spectra, their distributions from 10? data points
    for i, (targ, lat, axis) in enumerate(zip(multi_targs, multi_lats, axes.values())):
        # targ[spectrum number][value or error]

        for spec_num in range(num_specs):
            axis.errorbar(x=[targ[spec_num][0]]*n_points, xerr=[targ[spec_num][1]]*n_points, y=lat[spec_num][:n_points], linestyle='None', capsize=1, ms=7, alpha=0.05, marker='o')

        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

        axis.set_xlabel('inputs', fontsize=MINOR)
        axis.set_ylabel('outputs', fontsize=MINOR)
        axis.set_title(param_names[i], fontsize=MAJOR)

    plt.savefig(dir_name+'NF_comparison.png', dpi=300)

    # for each first num_spec spectra, we want to plot:
    # param_distributions, with eg. 4 samples shown (and then change predicted dist to a dist from xspec)
    # reconstructions for each spectra with each 10? samples plotted on the same graph (and then with compared to what xspec model predicts)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # loop over mun_specs spectra
    for spec_num in range(num_specs):
        # fig, ax = plt.subplot_mosaic('aabbccdddddd\eeefffdddddd', layout='constrained', figsize=(16,9))
        
        dist_targs = data['targets'][spec_num][0]
        dist_targ_errs = data['targets'][spec_num][1]
        dist_lats = data['latent'][spec_num].swapaxes(0,1)

        # samples parameters for reconstruction
        # finds 3? random indexes to index each parameter in latent space
        indexes = np.random.randint(0, len(dist_lats[0]), size=3)
        samples = [[dist_lats[param_num][indexes[sample_number]] for param_num in range(len(param_names))] for sample_number in range(n_recons)]

        # param_distributions, with eg. 4 samples shown (and then change predicted dist to a dist from xspec)
        dist_plots = sciplots.PlotDistributions(data=dist_lats, log=log_params, density=True, norm=True, titles=param_names, bins=200, colours=[colors[spec_num]]*5)

        # loop over parameters to plot true distributions and samples
        for i, (targ, targ_err, axis) in enumerate(zip(dist_targs, dist_targ_errs, dist_plots.axes.values())):
           
            targ_range = np.linspace(param_limits[i][0], param_limits[i][1], 10000)     # finding range of x values for true distribution from latent distribution
            gaussian = np.exp(-( ( (targ_range-targ)**2 ) / (2*(targ_err**2))) )             # calculating gaussian with this error
            axis.plot(targ_range, gaussian, color='grey')
            axis.fill_between(targ_range, gaussian, np.zeros(len(targ_range)), color='grey', alpha=0.3)

            for sample_num, sample in enumerate(torch.tensor(samples).swapaxes(0,1)[i]):
                axis.plot([sample, sample], [0,1], color = colors[sample_num])
        
        plt.savefig(dir_name+'NF_distribution'+str(spec_num)+'.png', dpi=300)


    # reconstructions for each spectra with each 3? samples plotted on the same graph (and then with compared to what xspec model predicts)

    # [Batch_size/total number of spectra, number of data points in spectra, number of parameters]
    inputs = data['inputs']
    for spec_num, (input, axis) in enumerate(zip (inputs, axes.values())):
        fig = plt.figure(figsize=(16,9))
        axis = fig.gca()

        inp = input[0]
        inp_err = input[1]

        # to get specta in terms of energies
        # Open the FITS file
        with fits.open('/Users/astroai/Downloads/spectra/'+data['ids'][spec_num]) as hdul:
            # Print information about the structure of the FITS file
            hdul.info()
            # Extract the data (assuming it is in the second HDU, adjust index if necessary)
            loaded_data = hdul[1].data  # Adjust the HDU index based on your file structure
            channels = loaded_data['CHANNEL']

        bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)
        channels=_binning(channels, bins)

        energies = ((channels * 10) + 5 ) / 1e3

        cut_off = (0.3, 10)
        cut_indices = np.argwhere((energies < cut_off[0]) | (energies > cut_off[1]))
        energies = np.delete(energies, cut_indices)
        
        axis.set_xlabel('Energy (keV)', fontsize=MINOR)
        axis.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        axis.set_xscale('log')
        # axis.set_yscale('log')

        # plotting data points
        axis.errorbar(x=energies[:240],y=inp, yerr=inp_err, linestyle="None", color='k', label='data')
        # plotting xspec reconstruction
        
        Xset.chatter = 0
        Xset.logChatter = 0

        os.chdir('/Users/astroai/Projects/FSPNet/data/spectra/')
        spectrum = Spectrum(data['ids'][spec_num])
        AllData.ignore("bad")
        spectrum.ignore("**-0.3 10.0-**")
        AllModels.lmod('simplcutx', dirPath='/Users/astroai/Downloads/simplcutx/')

        for plot_num, params in enumerate(torch.tensor(samples)):
            margin = np.minimum(1e-6, 1e-6 * (param_limits[:, 1] - param_limits[:, 0]))
            param_min = param_limits[:, 0] + margin
            param_max = param_limits[:, 1] - margin
            params = np.clip(params, a_min=param_min, a_max=param_max)

            # xspec_params = [str(params[param_num].item())+" 0.0" for param_num in range(len(params))]
            xspec_model = Model("tbabs(simplcutx(ezdiskbb))") #, setPars=xspec_params)
            xspec_model.componentNames
            xspec_model.TBabs.nH = str(params[0].item())+" 0.0"
            xspec_model.simplcutx.Gamma = str(params[1].item())+" 0.0"
            xspec_model.simplcutx.FracSctr = str(params[2].item())+" 0.0"
            xspec_model.simplcutx.ReflFrac.frozen = True
            xspec_model.simplcutx.kT_e.frozen = True
            xspec_model.ezdiskbb.T_max = str(params[3].item())+" 0.0"
            xspec_model.ezdiskbb.norm = str(params[4].item())+" 0.0"

            # Fit.nIterations=1
            # Fit.renorm()
            # Fit.perform()

            # Plot("ldata")??
            Plot.xAxis="keV"
            Plot.xLog=True
            # Plot.yLog=False
            Plot("model")
            xs_energies = Plot.x()
            edeltas = Plot.xErr()
            foldedmodel = Plot.model()

            nE = len(xs_energies)
            stepenergies = list()
            for i in range(nE):
                stepenergies.append(xs_energies[i] - edeltas[i])
            stepenergies.append(xs_energies[-1]+edeltas[-1])
            foldedmodel.append(foldedmodel[-1])
            
            axis.step(stepenergies, 10**(np.array(foldedmodel)),where='post', linestyle='--', color=colors[plot_num], label='xspec'+str(plot_num))

        # plotting decoder reconstruction
        sample_transform = network.transforms['targets']
        samples = sample_transform(np.array(samples))

        transform = network.transforms['inputs']
        # clear transforms
        # network.transforms['inputs'] = None
        recons = transform(decoder(samples), back=True)[0]
        #reset transforms
        # network.transforms['inputs'] = transform

        for recon_num, recon in enumerate(recons):
            axis.plot(energies[:240], recon, color=colors[recon_num], label='decoder'+str(recon_num))

        plt.legend()
        plt.savefig(dir_name+'NF_recons'+str(spec_num)+'.png', dpi=300)
        

def distribution_plot(
    targs: dict,
    lats: dict,
    param_names: str | list[str],
    log_params: list[int],
    dir_name: str,
    save_name: str,
    err: str = False, #whether to plot the error distributions instead
    show: bool = False,
    Range: ndarray = None
    ):

    """
    Plots comparison between input and output parameters

    Parameters
    ----------
    targs : dict
        Target parameters
    lats : dict
        Target uncertainties
    param_names : str | list[str]
        Names of each parameter
    log_params : list[int]
        Defines which parameters are logged
    dir_name : str
        Name of the directory to save plot
    save_name : str
        File name to save plot
    err : dict
        Whether error should be plotted and whether it should be a separate plot
    show : bool
        Whether to show the plot before saving
    """

    axes, fig =  _init_subplots(subplot_grid(len(param_names)))

    for i, (name, axis, lat_err, lat, targ_err, targ) in enumerate(zip(
        param_names,
        axes.values(),
        lats[:,1,:],
        lats[:,0,:],
        targs[:,1,:],
        targs[:,0,:])):

        if i in log_params:     # conversion for log parameters
            axis.set_xscale('log')
            axis.set_yscale('log')

        if err==True:
            axis.set_title('std' + name, fontsize=MAJOR)
            # if name=='$\Gamma$':
            #     axis.hist(targ_err, bins=10, label='target', range=[np.min(np.array([lat_err,targ_err])), np.max(np.array([lat_err,targ_err]))])
            
            # else:
            axis.hist(targ_err, bins=10, label='target',  range=[min(lat_err), max(lat_err)])
            axis.hist(lat_err,  bins=10, label='latent', alpha=0.3)

        else:
            axis.set_title('mean' + name, fontsize=MAJOR)
            axis.hist(targ, bins=10, label='target', range=Range)
            axis.hist(lat, alpha=0.3, bins=10, label='latent')

    plt.legend(fontsize=MINOR)
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)

    if show:
        fig.show()

    fig.savefig(dir_name+save_name, dpi=300)

def recon_plot_params(
    inputs: ndarray,
    preds: ndarray,
    ids: ndarray,
    save_name: str,
    dir_name: str,
    lats: dict,
    targs: dict,
    names,
    log: str = 'both'
    ):
    
    """
    Plots comparison between input and output parameters

    Parameters
    ----------
    Inputs : dict
        Input spectra
    preds : dict
        Reconstructed spectra
    """

    axes, fig =  _init_subplots(subplot_grid(2)) #, gridspec_kw = {'figure': {'tight_layout':True}})

    fig.suptitle('Reconstructed Spectra \n', fontsize=MAJOR)

    for i, (axis, pred, input, input_err, lat, targ, id) in enumerate(zip(
        axes.values(),
        preds[:,0,:],
        inputs[:,0,:],
        inputs[:,1,:],
        lats[:,0,:],
        targs[:,0,:],
        ids
        )):

        # convert x axis to energy:
        # energy = _channel_kev(np.arange(len(input)))

        # Fetch data from PyXspec
        # background = pd.DataFrame(xspec.AllData(1).background.values, columns=['RATE'])
        # spectrum = pd.DataFrame(np.stack((
        # np.arange(background['RATE'].size),
        # xspec.AllData(1).values,
        # ), axis=1), columns=['CHANNEL', 'RATE'])

        # # Initialize variables
        # bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)

        # if not cut_off:
        #     cut_off = (0.3, 10)

        # Open the FITS file
        with fits.open('/Users/astroai/Downloads/spectra/'+id) as hdul:
            # Print information about the structure of the FITS file
            hdul.info()

            # Extract the data (assuming it is in the second HDU, adjust index if necessary)
            data = hdul[1].data  # Adjust the HDU index based on your file structure

            # Assuming the data has a 'CHANNEL' column (check the structure using hdul.info())
            channels = data[data['QUALITY'] == 2]['CHANNEL']

        # Pre binned data
        energy = _channel_kev(channels[26:])
        
        axis.set_xlabel('Energy (keV)', fontsize=MINOR)
        axis.set_ylabel('cts / det / s/ keV', fontsize=MINOR) 

        if log=='log':
            axis.set_yscale('log')

        elif log=='both':
            if i==0:
                axis.set_yscale('log')

        axis.errorbar(energy, input, yerr=input_err, marker='o', markersize=3, capsize=CAPSIZE, linestyle='None', label='input')
        axis.scatter(energy, pred, label='reconstruction', s=5, c='#ff7f0e')

        #set text for parameters of given plot
        str_names = ''
        str_targ = 'target'
        str_lat = 'latent'
        for i, (name, l , t) in enumerate(zip(
            names,
            lat,
            targ
            )):
            str_names+='\n'+name+':'
            str_targ+='\n'+str(round(t,3))
            str_lat+='\n'+str(round(l,3))

        axis.text(-0.05, 1.05, str_names, verticalalignment='bottom', horizontalalignment='left', 
                   transform=axis.transAxes, fontsize=MINOR)
        axis.text(0.4, 1.05, str_targ, verticalalignment='bottom', horizontalalignment='left', 
                   transform=axis.transAxes, fontsize=MINOR)
        axis.text(0.7, 1.05, str_lat, verticalalignment='bottom', horizontalalignment='left', 
                   transform=axis.transAxes, fontsize=MINOR)
        
        axis.tick_params(axis='x', labelsize=MINOR)
        axis.tick_params(axis='y', labelsize=MINOR)
        axis.margins(0.05, 0.05)

        if i==1:
            axis.legend(fontsize=MINOR)

    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)

    fig.savefig(dir_name+save_name, dpi=300)

# def recon_plot_params_NF(
#     data: dict,
#     samples: int,
#     dir_name: str
#     ):

    


    

def recon_plot(
    inputs: dict,
    preds: dict,
    ids: ndarray,
    save_name: str,
    dir_name: str,
    log_plot: str = 'both'
    ):
    
    """
    Plots comparison between input and output parameters

    Parameters
    ----------
    Inputs : dict
        Input spectra
    preds : dict
        Reconstructed spectra
    """

    axes, fig =  _init_subplots(subplot_grid(4)) #, gridspec_kw = {'figure': {'tight_layout':True}})

    fig.suptitle('Reconstructed Spectra \n', fontsize=MAJOR)

    for i, (axis, pred, input, id) in enumerate(zip(
        axes.values(),
        preds,
        inputs,
        # inputs[:,1,:],
        ids
        )):

        # convert x axis to energy:
        # energy = _channel_kev(np.arange(len(input)))

        # Fetch data from PyXspec

        # # Initialize variables
        bins = np.array([[0, 20, 248, 600, 1200, 1494, 1500], [2, 3, 4, 5, 6, 2, 1]], dtype=int)


        # if not cut_off:
        #     cut_off = (0.3, 10)

        # Pre binned data

        # energy = _channel_kev(np.arange(20,len(input)+20))

        # Initialize variables
 
        # Open the FITS file
        with fits.open('/Users/astroai/Downloads/spectra/'+id) as hdul:
            # Print information about the structure of the FITS file
            hdul.info()

            # Extract the data (assuming it is in the second HDU, adjust index if necessary)
            data = hdul[1].data  # Adjust the HDU index based on your file structure

            # Assuming the data has a 'CHANNEL' column (check the structure using hdul.info())
            # channels = data[data['QUALITY'] == 2]['CHANNEL']
            channels = data['CHANNEL']

        #energies = _channel_kev(channels)

        channels=_binning(channels, bins)

        energies = ((channels * 10) + 5 ) / 1e3

        cut_off = (0.3, 10)
        cut_indices = np.argwhere((energies < cut_off[0]) | (energies > cut_off[1]))
        energies = np.delete(energies, cut_indices)
        
        axis.set_xlabel('Energy (keV)', fontsize=MINOR)
        axis.set_ylabel('cts / det / s/ keV', fontsize=MINOR) 

        if log_plot=='log':
            axis.set_yscale('log')

        elif log_plot=='both':
            if i==0 or i==1:
                axis.set_yscale('log')

            #axis.set_xscale('log')

        # for without uncertainties
        axis.errorbar(energies[0:240], input[0], yerr=input[1], label='input', marker='o', markersize=3,  capsize=CAPSIZE, linestyle='None')
        axis.scatter(energies[0:240], pred, label='reconstruction', s=5, c='#ff7f0e')
        # for with uncertainties 
        # axis.scatter(energies[0:240], input, label='input', s=5, c='b')
        
        axis.tick_params(axis='x', labelsize=MINOR)
        axis.tick_params(axis='y', labelsize=MINOR)

        axis.margins(0.05, 0.05)

        if i==1:
            axis.legend(fontsize=MINOR)

        plt.xticks(fontsize=MINOR)
        plt.yticks(fontsize=MINOR)

    fig.savefig(dir_name+save_name, dpi=300)