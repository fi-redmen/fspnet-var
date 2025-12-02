import numpy as np
import pandas as pd
import torch
import os
import lampe
import pickle

from numpy import ndarray
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as pltcolors
from matplotlib.colors import LogNorm

from chainconsumer import Chain, ChainConsumer, PlotConfig

from fspnet.utils.plots import plot_param_pairs, _plot_histogram

from NF_utils.plot_utils import get_energies, xspec_reconstruction, decoder_reconstruction, quantile_limits
from NF_utils.misc_utils import sample, reduced_PG

RECTANGLE: tuple[int, int] = (16, 9)
MAJOR: int = 32
MINOR: int = 28
PCC: int = 24
SCATTER_NUM: int = 1000
HI_RES: tuple[int, int] = (32, 18)
ERR_DIST: bool = False  # plot error in the distribution
CAPSIZE: int = 2
LOG_PARAMS: list[int] = [0, 2, 3, 4]  # parameters to be plotted in log scale
PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])
PARAM_NAMES : ndarray = np.array(['$N_{H}$ $(10^{22}\ cm^{-2})$', '$\Gamma$', '$f_{sc}$','$kT_{disk}$ $(keV)$','$N$'])
COLORS_LIST = ['#0C5DA5']*17 #'#00B945', '#FF9500', '#9159ab', '#00A7C6'] # to colour different spectr differently
SYNTHETIC_DIR: str = '/Users/work/Projects/FSPNet/data/synth_spectra_clean.pickle'

NAMES = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp','js_ni0103010102_0mpu7_goddard_GTI0.jsgrp','js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
         'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp','js_ni1100320119_0mpu7_goddard_GTI26.jsgrp','js_ni1200120203_0mpu7_goddard_GTI0.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI10.jsgrp','js_ni1200120203_0mpu7_goddard_GTI11.jsgrp','js_ni1200120203_0mpu7_goddard_GTI13.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI1.jsgrp','js_ni1200120203_0mpu7_goddard_GTI3.jsgrp','js_ni1200120203_0mpu7_goddard_GTI4.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI5.jsgrp','js_ni1200120203_0mpu7_goddard_GTI6.jsgrp','js_ni1200120203_0mpu7_goddard_GTI7.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI8.jsgrp','js_ni1200120203_0mpu7_goddard_GTI9.jsgrp']
OBJECT_NAMES = ['Cyg X-1 (2017)','GRS 1915+105','LMC X-3','MAXI J1535-571','Cyg X-1 (2018)','MAXI J1820 0','MAXI J1820 10','MAXI J1820 11','MAXI J1820 13','MAXI J1820 1','MAXI J1820 3','MAXI J1820 4','MAXI J1820 5','MAXI J1820 6','MAXI J1820 7','MAXI J1820 8','MAXI J1820 9']
COLORS_DICT: dict = {}
MIN_QUANT: dict = {
    NAMES[0]: 0.005,
    NAMES[1]: 0.005,
    NAMES[2]: 0.0001,
    NAMES[3]: 0.001,
    NAMES[4]: 0.001}
MAX_QUANT: dict = {
    NAMES[0]: 0.995,
    NAMES[1]: 0.995,
    NAMES[2]: 0.9999,
    NAMES[3]: 0.999,
    NAMES[4]: 0.999}
for key in NAMES:
    COLORS_DICT[key] = COLORS_LIST[NAMES.index(key)]
    if NAMES.index(key) >=5:
        MIN_QUANT[key] = 1-0.9975
        MAX_QUANT[key] = 0.9975

def line(x, m, c):
    return m*x+c

def line_justm(x, m):
    return m*x

def log_line_justm(x, m):
    return (x **(m))

def log_line(x, m, c):
    return (10**c) * (x **(m))

def performance_plot(
        y_label: str,
        loss_fns: dict,
        log_y: bool = True,
        plots_dir: str | None = None,
        save_name: str | None = 'performance.png') -> None:
    """
    Plots training and validation performance as a function of epochs

    Parameters
    ----------
    y_label : string
        Performance metric
    loss_fns : dict
        Validation performance
    log_y : boolean, default = True
        If y-axis should be logged
    plots_dir : string
        Directory to save plots
    plots_dir : str, default = None
        Directory to save plots
    save_name : string, default = 'performance.png'
        Name to save plot as
    """
    plt.figure(figsize=RECTANGLE, constrained_layout=True)

    text = ''
    total_loss_fn = np.zeros(len(loss_fns['reconstruct']))

    # makes function positive so logging possible
    move = abs(min([min(v) if v else 0 for k, v in loss_fns.items()]))+1
    loss_fns['reconstruct'] = list(np.array(loss_fns['reconstruct'])+move)
    loss_fns['latent'] = list(np.array(loss_fns['latent'])+move)
    loss_fns['flow'] = list(np.array(loss_fns['flow'])+move)
    loss_fns['bound'] = list(np.array(loss_fns['bound'])+move)

    # plotting loss function components if they contribute, and adding to total loss
    if loss_fns['reconstruct'] and np.array(loss_fns['reconstruct']).all()!=0:
        plt.plot(loss_fns['reconstruct'], label='Reconstruction')
        text+=f"Final recon: {loss_fns['reconstruct'][-1]:.3e} \n"
        total_loss_fn += np.array(loss_fns['reconstruct'])
    if loss_fns['flow']:
        plt.plot(loss_fns['flow'], label='Flow')
        text+=f"Final flow: {loss_fns['flow'][-1]:.3e} \n"
        total_loss_fn += np.array(loss_fns['flow'])
    if loss_fns['latent']:
        plt.plot(loss_fns['latent'], label='Latent')
        text+=f"Final latent: {loss_fns['latent'][-1]:.3e} \n"
        total_loss_fn += np.array(loss_fns['latent'])
    if loss_fns['bound']:
        plt.plot(loss_fns['bound'], label='Bound')
        text+=f"Final bound: {loss_fns['bound'][-1]:.3e} \n"
        total_loss_fn += np.array(loss_fns['bound'])
    if loss_fns['kl']:
        plt.plot(loss_fns['kl'], label='KL')
        text+=f"Final KL: {loss_fns['kl'][-1]:.3e} \n"
        total_loss_fn += np.array(loss_fns['kl'])

    plt.plot(total_loss_fn, label='Total Loss', color='k')

    # plt.plot(val, label='Validation ---')
    plt.xticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR)
    plt.yticks(fontsize=MINOR, minor=True)
    plt.xlabel('Epoch', fontsize=MINOR)
    plt.ylabel(y_label, fontsize=MINOR)
    plt.text(
        0.75, 0.75,
        text,
        fontsize=MINOR,
        transform=plt.gca().transAxes
    )

    plt.yscale('log')

    if all(np.all(np.array(loss) >= 0) for loss in loss_fns.values()):
        plt.yscale('log')

    plt.legend(fontsize=MAJOR)

    if plots_dir and save_name:
        plt.savefig(f'{plots_dir}'+save_name)

def comparison_plot(
    data: dict,
    dir_name: str,
    n_points: int = 100, # number of points to take from each distribution
    num_specs: int = 5,
    num_dist_specs: int = 100, # number of different distribution to plot
    param_names: str | list[str]=PARAM_NAMES,
    log_params: list[int]=LOG_PARAMS,
    log_colour_map: bool | None = False,
    colour_map: list[any] | None = None,
    colour_map_label: str |None = '',
    specific_data: dict | None = None, #to choose which specific spectrum to take, from a list of spectra names)
    ):
    """
    Plots:
    1) a random SCATTER_NUM points (in grey or colour map colours) corresponding to maximum of each parameter distribution
    2) num_points points sampled from distributions of n_dist_specs spectra to see spread of distributions
    3) 1:1 relation and fit line with 1 sigma region and Pearson correlation coefficient (PCC) for 2)
    4) coloured specific points that correspond to specific spectra if given specific_data to highlight where they lie in this plot

    Parameters
    ----------
    data: dict
        The input data containing the parameter distributions and spectra.
    dir_name: str
        The directory where the plots will be saved.
    n_points: int
        The number of points to take from each distribution.
    num_specs: int
        The number of specific spectra to highlight if specific_data is given.
    param_names: str | list[str]
        The names of the parameters to plot.
    log_params: list[int]
        The indices of the parameters to plot on a logarithmic scale.
    log_colour_map: bool | None
        Whether to use a logarithmic colour map.
    colour_map: list[any] | None
        The colour map to use for the points.
    colour_map_label: str | None
        The label for the colour map.
    specific_data: dict | None
        To choose which specific spectrum to take, from a list of spectra names.
    """

    num_specs = np.min([len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1)),num_specs]) if specific_data else None
    colors = COLORS_DICT if specific_data else COLORS_LIST

    fig, axes = plt.subplot_mosaic('aabbcc\ndddeee', figsize=(16,14))
    # fig.subplots_adjust(top=0.9, bottom=0.0, hspace=0.6, wspace=0.8, left=0.05, right=0.95)
    fig.subplots_adjust(top=0.92, bottom=0.2, hspace=0.7, wspace=0.7, left=0.05, right=0.95)

    # Adjust the title position
    fig.suptitle('Comparison Plot', fontsize=MAJOR, y=0.99)  # Move the title slightly higher

    # grey plot - data for the grey plots has 5 parameters which each have their own 1620 corresponding spectra (limited to SCATTER_NUM)
    grey_targs = data['targets'][:SCATTER_NUM,0,:].swapaxes(0,1)
    grey_lats = data['latent'][:SCATTER_NUM,0,:].swapaxes(0,1)

    original_cmap = plt.get_cmap('viridis')
    new_cmap = original_cmap(np.linspace(0.0, 0.85, 256))

    # loop through each set of parameters, plotting the outputs vs inputs for SCATTER_NUM points
    for i, (targ, lat, axis) in enumerate(zip(grey_targs, grey_lats, axes.values())):
        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')
        axis.set_xlabel('inputs', fontsize=MINOR)
        axis.set_ylabel('outputs', fontsize=MINOR)
        axis.set_title(param_names[i], fontsize=MAJOR)
        axis.tick_params(labelsize=20)
        axis.set_xlim(np.min(targ), np.max(targ))
        axis.set_ylim(np.min(targ), np.max(targ))
        axis.plot([np.min(targ), np.max(targ)], [np.min(targ), np.max(targ)], color='k')

        # for taking maximum from the distibution rather than frist sample
        # lat = [np.max(data['latent'].swapaxes(1,2).swapaxes(1,0)[i][j]) for j in range(0,SCATTER_NUM)] # or np.max(data['latent'][:SCATTERNUM,:,:], axis=1)

        if list(colour_map):
            if log_colour_map:
                Norm = pltcolors.LogNorm(vmin=np.min(colour_map), vmax=np.max(colour_map))
            else:
                Norm = pltcolors.Normalize(vmin=np.min(colour_map), vmax=np.max(colour_map))
            axis.scatter(x=targ, y=lat, linestyle='None', c=colour_map[:SCATTER_NUM], alpha=0.5, s=7, cmap=plt.cm.colors.ListedColormap(new_cmap), norm=Norm)
        else:
            axis.scatter(x=targ, y=lat, linestyle='None', color='grey', alpha=0.3, s=3)

    if list(colour_map):
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Norm, cmap=plt.cm.colors.ListedColormap(new_cmap)), ax=axes.values(), orientation='horizontal',  label=colour_map_label, aspect=40)
        cbar.ax.xaxis.label.set_size(MINOR)
        cbar.ax.tick_params(labelsize=MINOR)

    legend_elements = [Line2D([0], [0], color='grey', label='Single Sample', linestyle='None', marker='o', alpha=0.5, markersize=10)]
    plt.savefig(dir_name+'NF_comparison_grey'+colour_map_label.replace(' ', '_').replace('$','')+'.png', dpi=300)

    # adding more distributions corresponding to more spectra
    # loop through each set of parameters, plotting, for 5? different spectra, their distributions from 10? data points
    multi_targs = data['targets'].swapaxes(1,2).swapaxes(0,1)
    multi_lats = data['latent'].swapaxes(1,2).swapaxes(0,1)
    for i, (targ, lat, axis, grey_targ) in enumerate(zip(multi_targs, multi_lats, axes.values(), grey_targs)):
        # targ[spectrum number][0=value, 1=error]
        # clear axis from single samples
        axis.cla()

        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')
        axis.set_xlabel('inputs', fontsize=MINOR)
        axis.set_ylabel('outputs', fontsize=MINOR)
        axis.set_title(param_names[i], fontsize=MAJOR)
        axis.set_xlim(np.min(grey_targ), np.max(grey_targ))
        axis.set_ylim(np.min(grey_targ), np.max(grey_targ))
        axis.tick_params(axis='x', labelsize=20)
        axis.tick_params(axis='y', labelsize=20)

        axis.plot([np.min(grey_targ), np.max(grey_targ)], [np.min(grey_targ), np.max(grey_targ)], color='k')

        for spec_num in range(0,num_dist_specs):
            if list(colour_map):
                if log_colour_map:
                    Norm = pltcolors.LogNorm(vmin=np.min(colour_map), vmax=np.max(colour_map))
                else:
                    Norm = pltcolors.Normalize(vmin=np.min(colour_map), vmax=np.max(colour_map))
                axis.scatter(x=[targ[spec_num][0]]*n_points, y=lat[spec_num][:n_points], linestyle='None', c=[colour_map[spec_num]]*n_points, alpha=0.1, s=15, cmap=plt.cm.colors.ListedColormap(new_cmap), norm=Norm)
            else:
                axis.scatter(x=[targ[spec_num][0]]*n_points, y=lat[spec_num][:n_points], linestyle='None', color='grey', alpha=0.05, s=10)

        x_PCC_data = np.repeat(targ[:num_dist_specs,0], n_points, axis=0).flatten()
        y_PCC_data = lat[:num_dist_specs,:n_points].flatten()
        x_PCC_data = np.log10(x_PCC_data) if i in log_params else x_PCC_data
        y_PCC_data = np.log10(y_PCC_data) if i in log_params else y_PCC_data
        corr_coef = pearsonr(x_PCC_data, y_PCC_data)[0]

        ms = []
        m_errs = []
        cs = []
        c_errs = []
        fit_lines = []
        x_fit = np.linspace(np.min(grey_targ), np.max(grey_targ), 100)
        # Samples 1 point per distribution, len(lat[0]) times
        for sample_num in range(len(lat[0])):
            x = targ[:num_dist_specs,0]
            y = lat[:num_dist_specs,sample_num]

            x_data = np.log10(x) if i in log_params else x
            y_data = np.log10(y) if i in log_params else y

            # using just m and assuming c = 0
            bp, cov = curve_fit(line_justm, x_data, y_data)
            m = bp[0]
            m_err = np.sqrt(cov[0,0])
            ms.append(m)
            m_errs.append(m_err)
            cs.append(0)
            c_errs.append(0)
            fit_lines.append(log_line(x_fit, m, 0) if i in log_params else line(x_fit, m, 0))
        # fit_lines shape = (number of lines, number of points in a line)
        fit_lines = np.array(fit_lines).swapaxes(0,1) # now shape is (point in line, line number)

        one_sig_low = []
        one_sig_upp = []
        # loop through each point in the fit lines and takes the quantiles for lower and upper
        for point_num in range(fit_lines.shape[0]):
            one_sig_low.append(np.quantile(fit_lines[point_num], 0.16))
            one_sig_upp.append(np.quantile(fit_lines[point_num], 0.84))
        m_final = np.mean(ms)
        m_err_final = np.std(ms)

        axis.fill_between(x_fit, one_sig_low, one_sig_upp, color='#2765f5', alpha=0.2, label='1$\sigma$')
        axis.text(-0.13, -0.42, f"PCC: ${corr_coef:.3f}$\n$y=({m_final:.3f}\pm{m_err_final:.3f})x$", transform=axis.transAxes, fontsize=PCC)

    plt.savefig(dir_name+'NF_comparison_'+colour_map_label.replace(' ', '_')+'.png', dpi=300)

    # colors specific spectra by their distinguishing color - set in COLORS_DICT
    if specific_data:
        # shape of multi_targs : param_number, spectrum_number, mean+error
        multi_targs = specific_data['targets'].swapaxes(1,2).swapaxes(0,1)
        # shape of multi_lats : param_number, spectrum_number, number of samples in that spectrums distribution
        multi_lats = specific_data['latent'].swapaxes(1,2).swapaxes(0,1)
        # loop through each set of parameters, plotting, for 5? different spectra, their distributions from 10? data points
        for i, (targ, lat, axis) in enumerate(zip(multi_targs, multi_lats, axes.values())):
            # targ shape: (spectrum number)(0=value, 1=error)
            for spec_num, spec_name in enumerate(NAMES[:num_specs]):
                axis.errorbar(x=[targ[spec_num][0]]*n_points, xerr=[targ[spec_num][1]]*n_points, y=lat[spec_num][:n_points], 
                            linestyle='None', capsize=1, ms=7, alpha=0.05, marker='o', color=colors[spec_name])

            legend_elements = [Line2D([0], [0], color=COLORS_LIST[i], label=OBJECT_NAMES[i], linestyle='None', marker='o', alpha=0.5, markersize=10) for i in range(num_specs)]
            fig.legend(handles=list(legend_elements), bbox_to_anchor=(0.5, 1.05), fancybox=False, shadow=False,
                            ncol=num_specs+1, fontsize=20, handletextpad=0.05, columnspacing=0.1, loc='upper center')

        plt.savefig(dir_name+'NF_comparison_specific_'+colour_map_label.replace(' ', '_')+'.png', dpi=300, bbox_inches='tight')

def corner_plot(
    data: dict,
    param_names: str | list[str],
    dir_name: str,
    ):
    """
    Plots a corner plot to show parameter coverage of all input parameters and latent distributions (using one sample from each latent distribution).

    Parameters
    ----------
    data: dict
        The input data containing the latent distributions and target distributions.
    param_names: str | list[str]
        The names of the parameters to plot.
    dir_name: str
        The directory where the plot will be saved.
    """

    c = ChainConsumer()
    c.set_plot_config(PlotConfig(legend={'fontsize':20}))

    # putting latent data into a data frame
    NF_data = pd.DataFrame(data['latent'][:,0,:], columns=param_names)
    
    c.add_chain(Chain(samples=NF_data, name="Normalising Flow", color='#e0c700'))

    # getting the xspec data and putting into a data frame
    dist_targs = data['targets'][:,0,:]

    xspec_gauss_data = pd.DataFrame(dist_targs, columns=param_names).dropna()
    c.add_chain(Chain(samples=xspec_gauss_data, name="XSPEC (Gauss)", color='#e41a1c'))

    fig = c.plotter.plot()

    plt.savefig(os.path.join(dir_name, 'corner_plot.png'))

def latent_corner_plot(
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None, #to choose which specific spectrum to take, from a list of spectra names)
    xspec_data: dict | None = None,
    param_names: str | list[str] = PARAM_NAMES,
    in_param_samples: list = None,
    num_specs: int = 3,
    min_quant = 0.0025,
    max_quant = 0.9975,
    gaussian_truth: bool = False, # whether to plot the gaussian distribution from xspec targets
    ):
    """
    Plots corner plots of the latent distributions from the normalising flow with the target distributions from XSPEC MCMC and XSPEC Gaussian (if gaussian_truth=True).

    Parameters
    ----------
    dir_name: str
        The directory where the plots will be saved.
    data: dict | None
        The input data containing the latent distributions and target distributions.
    specific_data: dict | None
        Input data for specific spectra
    xspec_data: dict | None
        The XSPEC MCMC data containing the posterior distributions.
    param_names: str | list[str]
        The names of the parameters to plot.
    in_param_samples: list
        Parameter samples taken to plot on the corner plots.
    num_specs: int
        The number of specific spectra to highlight if specific_data is given.
    min_quant: float
        The minimum quantile to set the axis limits to. - working on this
    max_quant: float
        The maximum quantile to set the axis limits to. - working on this
    gaussian_truth: bool
        Whether to plot the gaussian distribution from xspec targets.
    """

    # To stop pandas dataframe from rounding
    pd.set_option('display.precision', 10)

    num_specs = len(specific_data['targets']) if specific_data else num_specs
    colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs

    for spec_num in range(num_specs):
        c = ChainConsumer()
        c.set_plot_config(PlotConfig(legend={'fontsize':MAJOR}, label_font_size=MINOR, tick_font_size=16, max_ticks=4)) #, spacing=2))

        # putting latent data into a data frame
        NF_data = pd.DataFrame(specific_data['latent'][spec_num], columns=param_names) if specific_data else pd.DataFrame(data['latent'][spec_num], columns=param_names)
        spec_name = specific_data['ids'][spec_num] if specific_data else data['ids'][spec_num]

        c.add_chain(Chain(samples=NF_data, name="Normalising Flow", color=colors[spec_name] if specific_data else '#0C5DA5', sigmas=[0,1,2]))

        # getting the xspec gauss data and putting into a data frame
        dist_targs = specific_data['targets'][spec_num][0] if specific_data else data['targets'][spec_num][0]
        dist_targ_errs = specific_data['targets'][spec_num][1] if specific_data else data['targets'][spec_num][1]
        if gaussian_truth==True:
            targ_samples = np.array([
                np.random.normal(loc=targ, scale=targ_err, size=3000) 
                for targ, targ_err in zip(dist_targs, dist_targ_errs)]).swapaxes(0,1)
            cut_indices = [np.argwhere((targ_sample<=0)) for targ_sample in targ_samples] 
            targ_samples = [np.delete(targ_sample, cut_index) for targ_sample, cut_index in zip(targ_samples, cut_indices)]
            xspec_gauss_data = pd.DataFrame(targ_samples, columns=param_names).dropna()
            c.add_chain(Chain(samples=xspec_gauss_data, name="XSPEC (Gauss)", color='#91e82e', sigmas=[0,1,2]))
        else:
            xspec_gauss_data = None

        # getting xspec MCMC data and putting into a dataframe and chain
        if xspec_data:
            # xspec_spec_num = int(np.argwhere(specific_data['ids'][spec_num]==xspec_data['ids']))
            xspec_MCMC_data = pd.DataFrame(np.array(xspec_data['posteriors'][spec_num]).swapaxes(0,1)[:5000,:], columns=param_names)
            c.add_chain(Chain(samples=xspec_MCMC_data, name='XSPEC (MCMC)', color='#e41a1c', sigmas=[0,1,2]))
        try:
            fig = c.plotter.plot()

            # settings for specific data plot vs general data plot
            if specific_data:
                fig.axes[2].set_title(specific_data['object'][spec_num], fontsize=MAJOR)
                min_quant = MIN_QUANT[spec_name]
                max_quant = MAX_QUANT[spec_name]

            # adding better limits to axes - working on this
            quantile_limits(
                fig,
                min_quant, 
                max_quant, 
                param_names,
                NF_data = NF_data,
                xspec_gauss_data= xspec_gauss_data if xspec_gauss_data is not None else None,
                xspec_MCMC_data= xspec_MCMC_data if xspec_data else None,
                object_name = specific_data['object'][spec_num] if specific_data else None
            )

            # make sure 

            plt.savefig(os.path.join(dir_name, 'latent_corner_plot'+str(spec_num)+'.png'), dpi=300)

            # add sample lines
            if in_param_samples:
                param_samples = in_param_samples[spec_num]
                for bottom_plot_num in range(0,len(param_samples[0])):
                    for left_plot_num in range(0,bottom_plot_num+1):
                        if bottom_plot_num!=left_plot_num:
                            if dist_targ_errs.all()==0:
                                fig.axes[bottom_plot_num*5+left_plot_num].plot(dist_targs[left_plot_num], dist_targs[bottom_plot_num], marker='*', linestyle='None', color=colour, markersize=20)
                            fig.axes[bottom_plot_num*5+left_plot_num].plot(param_samples[0][left_plot_num], param_samples[0][bottom_plot_num], marker='*', linestyle='None', color='k', markersize=20)
                            fig.axes[bottom_plot_num*5+left_plot_num].axvline(param_samples[0][left_plot_num], color='k', linewidth=2) #colors[spec_name])
                            fig.axes[bottom_plot_num*5+left_plot_num].axhline(param_samples[0][bottom_plot_num], color='k', linewidth=2) #colors[spec_name])
                        else:
                            fig.axes[bottom_plot_num*5+left_plot_num].axvline(param_samples[0][bottom_plot_num], color='k', linewidth=2)
            
                plt.savefig(os.path.join(dir_name, 'latent_corner_plot'+str(spec_num)+'_1samp.png'), dpi=300)
    
        except IndexError:
            print('Index error')

def latent_scatter_plot(
    data1,
    plots_directory,
    ):
    """
    Plots scatter plots and histograms of the latent space for one spectrum. - latent corner plot might be better for this

    Parameters
    ----------
    data1: dict
        The input data containing the latent distributions and target distributions.
    plots_directory: str
        The directory where the plots will be saved.
    """

    log_params = LOG_PARAMS
    param_names = PARAM_NAMES

    param_pair_axes0 = plot_param_pairs(
        data=data1['latent'][0],
        plots_dir=plots_directory,
        save_name = 'latent_space_0',
        log_params=log_params,
        param_names=param_names,
        colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
        scatter_colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
        plot_hist=False
    )

    dist_targs = data1['targets'][0][0]
    dist_targ_errs = data1['targets'][0][1]
    targ_samples = [10**np.random.normal(loc=np.log10(targ), scale=(1/np.log(10))*(targ_err/targ), size=1000) if param_num in log_params
                            else np.random.normal(loc=targ, scale=targ_err, size=1000)
                            for param_num, (targ, targ_err) in enumerate(zip(dist_targs, dist_targ_errs))]
    param_pair_data0=np.array(targ_samples)
    ranges = [None] * param_pair_data0.shape[0]
    # Plot scatter plots & histograms
    for i, (axes_row, y_data, y_range) in enumerate(zip(param_pair_axes0, param_pair_data0, ranges)):
            for j, (axis, x_data, x_range) in enumerate(zip(axes_row, param_pair_data0, ranges)):
                if i == j:
                    _plot_histogram(x_data, axis, log=i in log_params, data_range=x_range, colour='grey')
                    axis.tick_params(labelleft=False, left=False)
                if j < i:
                    axis.scatter(
                        x_data[:1000],
                        y_data[:1000],
                        s=20,
                        alpha=0.2,
                        color='grey'
                    )
                else:
                    axis.set_visible(False)
    plt.savefig(plots_directory+'latent_space_0.png', dpi=600)


def param_pairs_plot(    
    data: dict,
    dir_name: str,
    log_params: list[int]=LOG_PARAMS,
    param_names: list[str] = PARAM_NAMES,
    ): 
    """
    Plots a corner (scatter) plot to show parameter coverage of all input parameters and latent distributions (using one sample from each latent distribution).

    Parameters
    ----------
    data: dict
        The input data containing the latent distributions and target distributions.
    dir_name: str
        The directory where the plots will be saved.
    log_params: list[int]
        The indices of the parameters to plot on a logarithmic scale.
    param_names: list[str]
        The names of the parameters to plot.
    """
    param_pair_axes = plot_param_pairs(
    data=np.array(data['targets'][:,0,:]),
    plots_dir=dir_name,
    save_name='param_pair_plot',
    log_params=log_params,
    param_names=param_names,
    colour='#4cb555',
    scatter_colour='#4cb555',
    alpha=0.7,
    )

    param_pair_data=np.squeeze(data['latent']).swapaxes(0,1)
    ranges = [None] * param_pair_data.shape[0]
    # Plot scatter plots & histograms
    for i, (axes_row, y_data, y_range) in enumerate(zip(param_pair_axes, param_pair_data, ranges)):
            for j, (axis, x_data, x_range) in enumerate(zip(axes_row, param_pair_data, ranges)):
                if i == j:
                    _plot_histogram(x_data, axis, log=i in log_params, data_range=x_range, colour='#8445cc', alpha=0.5)
                    axis.tick_params(labelleft=False, left=False)
                elif j < i:
                    axis.scatter(
                        x_data[:1000],
                        y_data[:1000],
                        s=20,
                        alpha=0.3,
                        color='#8445cc'
                    )
                else:
                    axis.set_visible(False)

    plt.savefig(dir_name+'param_pair_plot.png', dpi=600)


def recon_plot(
    decoder,
    network,
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    all_param_samples: list | None = None,
    num_specs: int = 3,
    data_dir: str = '/Users/work/Projects/FSPNet/data/spectra/',
    spec_scroll = 0,
    ):
    """
    Plots reconstructions from decoder and xspec for either all data or specific data given.
    Reconstructions are:
    1) decoder reconstruction from latent samples
    2) xspec reconstruction from latent samples
    3) decoder reconstruction from target values
    4) xspec reconstruction from target values
    reduced PG statistics also shown for xspec reconstructions from targets or latent samples

    Parameters
    ----------
    decoder:
        The decoder model.
    network:
        The normalising flow network.
    dir_name: str
        The directory where the plots will be saved.
    data: dict | None
        The input data containing the latent distributions and target distributions.
    specific_data: dict | None
        Input data for specific spectra.
    all_param_samples: list | None
        List of sampled parameters to use for reconstructions instead of first value from latent distributions. (shape: (Batch_size/total number of spectra, number of data points in spectra, number of parameters))
    num_specs: int
        The number of spectra to reconstruct, if not reconstructing number of specific spectra.
    data_dir: str
        The directory where the spectra data is stored.
    spec_scroll: int
        The number to start the spectrum indexing from.
    """

    # changes num_specs and spec_scroll if specific data given
    num_specs = min(len(specific_data['targets']), num_specs) if specific_data else num_specs
    spec_scroll = 0 if specific_data else spec_scroll

    # loop through spectra, getting reconstructions and plotting for each 
    for spec_num in range(spec_scroll, num_specs+spec_scroll):

        # if specific data given, use that. use normal data otherwise. Note: [0] is to select one set of parameter samples when there may be more
        input = specific_data['inputs'][spec_num] if specific_data else data['inputs'][spec_num]
        targs = specific_data['targets'][spec_num][0] if specific_data else data['targets'][spec_num][0]
        lats = specific_data['latent'][spec_num][0] if specific_data else data['latent'][spec_num][0]
        spec_name = specific_data['ids'][spec_num] if specific_data else data['ids'][spec_num]
        color_id = spec_name if specific_data else None
        colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs
        
        # if we have given samples, use those instead of data['lats'][0]
        if all_param_samples:
            lats = all_param_samples[spec_num-spec_scroll][0]

        # input spectrum and errors
        inp = input[0]
        inp_err = input[1]

        # getting energy for decoder plots and spectra from decoder and xspec reconstructions
        dec_energy = get_energies(spec_name if type(spec_name)==str else 'js_ni0100320101_0mpu7_goddard_GTI0.jsgrp')[:240]

        lat_dec_recon = decoder_reconstruction(lats, decoder, network)
        true_dec_recon = decoder_reconstruction(targs, decoder, network)

        lat_xs_energy, lat_xs_recon = xspec_reconstruction(lats, spec_name)   
        true_xs_energy, true_xs_recon = xspec_reconstruction(targs, spec_name)

        # getting reduced pg stat
        if os.path.exists(data_dir+spec_name):
            PG_nf = reduced_PG(lats, spec_name)
            PG_xspec = reduced_PG(targs, spec_name)

        # setting up figure
        fig = plt.figure(figsize=(16,9))
        axis = fig.gca()
        if specific_data:
            axis.set_title(specific_data['object'][spec_num], fontsize=MAJOR)
        else:
            axis.set_title('Reconstruction', fontsize=MAJOR)
        axis.set_xlabel('Energy (keV)', fontsize=MINOR)
        axis.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        axis.set_xscale('log')
        axis.set_yscale('log')
        plt.xticks(fontsize=MINOR)
        plt.yticks(fontsize=MINOR)
        axis.set_xlim(dec_energy[0], dec_energy[-1])
        # axis.set_ylim(0, 1.1*np.max(np.concatenate([np.squeeze(lat_dec_recon), lat_xs_recon, np.squeeze(true_dec_recon), true_xs_recon, inp+inp_err])))
            
        # plot data points
        axis.errorbar(x=dec_energy,y=inp, yerr=inp_err, linestyle="None", color='k', label='data', capsize=3, elinewidth=2)
        plt.legend(fontsize=MINOR)
        plt.savefig(dir_name+'NF_spectra'+str(spec_num)+'.png', dpi=300)

        # decoder reconstructions with latent
        axis.plot(dec_energy, lat_dec_recon[0], markersize=5, label='decoder latent', color=colors[color_id], linewidth=2)
        plt.legend(fontsize=MINOR)
        plt.savefig(dir_name+'NF_spectra_recon'+str(spec_num)+'.png', dpi=300)

        # xspec reconstructions with latent
        axis.plot(lat_xs_energy, lat_xs_recon, linestyle="--", linewidth=2, color=colors[color_id], label='xspec latent')
        plt.legend(fontsize=MINOR)
        if os.path.exists(data_dir+spec_name): 
            text1 = plt.text(-0.1, -0.05, f'Reduced PG (NF): {np.format_float_positional(PG_nf, precision=3)}', transform=axis.transAxes, fontsize=MINOR)
        plt.savefig(dir_name+'NF_recon_latentonly'+str(spec_num)+'.png', dpi=300)
        if os.path.exists(data_dir+spec_name): 
            text1.remove()

        # decoder reconstructions with ground truth
        axis.plot(dec_energy, true_dec_recon[0], markersize=5, label='decoder ground truth', color='#de5454', linewidth=2)
        # xspec reconstructions with ground truth
        axis.plot(true_xs_energy, true_xs_recon, linestyle=":", linewidth=2, color='#de5454', label='xspec ground truth')
        plt.text(-0.1, -0.05, f'Reduced PG (Xspec): {np.format_float_positional(PG_xspec, precision=3)}\nReduced PG (NF): {np.format_float_positional(PG_nf, precision=3)}', transform=axis.transAxes, fontsize=MINOR)
        plt.legend(fontsize=MINOR)
        plt.savefig(dir_name+'NF_recons'+str(spec_num)+'.png', dpi=300)


def post_pred_plot(
    decoder,
    network,
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    n_samples: int | None = 100,
    post_pred_samples: list | ndarray | None = None,
    num_specs: int = 3,
    data_dir: str = '/Users/work/Projects/FSPNet/data/spectra/'
    ):
    """
    Plots posterior predictive plots for either all data or specific data given.
    Reconstructions are:
    1) decoder reconstruction from latent samples
    2) xspec reconstruction from target values

    Parameters
    ----------
    decoder:
        The decoder model.
    network:
        The normalising flow network.
    dir_name: str
        The directory where the plots will be saved.
    data: dict | None
        The input data containing the latent distributions and target distributions.
    specific_data: dict | None
        Input data for specific spectra.
    n_samples: int | None
        The number of posterior predictive samples to take.
    post_pred_samples: list | ndarray | None
        List of sampled parameters to use for reconstructions instead of sampling from the latent distributions. 
    num_specs: int
        The number of spectra to reconstruct, if not reconstructing number of specific spectra.
    data_dir: str
        The directory where the spectra data is stored.
    """

    # number of spectra which is either the number of specific spectra or min(num_specs, 5)
    num_specs = min(len(specific_data['targets']),num_specs) if specific_data else num_specs
    colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs

    # if we haven't been given the samples already
    if not post_pred_samples:
        if  specific_data:
            post_pred_samples = sample(specific_data, num_specs=num_specs, num_samples=n_samples)
        else:
            post_pred_samples = sample(data, num_specs=num_specs, num_samples=n_samples)

    # looping over each spectrum
    for spec_num, param_samples in enumerate(post_pred_samples):
        spec_name = spec_num if specific_data is None else specific_data['ids'][spec_num]

        colour = colors[spec_name] # if specific_data else '#0C5DA5'

        # true values, their errors and our latent distribution
        if specific_data:
            targs = specific_data['targets'][spec_num][0]
            spec_name = specific_data['ids'][spec_num]
            input = specific_data['inputs'][spec_num]
        else:
            targs = data['targets'][spec_num][0]
            spec_name = data['ids'][spec_num]
            input = data['inputs'][spec_num]

        # input_data
        inp = input[0]
        inp_err = input[1]
        
        # getting energy for decoder plots and spectra from decoder and xspec reconstructions
        if type(spec_name)==str:
            dec_energy = get_energies(spec_name, data_dir)[:240]
        else:
            dec_energy = get_energies('js_ni0100320101_0mpu7_goddard_GTI0.jsgrp')[:240]

        # decoder reconstructions
        lat_dec_recons=[]
        for params in torch.tensor(param_samples):
            lat_dec_recons.append(decoder_reconstruction(params, decoder, network, spec_name, data_dir if data_dir else None))

        # xspec ground truth reconstructions
        true_xs_energy, true_xs_recon = xspec_reconstruction(targs, spec_name)

        # plotting
        fig = plt.figure(figsize=(16,9))
        axis = fig.gca()
        if specific_data:
            axis.set_title(specific_data['object'][spec_num], fontsize=MAJOR)
        else:
            axis.set_title('Reconstruction', fontsize=MAJOR)
        axis.set_xlabel('Energy (keV)', fontsize=MINOR)
        axis.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        axis.set_xscale('log')
        axis.set_yscale('log')
        plt.xticks(fontsize=MINOR)
        plt.yticks(fontsize=MINOR)
        axis.set_xlim(dec_energy[0], dec_energy[239])
        # axis.set_ylim(0, 1.1*np.max(np.concatenate([[np.max(lat_dec_recons)], true_xs_recon, inp+inp_err])))

        # data
        err=axis.errorbar(x=dec_energy[:240],y=inp, yerr=inp_err, linestyle="None", color='k', capsize=3, label='data')

        # decoder reconstructions
        axis.plot(dec_energy, np.squeeze(lat_dec_recons[0]), color=colour, marker=None, label='decoder latent', alpha=0.1, linewidth=3)
        for lat_dec_recon in lat_dec_recons[1:]:
            axis.plot(dec_energy, np.squeeze(lat_dec_recon), color=colour, alpha=0.1, linewidth=3)

        # xspec reconstructions
        axis.plot(true_xs_energy, true_xs_recon, linestyle="--", linewidth=3, color='#de5454', label='xspec ground truth')

        legend_elements = [Line2D([0], [0], color=colour, label='Latent', linestyle='-', marker='None', alpha=1, markersize=10, linewidth=3)] \
            +[err]  \
            +[Line2D([0], [0], color='#de5454', label='Target', linestyle='--', marker='None', alpha=1, markersize=10, linewidth=3)]

        fig.legend(handles=list(legend_elements), fancybox=False, shadow=False, fontsize=MINOR)

        plt.savefig(dir_name+'NF_post_pred_plot'+str(spec_num)+'.png', dpi=300)

    return post_pred_samples

def post_pred_plot_xspec(
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    n_samples: int | None = 100,
    post_pred_samples: list | ndarray | None = None,
    num_specs: int = 3,
    data_dir: str = '/Users/work/Projects/FSPNet/data/spectra/',
    net=None,
    decoder=None
    ):
    """
    Plots posterior predictive plots for either all data or specific data given.
    Reconstructions are:
    1) decoder reconstruction from latent samples (if net and decoder are given)
    2) xspec reconstruction from latent samples
    3) xspec reconstruction from target values
    
    Parameters
    ----------
    dir_name: str
        The directory where the plots will be saved.
    data: dict | None
        The input data containing the latent distributions and target distributions.
    specific_data: dict | None
        Input data for specific spectra.
    n_samples: int | None
        The number of posterior predictive samples to take.
    post_pred_samples: list | ndarray | None
        List of sampled parameters to use for reconstructions instead of sampling from the latent distributions.
    num_specs: int
        The number of spectra to reconstruct, if not reconstructing number of specific spectra.
    data_dir: str
        The directory where the spectra data is stored.
    net:
        The normalising flow network. - if doing decoder reconstructions
    decoder:
        The decoder model. - if doing decoder reconstructions
    """

    # number of spectra which is either the number of specific spectra or num_specs
    num_specs = np.min([len(specific_data['targets']),num_specs]) if specific_data else num_specs
    colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs

    # if we haven't been given the samples already
    if not post_pred_samples:
        post_pred_samples = sample(specific_data if specific_data else data, num_specs=num_specs, num_samples=n_samples)

    # looping over each spectrum
    for spec_num, param_samples in enumerate(post_pred_samples):
        spec_name = spec_num if specific_data is None else specific_data['ids'][spec_num]
        color_id = spec_name if specific_data else spec_num
        colour = colors[color_id]

        # true values, their errors and our latent distribution
        if specific_data:
            targs = specific_data['targets'][spec_num][0]
            spec_name = specific_data['ids'][spec_num]
            input = specific_data['inputs'][spec_num]
        else:
            targs = data['targets'][spec_num][0]
            spec_name = data['ids'][spec_num]
            input = data['inputs'][spec_num]

        # input_data
        inp = input[0]
        inp_err = input[1]
        
        # getting energy for decoder plots and spectra from decoder and xspec reconstructions
        dec_energy = get_energies(spec_name, data_dir)[:240]

        # xspec latent reconstructions
        lat_xs_recons=[]
        PGs_nf = []
        for params in torch.tensor(param_samples):
            lat_xs_recons.append(xspec_reconstruction(params, spec_name))
            PGs_nf.append(reduced_PG(params, spec_name))
        lat_xs_recons = np.array(lat_xs_recons)
        median_PG_nf = np.median(PGs_nf)

        # xspec ground truth reconstructions
        true_xs_energy, true_xs_recon = xspec_reconstruction(targs, spec_name)
        PG_xspec = reduced_PG(targs, spec_name)

        if net is not None and decoder is not None:
            # decoder reconstructions
            lat_dec_recons=[]
            for params in torch.tensor(param_samples):
                lat_dec_recons.append(decoder_reconstruction(params, decoder, net, spec_name, data_dir if data_dir else None))

        # plotting
        fig = plt.figure(figsize=(10,7), layout='tight')
        axis = fig.gca()
        if specific_data:
            axis.set_title(specific_data['object'][spec_num], fontsize=22)
        else:
            axis.set_title('Reconstruction', fontsize=22)
        axis.set_xlabel('Energy (keV)', fontsize=20)
        axis.set_ylabel('cts / det / s/ keV', fontsize=20)
        axis.set_xscale('log')
        axis.set_yscale('log')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        axis.set_xlim(dec_energy[0], dec_energy[239])
 
        # decoder reconstructions
        if net is not None and decoder is not None:
            axis.plot(dec_energy, np.squeeze(lat_dec_recons[0]), color='#39db39', marker=None, label='decoder latent', alpha=0.1, linewidth=3)
            for lat_dec_recon in lat_dec_recons[1:]:
                axis.plot(dec_energy, np.squeeze(lat_dec_recon), color='#39db39', alpha=0.1, linewidth=3)
        # data
        err=axis.errorbar(x=dec_energy[:240],y=inp, yerr=inp_err, linestyle="None", color='k', capsize=3, elinewidth=2, label='data')

        # xspec latent reconstructions
        axis.plot(lat_xs_recons[0,0,:], lat_xs_recons[0,1,:], color=colour, marker=None, label='latent', alpha=0.1, linewidth=3)
        for lat_xs_recon in lat_xs_recons[1:]:
            axis.plot(lat_xs_recon[0], lat_xs_recon[1], color=colour, alpha=0.1, linewidth=3)

        # xspec target reconstructions
        axis.plot(true_xs_energy, true_xs_recon, linestyle=":", linewidth=5, color='#de5454', label='target')
        plt.text(-0.1, -0.15, f'Reduced PG (Target): {np.format_float_positional(PG_xspec, precision=3)}\nReduced PG (Network): {np.format_float_positional(median_PG_nf, precision=3)}', transform=axis.transAxes, fontsize=20)
        
        legend_elements = [err]\
            +[Line2D([0], [0], color='#de5454', label='Target parameters, Xspec reconstructions', linestyle='--', marker='None', alpha=1, markersize=10, linewidth=3)]\
            +[Line2D([0], [0], color=colour, label='Network parameters, Xspec Rreconstructions', linestyle='-', marker='None', alpha=1, markersize=10, linewidth=3)] \
            +[Line2D([0], [0], color='#39db39', label='Network parameters, Decoder reconstructions', linestyle='-', marker='None', alpha=1, markersize=10, linewidth=3)] if decoder else []
            
        fig.legend(handles=list(legend_elements), fancybox=False, shadow=False, fontsize=18)

        plt.savefig(dir_name+'NF_post_pred_plot_xspec'+str(spec_num)+'.png', dpi=300)

    return post_pred_samples

def rec_2d_plot(
    decoder,
    network,
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    data_dir: str = '/Users/work/Projects/FSPNet/data/spectra/'
    ):
    '''
    Plot 2D color plots of spectra reconstructions from latent parameters to show multiple spectra together

    parameters
    ----------
    decoder: 
        trained decoder model
    network: 
        trained normalising flow model
    dir_name: 
        directory to save plots to
    data: 
        dictionary of data containing inputs, targets, latents and ids
    specific_data: 
        dictionary of specific data containing inputs, targets, latents and ids for specific
    data_dir: 
        directory of spectra data files
    '''

    # if we have specific data, use that. otherwise use normal data - make length of spectra 240 for non specific and length of specific data for specific
    if specific_data:
        num_specs = len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1))
        latents = specific_data['latent'][:num_specs,0,:]
        targets = specific_data['targets'][:num_specs,0,:]
        inputs = specific_data['inputs'][:num_specs,0,:]
        spec_names = specific_data['ids'][:num_specs]
    else:
        num_specs = 240
        if len(data['targets'].shape)==2:
            targets=data['targets'][:num_specs,:]
        else:
            targets = data['targets'][:num_specs,0,:]
        latents = data['latent'][:num_specs,0,:]
        inputs = data['inputs'][:num_specs,0,:]
        spec_names = data['ids'][:num_specs]
        
    
    # getting energy for decoder plots and spectra from decoder and xspec reconstructions
    dec_energy = get_energies()[:240]
    spec_nums = np.arange(0,num_specs)

    lat_dec_recons = np.array([decoder_reconstruction(params, decoder, network, spec_name, data_dir if data_dir else None) 
                               for spec_num, (params, spec_name) in enumerate(zip(latents, spec_names))])[:,0,:]
    
    fig, axes = plt.subplot_mosaic('abc')
    fig.set_size_inches(9,3)
    min_value = min([np.min(inputs),np.min(lat_dec_recons)])
    max_value = max([np.max(inputs),np.max(lat_dec_recons)])

    axes['a'].set_title('data')
    axes['a'].pcolor(dec_energy,spec_nums,inputs, vmin=min_value, vmax=max_value)
    axes['a'].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axes['a'].set_xscale('log')

    axes['b'].set_title('reconstructions')
    axes['b'].pcolor(dec_energy, spec_nums,lat_dec_recons, vmin=min_value, vmax=max_value)
    axes['b'].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axes['b'].set_xscale('log')

    axes['c'].set_title('residuals')
    axes['c'].pcolor(dec_energy, spec_nums,inputs-lat_dec_recons, norm=LogNorm(vmin=min_value, vmax=max_value))
    axes['c'].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axes['c'].set_xscale('log')

def resid_params_plot(
    data: dict,
    x_param: list | ndarray | int,
    x_param_name: str,
    dir_name: str | None = '',
    ):
    ''' WORK IN PROGRESS
    Plot residuals of parameters (true - predicted) against one parameter to show trends between prediction error and that parameter
    
    parameters
    ----------
    data: 
        dictionary of data containing inputs, targets, latents and ids
    x_param: 
        list or ndarray of parameter values to plot against or int of parameter index in
    x_param_name: 
        name of parameter to plot against
    dir_name: 
        directory to save plots to
    '''

    resids = np.max(data['latent'][:,:,:], axis=1)-data['targets'][:,1,:]

    fig = plt.figure(figsize=(16,9))
    axis = fig.gca()

    axis.set_title('Residual parameters')
    axis.set_xlabel(x_param_name)
    axis.set_ylabel('true-pred')
    axis.plot()

def coverage_plot(
    dataset,
    loaders,
    network,
    dir_name,
    coverage_dir = '/Users/work/Projects/FSPNet/coverages/',
    pred_savename='noname',
    overwrite=False
    ):
    """
    Plot coverage plot for normalising flow network

    parameters
    ----------
    dataset: 
        full dataset object
    loaders: 
        list of dataloaders for train, val, test
    network: 
        trained normalising flow network
    dir_name: 
        directory to save plots to
    coverage_dir: 
        directory to save coverage data to
    pred_savename: 
        name to save coverage data as
    overwrite: whether to overwrite existing coverage data
    """

    subset = dataset[loaders[1].dataset.indices]

    # List of pairs
    testset = list(zip(subset[1], subset[2].swapaxes(0, 1)[:, None]))

    # Generate levels and coverages
    if os.path.exists(os.path.join(coverage_dir,pred_savename+'.pickle')) and overwrite==False:
        with open(os.path.join(coverage_dir,pred_savename+'.pickle'), 'rb') as file:
            data = pickle.load(file)

    else:
        levels, coverages = lampe.diagnostics.expected_coverage_mc(network.net.net[0], testset)
        data = [levels, coverages]

        with open(os.path.join(coverage_dir,pred_savename+'.pickle'), 'wb') as file:
            pickle.dump(data, file)

    levels = data[0]
    coverages = data[1]

    plt.figure(figsize=(4,4))
    plt.plot([0,1], [0,1], linestyle='--', color='grey', label='perfect coverage')
    plt.plot(levels, coverages, label='network coverage')
    plt.xlabel('Credible Level', fontsize=14)
    plt.ylabel('Coverage', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(dir_name, 'coverage'), dpi=300)
