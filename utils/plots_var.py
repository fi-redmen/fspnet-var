import torch
import pandas as pd
import os
import lampe
import pickle
import random
import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as pltcolors
import matplotlib.ticker as mticker
import matplotlib as mpl
from chainconsumer import Chain, ChainConsumer, PlotConfig

from fspnet.utils.plots import plot_param_pairs, _plot_histogram
from netloader.utils.utils import get_device

from utils.plot_utils import get_energies, decoder_reconstruction, quantile_limits, MidPointLogNorm
from utils.misc_utils import sample, add_state_labels, order_by_state
from utils.xspec_utils import xspec_reconstruction, reduced_PG, xspec_data

mpl.rcParams['agg.path.chunksize'] = 10000

ROOT = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DIR: str = ROOT+'/data/synth_spectra_clean.pickle'

RECTANGLE: tuple[int, int] = (16, 9)
MAJOR: int = 32
MINOR: int = 28
PCC: int = 20
SCATTER_NUM: int = 1000
LOG_PARAMS: list[int] = [0, 2, 3, 4]  # parameters to be plotted in log scale
PARAM_NAMES : ndarray = np.array(['$N_{H}$ $(10^{22} cm^{-2})$', '$\Gamma$', '$f_{sc}$','$kT_{disk}$ $(keV)$','$N$'])
COLORS_LIST = ['#0C5DA5']*17 #'#00B945', '#FF9500', '#9159ab', '#00A7C6'] # to colour different spectr differently
NAMES = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp','js_ni0103010102_0mpu7_goddard_GTI0.jsgrp','js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
         'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp','js_ni1100320119_0mpu7_goddard_GTI26.jsgrp','js_ni1200120203_0mpu7_goddard_GTI0.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI10.jsgrp','js_ni1200120203_0mpu7_goddard_GTI11.jsgrp','js_ni1200120203_0mpu7_goddard_GTI13.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI1.jsgrp','js_ni1200120203_0mpu7_goddard_GTI3.jsgrp','js_ni1200120203_0mpu7_goddard_GTI4.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI5.jsgrp','js_ni1200120203_0mpu7_goddard_GTI6.jsgrp','js_ni1200120203_0mpu7_goddard_GTI7.jsgrp',
         'js_ni1200120203_0mpu7_goddard_GTI8.jsgrp','js_ni1200120203_0mpu7_goddard_GTI9.jsgrp']
OBJECT_NAMES = ['Cyg X-1 (2017)','GRS 1915+105','LMC X-3','MAXI J1535-571','Cyg X-1 (2018)','MAXI J1820 0','MAXI J1820 10','MAXI J1820 11','MAXI J1820 13','MAXI J1820 1','MAXI J1820 3','MAXI J1820 4','MAXI J1820 5','MAXI J1820 6','MAXI J1820 7','MAXI J1820 8','MAXI J1820 9']
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
COLORS_DICT: dict = {}
for key in NAMES:
    COLORS_DICT[key] = COLORS_LIST[NAMES.index(key)]
    if NAMES.index(key) >=5:
        MIN_QUANT[key] = 1-0.9975
        MAX_QUANT[key] = 0.9975

# def line(x, m, c):
#     return (m*x) + c

def line(x, m):
    return m*x

def log_line(x, m, c):
    return (x **(m)) * c

def performance_plot(
        y_label: str,
        loss_fns_train: dict,
        loss_fns_val: dict,
        log_y: bool = True,
        plots_dir: str | None = None,
        save_name: str | None = 'performance.png') -> None:
    """
    Plots training and validation performance as a function of epochs

    Parameters
    ----------
    y_label : string
        Performance metric
    loss_fns_train : dict
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
    total_loss_fn = np.zeros(len(loss_fns_train['reconstruct']))

    # makes function positive so logging possible
    move = abs(min([min(v) if v else 0 for k, v in loss_fns_train.items()]))+1
    if 'reconstruct' in loss_fns_train.keys(): loss_fns_train['reconstruct'] = list(np.array(loss_fns_train['reconstruct'])+move)
    if 'latent' in loss_fns_train.keys(): loss_fns_train['latent'] = list(np.array(loss_fns_train['latent'])+move)
    if 'flow' in loss_fns_train.keys(): loss_fns_train['flow'] = list(np.array(loss_fns_train['flow'])+move)
    if 'bound' in loss_fns_train.keys(): loss_fns_train['bound'] = list(np.array(loss_fns_train['bound'])+move)

    # plotting loss function components if they contribute, and adding to total loss
    if 'reconstruct' in loss_fns_train.keys() and np.array(loss_fns_train['reconstruct']).all()!=0:
        rec = plt.plot(loss_fns_train['reconstruct'], label='Reconstruction')
        plt.plot(loss_fns_val['reconstruct'], label='val', linestyle='--', c=rec[0].get_color())
        text+=f"Final recon: {loss_fns_train['reconstruct'][-1]:.3e} \n"
    if 'flow' in loss_fns_train.keys():
        flow = plt.plot(loss_fns_train['flow'], label='Flow')
        plt.plot(loss_fns_val['flow'], label='val', linestyle='--', c=flow[0].get_color())
        text+=f"Final flow: {loss_fns_train['flow'][-1]:.3e} \n"
    if 'latent' in loss_fns_train.keys():
        latent = plt.plot(loss_fns_train['latent'], label='Latent')
        plt.plot(loss_fns_val['latent'], label='val', linestyle='--', c=latent[0].get_color())
        text+=f"Final latent: {loss_fns_train['latent'][-1]:.3e} \n"
    if 'bound' in loss_fns_train.keys():
        bound = plt.plot(loss_fns_train['bound'], label='Bound')
        plt.plot(loss_fns_val['bound'], label='val', linestyle='--', c=bound[0].get_color())
        text+=f"Final bound: {loss_fns_train['bound'][-1]:.3e} \n"
    if 'kl' in loss_fns_train.keys():
        kl = plt.plot(loss_fns_train['kl'], label='KL')
        plt.plot(loss_fns_val['kl'], label='val', linestyle='--', c=kl[0].get_color())
        text+=f"Final KL: {loss_fns_train['kl'][-1]:.3e} \n"

    plt.plot(loss_fns_train['total'], label='Total', color='k')
    plt.plot(loss_fns_val['total'], label='val', linestyle='--', c='k')

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

    if all(np.all(np.array(loss) >= 0) for loss in loss_fns_train.values()) and all(np.all(np.array(loss) >= 0) for loss in loss_fns_val.values()):
        plt.yscale('log')

    plt.legend(fontsize=MAJOR, ncol=2)

    if plots_dir and save_name:
        plt.savefig(f'{plots_dir}'+save_name)
    # plt.close()

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

    num_specs = len(specific_data['targets']) if specific_data else None
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
    new_cmap = original_cmap(np.linspace(0.0, 1.0, 256))

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
        axis.plot([np.min(targ), np.max(targ)], [np.min(targ), np.max(targ)], color='k', alpha=0.5, linestyle='--')

        # for taking maximum from the distibution rather than frist sample
        # lat = [np.max(data['latent'].swapaxes(1,2).swapaxes(1,0)[i][j]) for j in range(0,SCATTER_NUM)] # or np.max(data['latent'][:SCATTERNUM,:,:], axis=1)

        if list(colour_map):
            Norm = pltcolors.LogNorm(vmin=np.min(colour_map), vmax=np.max(colour_map)) if log_colour_map else pltcolors.Normalize(vmin=np.min(colour_map), vmax=np.max(colour_map))
            axis.scatter(x=targ, y=lat, linestyle='None', c=colour_map[:SCATTER_NUM], alpha=0.5, s=7, cmap=plt.cm.colors.ListedColormap(new_cmap), norm=Norm)
        else:
            axis.scatter(x=targ, y=lat, linestyle='None', color='grey', alpha=0.3, s=3)

    if list(colour_map):
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Norm, cmap=plt.cm.colors.ListedColormap(new_cmap)), ax=axes.values(), orientation='horizontal',  label=colour_map_label, aspect=40)
        cbar.ax.xaxis.label.set_size(MINOR)
        cbar.ax.tick_params(labelsize=MINOR)

    legend_elements = [Line2D([0], [0], color='grey', label='Single Sample', linestyle='None', marker='o', alpha=0.5, markersize=10)]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    plt.savefig(dir_name+'NF_comparison_grey'+colour_map_label.replace(' ', '_').replace('$','')+'.png', dpi=300)

    # loop through parameters, plotting n_points sampled from num_dist_specs spectra
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

        axis.plot([np.min(grey_targ), np.max(grey_targ)], [np.min(grey_targ), np.max(grey_targ)], color='k', alpha=0.5, linestyle='--')

        for spec_num in range(0,num_dist_specs):
            if list(colour_map):
                Norm = pltcolors.LogNorm(vmin=np.min(colour_map), vmax=np.max(colour_map)) if log_colour_map else pltcolors.Normalize(vmin=np.min(colour_map), vmax=np.max(colour_map))
                axis.scatter(x=[targ[spec_num][0]]*n_points, y=lat[spec_num][:n_points], linestyle='None', c=[colour_map[spec_num]]*n_points, alpha=0.1, s=15, cmap=plt.cm.colors.ListedColormap(new_cmap), norm=Norm)
            else:
                axis.scatter(x=[targ[spec_num][0]]*n_points, y=lat[spec_num][:n_points], linestyle='None', color='grey', alpha=0.05, s=10)
        
        n_points_an = 100
        num_dist_specs_an = 250
        x_PCC_data = np.repeat(targ[:num_dist_specs_an,0], n_points_an, axis=0).flatten()
        y_PCC_data = lat[:num_dist_specs_an,:n_points_an].flatten()
        x_PCC_data = np.log10(x_PCC_data) if i in log_params else x_PCC_data
        y_PCC_data = np.log10(y_PCC_data) if i in log_params else y_PCC_data
        # x_PCC_data = x_PCC_data
        # y_PCC_data = y_PCC_data
        corr_coef = pearsonr(x_PCC_data, y_PCC_data)[0]

        # cs = []
        # c_errs = []
        ms = []
        m_errs = []
        fit_lines = []
        x_fit = np.linspace(np.min(grey_targ), np.max(grey_targ), 100000)
        # Samples 100 points out of each of the num_disst_specs_an distributions, and does 1000 fits
        for i in range(100):
            idxs = random.sample(range(len(lat[:num_dist_specs_an,:].flatten())), 50)

            x = np.array([targ[:num_dist_specs_an,0] for _ in range(len(lat[0]))]).swapaxes(0,1).flatten()[idxs]
            y = lat[:num_dist_specs_an,:].flatten()[idxs]

            x_data = np.log10(x) if i in log_params else x
            y_data = np.log10(y) if i in log_params else y
            # x_data = x
            # y_data = y

            # using just gradient m and assuming intercept = 0
            bp, cov = curve_fit(line, x_data, y_data)
            m = bp[0]
            m_err = np.sqrt(cov[0,0])
            # c = bp[1]
            # c_err = np.sqrt(cov[1,1])
            ms.append(m)
            m_errs.append(m_err)
            # cs.append(c)
            # c_errs.append(c_err)
            # fit_lines.append(line(x_fit, m, c))
            fit_lines.append(line(x_fit, m))
            # axis.plot(x_fit, line(x_fit, m, c), color='#2765f5', alpha=0.05)
            # fit_lines.append(log_line(x_fit, m, c) if i in log_params else line(x_fit, m, c))
            # axis.plot(x_fit, log_line(x_fit, m, c) if i in log_params else line(x_fit, m, c), color='#2765f5', alpha=0.05)
        # fit_lines shape = (number of lines, number of points in a line)
        fit_lines = np.array(fit_lines).swapaxes(0,1) # now shape is (point in line, line number)

        # takes standard deviation of all fit lines and uses it to get lower and upper quantile
        std = np.std(fit_lines, axis=1)
        one_sig_low = np.mean(fit_lines, axis=1) - std
        one_sig_upp = np.mean(fit_lines, axis=1) + std

        m_final = np.mean(ms)
        m_err_final = np.std(ms)
        # c_final = np.mean(cs)
        # c_err_final = np.std(cs)

        axis.fill_between(x_fit, one_sig_low, one_sig_upp, color='#2765f5', alpha=0.2, label='1$\sigma$')
        axis.text(-0.13, -0.42, f"PCC: ${corr_coef:.3f}$\n$y=({m_final:.3f}\pm{m_err_final:.3f})x$", transform=axis.transAxes, fontsize=PCC)
        # if c_final >= 0:
        #     axis.text(-0.13, -0.42, rf"PCC: ${corr_coef:.3f}$"+"\n"+rf"$y=({m_final:.3f}\pm{m_err_final:.2f})x + ({c_final:.3f}\pm{c_err_final:.2f})$", transform=axis.transAxes, fontsize=PCC)
        # else:
        #     c_final0 = abs(c_final)
        #     axis.text(-0.13, -0.42, rf"PCC: ${corr_coef:.3f}$"+"\n"+rf"$y=({m_final:.3f}\pm{m_err_final:.2f})x - ({c_final0:.3f}\pm{c_err_final:.2f})$", transform=axis.transAxes, fontsize=PCC)

    plt.savefig(dir_name+'NF_comparison_'+colour_map_label.replace(' ', '_')+'.png', dpi=300)

    # add points coloursed by above and belot kTe
    belows = []
    aboves = []
    bad_names = []
    bad_obj_names = []
    for i, (targ, lat, axis, grey_targ) in enumerate(zip(multi_targs, multi_lats, axes.values(), grey_targs)):
        for spec_num in range(0,num_dist_specs):
            # colors points below 1:1 line for kTe in pink and above in yellowish orange
            below_idxs = np.argwhere(multi_lats[3,spec_num,:n_points]<multi_targs[3,spec_num,0]*0.5)[:,0]
            above_idxs = np.argwhere(multi_lats[3,spec_num,:n_points]>multi_targs[3,spec_num,0]*1.5)[:,0]

            below = axis.scatter([targ[spec_num][0]]*len(below_idxs), lat[spec_num][below_idxs], linestyle='None', c='r', alpha=0.7)
            above = axis.scatter([targ[spec_num][0]]*len(above_idxs), lat[spec_num][above_idxs], linestyle='None', c='y', alpha=0.7)
    
            belows.append(below)
            aboves.append(above)

            # gets file names and object names for the bad predictions
            # if max(multi_lats[i,spec_num,:])

    plt.savefig(dir_name+'NF_comparison_'+colour_map_label.replace(' ', '_')+'_devs.png')

    for below, above in zip(belows, aboves):
        below.remove()
        above.remove()

    # colors specific spectra by their distinguishing color - set in COLORS_DICT
    if specific_data:
        # shape of multi_targs : param_number, spectrum_number, mean/error if real data else param_number, spectrum_number
        multi_targs = specific_data['targets'].swapaxes(0,1) if len(specific_data['targets'].shape) == 2 else specific_data['targets'].swapaxes(1,2).swapaxes(0,1)
        # shape of multi_lats : param_number, spectrum_number, number of samples in that spectrums distribution
        multi_lats = specific_data['latent'].swapaxes(1,2).swapaxes(0,1)
        # loop through each set of parameters, plotting, for 5? different spectra, their distributions from 10? data points
        for i, (targ, lat, axis) in enumerate(zip(multi_targs, multi_lats, axes.values())):
            # targ shape: (spectrum number)(0=value, 1=error)
            for spec_num, spec_name in enumerate(NAMES[:num_specs]):
                if  len(specific_data['targets'].shape) == 3:
                    axis.errorbar(x=[targ[spec_num][0]]*n_points, xerr=[targ[spec_num][1]]*n_points, y=lat[spec_num][:n_points], 
                            linestyle='None', capsize=1, ms=7, alpha=0.05, marker='o', color=colors[spec_name])

            legend_elements = [Line2D([0], [0], color=COLORS_LIST[i], label=OBJECT_NAMES[i], linestyle='None', marker='o', alpha=0.5, markersize=10) for i in range(num_specs)]
            fig.legend(handles=list(legend_elements), bbox_to_anchor=(0.5, 1.05), fancybox=False, shadow=False,
                            ncol=num_specs+1, fontsize=20, handletextpad=0.05, columnspacing=0.1, loc='upper center')

        plt.savefig(dir_name+'NF_comparison_specific_'+colour_map_label.replace(' ', '_')+'.png', dpi=300, bbox_inches='tight')
        plt.close()

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
    plt.close()

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
        print('Plotting for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num], '...')

        colour=COLORS_DICT[specific_data['ids'][spec_num]] if specific_data else COLORS_LIST[spec_num]

        c = ChainConsumer()
        c.set_plot_config(PlotConfig(legend={'fontsize':MAJOR}, label_font_size=MINOR, tick_font_size=16, max_ticks=4)) 
                                    #  log_scales=['$N_{H}$ $(10^{22} cm^{-2})$', '$f_{sc}$','$kT_{disk}$ $(keV)$','$N$'],))

        # removing largest outliers from latent distribution for better plotting
        if specific_data: NF_data = np.delete(specific_data['latent'][spec_num], np.argwhere(specific_data['latent'][spec_num]>np.quantile(specific_data['latent'][spec_num],0.9998)) , axis=0)
        else: NF_data = np.delete(data['latent'][spec_num], np.argwhere(data['latent'][spec_num]>np.quantile(data['latent'][spec_num],0.9998) , axis=0))

        # putting latent data into a data frame
        NF_data = pd.DataFrame(NF_data, columns=param_names)
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
                object_name = specific_data['object'][spec_num] if specific_data else None)

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.savefig(os.path.join(dir_name, 'latent_corner_plot'+str(spec_num)+'.png'), dpi=300)
            
            # add sample lines
            if list(in_param_samples):
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
                plt.close()
        
            print('Saved latent corner plot for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num])
    
        except IndexError:
            print('Index error for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num], 'skipping plot')

        

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

    param_pair_data=data['latent'][:,0,:].swapaxes(0,1)
    ranges = [None] * param_pair_data.shape[0]
    # Plot scatter plots & histograms
    # for i, (axes_row, y_data, y_range) in enumerate(zip(param_pair_axes, param_pair_data, ranges)):
    #         for j, (axis, x_data, x_range) in enumerate(zip(axes_row, param_pair_data, ranges)):
    #             if i == j:
    #                 _plot_histogram(x_data, axis, log=i in log_params, data_range=x_range, colour='#8445cc', alpha=0.5)
    #                 axis.tick_params(labelleft=False, left=False)
    #             elif j < i:
    #                 axis.scatter(
    #                     x_data[:1000],
    #                     y_data[:1000],
    #                     s=20,
    #                     alpha=0.3,
    #                     color='#8445cc'
    #                 )
    #             else:
    #                 axis.set_visible(False)
    
    legend_elements = [Line2D([0], [0], color='#4cb555', label="Targets", linestyle='None', marker='o', alpha=0.5, markersize=10),
                       Line2D([0], [0], color='#8445cc', label="Predicted", linestyle='None', marker='o', alpha=0.5, markersize=10)]
    param_pair_axes[0,4].get_figure().legend(handles=legend_elements, fontsize=24)

    plt.savefig(dir_name+'param_pair_plot.png', dpi=600)


def recon_plot(
    decoder,
    network,
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    all_param_samples: list | None = None,
    num_specs: int = 3,
    data_dir: str | None = None,
    spectra_idxs: list | None = None):
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
    num_specs = len(specific_data['targets']) if specific_data else num_specs
    spectra_idxs = list(range(num_specs)) if not spectra_idxs else spectra_idxs

    # loop through spectra, getting reconstructions and plotting for each 
    print('Plotting reconstructions...')
    for spec_num in spectra_idxs:

        # if specific data given, use that. use normal data otherwise. Note: [0] is to select one set of parameter samples when there may be more
        input = specific_data['inputs'][spec_num] if specific_data else data['inputs'][spec_num]
        targs = specific_data['targets'][spec_num][0] if specific_data else data['targets'][spec_num][0]
        lats = specific_data['latent'][spec_num][0] if specific_data else data['latent'][spec_num][0]
        spec_name = specific_data['ids'][spec_num] if specific_data else data['ids'][spec_num]
        color_id = spec_name if specific_data else 0
        colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs
        
        # if we have given samples, use those instead of data['lats'][0]
        if list(all_param_samples):
            lats = all_param_samples[spec_num][0]

        # input spectrum and errors
        inp = input[0]
        inp_err = input[1]

        # getting energy for decoder plots and spectra from decoder and xspec reconstructions
        dec_energy = get_energies(spec_name if type(spec_name)==str else 'js_ni0100320101_0mpu7_goddard_GTI0.jsgrp')[:240]

        lat_dec_recon = decoder_reconstruction(lats, decoder, network)[0]
        true_dec_recon = decoder_reconstruction(targs, decoder, network)[0]

        lat_xs_energy, lat_xs_recon, lat_xs_resid_energy, lat_xs_resid, lat_xs_resid_err = xspec_reconstruction(lats, spec_name, data_dir=data_dir)   
        true_xs_energy, true_xs_recon, true_xs_resid_energy, true_xs_resid, true_xs_resid_err = xspec_reconstruction(targs, spec_name, data_dir=data_dir)
        
        # getting reduced pg stat
        if os.path.exists(data_dir+spec_name):
            PG_nf = reduced_PG(lats, spec_name)
            PG_xspec = reduced_PG(targs, spec_name)

        # setting up figure
        fig, axes = plt.subplot_mosaic('a\na\na\nb\nb', figsize=(16,14))
        fig.subplots_adjust(hspace=0.0)
        recon_ax = axes['a']
        resid_ax = axes['b']
        if specific_data:
            recon_ax.set_title(specific_data['object'][spec_num], fontsize=MAJOR)
        else:
            recon_ax.set_title('Reconstruction', fontsize=MAJOR)
        recon_ax.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        recon_ax.set_xscale('log')
        recon_ax.set_yscale('log')
        recon_ax.tick_params('both', labelsize=MINOR)
        recon_ax.set_xlim(dec_energy[0], dec_energy[-1])

        resid_ax.tick_params('both', labelsize=MINOR)
        resid_ax.set_xlim(dec_energy[0], dec_energy[-1])
        resid_ax.set_xscale('log')
        resid_ax.set_ylabel('Data / Model', fontsize=MINOR)
        resid_ax.set_xlabel('Energy (keV)', fontsize=MINOR)

        # plot data points
        recon_ax.errorbar(x=dec_energy,y=inp, yerr=inp_err, linestyle="None", color='k', label='data', capsize=3, elinewidth=2)
        fig.legend(fontsize=MINOR)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name+'NF_spectra'+str(spec_num)+'.png', dpi=300)

        # decoder reconstructions with latent
        recon_ax.plot(dec_energy, lat_dec_recon, markersize=5, label='decoder latent', color=colors[color_id], linewidth=2)
        resid_ax.errorbar(dec_energy, inp/lat_dec_recon, inp_err/lat_dec_recon, linestyle='None', marker='o', ms=1, color=colors[color_id], linewidth=2)
        fig.legend(fontsize=MINOR)
        plt.savefig(dir_name+'NF_spectra_recon'+str(spec_num)+'.png', dpi=300)

        # xspec reconstructions with latent
        recon_ax.plot(lat_xs_energy, lat_xs_recon, linewidth=2, color='#FF9500', label='xspec latent')
        resid_ax.errorbar(lat_xs_resid_energy, lat_xs_resid, abs(lat_xs_resid_err), linestyle='None', marker='o', ms=1, color='#FF9500', linewidth=2)
        fig.legend(fontsize=MINOR)
        if os.path.exists(data_dir+spec_name): 
            text = plt.text(0, 0, f'Reduced PG (NF): {np.format_float_positional(PG_nf, precision=3)}', transform=fig.transFigure, fontsize=MINOR)
        plt.savefig(dir_name+'NF_recon_latentonly'+str(spec_num)+'.png', dpi=300)
        if os.path.exists(data_dir+spec_name): # remove text as this is covered by next plt.text
            text.remove() 

        # decoder reconstructions with ground truth
        recon_ax.plot(dec_energy, true_dec_recon, markersize=5, label='decoder ground truth', color='#de5454', linewidth=2)
        resid_ax.errorbar(dec_energy, inp/true_dec_recon, inp_err/true_dec_recon, linestyle='None', marker='o', ms=1, color='#de5454', linewidth=2)
        # xspec reconstructions with ground truth
        recon_ax.plot(true_xs_energy, true_xs_recon, linewidth=2, color='#00B945', label='xspec ground truth')
        resid_ax.errorbar(true_xs_resid_energy, true_xs_resid, true_xs_resid_err, linestyle='None', marker='o', ms=1, color='#00B945', linewidth=2)
        plt.text(0, 0, f'Reduced PG (Xspec): {np.format_float_positional(PG_xspec, precision=3)}\nReduced PG (NF): {np.format_float_positional(PG_nf, precision=3)}', transform=fig.transFigure, fontsize=MINOR)
        fig.legend(fontsize=MINOR)
        plt.savefig(dir_name+'NF_recons'+str(spec_num)+'.png', dpi=300)

        print('Saved reconstructions for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num])

def post_pred_plot(
    decoder,
    network,
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    n_samples: int | None = 100,
    post_pred_samples: list | ndarray | None = None,
    num_specs: int = 3,
    data_dir: str | None = None):
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

    Returns
    -------
    post_pred_samples: list | ndarray
        The list of sampled parameters used for reconstructions.
    """

    # number of spectra which is either the number of specific spectra or min(num_specs, 5)
    num_specs = len(specific_data['targets']) if specific_data else num_specs
    colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs

    # if we haven't been given the samples already
    if post_pred_samples is None:
        if  specific_data:
            post_pred_samples = sample(specific_data, num_specs=num_specs, num_samples=n_samples)
        else:
            post_pred_samples = sample(data, num_specs=num_specs, num_samples=n_samples)

    # looping over each spectrum
    print('starting posterior predictive plots with decoder reconstructions...')
    for spec_num, param_samples in enumerate(post_pred_samples):
        print('Plotting posterior predictive plot for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num],'...')
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
            dec_energy = get_energies(spec_name, data_dir)
        else:
            dec_energy = get_energies('js_ni0100320101_0mpu7_goddard_GTI0.jsgrp')

        # decoder reconstructions
        lat_dec_recons=[]
        for params in torch.tensor(param_samples):
            lat_dec_recons.append(decoder_reconstruction(params, decoder, network, spec_name, data_dir if data_dir else None))

        # xspec ground truth reconstructions
        true_xs_energy, true_xs_recon, true_xs_resid_energy, true_xs_resid, true_xs_resid_err = xspec_reconstruction(targs, spec_name, data_dir=data_dir)

        # plotting
        fig, axes = plt.subplot_mosaic('a\na\na\nb\nb', figsize=(16,14))
        recon_ax = axes['a']
        resid_ax = axes['b']
        fig.subplots_adjust(hspace=0.0)
        if specific_data:
            recon_ax.set_title(specific_data['object'][spec_num], fontsize=MAJOR)
        else:
            recon_ax.set_title('Reconstruction', fontsize=MAJOR)
        recon_ax.set_xlabel('Energy (keV)', fontsize=MINOR)
        recon_ax.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        recon_ax.set_xscale('log')
        recon_ax.set_yscale('log')
        recon_ax.tick_params(labelsize=MINOR)
        recon_ax.set_xlim(dec_energy[0], dec_energy[239])

        resid_ax.tick_params(labelsize=MINOR)
        resid_ax.set_xlabel('Energy (keV)', fontsize=MINOR)
        resid_ax.set_ylabel('Data / Model', fontsize=MINOR)
        resid_ax.set_xscale('log')
        resid_ax.set_xlim(dec_energy[0], dec_energy[239])

        # data
        err=recon_ax.errorbar(x=dec_energy[:240],y=inp, yerr=inp_err, linestyle="None", color='k', capsize=3, label='data')

        # decoder reconstructions
        recon_ax.plot(dec_energy, np.squeeze(lat_dec_recons[0]), color=colour, marker=None, label='decoder latent', alpha=0.1, linewidth=3)
        for lat_dec_recon in lat_dec_recons[1:]:
            recon_ax.plot(dec_energy, np.squeeze(lat_dec_recon), color=colour, alpha=0.1, linewidth=3)
            resid_ax.errorbar(dec_energy, inp/np.squeeze(lat_dec_recon), inp_err/np.squeeze(lat_dec_recon), linestyle='None', marker=None, color=colour, alpha=0.1, elinewidth=3)

        # xspec reconstructions
        recon_ax.plot(true_xs_energy, true_xs_recon, linestyle="--", linewidth=3, color='#de5454', label='xspec ground truth')
        resid_ax.errorbar(true_xs_resid_energy, true_xs_resid, true_xs_resid_err, linestyle='None', marker=None, color='#de5454', elinewidth=3)

        legend_elements = [Line2D([0], [0], color=colour, label='Latent parameters, Decoder Reconstruction', linestyle='-', marker='None', alpha=1, markersize=10, linewidth=3)] \
            +[err]  \
            +[Line2D([0], [0], color='#de5454', label='Target parameters, Xspec Reconstruction ', linestyle='--', marker='None', alpha=1, markersize=10, linewidth=3)]

        fig.legend(handles=list(legend_elements), fancybox=False, shadow=False, fontsize=MINOR)

        if dir_name:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            plt.savefig(dir_name+'NF_post_pred_plot'+str(spec_num)+'.png', dpi=300)
            print('Saved posterior predictive plot for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num])

    return post_pred_samples

def post_pred_plot_xspec(
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    n_samples: int | None = 100,
    post_pred_samples: list | ndarray | None = None,
    num_specs: int = 3,
    data_dir: str | None = None,
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

    Returns
    -------
    post_pred_samples: list | ndarray
        The list of sampled parameters used for reconstructions.
    """

    # number of spectra which is either the number of specific spectra or num_specs
    num_specs = len(specific_data['targets']) if specific_data else num_specs
    colors = COLORS_DICT if specific_data else COLORS_LIST*num_specs

    # if we haven't been given the samples already
    if post_pred_samples is None:
        post_pred_samples = sample(specific_data if specific_data else data, num_specs=num_specs, num_samples=n_samples)

    # looping over each spectrum
    print('starting posterior predictive plots with xspec reconstructions...')
    for spec_num, param_samples in enumerate(post_pred_samples):
        print('Plotting xspec posterior predictive plot for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num], '...')
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
        lat_xs_resids=[]
        PGs_nf = []
        for params in torch.tensor(param_samples):
            lat_xs_energy, lat_xs_recon, lat_xs_resid_energy, lat_xs_resid, lat_xs_resid_err = xspec_reconstruction(params, spec_name, data_dir=data_dir)
            lat_xs_recons.append((lat_xs_energy, lat_xs_recon))
            lat_xs_resids.append((lat_xs_resid_energy, lat_xs_resid, lat_xs_resid_err))
            PGs_nf.append(reduced_PG(params, spec_name, data_dir=data_dir))
        lat_xs_recons = np.array(lat_xs_recons)
        lat_xs_resids = np.array(lat_xs_resids)
        median_PG_nf = np.median(PGs_nf)

        # xspec ground truth reconstructions
        true_xs_energy, true_xs_recon, true_xs_resid_energy, true_xs_resid, true_xs_resid_err = xspec_reconstruction(targs, spec_name, data_dir=data_dir)
        PG_xspec = reduced_PG(targs, spec_name)

        if net is not None and decoder is not None:
            # decoder reconstructions
            lat_dec_recons=[]
            for params in torch.tensor(param_samples):
                lat_dec_recons.append(decoder_reconstruction(params, decoder, net, spec_name, data_dir))

        # plotting
        fig, axes = plt.subplot_mosaic('a\na\na\nb\nb', figsize=(16,14))
        recon_ax = axes['a']
        resid_ax = axes['b']
        fig.subplots_adjust(hspace=0.0)
        if specific_data:
            recon_ax.set_title(specific_data['object'][spec_num], fontsize=MAJOR)
        else:
            recon_ax.set_title('Reconstruction', fontsize=MAJOR)
        recon_ax.set_xlabel('Energy (keV)', fontsize=MINOR)
        recon_ax.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        recon_ax.set_xscale('log')
        recon_ax.set_yscale('log')
        recon_ax.tick_params(labelsize=MINOR)
        recon_ax.set_xlim(dec_energy[0], dec_energy[239])

        resid_ax.tick_params(labelsize=MINOR)
        resid_ax.set_xlabel('Energy (keV)', fontsize=MINOR)
        resid_ax.set_ylabel('Data / Model', fontsize=MINOR)
        resid_ax.set_xscale('log')
        resid_ax.set_xlim(dec_energy[0], dec_energy[239])
 
        # decoder reconstructions
        if net is not None and decoder is not None:
            recon_ax.plot(dec_energy, np.squeeze(lat_dec_recons[0]), color='#39db39', marker=None, label='decoder latent', alpha=0.1, linewidth=3)
            resid_ax.errorbar(dec_energy, inp/np.squeeze(lat_dec_recons[0]), inp_err/np.squeeze(lat_dec_recons[0]), linestyle='None', color='#39db39', marker=None, alpha=0.1, linewidth=3)
            for lat_dec_recon in lat_dec_recons[1:]:
                recon_ax.plot(dec_energy, np.squeeze(lat_dec_recon), color='#39db39', alpha=0.1, linewidth=3)
                resid_ax.errorbar(dec_energy, inp/np.squeeze(lat_dec_recon), inp_err/np.squeeze(lat_dec_recon), linestyle='None', color='#39db39', alpha=0.1, linewidth=3)

        # data
        err=recon_ax.errorbar(x=dec_energy[:240],y=inp, yerr=inp_err, linestyle="None", color='k', capsize=3, elinewidth=2, label='data')

        # xspec latent reconstructions
        recon_ax.plot(lat_xs_recons[0,0,:], lat_xs_recons[0,1,:], color=colour, marker=None, label='latent', alpha=0.1, linewidth=3)
        resid_ax.errorbar(lat_xs_resids[0,0,:], lat_xs_resids[0,1,:], yerr=lat_xs_resids[0,2,:], color=colour, alpha=0.1, elinewidth=3, linestyle='None')
        print('\rreconstruction #', 0)
        for i, (lat_xs_recon, lat_xs_resid) in enumerate(zip(lat_xs_recons[1:], lat_xs_resids[1:])):
            print('\rreconstruction # '+str(i), end='\r')
            recon_ax.plot(lat_xs_recon[0], lat_xs_recon[1], color=colour, alpha=0.1, linewidth=3)
            resid_ax.errorbar(lat_xs_resid[0], lat_xs_resid[1], yerr=abs(lat_xs_resid[2]), color=colour, alpha=0.1, elinewidth=3, linestyle='None')

        # xspec target reconstructions
        recon_ax.plot(true_xs_energy, true_xs_recon, linewidth=5, color='#de5454', label='target')
        resid_ax.errorbar(true_xs_resid_energy, true_xs_resid, true_xs_resid_err, linestyle='None', marker=None, color='#de5454', elinewidth=3)

        plt.text(0, 0, f'Reduced PG (Target): {np.format_float_positional(PG_xspec, precision=3)}\nReduced PG (Network): {np.format_float_positional(median_PG_nf, precision=3)}', transform=fig.transFigure, fontsize=20)

        legend_elements = [err]\
            +[Line2D([0], [0], color='#de5454', label='Target parameters, Xspec reconstructions', linestyle='--', marker='None', alpha=1, markersize=10, linewidth=3)]\
            +[Line2D([0], [0], color=colour, label='Network parameters, Xspec Rreconstructions', linestyle='-', marker='None', alpha=1, markersize=10, linewidth=3)]
        if decoder: legend_elements+=[Line2D([0], [0], color='#39db39', label='Network parameters, Decoder reconstructions', linestyle='-', marker='None', alpha=1, markersize=10, linewidth=3)]

        fig.legend(handles=list(legend_elements), fancybox=False, shadow=False, fontsize=18)

        if dir_name:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            plt.savefig(dir_name+'NF_post_pred_plot_xspec'+str(spec_num)+'.png', dpi=300)
            print('Saved xspec posterior predictive plot for', specific_data['object'][spec_num] if specific_data else data['ids'][spec_num])

    return post_pred_samples

def rec_2d_plot(
    data: dict,
    plot_dir: str | None = None,
    n_spectra: int = 200,
    data_dir: str | None = None):
    '''
    Plot 2D color plots of spectra reconstructions from latent parameters to show multiple spectra together

    parameters
    ----------
    plot_dir: 
        directory to save plots to
    data: 
        dictionary of data containing inputs, targets, latents and ids
    specific_data: 
        dictionary of specific data containing inputs, targets, latents and ids for specific
    data_dir: 
        directory of spectra data files - if we want to do xspec reconstruction (working on this)
    '''
    title_size = 16
    label_size = 14
    tick_size = 12

    energies = get_energies()[:240] # energies

    # shortens data to what we want to plot
    shortened_data = {key: data[key][:n_spectra] for key in data.keys()}

    # add state labels and order data by state
    ordered_data = order_by_state(shortened_data)

    states = ordered_data['state']  # spectra states
    inputs = ordered_data['inputs'][:,0,:] # spectra data
    preds = ordered_data['preds'][:,0,:]   # spectra predictions
    # resids = (inputs-preds)/(np.max(preds, axis=1)[np.newaxis].swapaxes(0,1))
    resids = inputs/preds # plot ratio - as suggested by xspec

    # indexes in ordered_data['state'] where state changes
    lines = [0]+list(np.where(np.array(states[:-1]) !=  np.array(states[1:]))[0])+[n_spectra]

    # plot format
    fig = plt.figure(figsize=(8,6), layout='constrained')
    ax = fig.gca()
    ax.set_xlabel('Energy (keV)', fontsize=label_size)
    ax.set_xscale('log')
    ax.set_xlim(energies[0],energies[-1])
    ax.set_ylim(0,len(inputs))

    ax.hlines(lines, energies[0], energies[-1], colors='#f2c42c', linestyles='-', linewidths=1) # adding lines to show state boundaries
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # ax.set_title('log(Inputs)-log(Preds)/log(Peak Value)', fontsize=title_size) 
    ax.set_title('Data / Decoder Reconstruction', fontsize=title_size) 

    for i, state in enumerate(list(dict.fromkeys(states))):   # add state labels
        ax.text(10.5, lines[i] + (lines[i+1]-lines[i])/2, state)

    # resid_Norm = MidPointLogNorm(vmin=np.min(resids), vmax=np.max(resids), midpoint=1.0)
    resid_Norm = pltcolors.TwoSlopeNorm(vcenter=1, vmin=resids.min(), vmax=resids.max())
    # resid_Norm = pltcolors.CenteredNorm(vcenter=1, halfrange=1)
    # plotting data
    plot_kwargs = {'aspect':'auto', 'extent':(energies[0], energies[-1], 0, len(preds)),
                'origin':'lower', 'cmap':'seismic', 'interpolation':'none', 'norm': resid_Norm}
    image = ax.imshow(resids, **plot_kwargs)
    cbar_resid = fig.colorbar(image, ax=ax, ticks=[0,0.25,0.5,0.75,1,2,3,4,5,6,7,8], orientation='horizontal', aspect=40)
    cbar_resid.ax.tick_params(labelsize=tick_size)
    cbar_resid.set_label(label='Flux (Counts s$^{-1}$ keV$^{-1}$)', size=label_size)

    if plot_dir:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        plt.savefig(os.path.join(plot_dir,'2d_recon.png'), dpi=300)
    plt.close()

def resid_params_plot(
    data: dict,
    x_param: list | ndarray,
    x_param_name: str,
    n_samples: int | None =  100,
    n_spectra: int| None = 250,
    log_params: list = LOG_PARAMS,
    plot_dir: str | None = None,
    save_name: str | None = 'Param_resid.png'):
    ''' WORK IN PROGRESS
    Plot residuals of parameters (true - predicted) against one parameter to show trends between prediction error and that parameter
    Main idea is to have count rate or normalisation/disk_temperature on x-axis as these are somewhat proxies on strength of the components
    
    parameters
    ----------
    data: 
        dictionary of data containing inputs, targets, latents and ids
    resid_param_idx:
        index of parameter we want to take uncertainty of
    x_param: 
        list or ndarray of parameter values to plot against or int of parameter index in
    x_param_name: 
        name of parameter to plot against
    dir_name: 
        directory to save plots to
    '''

    title_size = 28
    tick_size = 20
    label_size = 24

    # defines colours for each state
    hard_color = '#1a66d9'
    thermal_color = '#d93d2b'
    SPL_color = '#ebcb2a'
    misc_color = '#49a333'

    # labels data if not already labelled
    labelled_data = data.copy() if 'state' in data else add_state_labels(data).copy()

    # ensures correct dimensions of data
    true = labelled_data['targets'][:n_spectra,0, np.newaxis,:]
    pred = labelled_data['latent'][:n_spectra,:n_samples,:]
    state = labelled_data['state'][:n_spectra]
    x_param = x_param[:n_spectra]

    log_true = true.copy()  # logs values that we want to look at on log scale
    log_pred = pred.copy()
    log_true[:,:,log_params] = np.log10(log_true[:,:,log_params])
    log_pred[:,:,log_params] = np.log10(log_pred[:,:,log_params])

    # computes residuals 
    resids = ((log_true-log_pred)).swapaxes(0,2)

    # indexes of pegged and non-pegged_values
    pegged_idxs = np.argwhere(np.char.find(state, 'Pegged')!=-1).flatten()
    free_idxs = np.argwhere(np.char.find(state, 'Pegged')==-1).flatten()

    # makes colours list corresponding to state labels
    colors = np.array([hard_color if 'Hard' in label \
                else thermal_color if 'Thermal' in label \
                else SPL_color if 'SPL' in label \
                else misc_color if 'Misc' in label \
                else '#9c33a3' for label in state])

    fig, ax_dict = plt.subplot_mosaic('aabbcc\ndddeee', figsize=(16,14), layout="constrained")
    fig.suptitle('Parameter Residudals', fontsize=title_size)

    for i, (ax, resid) in enumerate(zip(ax_dict.values(), resids)):
        ax.set_title(PARAM_NAMES[i], fontsize=title_size)
        ax.set_xlabel(x_param_name, fontsize=label_size)
        if i in log_params: ax.set_ylabel('log(true)-log(pred)', fontsize=label_size)
        else: ax.set_ylabel('true-pred', fontsize=label_size)
        ax.set_xscale('log')
        ax.tick_params(labelsize=tick_size)

        for sample in resid:    # for each sample we make a plot across all selected spectra
            ax.scatter(x_param[pegged_idxs], sample[pegged_idxs], alpha=0.05, color=colors[pegged_idxs], marker='x', s=20)
            ax.scatter(x_param[free_idxs], sample[free_idxs], alpha=0.05, color=colors[free_idxs], marker='o', s=20)

    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir,save_name), dpi=300)
    plt.close()

def coverage_plot(
    dataset,
    loaders,
    network,
    dir_name,
    coverage_dir = './coverages/',
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
        levels, coverages = lampe.diagnostics.expected_coverage_mc(network.net.net[0], testset)#, device=get_device()[1])
        data = [levels, coverages]

        if not os.path.exists(coverage_dir):
            os.makedirs(coverage_dir)
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
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(os.path.join(dir_name, 'coverage'), dpi=300)
    plt.close()

def labels_plot(data: dict,
                n_spectra: int = 10000,
                plot_dir: str | None = None,
                save_name: str | None = None):

    """
    Plot target scattered fraction against gamma and colour by label to show whether misc labels correspond with pegged values

    Parameters
    ----------

    """

    title_size = 20
    label_size = 18
    tick_size = 14

    # defines colours for each state
    hard_color = '#1a66d9'
    thermal_color = '#d93d2b'
    SPL_color = '#ebcb2a'
    misc_color = '#49a333'

    # shortens data to what we want to plot
    shortened_data = {key: data[key][:n_spectra] for key in data.keys()}

    # groups different spectra based on parameters
    labelled_data = add_state_labels(shortened_data)

    pegged_idxs = np.argwhere(np.char.find(labelled_data['state'], 'Pegged')!=-1).flatten()
    free_idxs = np.argwhere(np.char.find(labelled_data['state'], 'Pegged')==-1).flatten()

    # makes colours list corresponding to state labels
    colors = np.array([hard_color if 'Hard' in label \
               else thermal_color if 'Thermal' in label \
               else SPL_color if 'SPL' in label \
               else misc_color if 'Misc' in label \
               else '#9c33a3' for label in labelled_data['state']])

    # plots parameters with corresponding color
    fig = plt.figure(figsize=(8,8))

    # 2d plot
    axis = fig.gca()
    axis.scatter(labelled_data['targets'][pegged_idxs,0,1], labelled_data['targets'][[pegged_idxs],0,2], c=colors[pegged_idxs], marker='x', alpha=0.3)
    axis.scatter(labelled_data['targets'][free_idxs,0,1], labelled_data['targets'][free_idxs,0,2], color=colors[free_idxs], marker='o', alpha=0.3)
    
    axis.set_title('Parameter comparisons', fontsize=title_size)
    axis.set_xlabel('Gamma', fontsize=label_size)
    axis.set_ylabel('Scattered Fraction', fontsize=label_size)
    axis.tick_params(axis='both', which='major', labelsize=tick_size)

    axis.set_yscale('log')

    if plot_dir and save_name:
        plt.savefig(os.path.join(plot_dir,save_name), dpi=300)

    # 3d plot
    axis = fig.add_subplot(projection='3d')
    axis.scatter(labelled_data['targets'][pegged_idxs,0,1], np.log10(labelled_data['targets'][pegged_idxs,0,2]),
                  np.log10(labelled_data['targets'][pegged_idxs,0,3]), color=colors[pegged_idxs], marker='x', alpha=0.3)
    axis.scatter(labelled_data['targets'][free_idxs,0,1], np.log10(labelled_data['targets'][free_idxs,0,2]),
                  np.log10(labelled_data['targets'][free_idxs,0,3]), color=colors[free_idxs], marker='o', alpha=0.3)
    
    axis.set_title('Parameter comparisons', fontsize=title_size)
    axis.set_xlabel('Gamma', fontsize=label_size)
    axis.set_ylabel('Scattered Fraction', fontsize=label_size)
    axis.set_zlabel('Disk Temperature', fontsize=label_size)
    axis.tick_params(axis='both', which='major', labelsize=tick_size)

    axis.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    axis.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if plot_dir and save_name:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir,save_name.replace('.', '_3d.')), dpi=300)

    plt.close()
