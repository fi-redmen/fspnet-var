import numpy as np
import pandas as pd
import sciplots
import xspec
import torch
import os

from typing import Any
from numpy import ndarray
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from chainconsumer import Chain, ChainConsumer, PlotConfig, ChainConfig, Truth

from fspnet.utils.plots import plot_param_pairs, _plot_histogram

from plot_utils import get_energies, xspec_reconstruction, decoder_reconstruction, quantile_limits
from misc_utils import sample

RECTANGLE: tuple[int, int] = (16, 9)
MAJOR: int = 28
MINOR: int = 24
SCATTER_NUM: int = 1000
HI_RES: tuple[int, int] = (32, 18)
ERR_DIST: bool = False  # plot error in the distribution
CAPSIZE: int = 2
LOG_PARAMS: list[int] = [0, 2, 3, 4]  # parameters to be plotted in log scale
PARAM_LIMS: ndarray = np.array([[5.0e-3,75],[1.3,4],[1.0e-3,1],[2.5e-2, 4],[1.0e-2, 1.0e+10]])
PARAM_NAMES : ndarray = np.array(['$N_{H}$ $(10^{22}\ cm^{-2})$', '$\Gamma$', '$f_{sc}$','$kT_{disk}$ $(keV)$','$N$'])
COLORS_LIST = ['#0C5DA5', '#00B945', '#FF9500', '#9159ab', '#00A7C6']
SYNTHETIC_DIR: str = '/Users/astroai/Projects/FSPNet/data/synth_spectra_clean.pickle'

NAMES = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp', 
           'js_ni0103010102_0mpu7_goddard_GTI0.jsgrp',
           'js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
           'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp',
           'js_ni1100320119_0mpu7_goddard_GTI26.jsgrp']
COLORS: dict = {
    NAMES[0]: COLORS_LIST[0],
    NAMES[1]: COLORS_LIST[1],
    NAMES[2]: COLORS_LIST[2],
    NAMES[3]: COLORS_LIST[3],
    NAMES[4]: COLORS_LIST[4]
}

MIN_QUANT: dict = {
    NAMES[0]: 0.005,
    NAMES[1]: 0.005,
    NAMES[2]: 0.0001,
    NAMES[3]: 0.001,
    NAMES[4]: 0.001
}

MAX_QUANT: dict = {
    NAMES[0]: 0.995,
    NAMES[1]: 0.995,
    NAMES[2]: 0.9999,
    NAMES[3]: 0.999,
    NAMES[4]: 0.999
}


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
    plots_dir : string
        Directory to save plots
    y_label : string
        Performance metric
    val : list[Any] | ndarray
        Validation performance
    log_y : boolean, default = True
        If y-axis should be logged
    plots_dir : str, default = None
        Directory to save plots
    train : list[Any] | ndarray, default = None
        Training performance
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
    n_points: int = 100,
    num_specs: int = 3,
    param_names: str | list[str]=PARAM_NAMES,
    log_params: list[int]=LOG_PARAMS,
    specific_data: dict | None = None, #to choose which specific spectrum to take, from a list of spectra names)
    ):
    '''
    Plots:
    1) a random 1000? points (in grey) corresponding to maximum of each parameter distribution
    2) num_specs points (in different colours) where each set of points is sampled from different sets of parameter distributions
    3) 1:1 relation
    '''

    
    if specific_data:
        num_specs = np.min([len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1)),5])
        colors = COLORS

    else:
        num_specs = np.min([num_specs, 5])
        colors = COLORS_LIST

    fig, axes = plt.subplot_mosaic('aabbcc\ndddeee', figsize=(16,9))
    fig.subplots_adjust(top=0.8, hspace=0.5, wspace=0.8, left=0.05, right=0.95)
    
    # Adjust the title position
    fig.suptitle('Comparison Plot', fontsize=MAJOR, y=0.99)  # Move the title slightly higher

    # grey plot - data for the grey plots has 5 parameters which each have their own 1620 corresponding spectra (limited to SCATTER_NUM)
    grey_targs = data['targets'].swapaxes(0,1).swapaxes(1,2)[0][:,:SCATTER_NUM]
    grey_lats = data['latent'].swapaxes(0,1).swapaxes(1,2)[0][:,:SCATTER_NUM]


    # loop through each set of parameters, plotting the outputs vs inputs for SCATTER_NUM points
    for i, (targ, lat, axis) in enumerate(zip(grey_targs, grey_lats, axes.values())):
        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

        axis.set_xlabel('inputs', fontsize=MINOR)
        axis.set_ylabel('outputs', fontsize=MINOR)
        axis.set_title(param_names[i], fontsize=MAJOR)

        axis.tick_params(labelsize=20)

        # for taking maximum from the distibution rather than frist sample
        # lat = [np.max(data['latent'].swapaxes(1,2).swapaxes(1,0)[i][j]) for j in range(0,SCATTER_NUM)]

        axis.set_xlim(np.min(targ), np.max(targ))
        axis.set_ylim(np.min(targ), np.max(targ))

        axis.tick_params(axis='x', labelsize=20)
        axis.tick_params(axis='y', labelsize=20)
        
        axis.plot([np.min(targ), np.max(targ)], [np.min(targ), np.max(targ)], color='k')
        axis.scatter(x=targ, y=lat, linestyle='None', color='grey', alpha=0.3, s=3)

    # fig.legend(fontsize=20)
    
    legend_elements = [Line2D([0], [0], color='grey', label='Single Sample', linestyle='None', marker='o', alpha=0.5, markersize=10)]

    fig.legend(handles=list(legend_elements), bbox_to_anchor=(0.5, 0.95), fancybox=False, shadow=False,
                                ncol=4, fontsize=20, handletextpad=0.4, columnspacing=0.4, loc='upper center')

    plt.savefig(dir_name+'NF_comparison_grey.png', dpi=300)

    # adding more distributions corresponding to more spectra
    # loop through each set of parameters, plotting, for 5? different spectra, their distributions from 10? data points
    multi_targs = data['targets'].swapaxes(1,2).swapaxes(0,1)
    multi_lats = data['latent'].swapaxes(1,2).swapaxes(0,1)
    for i, (targ, lat, axis, grey_targ) in enumerate(zip(multi_targs, multi_lats, axes.values(), grey_targs)):
        # targ[spectrum number][value or error]
        # clear acis from single samples
        axis.cla()

        if i in log_params:
            axis.set_xscale('log')
            axis.set_yscale('log')

        axis.set_xlabel('inputs', fontsize=MINOR)
        axis.set_ylabel('outputs', fontsize=MINOR)
        axis.set_title(param_names[i], fontsize=MAJOR)

        axis.tick_params(labelsize=20)

        # for taking maximum from the distibution rather than frist sample
        # lat = [np.max(data['latent'].swapaxes(1,2).swapaxes(1,0)[i][j]) for j in range(0,SCATTER_NUM)]

        axis.set_xlim(np.min(grey_targ), np.max(grey_targ))
        axis.set_ylim(np.min(grey_targ), np.max(grey_targ))

        axis.tick_params(axis='x', labelsize=20)
        axis.tick_params(axis='y', labelsize=20)
        
        axis.plot([np.min(grey_targ), np.max(grey_targ)], [np.min(grey_targ), np.max(grey_targ)], color='k')

        for spec_num in range(0,100):
            axis.errorbar(x=[targ[spec_num][0]]*n_points, xerr=[targ[spec_num][1]]*n_points, y=lat[spec_num][:n_points], 
                        linestyle='None', capsize=1, ms=5, alpha=0.01, marker='o', color='grey')
    
    legend_elements = [Line2D([0], [0], color='grey', label='Multiple Samples', linestyle='None', marker='o', alpha=0.5, markersize=10)]

    fig.legend(handles=list(legend_elements), bbox_to_anchor=(0.5, 0.95), fancybox=False, shadow=False,
                                ncol=4, fontsize=20, handletextpad=0.4, columnspacing=0.4, loc='upper center')
    
    plt.savefig(dir_name+'NF_comparison_greyextra.png', dpi=300)

    # multi-color plot
    if specific_data:
        # shape of multi_targs : param_number, spectrum_number, mean+error
        multi_targs = specific_data['targets'].swapaxes(1,2).swapaxes(0,1)
        # shape of multi_lats : param_number, spectrum_number, number of samples in that spectrums distribution
        multi_lats = specific_data['latent'].swapaxes(1,2).swapaxes(0,1)
        spec_names = specific_data['id']
        object_names = specific_data['object']
    else:
        multi_targs = data['targets'].swapaxes(1,2).swapaxes(0,1)
        multi_lats = data['latent'].swapaxes(1,2).swapaxes(0,1)
        spec_names = data['ids']

    # loop through each set of parameters, plotting, for 5? different spectra, their distributions from 10? data points
    for i, (targ, lat, axis) in enumerate(zip(multi_targs, multi_lats, axes.values())):
        # targ[spectrum number][value or error]

        if specific_data:
            for spec_num in range(len(targ)):
                spec_name = spec_names[spec_num]

                axis.errorbar(x=[targ[spec_num][0]]*n_points, xerr=[targ[spec_num][1]]*n_points, y=lat[spec_num][:n_points], 
                            linestyle='None', capsize=1, ms=7, alpha=0.05, marker='o', color=colors[spec_name])

        else:
            for spec_num in range(num_specs):

                axis.errorbar(x=[targ[spec_num][0]]*n_points, xerr=[targ[spec_num][1]]*n_points, y=lat[spec_num][:n_points], 
                            linestyle='None', capsize=1, ms=7, alpha=0.05, marker='o', color=colors[spec_num])

        if specific_data:

            legend_elements = np.concat([
                [Line2D([0], [0], color='grey', label='Multiple', linestyle='None', marker='o', alpha=0.5, markersize=10)],
                # [Line2D([0], [0], color=colors[i], label='Multiple '+str(0), linestyle='None', marker='o', alpha=0.5, markersize=10)],
                [Line2D([0], [0], color=colors[spec_names[i]], label=object_names[i], linestyle='None', marker='o', alpha=0.5, markersize=10)
                for i in range(num_specs)]
            ])
        else:
            legend_elements = np.concat([
                [Line2D([0], [0], color='grey', label='Multiple', linestyle='None', marker='o', alpha=0.5, markersize=10)],
                # [Line2D([0], [0], color=colors[i], label='Multiple '+str(0), linestyle='None', marker='o', alpha=0.5, markersize=10)],
                [Line2D([0], [0], color=colors[i], label='Multiple'+str(i), linestyle='None', marker='o', alpha=0.5, markersize=10)
                for i in range(num_specs)]
            ])

    fig.legend(handles=list(legend_elements), bbox_to_anchor=(0.5, 0.95), fancybox=False, shadow=False,
                                ncol=num_specs+1, fontsize=20, handletextpad=0.05, columnspacing=0.1, loc   ='upper center')
    
    # fig.legend(fontsize=MINOR)
    plt.savefig(dir_name+'NF_comparison.png', dpi=300, bbox_inches='tight')


def corner_plot(
    data: dict,
    param_names: str | list[str],
    log_params: list[int],
    dir_name: str,
    ):

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

    # fig.axes[20].set_xlim(-10, 30)


    plt.savefig(os.path.join(dir_name, 'corner_plot.png'))

def latent_corner_plot(
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None, #to choose which specific spectrum to take, from a list of spectra names)
    xspec_data: dict | None = None,
    param_names: str | list[str] = PARAM_NAMES,
    in_param_samples: list = None,
    num_specs: int = 3,
    ):

    # To stop pandas dataframe from rounding
    pd.set_option('display.precision', 10)

    if specific_data:
        num_specs = len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1))
        colors = COLORS
    else:
        num_specs = np.min([num_specs, 5])
        colors = COLORS_LIST


    for spec_num in range(num_specs):
        colour = colors[specific_data['id'][spec_num]] if specific_data else colors[spec_num]

        c = ChainConsumer()
        c.set_plot_config(PlotConfig(legend={'fontsize':MAJOR}, label_font_size=MINOR, tick_font_size=16, max_ticks=4)) #, spacing=2))

        # putting latent data into a data frame
        if specific_data:
            NF_data = pd.DataFrame(specific_data['latent'][spec_num], columns=param_names)
            spec_name = specific_data['id'][spec_num]
        else:
            NF_data = pd.DataFrame(data['latent'][spec_num], columns=param_names)
            spec_name = data['ids'][spec_num]
        
        c.add_chain(Chain(samples=NF_data, name="Normalising Flow", color=colour, sigmas=[0,1,2]))

        # getting the xspec gauss data and putting into a data frame
        if specific_data:
            dist_targs = specific_data['targets'][spec_num][0]
            dist_targ_errs = specific_data['targets'][spec_num][1]
        else: 
            dist_targs = data['targets'][spec_num][0]
            dist_targ_errs = data['targets'][spec_num][1]

        if dist_targ_errs.all()!=0:
            targ_samples = np.array([
                np.random.normal(loc=targ, scale=targ_err, size=3000) 
                for targ, targ_err in zip(dist_targs, dist_targ_errs)]).swapaxes(0,1)
            cut_indices = [np.argwhere((targ_sample<=0)) for targ_sample in targ_samples] 
            targ_samples = [np.delete(targ_sample, cut_index) for targ_sample, cut_index in zip(targ_samples, cut_indices)]
                
            xspec_gauss_data = pd.DataFrame(targ_samples, columns=param_names).dropna()
            c.add_chain(Chain(samples=xspec_gauss_data, name="XSPEC (Gauss)", color='#e41a1c', sigmas=[0,1,2]))
        else:
            xspec_gauss_data = None

        # getting xspec MCMC data and putting into a dataframe and chain
        if xspec_data:
            # xspec_spec_num = int(np.argwhere(specific_data['id'][spec_num]==xspec_data['id']))
            xspec_MCMC_data = pd.DataFrame(np.array(xspec_data['posteriors'][spec_num]).swapaxes(0,1)[:3000,:], columns=param_names)
            
            c.add_chain(Chain(samples=xspec_MCMC_data, name='XSPEC (MCMC)', color='gray', sigmas=[0,1,2]))

        fig = c.plotter.plot()

        # settings for specific data plot vs general data plot
        if specific_data:
            fig.axes[2].set_title(specific_data['object'][spec_num], fontsize=MAJOR)
            min_quant = MIN_QUANT[spec_name]
            max_quant = MAX_QUANT[spec_name]
        else:
            min_quant=0.01
            max_quant=0.99

        # adding better limits to axes
        quantile_limits(
            fig,
            min_quant, 
            max_quant, 
            param_names,
            NF_data = NF_data,
            xspec_gauss_data = xspec_gauss_data if xspec_gauss_data else None,
            # xspec_MCMC_data
            object_name = specific_data['object'][spec_num] if specific_data else None
        )
        
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
        
        # fig = c.plotter.plot()
        plt.savefig(os.path.join(dir_name, 'latent_corner_plot'+str(spec_num)+'_1samp.png'), dpi=300)

def param_pairs_plot(    
    data: dict,
    dir_name: str,
    log_params: list[int]=LOG_PARAMS,
    param_names: list[str] = PARAM_NAMES,
    ): 

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
    data_dir: str = '/Users/astroai/Projects/FSPNet/data/spectra/',
    synthetic_dir: str = SYNTHETIC_DIR,
    spec_scroll = 0,
    ):

    # decoder and xspec reconstructions for each spectra from one set of sampled parameters
    # [Batch_size/total number of spectra, number of data points in spectra, number of parameters] - all_params_samples shape
    
    # gets the input spectra from either data or specific data
    if specific_data:    
        num_specs = len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1))

    # loop through spectra, getting reconstructions and plotting for each 
    for spec_num in range(spec_scroll, num_specs+spec_scroll):

        # if specific data given, use that. use normal data otherwise. Note: [0] is to select one set of parameter samples when there may be more
        if specific_data:
            input = specific_data['inputs'][spec_num]
            targs = specific_data['targets'][spec_num][0]
            lats = specific_data['latent'][spec_num][0]
            spec_name = specific_data['id'][spec_num]
            color_id = spec_name
            colors = COLORS
        else:
            input = data['inputs'][spec_num]
            targs = data['targets'][spec_num][0]
            lats = data['latent'][spec_num][0]
            spec_name = data['ids'][spec_num]
            color_id = spec_num-spec_scroll
            colors = COLORS_LIST
        
        
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
        axis.set_ylim(0, 1.1*np.max(np.concatenate([np.squeeze(lat_dec_recon), lat_xs_recon, np.squeeze(true_dec_recon), true_xs_recon, inp+inp_err])))
                
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
        plt.savefig(dir_name+'NF_recon_latentonly'+str(spec_num)+'.png', dpi=300)

        # decoder reconstrucitons with ground truth
        axis.plot(dec_energy, true_dec_recon[0], markersize=5, label='decoder ground truth', color='#de5454', linewidth=2)
        # xspec reconstrucitons with ground truth
        axis.plot(true_xs_energy, true_xs_recon, linestyle="--", linewidth=2, color='#de5454', label='xspec ground truth')
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
    data_dir: str = '/Users/astroai/Projects/FSPNet/data/spectra/'
    ):

    colors: dict = COLORS

    # number of spectra which is either the number of specific spectra or min(num_specs, 5)
    if specific_data:
        num_specs = len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1))
        colors = COLORS
    else:
        num_specs = np.min([num_specs, 5])
        colors = COLORS_LIST
        
    # if we haven't been given the samples already
    if not post_pred_samples:
        if  specific_data:
            post_pred_samples = sample(specific_data, num_specs=num_specs, num_samples=n_samples)
        else:
            post_pred_samples = sample(data, num_specs=num_specs, num_samples=n_samples)

    # looping over each spectrum
    for spec_num, param_samples in enumerate(post_pred_samples):

        colour = colors[spec_name] if specific_data else colors[spec_num]

        # true values, their errors and our latent distribution
        if specific_data:
            targs = specific_data['targets'][spec_num][0]
            spec_name = specific_data['id'][spec_num]
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
        for plot_num, params in enumerate(torch.tensor(param_samples)):
            if spec_name==str:
                lat_dec_recons.append(decoder_reconstruction(params, decoder, network, spec_name, data_dir if data_dir else None))
            else:
                lat_dec_recons.append(decoder_reconstruction(params, decoder, network))

        # xspec ground truth reconstructions
        true_xs_energy, true_xs_recon = xspec_reconstruction(targs, spec_name)

        # plotting
        fig = plt.figure(figsize=(16,9))
        axis = fig.gca()
        if specific_data:
            axis.set_title(specific_data['object'][spec_num], fontsize=MAJOR)
        else:
            axis.set_title('Reconstruciton', fontsize=MAJOR)
        axis.set_xlabel('Energy (keV)', fontsize=MINOR)
        axis.set_ylabel('cts / det / s/ keV', fontsize=MINOR)
        axis.set_xscale('log')
        axis.set_yscale('log')
        plt.xticks(fontsize=MINOR)
        plt.yticks(fontsize=MINOR)
        axis.set_xlim(dec_energy[0], dec_energy[239])
        axis.set_ylim(0, 1.1*np.max(np.concatenate([[np.max(lat_dec_recons)], true_xs_recon, inp+inp_err])))

        # decoder reconstructions
        axis.plot(dec_energy, np.squeeze(lat_dec_recons[0]), color=colour, marker=None, label='decoder latent', alpha=0.1)
        for lat_dec_recon in lat_dec_recons[1:]:
            axis.plot(dec_energy, np.squeeze(lat_dec_recon), color=colour, alpha=0.1)

        # xspec reconstructions
        axis.plot(true_xs_energy, true_xs_recon, linestyle="--", linewidth=2, color='#de5454', label='xspec ground truth')

        # data
        axis.errorbar(x=dec_energy[:240],y=inp, yerr=inp_err, linestyle="None", color='k', capsize=3, elinewidth=2, label='data')

        plt.legend(fontsize=MINOR)
        plt.savefig(dir_name+'NF_post_pred_plot'+str(spec_num)+'.png', dpi=300)

    return post_pred_samples

def rec_2d_plot(
    decoder,
    network,
    dir_name: str,
    data: dict | None = None,
    specific_data: dict | None = None,
    data_dir: str = '/Users/astroai/Projects/FSPNet/data/spectra/'
    ):

    # if we have specific data, use that. otherwise use normal data - make length of spectra 240 for non specific and length of specific data for specific
    if specific_data:
        latents = specific_data['latent'][:240,0,:]
        targets = specific_data['targets'][:240,0,:]
        spec_names = specific_data['id'][:240]
        num_specs = len(specific_data['targets'].swapaxes(1,2).swapaxes(0,1))
    else:
        latents = data['latent'][:240,0,:]
        targets = data['targets'][:240,0,:]
        spec_names = data['ids'][:240]
        num_specs = 240

    
    # getting energy for decoder plots and spectra from decoder and xspec reconstructions
    dec_energy = get_energies(spec_names[1], data_dir)[:240]

    lat_dec_recons = []
    for spec_num, (params, spec_name) in enumerate(zip(latents, spec_names,)):
        lat_dec_recons.append(decoder_reconstruction(params, decoder, network, spec_name, data_dir if data_dir else None))
    
    
    X, Y = np.meshgrid(dec_energy, np.linspace(0, num_specs))
    Z = np.meshgrid(np.array(lat_dec_recons).squeeze())

    fig = plt.figure(figsize=(16,9))
    ax = plt.gca()

    pc = ax.imshow(Z, cmap='plasma') # norm=LogNorm(vmin=0.01, vmax=100))
    fig.colorbar(pc, ax=ax, extend='both')
    ax.set_title('imshow() with LogNorm()')