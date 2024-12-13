
from fspnet.spectrum_fit import init
from fspnet.spectrum_fit import AutoencoderNet 
import xspec
#from fspnet.utils.data import __get_tiem__

import os
import pickle
from typing import Any, Self, BinaryIO

import torch
import numpy as np
from netloader.networks import Decoder
import netloader.networks as nets
from netloader.network import Network
from netloader.utils import transforms
from netloader.utils.utils import save_name, get_device
from torch.utils.data import DataLoader
from torch import nn, Tensor
from numpy import ndarray

from fspnet.utils import plots
from fspnet.utils.utils import open_config
from fspnet.utils.data import SpectrumDataset, loader_init
from fspnet.utils.analysis import autoencoder_saliency, decoder_saliency, pyxspec_test

#gaussian loss for variational autoencoder with uncertainties
def gaussian_loss(predictions, target):
    #return function from gaussian loss website
    return nn.GaussianNLLLoss()(predictions, target[:,0])#, target[:,1]**2) #(spectra)(B,240), target[:,1]**2 (B,2,240) (uncertainty)) - for variational with uncertaimties??
    # return nn.GaussianNLLLoss()(predictions, target, target)

#MSE loss for variational autoencoder
def mse_loss(predictions, target):
    return nn.MSELoss(predictions, target[:,0]) #(spectra))(B,240)

def net_init(
        datasets: tuple[SpectrumDataset, SpectrumDataset],
        config: str | dict[str, Any] = './config.yaml',
) -> tuple[nets.BaseNetwork, nets.BaseNetwork]:
    """
    Initialises the network

    Parameters
    ----------
    datasets : tuple[SpectrumDataset, SpectrumDataset]
        Encoder and decoder datasets
    config : string | dictionary, default = './config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[BaseNetwork, BaseNetwork]
        Constructed decoder and autoencoder
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    # Load config parameters
    e_save_num = config['training']['encoder-save']
    e_load_num = config['training']['encoder-load']
    d_save_num = config['training']['decoder-save']
    d_load_num = config['training']['decoder-load']
    learning_rate = config['training']['learning-rate']
    encoder_name = config['training']['encoder-name']
    decoder_name = config['training']['decoder-name']
    description = config['training']['network-description']
    networks_dir = config['training']['network-configs-directory']
    log_params = config['model']['log-parameters']
    states_dir = config['output']['network-states-directory']
    device = get_device()[1]

    print('device:', device)

    if d_load_num:
        decoder = nets.load_net(d_load_num, states_dir, decoder_name)
        decoder.description = description
        decoder.save_path = save_name(d_save_num, states_dir, decoder_name)
        transform = decoder.header['targets']
        param_transform = decoder.in_transform
    else:
        transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=-1),
            transforms.Log(),
        ])
        transform.transforms.append(transforms.Normalise(
            transform(datasets[1].spectra),
            mean=False,
        ))
        param_transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=0, idxs=log_params),
            transforms.Log(idxs=log_params),
        ])
        param_transform.transforms.append(transforms.Normalise(
            param_transform(datasets[1].params),
            dim=0,
        ))
        decoder = Network(
            decoder_name,
            networks_dir,
            list(datasets[1][0][1].shape),
            list(datasets[1][0][2].shape),
        )
        decoder = nets.Decoder(
            d_save_num,
            states_dir,
            decoder,
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
        )

        decoder.in_transform = param_transform
        #changes decoder loss to gaussian for variational autoencoder
        # decoder.loss_func = gaussian_loss    #changes loss to gaussian loss - can look into other typs of loss

    if e_load_num:
        net = nets.load_net(e_load_num, states_dir, encoder_name)
        net.description = description
        net.save_path = save_name(e_save_num, states_dir, encoder_name)
    else:
        net = Network(
            encoder_name,
            networks_dir,
            list(datasets[0][0][2].shape),
            list(datasets[0][0][1].shape),
        )
        net = nets.Autoencoder(
            e_save_num,
            states_dir,
            AutoencoderNet(net, decoder.net, name=encoder_name),
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
            latent_transform=param_transform,
        )
        net.bound_loss = 0
        #net.kl_loss = 0

        #changes autoencoder loss for variational autoencoder, mse for non-variational autoencoder
        # net.reconstruct_func = gaussian_loss   
        # net.latent_func = mse_loss              #adds latent loss as mse

        # net = nets.Encoder(
        #     e_save_num,
        #     states_dir,
        #     net,
        #     learning_rate=learning_rate,
        #     description=description,
        #     transform=param_transform,
        # )
        # net.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     net.optimiser,
        #     factor=0.5,
        #     min_lr=1e-5,
        # )
        # net._loss_function = encoder_loss

    for dataset in datasets:
        dataset.spectra, dataset.uncertainty = transform(
            dataset.spectra,
            uncertainty=dataset.uncertainty,
        ) # same as before

        # for non-variational autoencoder
        # dataset.params = param_transform(dataset.params)

        # for variational autoencoder (with uncertainties??)
        dataset.params, dataset.param_uncertainty = param_transform(
            dataset.params,
            uncertainty=dataset. param_uncertainty,
        ) # this has been changed
        #note: changed from: dataset.params = param_transform(dataset.params)

    return decoder.to(device), net.to(device)
    '''
    for dataset in datasets:
        dataset.spectra, dataset.uncertainties = transform(
            dataset.spectra,
            uncertainties=dataset.uncertainties,
        )
        dataset.params = param_transform(dataset.params)
    return decoder.to(device), net.to(device)
    '''
def init(config: dict | str = './config.yaml') -> tuple[
        tuple[DataLoader, DataLoader],
        tuple[DataLoader, DataLoader],
        nets.BaseNetwork,
        nets.BaseNetwork]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = './config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], tuple[Dataloader, Dataloader], BaseNetwork, BaseNetwork]
        Train & validation dataloaders for decoder and autoencoder, decoder, and autoencoder
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    # Load config parameters
    batch_size = config['training']['batch-size']
    val_frac = config['training']['validation-fraction']
    e_data_path = config['data']['encoder-data-path']
    d_data_path = config['data']['decoder-data-path']
    log_params = config['model']['log-parameters']

    # Fetch dataset & network
    e_dataset = SpectrumDataset(e_data_path, log_params)
    d_dataset = SpectrumDataset(d_data_path, log_params)
    decoder, net = net_init((e_dataset, d_dataset), config)

    # Initialise datasets
    e_loaders = loader_init(e_dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    d_loaders = loader_init(d_dataset, batch_size=batch_size, val_frac=val_frac, idxs=decoder.idxs)
    net.idxs = e_dataset.idxs
    decoder.idxs = d_dataset.idxs
    return e_dataset, d_dataset, e_loaders, d_loaders, decoder, net
    #added e_dataset, d_dataset changed. put this function into code
    #do the same with net_init - so you dont have to untransfrom and retransform


'''shows what the data file looks like
with open('./data/spectra.pickle', 'rb') as file:
    data = pickle.load(file)
print(data)
'''

num_epochs = 1

#initialise data loaders and networks
e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init()

#generate predictions for the autoencoder validation data loader (e_loaders contains training and validation data loaders)
#data = net.predict(e_loaders[-1], path='/Users/astroai/Downloads/Spectrum-Machine-Learning-master/predictions')

#train decoder
decoder.training(num_epochs, d_loaders)

#fix decoder's weights so they dont change while training the encoder
net.net.net[1].requires_grad_(False)

#train encodeer as autoencoder
net.training(num_epochs, e_loaders)

#---manually transforming data---#
# save transforms
transform = net.header['inputs']
param_transform = net.header['targets']

# clear transforms - only for vatiational autoencoeders
# net.header['inputs'] = None
# net.header['targets'] = None

# predict & transform
data = net.predict(e_loaders[-1], path='./predictions/prediction')
# for variational autoencoders (with uncertainties)
# data['targets'] = np.stack(param_transform(data['targets'][:,0], back=True,
#                                             uncertainty=data['targets'][:,1]), axis=1)
# data['inputs'] = np.stack(transform(data['inputs'][:,0], back=True,
#                                            uncertainty=data['inputs'][:,1]), axis=1)
# might not need to transform these..?
# data['preds'] = np.stack(transform(data['preds'][:,0], back=True,
#                                    uncertainty=data['preds']), axis=1)
# data['latent'] = np.stack(param_transform(data['latent'][:,0], back=True,
#                                             uncertainty=data['latent'][:,1]), axis=1)

# using checkpoints for parameters with error instead of latenet
# err_dat = np.stack(param_transform(net.net.net[0].checkpoints[-1][:,0], back=True,
#                                    uncertainty=net.net.net[0].checkpoints[-1][:,1]), axis=1)
#reset transforms
net.header['inputs'] = transform
net.header['targets'] = param_transform


#generate predictions
#data = net.predict(e_loaders[-1], path='./predictions/prediction')


'''---------- PLOTTING LOSS PER EPOCH USING plot_performance ----------'''
#plots performance of decoder
plots.plot_performance(
    'Loss',
    decoder.losses[1],
    plots_dir='/Users/astroai/Projects/FSPNet/plots-backtoAE/',
    train=decoder.losses[0],
)


# '''---------- COMPARING PARAMETER PREDICTIONS AGAINST FITTED PARAMETERS ----------'''
log_params = [0,2,3,4]
param_names = ['$N_{H}$ $(10^{22}\ cm^{-2})$', '$\Gamma$', '$f_{sc}$',r'$kT_{\rm disk}$ (keV)','$N$']


#plots parameter comparisons
# for variational autencoder (with uncertainties), use [:,0] after targets and latent, for non-var, don't
plots.plot_param_comparison(
    log_params,
    param_names,
    data['targets'],
    data['latent'],
    plots_dir='/Users/astroai/Projects/FSPNet/plots-backtoAE/',
)

# plots parameter distributions using plot_multi_plot
plots.plot_multi_plot(
    ['Target', 'Prediction'],
    [data['targets'], data['latent']],
    plots.plot_param_distribution,
    plots_dir='/Users/astroai/Projects/FSPNet/plots-backtoAE/',
    y_axis=False,
    log_params=log_params,
    param_names=param_names,
    # data_range=[[],[],[],[],[], [],[],[],[]]
)

# plots parameter pair plot uing plot_multi_plot
plots.plot_multi_plot(
    ['Targets', 'Predictions'],
    [data['targets'], data['latent']],
    plots.plot_param_pairs,
    plots_dir='/Users/astroai/Projects/FSPNet/plots-backtoAE/',
    log_params=log_params,
    param_names=param_names
)

# '''---------- CALCULATING AND PLOTTING SALIENCY ----------'''
from fspnet.utils.plots import plot_saliency
from fspnet.utils.analysis import autoencoder_saliency, decoder_saliency

# #Initialise networks and data loaders
# e_loaders, d_loaders, decoder, net = init()

#calculate saliencies
# decoder_saliency(d_loaders[1], decoder.net)
# saliency_output = autoencoder_saliency(e_loaders[1], net.net)
# plot_saliency(*saliency_output, plots_dir='/Users/astroai/Projects/FSPNet/plots-backtoAE/')


# '''---------- PLOTTING LINEAR WEIGHTS USING plot_linear_weights ----------'''
# # #initialise decoder
# # decoder = init()[-2]

# #plot linear weights
# plots.plot_linear_weights(param_names, decoder.net, plots_dir='/Users/astroai/Projects/FSPNet/plots-backtoAE/')


'''---------- REDUCED PGSTAT OF ENCODER'S PREDICTIONS USING pyxspec_tests ----------'''
# from fspnet.spectrum_fit import pyxspec_tests

# #initialise autoencoder and data loader
# e_loaders, *_, = init()

# #generate predictions
# data = net.predict(e_loaders[-1])

# #start pyxspec tests
# pyxspec_tests(data)


'''---------- getting uncertainties out ----------'''

#param_transform = decoder.in_transform
#spectra_transform = decoder.header['targets']

#torch.mean(dataset.params) == 0
#torch.std(dataset.params) == 1


#print("params: ", param_transform(d_dataset.params, back=True, uncertainty=d_dataset.param_uncertainty))



''' changing loss function'''

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from fspnet.utils.utils import subplot_grid

RECTANGLE: tuple[int, int] = (16, 9)

def _init_subplots(
        subplots: str | tuple[int, int] | list | ndarray,
        fig: FigureBase | None = None,
        fig_size: tuple[int, int] = RECTANGLE,
        **kwargs: Any) -> tuple[dict[str, Axes] | ndarray[Axes], FigureBase]:
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

#manual param comparison of errors -- could just do this using ethans param comparison - just change param names to error
# plt.title('param err comparison - n_H')
# plt.xlabel('target')
# plt.ylabel('pred')
# plt.scatter(data['targets'][:,0,0], err_dat[:,0,0]) - doesnt work bc targets has 1620 data points and err_dat has 60...?

# swapaxes(0,1) for non-variational (with uncertainties), swapaxes(0,2) for variational
# err_dat = data['latent'].swapaxes(0,1)
# err_targets = data['targets'].swapaxes(0,1)
# # err_dat = err_dat.swapaxes(0,2)
# axes, fig =  _init_subplots(subplot_grid(len(param_names)))
# for i, (name, axis, dat, targ) in enumerate(zip(
#     param_names,
#     axes.values(),
#     err_dat,
#     err_targets
#     )):
#     axis.set_title('means of '+name)
#     # axis.hist(targ[0])
#     axis.hist(dat[0], alpha=0.3)

# plt.show()

# plt.title('errors')
# plt.hist(err_dat[:,1,0])
# plt.show()
