
from fspnet.spectrum_fit import init
from fspnet.spectrum_fit import AutoencoderNet 
import xspec
#from fspnet.utils.data import __get_tiem__
from astropy.io import fits

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
from torch import nn, optim, Tensor
from numpy import ndarray

from fspnet.utils import plots
from fspnet.utils.utils import open_config
from fspnet.utils.data import SpectrumDataset, loader_init
from fspnet.utils.analysis import autoencoder_saliency, decoder_saliency, pyxspec_test

import matplotlib.pyplot as plt

from VAE_plots import comparison_plot, distribution_plot, reconstruction_plot, reconstruction_plot_w_params

#gaussian loss for variational autoencoder with uncertainties
def gaussian_loss(predictions, target):
    #return function from gaussian loss website
    return nn.GaussianNLLLoss()(predictions[:,0], target[:,0], target[:,1]**2) #(spectra)(B,240), target[:,1]**2 (B,2,240) (uncertainty)) - for variational with uncertaimties??
    # return nn.GaussianNLLLoss()(predictions target, target)

#MSE loss for variational autoencoder
def mse_loss(predictions, target):
    return nn.MSELoss()(predictions[:,0], target[:,0]) #(spectra))(B,240)


class TestDataset(torch.utils.data.Dataset):
    """
    Fake dataset to test netloader.networks
    """
    def __init__(self, in_shape: list[int]):
        self._in_shape: list[int] = in_shape
        self._device: torch.device = get_device()[1]
        self.transform: transforms.BaseTransform | None = None

    def __len__(self) -> int:
        return 600

    def __getitem__(self, item: int):
        # target: torch.Tensor = torch.randint(0, 10, size=(1, 1)).to(self._device).float()
        # in_tensor: torch.Tensor = (torch.ones(size=(1, *self._in_shape[1:])).to(self._device) *
        #                            target[..., None, None])

        target: torch.Tensor = torch.randint(0, 10, size=(1, 1)).to(self._device).float()
        in_tensor: torch.Tensor = torch.ones(size=(1, *self._in_shape[1:])).to(self._device) * target


        if self.transform:
            target = self.transform(target)

        return 0, target[0], in_tensor[0]


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
        
        # transform = transforms.MultiTransform([
        #     transforms.NumpyTensor(),
        #     transforms.MinClamp(dim=-1),
        #     transforms.Log(),
        # ])
        # transform.transforms.append(transforms.Normalise(
        #     transform(datasets[1].spectra),
        #     mean=False,
        # ))
        # for fake dataset
        transform = None

        # param_transform = transforms.MultiTransform([
        #     transforms.NumpyTensor(),
        #     transforms.MinClamp(dim=0, idxs=log_params),
        #     transforms.Log(idxs=log_params),
        # ])
        # param_transform.transforms.append(transforms.Normalise(
        #     param_transform(datasets[1].params),
        #     dim=0,
        # ))
        # for fake dataset
        param_transform = None

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
        decoder.loss_func = gaussian_loss    #changes loss to gaussian loss - can look into other typs of loss

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
        # net = nets.Autoencoder(
        #     e_save_num,
        #     states_dir,
        #     AutoencoderNet(net, decoder.net, name=encoder_name),
        #     learning_rate=learning_rate,
        #     description=description,
        #     verbose='epoch',
        #     transform=transform,
        #     latent_transform=param_transform,
        # )
        
        # chooses to just train encoder
        net = nets.Encoder(
            e_save_num,
            states_dir,
            net,
            learning_rate=learning_rate,
            description=description,
            transform=param_transform,
        )

        #changes autoencoder loss for variational autoencoder, mse for non-variational autoencoder
        # net.reconstruct_func = gaussian_loss   
        # net.latent_func = mse_loss              #adds latent loss as mse

        # net.latent_loss = 5e-1
        # net.kl_loss = 0
        # net.reconstruct_loss = 5e-1
        # net.bound_loss = 3e-1

        net.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #included scheduler to implement minimum learning rate
            net.optimiser,
            factor=0.5,
            min_lr=1e-5,
        )
        # net._loss_function = encoder_loss

    # for dataset in datasets:
        # # for with uncertainties
        # dataset.spectra, dataset.uncertainty = transform(
        #     dataset.spectra,
        #     uncertainty=dataset.uncertainty,
        # )

        # for fake spectra
        # dataset = transform(
        #     dataset)

        # for without uncertainties
        # dataset.params = param_transform(dataset.params)

        # # for with uncertainties
        # dataset.params, dataset.param_uncertainty = param_transform(
        #     dataset.params,
        #     uncertainty=dataset. param_uncertainty,
        # )

        # for fake spectra
        # dataset = param_transform(
        #     datasets
        # )

        # for without uncertainties
        # dataset.params = param_transform(dataset.params)

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
    # e_dataset = SpectrumDataset(e_data_path, log_params)
    # d_dataset = SpectrumDataset(d_data_path, log_params)
    # decoder, net = net_init((e_dataset, d_dataset), config)

    # Initialise datasets
    # e_loaders = loader_init(e_dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    # d_loaders = loader_init(d_dataset, batch_size=batch_size, val_frac=val_frac, idxs=decoder.idxs)
    # net.idxs = e_dataset.idxs
    # decoder.idxs = d_dataset.idxs
    # return e_dataset, d_dataset, e_loaders, d_loaders, decoder, net
    #added e_dataset, d_dataset changed. put this function into code
    #do the same with net_init - so you dont have to untransfrom and retransform

    # for fake dataset:
    dataset = TestDataset((1,1,240))
    decoder, net = net_init((dataset, dataset), config)

    loader = DataLoader(dataset, batch_size=60, shuffle=False)

    # net.idxs = dataset.idxs
    # decoder.idxs = dataset.idxs

    return dataset, dataset, (loader, loader), (loader, loader), decoder, net


'''shows what the data file looks like
with open('./data/spectra.pickle', 'rb') as file:
    data = pickle.load(file)
print(data)
'''

num_epochs = 100
learning_rate = 1.0e-4 #same as in config
directory = '/Users/astroai/Projects/FSPNet/plots-VAE/'
log_params = [0,2,3,4]
param_names = ['$N_{H}$ $(10^{22}\ cm^{-2})$', '$\Gamma$', '$f_{sc}$',r'$kT_{\rm disk}$ (keV)','$N$']
just_err = {
    'plot': True,   # plots error
    'sep': True     # plots it separate to parameters
}
no_err = {
    'plot': False, 
    'sep': False    
}

#initialise data loaders and networks
e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init()

#generate predictions for the autoencoder validation data loader (e_loaders contains training and validation data loaders)
# data = net.predict(e_loaders[-1], path='/Users/astroai/Downloads/Spectrum-Machine-Learning-master/predictions')

#train decoder
# decoder.training(num_epochs, d_loaders)

# #fix decoder's weights so they dont change while training the encoder
# net.net.net[1].requires_grad_(False)

#setting up optimiser correctly

net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-5,)

# train encodeer as autoencoder
net.training(num_epochs, e_loaders)

# for training outputs
# net.training(101, e_loaders)
# directory+="e_training/"

# for validation outputs
# net.training(1, e_loaders)
# directory+="e_validation/"

#---manually transforming data---#

# save transforms
# transform = net.header['inputs']
param_transform = net.header['targets']

# clear transforms
# net.header['inputs'] = None
net.header['targets'] = None


# predict & transform
data = net.predict(e_loaders[-1], path='./predictions/prediction')
d_data = decoder.predict(d_loaders[-1])


# for with uncertainties
# data['targets'] = np.stack(param_transform(data['targets'][:,0], back=True,
#                                             uncertainty=data['targets'][:,1]), axis=1)
# data['inputs'] = np.stack(transform(data['inputs'][:,0], back=True,
#                                            uncertainty=data['inputs'][:,1]), axis=1)


# using checkpoints for parameters with error instead of latent
# err_dat = np.stack(param_transform(net.net.net[0].checkpoints[-1][:,0], back=True,
#                                    uncertainty=net.net.net[0].checkpoints[-1][:,1]), axis=1)

#reset transforms
# net.header['inputs'] = transform
net.header['targets'] = param_transform

# swapaxes(0,1)  without uncertainties, swapaxes(0,2) with uncertainties - for plots
# lats = data['latent'].swapaxes(0,2)   #latent space parameters
targs = data['targets'].swapaxes(0,1)   #target reconstruction
preds = data['preds'].swapaxes(0,1)     #predicted reconstruction
# inps = data['inputs'].swapaxes(0,2)   #input parameters


d_targs = d_data['targets'].swapaxes(0,2)
d_preds = d_data['preds'].swapaxes(0,2)

'''---------- PLOTTING PERFORMANCE ----------'''
#autoencoder performance - remember to change part of net_init to include autoencoder and not encoder
# plots.plot_performance(
#     'Loss',
#     net.losses[1][1:],
#     plots_dir=directory,
#     train=net.losses[0][1:],
#     savename='perfomance.png'
# )

#encoder performance - remember to change part of net_init to include encoder and not autoencoder
plots.plot_performance(
    'Loss',
    net.losses[1][1:],
    plots_dir=directory,
    train=net.losses[0][1:],
    save_name='e_perfomance.png'
)

#decoder performance
# plots.plot_performance(
#     'Loss',
#     decoder.losses[1][1:],
#     plots_dir=directory,
#     train=decoder.losses[0][1:],
#     save_name='d_performance'
# )

# '''---------- PLOTTING PARAMETER DISTRIBUTIONS ----------'''
# # manual param distribution
# distribution_plot(
#     targs = targs,
#     lats = lats,
#     param_names = param_names,
#     log_params = log_params,
#     dir_name = directory,
#     save_name = 'param_distribution.png',
#     err = False)

# # manual error distribution
# distribution_plot(
#     targs = targs,
#     lats = lats,
#     param_names = param_names,
#     log_params = log_params,
#     dir_name = directory,
#     save_name = 'err_distribution.png',
#     err = True)


# #zoomed in gamma distribution
# plt.figure()
# plt.title('$\Gamma$')
# plt.hist(targs[1,1,:], label='targets')
# plt.hist(lats[1,1,:], label ='latent')
# plt.legend()

# plt.savefig(directory+'photon_index_dist', dpi=300)

# '''---------- PLOTTING PARAMETER COMPARISON ----------'''
# # plotting with error as error bars
# comparison_plot(
#     targs = targs,
#     lats = lats,
#     param_names = param_names,
#     log_params = log_params,
#     dir_name = directory,
#     save_name = 'param_comparison.png',
#     err = 'both'
#     )

# parameter comparison for encoder
# comparison_plot(
#     targs = targs,
#     lats = preds,
#     param_names = param_names,
#     log_params = log_params,
#     dir_name = directory,
#     save_name = 'e_param_comparison_valrange.png',
#     err = 'both',
#     lim_valrange=True
#     )

# comparison_plot(
#     targs = targs,
#     lats = preds,
#     param_names = param_names,
#     log_params = log_params,
#     dir_name = directory,
#     save_name = 'e_param_comparison.png',
#     err = 'both',
#     lim_valrange=False
#     )

'''---------- PLOTTING RECONSTRUCTED SPECTRA ----------'''
# # plots reconstructed spectra
# reconstruction_plot(
#     preds=data['preds'],
#     inputs=data['inputs'],
#     ids=data['ids'],
#     save_name = 'reconstruction',
#     dir_name=directory,
#     log_plot = 'both'
# )

#encoder reconstructions..?
# reconstruction_plot(
#     preds=d_data['preds'],
#     inputs=d_data['targets'],
#     ids=d_data['ids'],
#     save_name = 'e_reconstruction',
#     dir_name=directory,
#     log_plot = 'both',
#     inc_unc = False
# )

# reconstruction_plot_w_params(
#     inputs=data['inputs'],
#     preds=data['preds'],
#     lats=data['latent'],
#     targs=data['targets'],
#     ids=data['ids'],
#     names=param_names,
#     log='both',
#     save_name = 'reconstruction_w_params',
#     dir_name=directory,
# )


'''---------- PLOTTING GAMMA VS fsc ----------'''
'''
fsc_targ = targs[:,0,:][2]
gam_targ = targs[:,0,:][1]
fsc_lat = lats[:,0,:][2]
gam_lat =  lats[:,0,:][1]

fsc_targ_err = targs[:,1,:][2]
gam_targ_err = targs[:,1,:][1]
fsc_lat_err = lats[:,1,:][2]
gam_lat_err =  lats[:,1,:][1]

log=True

fig, ax = plt.subplots()   
ax.set_title('$\Gamma$ vs. $f_{sc}$')
ax.errorbar(fsc_targ, gam_targ, xerr=fsc_targ_err, yerr=gam_targ_err, linestyle='None', label='inputs', markersize=3, alpha=0.3)
ax.errorbar(fsc_lat, gam_lat, xerr=fsc_lat_err, yerr=gam_lat_err, linestyle='None', label='outputs', markersize=3, alpha=0.3)

if log==True:
    ax.set_xlabel('$f_{sc}$')
    ax.set_ylabel('$\Gamma$')
    ax.set_xscale('log')
else:
    ax.set_xlabel('$f_{sc}$')
    ax.set_ylabel('$\Gamma$')
    ax.set_xscale('log')

ax.legend()
plt.savefig(directory+'gam_vs_fsc', dpi=300)
'''