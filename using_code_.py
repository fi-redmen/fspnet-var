
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
from fspnet.spectrum_fit import pyxspec_tests
from fspnet.utils.plots import plot_param_pairs

# from netloader.layers.convolutional

import matplotlib.pyplot as plt

from VAE_plots import comparison_plot, distribution_plot, recon_plot, recon_plot_params, comparison_plot_NF

import sciplots

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
    device = 'cpu' # get_device()[1]

    print('device:', device)

    if d_load_num:
        decoder = nets.load_net(d_load_num, states_dir, decoder_name, weights_only=False)
        decoder.description = description
        decoder.save_path = save_name(d_save_num, states_dir, decoder_name)
        transform = decoder.transforms['targets']
        param_transform = decoder.transforms['inputs']
    else:
        
        transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=-1),
            transforms.Log(),
        ])
        transform.transforms.append(transforms.Normalise(
            data=transform(datasets[1].spectra), #transform(datasets[1].spectra)
            mean=False,
        ))
        # for fake dataset
        # transform = None

        param_transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=0, idxs=log_params),
            transforms.Log(idxs=log_params),
        ])
        param_transform.transforms.append(transforms.Normalise(
            param_transform(datasets[1].params), #
            dim=0,
        ))
        # for fake dataset
        # param_transform = None

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

        decoder.transforms['inputs'] = param_transform
        #changes decoder loss to gaussian for variational autoencoder
        decoder.loss_func = gaussian_loss    #changes loss to gaussian loss - can look into other typs of loss

    if e_load_num:
        net = nets.load_net(e_load_num, states_dir, encoder_name, weights_only=False)
        net.description = description
        net.save_path = save_name(e_save_num, states_dir, encoder_name)
    else:
        net = Network(
            encoder_name,
            networks_dir,
            list(datasets[0][0][2].shape),
            list(datasets[0][0][1].shape),
        )

        # for transformer
        # net = Network(
        #     encoder_name,
        #     networks_dir,
        #     list((datasets[0][0][2][0].shape,
        #     datasets[0][0][1].shape)),
        #     list(datasets[0][0][1].shape)
        # )

        # chooses to train autoencoder
        net = NFautoencoder(
            e_save_num,
            states_dir,
            NFautoencoderNetwork(net, decoder.net, name=encoder_name),
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
            latent_transform=param_transform,
        )
        
        # chooses to just train encoder
        # net = nets.Encoder(
        #     e_save_num,
        #     states_dir,
        #     net,
        #     learning_rate=learning_rate,
        #     description=description,
        #     transform=param_transform,
        # )


        #changes autoencoder loss for variational autoencoder, mse for non-variational autoencoder
        net.reconstruct_func = gaussian_loss   
        net.latent_func = mse_loss              #adds latent loss as mse

        net.latent_loss = 3.0e-1
        net.flowlossweight =  3.0e-1
        net.reconstruct_loss = 4.0e-1
        # net.kl_loss = 0
        # net.reconstruct_loss = 5e-1
        # net.bound_loss = 0 # 3e-1 

        net.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #included scheduler to implement minimum learning rate
            net.optimiser,
            factor=0.5,
            min_lr=1e-5,
        )
        # net._loss_function = encoder_loss

    # removed for fake data
    for dataset in datasets:
        # for with uncertainties
        dataset.spectra, dataset.uncertainty = transform(
            dataset.spectra,
            uncertainty=dataset.uncertainty,
        )

        # for fake spectra
        # dataset = transform(
        #     dataset)

        # for without uncertainties
        # dataset.spectra = transform(dataset.spectra)

        # # for with uncertainties
        dataset.params, dataset.param_uncertainty = param_transform(
            dataset.params,
            uncertainty=dataset. param_uncertainty,
        )

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
    e_dataset = SpectrumDataset(e_data_path, log_params)
    d_dataset = SpectrumDataset(d_data_path, log_params)
    decoder, net = net_init((e_dataset, d_dataset), config)

    # Initialise datasets
    e_loaders = loader_init(e_dataset, batch_size=batch_size, val_frac=val_frac, idxs=net.idxs)
    d_loaders = loader_init(d_dataset, batch_size=batch_size, val_frac=val_frac, idxs=decoder.idxs)
    net.idxs = e_dataset.idxs
    decoder.idxs = d_dataset.idxs
    
    return e_dataset, d_dataset, e_loaders, d_loaders, decoder, net
    # added e_dataset, d_dataset changed. put this function into code
    # do the same with net_init - so you dont have to untransfrom and retransform

    # for fake dataset:
    # dataset = TestDataset((1,1,240))
    # decoder, net = net_init((dataset, dataset), config)

    # loader = DataLoader(dataset, batch_size=60, shuffle=False)

    # net.idxs = dataset.idxs
    # decoder.idxs = dataset.idxs

    # return dataset, dataset, (loader, loader), (loader, loader), decoder, net

class NFautoencoder(nets.Autoencoder):
    def __init__(self, save_num, states_dir, net, mix_precision = False, learning_rate = 0.001, description = '', verbose = 'epoch', transform = None, latent_transform = None, in_transform = None):
        super().__init__(save_num, states_dir, net, mix_precision, learning_rate, description, verbose, transform, latent_transform, in_transform)
        self.flowlossweight = 0.5
    
    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'flowlossweight': self.flowlossweight
        }
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.flowlossweight = state['flowlossweight']

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the autoencoder's predictions

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N,...) Tensor
            Latent target low dimensional data of batch size N and the remaining dimensions depend
            on the network used

        Returns
        -------
        float
            Loss from the autoencoder's predictions'
        """
        loss: Tensor
        latent: Tensor | None = None
        bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
        output: Tensor = self.net(in_data) #,target (for inheriting class)

        if self.net.checkpoints:
            latent = self.net.checkpoints[-1].sample([1])[0]

        loss = self.reconstruct_loss * self.reconstruct_func(output, in_data) 

        loss -= self.flowlossweight * self.net.checkpoints[-1].log_prob(target).mean()

        if self.latent_loss and latent is not None:
            loss += self.latent_loss * self.latent_func(latent, target)

        if self.bound_loss and latent is not None:
            loss += self.bound_loss * torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        if self.kl_loss:
            loss += self.kl_loss * self.net.kl_loss

        # print("reconstruct: ", self.reconstruct_loss, '*', self.reconstruct_func(output, in_data),'=', self.reconstruct_loss*self.reconstruct_func(output, in_data))
        # print('latent: ', self.latent_loss, '*', self.latent_func(latent, target), '=', self.latent_loss*self.latent_func(latent, target))
        # print('flow: ', self.flowlossweight, '*', self.net.checkpoints[-1].log_prob(target).mean(), '=', self.flowlossweight*self.net.checkpoints[-1].log_prob(target).mean())

        self._update(loss)
        return loss.item()
    
    def batch_predict(self, data: Tensor, num_samples, **_: Any) -> tuple[ndarray, ...]:
        """
        Generates predictions for the given data batch

        Parameters
        ----------
        data : (N,...) Tensor
            N data to generate predictions for

        Returns
        -------
        tuple[(N,...) ndarray, ...]
            N predictions for the given data
        """
        # for nonvariational wihtout uncertainties??
        # return (
        #     self.net(data).detach().cpu().numpy(),
        #     self.net.checkpoints[-1].detach().cpu().numpy(),
        #     data.detach().cpu().numpy(),
        # )
    
        # for non variational without uncertainties??:
        # return (
        #     self.net(data).detach().cpu().numpy(),
        #     self.net[0].detach().cpu().numpy(),
        #     data.detach().cpu().numpy(),
        # )

        # NFs
        return (
            self.net(data).detach().cpu().numpy(),
            self.net.checkpoints[-1].sample([num_samples]).swapaxes(0,1).detach().cpu().numpy(),
            data.detach().cpu().numpy(),
        )


class NFautoencoderNetwork(AutoencoderNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor: # add ,target_uncertainty to arguments
        """
        Forward pass of the autoencoder

        Parameters
        ----------
        x : (N,...) list[Tensor] | Tensor
            Input tensor(s) with batch size N

        Returns
        -------
        (N,...) list[Tensor] | Tensor
            Output tensor from the network
        """
        x = self.net[0](x)
        self.checkpoints.append(x)
        x = x.sample([1])
        x = x[0]

        # if hasattr(self.net[0], 'kl_loss'): - don't need kl loss?
        #     self.kl_loss = self.net[0].kl_loss

        return self.net[1](x)   # for non variational
        # return self.net[1](torch.cat((x, self.net[0].checkpoints[-1][:,-1:]), dim=1))     #for variational - when we want mu and sigma from checkpoint layer in encoder



'''shows what the data file looks like
with open('./data/spectra.pickle', 'rb') as file:
    data = pickle.load(file)
print(data)
'''

# ctrl (or command) click on 'SplineFlow' here to see how Ethan implements the normalising flow
from netloader.layers.flows import SplineFlow

# settings
MAJOR = 26
MINOR = 20
TICK = 16
num_epochs = 1
learning_rate = 1.0e-5 #in config: 1e-4
directory = '/Users/astroai/Projects/FSPNet/plots_NF/'
log_params = [0,2,3,4]
param_names = ['$N_{H}$ $(10^{22}\ cm^{-2})$', '$\Gamma$', '$f_{sc}$',r'$kT_{\rm disk}$ (keV)','$N$']
# These are likely to be redundant - for my plotting functions which I need to tidy up
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

# train decoder
# decoder.training(num_epochs, d_loaders)

# #fix decoder's weights so they dont change while training the encoder
# net.net.net[1].requires_grad_(False)

#setting up autoencoder optimiser correctly - likely to not be needed
# net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
# net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-5,)

# train autoencoder
# net.training(num_epochs, e_loaders)

# for training outputs
# net.training(101, e_loaders)
# directory+="e_training/"
# for validation outputs
# net.training(1, e_loaders)
# directory+="e_validation/"

# save transforms
# transform = net.transforms['inputs']
# param_transform = net.transforms['targets']
# # clear transforms
# net.transforms['inputs'] = None
# net.transforms['targets'] = None

#--- making predictions with manually transforming data ---# -- for VAE
# # predict & transform
# data = net.predict(e_loaders[-1], path='./predictions/prediction')
# d_data = decoder.predict(d_loaders[-1])
# #reset transforms
# net.transforms['inputs'] = transform
# net.transforms['targets'] = param_transform
# # untransforms and stacks uncertainties
# data['targets'] = np.stack(param_transform(data['targets'][:,0], back=True, 
#                            uncertainties=data['targets'][:,1]), axis=1)
# data['inputs'] = np.stack(transform(data['inputs'][:,0], back=True,
#                                            uncertainty=data['inputs'][:,1]), axis=1)
# distributions: transform, samples from the predicted distributions
# probs: None, probabilities for the targets in the given distributions
# max: transform, values with the maximum probabilities in the distributions
# meds: transform, median values of the distributions

#--- making predictions with manually transforming data ---# -- for NF
# save transforms
# transform = net.transforms['inputs']
# param_transform = net.transforms['targets']
# # clear transforms
# net.transforms['inputs'] = None
# net.transforms['targets'] = None
# data = net.predict(e_loaders[-1], num_samples=1, input_=True)
# data1 = net.predict(e_loaders[-1], num_samples=1000, input_=True)
# param_uncertainties = e_dataset.param_uncertainty[np.isin(e_dataset.names, data['ids'])]  # get uncertainties in 'ground truth' parameters
# # untransforms and stacks uncertainties
# data['targets'] = np.stack([param_transform(data['targets'], back=True), 
#                            param_uncertainties], axis=1)
# data['inputs'] = np.stack(transform(data['inputs'][:,0], back=True,
#                                            uncertainty=data['inputs'][:,1]), axis=1)
# data1['targets'] = np.stack([param_transform(data1['targets'], back=True), 
#                             param_uncertainties], axis=1)
# data1['inputs'] = np.stack(transform(data1['inputs'][:,0], back=True,
#                                            uncertainty=data1['inputs'][:,1]), axis=1)

# --- making predictions with manually transforming data ---# -- for tranformer
# data1 = net.predict(e_loaders[-1], input_=False)
# idxs = np.isin(e_dataset.names, data1['ids'])
# spectra = np.stack((transform(e_dataset.spectra[idxs], back=True, uncertainty=e_dataset.uncertainties[idxs])), axis=1)
# untransforms
# data['targets'] = param_transform(data['targets'], back=True)
# data['inputs'] = transform(data['inputs'], back=True)
# data1['targets'] = param_transform(data1['targets'], back=True)
# data1['inputs'] = transform(data1['inputs'], back=True)

# #reset transforms
# net.transforms['inputs'] = transform
# net.transforms['targets'] = param_transform

# saves predictions to pickle file
# with open('/Users/astroai/Projects/FSPNet/predictions/my_preds1.pickle', 'wb') as file:
#     pickle.dump(data1, file)
# with open('/Users/astroai/Projects/FSPNet/predictions/my_preds.pickle', 'wb') as file:
#     pickle.dump(data, file)
# loads predictions from saved pickle file
with open('/Users/astroai/Projects/FSPNet/predictions/my_preds1.pickle', 'rb') as file:
        data1 = pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/predictions/my_preds.pickle', 'rb') as file:
        data = pickle.load(file)


# swapaxes(0,1)  without uncertainties, swapaxes(0,2) with uncertainties - for plots
lats = data['latent'].swapaxes(0,1)[0].swapaxes(0,1)   #latent space parameters
targs = data['targets'].swapaxes(0,1) #target reconstruction
preds = data['preds'].swapaxes(0,1)   #predicted reconstruction
inps = data['inputs'].swapaxes(0,2)   #input parameters

# d_targs = d_data['targets'].swapaxes(0,2)
# d_preds = d_data['preds'].swapaxes(0,2)

'''---------- PLOTTING PERFORMANCE ----------'''
#autoencoder performance - remember to change part of net_init to include autoencoder and not encoder
plots.plot_performance(
    'Loss',
    net.losses[1][1:],
    plots_dir=directory,
    train=net.losses[0][1:],
    save_name='NFperfomance.png'
)

#encoder performance - remember to change part of net_init to include encoder and not autoencoder
# plots.plot_performance(
#     'Loss',
#     net.losses[1][1:],
#     plots_dir=directory,
#     train=net.losses[0][1:],
#     save_name='e_perfomance.png'
# )

#decoder performance
plots.plot_performance(
    'Loss',
    decoder.losses[1][1:],
    plots_dir=directory,
    train=decoder.losses[0][1:],
    save_name='NF_d_performance'
)

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
#     save_name = 'NFparam_comparison.png',
#     err = 'none'
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

plt.plot(decoder.net(torch.tensor([[1,2,0.5,0.5,100]]).detach().numpy() ))
plt.savefig(directory+'recon.png', dpi=300)

comparison_plot_NF(
    data1,
    param_names,
    log_params,
    decoder.net,
    dir_name=directory,
    save_name='NF_comparison.png',
)

'''---------- PLOTTING RECONSTRUCTED SPECTRA ----------'''
# # plots reconstructed spectra - random spectra produced while training
recon_plot(
    preds=data['preds'],
    inputs=data['inputs'],
    ids=data['ids'],
    save_name = 'NFreconstruction',
    dir_name=directory,
    log_plot = 'both'
)

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

# recon_plot_params(
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

# recon_plot_params_NF(
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

'''--------- PLOTTING PARAMETER PAIRS ----------'''
param_pair_axes = plot_param_pairs(
    data=data1['latent'][0],
    plots_dir=directory,
    log_params=log_params,
    param_names=param_names
)



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

fig, ax = plt.subplot_mosaic('aabbcc\ndddeee', layout='constrained', figsize=(16,9))

for i, (axis, latent, target) in enumerate(zip(ax.values(), data1['latent'][0].swapaxes(0,1), data1['targets'][0])):
    axis.set_title(param_names[i], fontsize=MAJOR)

    if i in log_params:
        axis.set_xscale('log')
   
        # target = np.log10(target)
    hist_values = axis.hist(latent, bins=100, range=[np.quantile(latent,0.05), np.quantile(latent,0.95)]) #, log=(i in log_params))
    axis.plot([target, target], [0, np.max(hist_values[0])], color='k', linestyle='--', alpha=0.4)
    
    axis.tick_params(labelsize=TICK)


plt.savefig(directory+'NFparam_dist.png', dpi=300)

distribution_plot = sciplots.PlotDistributions(data=data1['latent'][0].swapaxes(0,1), log=log_params, density=True, norm=True, titles=param_names, bins=200) 
# Loop through the axes which I think is a dictionary (otherwise it is a list) - to plot targets as discrete values
for i, (target, err, axis) in enumerate(zip(data1['targets'][0][0], data1['targets'][0][1], distribution_plot.axes.values())):
    target_range = np.linspace(np.min(data['latent'][0].swapaxes(0,1)), np.max(data['latent'][0].swapaxes(0,1)), 1000)
    gaussian = np.exp(-( ( (target_range-target)**2 ) / (2*(err**2))) )

    axis.plot(target_range, gaussian, color='g') # Or however you want to do it/did it before
    axis.fill_between(target_range, gaussian, np.zeros(len(target_range)), color='g', alpha=0.3)



# distribution_plot.plot_twin_data(data1['targets'][0], log=log_params, density=False)

distribution_plot.savefig(directory, name='distributions')

data['latent'] = np.squeeze(data['latent'])
# pyxspec_tests(data)
