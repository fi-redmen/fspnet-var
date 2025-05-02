
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
from fspnet.utils.plots import plot_param_pairs, _plot_histogram

# from netloader.layers.convolutional


import matplotlib.pyplot as plt

from VAE_plots import comparison_plot_NF, distribution_plot_NF, recon_plot_NF, post_pred_plot_NF, latent_space_scatter_NF, param_pair_plot_NF, plot_performance_NF

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

        net.latent_loss = 0 #3.0e-1
        net.flowlossweight = 1 #3.0e-1
        net.reconstruct_loss = 0 #4.0e-1
        net.kl_loss = 0 #
        net.bound_loss = 0 # 3e-1 


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
        self.separate_losses = {
            'reconstruct': [],
            'flow': [],
            'latent': [],
            'bound': [],
            'kl': []
        }
    
    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'flowlossweight': self.flowlossweight,
            'separate_losses': self.separate_losses
        }
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.flowlossweight = state['flowlossweight']
        self.separate_losses = state['separate_losses']

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

        separate_loss = []

        if self.net.checkpoints:
            latent = self.net.checkpoints[-1].sample([1])[0]

        # # loss = self.reconstruct_loss * self.reconstruct_func(output, in_data) 
        # separate_loss.append(self.reconstruct_loss * self.reconstruct_func(output, in_data))
        # self.separate_losses['reconstruct'].append(self.reconstruct_loss * self.reconstruct_func(output, in_data).clone().item())

        # # loss -= self.flowlossweight * self.net.checkpoints[-1].log_prob(target).mean()
        # separate_loss.append(self.flowlossweight * self.net.checkpoints[-1].log_prob(target).mean())
        # self.separate_losses['flow'].append(self.flowlossweight * self.net.checkpoints[-1].log_prob(target).mean().clone().item())

        # if self.latent_loss and latent is not None:
        #     # loss += self.latent_loss * self.latent_func(latent, target)
        #     separate_loss.append(self.latent_loss * self.latent_func(latent, target))
        #     self.separate_losses['latent'].append(self.latent_loss * self.latent_func(latent, target).clone().item())    
        # # else: 
        # #     losses.append(0)

        # if self.bound_loss and latent is not None:
        #     # loss += self.bound_loss * torch.mean(torch.cat((
        #     #     (bounds[0] - latent) ** 2 * (latent < bounds[0]),
        #     #     (latent - bounds[1]) ** 2 * (latent > bounds[1]),
        #     # )))
        #     separate_loss.append(self.bound_loss * torch.mean(torch.cat((
        #         (bounds[0] - latent) ** 2 * (latent < bounds[0]),
        #         (latent - bounds[1]) ** 2 * (latent > bounds[1]),
        #     ))))
        #     self.separate_losses['bound'].append(self.bound_loss * torch.mean(torch.cat((
        #         (bounds[0] - latent) ** 2 * (latent < bounds[0]),
        #         (latent - bounds[1]) ** 2 * (latent > bounds[1]),
        #     ))).clone().item())

        # if self.kl_loss:
        #     # loss += self.kl_loss * self.net.kl_loss
        #     separate_loss.append(self.kl_loss * self.net.kl_loss)
        #     self.separate_losses['kl'].append(self.kl_loss * self.net.kl_loss.clone().item())

        # loss = torch.sum(torch.stack(separate_loss))
        
        # appends losses
        # if self._train_state:
        #     separate_loss = [loss_value.clone().item() for loss_value in separate_loss]

        # New method (also doesn't work..):
        loss_components = {
            'reconstruct': self.reconstruct_loss * self.reconstruct_func(output, in_data),
            'flow': self.flowlossweight * self.net.checkpoints[-1].log_prob(target).mean(),
            'latent': self.latent_loss * self.latent_func(latent, target) if self.latent_loss and latent is not None else None,
            'bound': self.bound_loss * torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            ))) if self.bound_loss and latent is not None else None,
            'kl': self.kl_loss * self.net.kl_loss if self.kl_loss else None,
        }

        # Iterate through the dictionary to process each loss component
        separate_loss = []
        for key, loss_value in loss_components.items():
            if loss_value is not None:  # Only process non-None losses
                separate_loss.append(loss_value)
                if self._train_state:
                    self.separate_losses[key].append(loss_value.clone().item())

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
num_epochs = 100
learning_rate = 1.0e-3 #in config: 1e-4

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

# saves name of predictions as encoder name_decoder name
pred_savename = os.path.basename(net.save_path)[:-4]+' '+os.path.basename(decoder.save_path)[:-4]
plots_directory = '/Users/astroai/Projects/FSPNet/plots/'+pred_savename+'/'
os.makedirs(plots_directory, exist_ok=True)
os.makedirs(plots_directory+'reconstructions/', exist_ok=True)
os.makedirs(plots_directory+'distributions/', exist_ok=True)

# train decoder
# decoder.training(num_epochs, d_loaders) 

# # #fix decoder's weights so they dont change while training the encoder
# net.net.net[1].requires_grad_(False)

# #setting up autoencoder optimiser correctly - likely to not be needed
# net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
# net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-6,)

# # train autoencoder
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
transform = net.transforms['inputs']
param_transform = net.transforms['targets']
# # clear transforms
net.transforms['inputs'] = None
net.transforms['targets'] = None
data = net.predict(e_loaders[-1], num_samples=1, input_=True)
data1 = net.predict(e_loaders[-1], num_samples=1000, input_=True)

names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp', 
           'js_ni0103010102_0mpu7_goddard_GTI0.jsgrp',
           'js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
           'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp',
           'js_ni1100320119_0mpu7_goddard_GTI26.jsgrp']
train_data = net.predict(e_loaders[0], num_samples=1000, input_=True)
val_data = net.predict(e_loaders[1], num_samples=1000, input_=True)
train_idxs = np.isin(train_data['ids'], names)
val_idxs = np.isin(val_data['ids'], names)
# # Then with the idxs, you can get the posteriors (or whatever else you want from the predicitons) by:
targets = np.concat((train_data['targets'][train_idxs],val_data['targets'][val_idxs]), axis=0)
latent = np.concat((train_data['latent'][train_idxs],val_data['latent'][val_idxs]), axis=0)
inputs = np.concat((train_data['inputs'][train_idxs], val_data['inputs'][val_idxs]), axis=0)
preds = np.concat((train_data['preds'][train_idxs], val_data['preds'][val_idxs]), axis=0)
names = np.concat((train_data['ids'][train_idxs], val_data['ids'][val_idxs]), axis=0)

specific_data = {
    'id': names,
    'targets': targets,
    'latent': latent,
    'inputs': inputs,
    'preds': preds,
}

param_uncertainties = e_dataset.param_uncertainty[np.isin(e_dataset.names, data['ids'])]  # get uncertainties in 'ground truth' parameters
param_uncertainties1 = e_dataset.param_uncertainty[np.isin(e_dataset.names, data1['ids'])]
specific_param_uncertainties = e_dataset.param_uncertainty[np.isin(e_dataset.names, specific_data['id'])]
# # untransforms and stacks uncertainties
data['targets'] = np.stack(param_transform(data['targets'], back=True,
                                            uncertainty=param_uncertainties), axis=1)
data['inputs'] = np.stack(transform(data['inputs'][:,0], back=True,
                                           uncertainty=data['inputs'][:,1]), axis=1)
data1['targets'] = np.stack(param_transform(data1['targets'], back=True,
                                             uncertainty=param_uncertainties1), axis=1)
data1['inputs'] = np.stack(transform(data1['inputs'][:,0], back=True,
                                           uncertainty=data1['inputs'][:,1]), axis=1)
specific_data['targets'] = np.stack(param_transform(specific_data['targets'], back=True,
                                            uncertainty=specific_param_uncertainties), axis=1)
specific_data['inputs'] = np.stack(transform(specific_data['inputs'][:,0], back=True,
                                             uncertainty=specific_data['inputs'][:,1]), axis=1)

# # --- making predictions with manually transforming data ---# -- for tranformer
# # data1 = net.predict(e_loaders[-1], input_=False)
# # idxs = np.isin(e_dataset.names, data1['ids'])
# # spectra = np.stack((transform(e_dataset.spectra[idxs], back=True, uncertainty=e_dataset.uncertainties[idxs])), axis=1)
# # untransforms
# # data['targets'] = param_transform(data['targets'], back=True)
# # data['inputs'] = transform(data['inputs'], back=True)
# # data1['targets'] = param_transform(data1['targets'], back=True)
# # data1['inputs'] = transform(data1['inputs'], back=True)

# # #reset transforms
# net.transforms['inputs'] = transform
# net.transforms['targets'] = param_transform

# # saves predictions to pickle file TRANSFORMED PARAM_UNCERTAINTIES
with open('/Users/astroai/Projects/FSPNet/predictions/preds1_'+pred_savename+'.pickle', 'wb') as file:
    pickle.dump(data1, file)
with open('/Users/astroai/Projects/FSPNet/predictions/preds_'+pred_savename+'.pickle', 'wb') as file:
    pickle.dump(data, file)
with open('/Users/astroai/Projects/FSPNet/predictions/preds_specific_'+pred_savename+'.pickle', 'wb') as file:
    pickle.dump(specific_data, file)

# loads predictions from saved pickle file
with open('/Users/astroai/Projects/FSPNet/predictions/preds1_'+pred_savename+'.pickle', 'rb') as file:
        data1 =pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/predictions/preds_'+pred_savename+'.pickle', 'rb') as file:
        data = pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/predictions/preds_specific_'+pred_savename+'.pickle', 'rb') as file:
    specific_data = pickle.load(file)

# loading in xspec predictions
with open('/Users/astroai/Projects/FSPNet/xspec_predictions/predictions_0.pickle', 'rb') as file:
    xspec_data0 = pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/xspec_predictions/predictions_1.pickle', 'rb') as file:
    xspec_data1 = pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/xspec_predictions/predictions_2.pickle', 'rb') as file:
    xspec_data2 = pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/xspec_predictions/predictions_3.pickle', 'rb') as file:
    xspec_data3 = pickle.load(file)
with open('/Users/astroai/Projects/FSPNet/xspec_predictions/predictions_4.pickle', 'rb') as file:
    xspec_data4 = pickle.load(file)

# putting each dataset into one dictionary
xspec_data={
    id: [],
    'xspec_recon': [],
    'true_posteriors': []
}
xspec_data['id'] = np.stack((xspec_data0['id'], xspec_data1['id'], xspec_data2['id'], xspec_data3['id'], xspec_data4['id']), axis=0)   
# note: these lists are all different lengths so we keep as a list
xspec_data['xspec_recon'] = [xspec_data0['xspec_recon'], 
                             xspec_data1['xspec_recon'], 
                             xspec_data2['xspec_recon'], 
                             xspec_data3['xspec_recon'], 
                             xspec_data4['xspec_recon']]
# note: xspec_data['true_posteriors'] shape = number of spectra, number of parameters, number of samples
xspec_data['true_posteriors'] = np.stack((xspec_data0['true_posteriors'], xspec_data1['true_posteriors'], xspec_data2['true_posteriors'], xspec_data3['true_posteriors'], xspec_data4['true_posteriors']), axis=0)


# note: data['latent'].shape = 1080,1,5 = number of spectra, number of samples, number of parameters


# separate_losses = net.separate_losses # shape (number of iterations ((dataset/batch)*epochs), number of loss terms)
# averaging loss over batch size
# separate_losses = [np.mean(separate_losses[i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses), len(e_loaders[0]))]

# separate_losses['reconstruct'] = [np.mean(separate_losses['reconstruct'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['reconstruct']), len(e_loaders[0]))]
# separate_losses['flow'] = [np.mean(separate_losses['flow'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['flow']), len(e_loaders[0]))]  
# separate_losses['latent'] = [np.mean(separate_losses['latent'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['latent']), len(e_loaders[0]))]
# separate_losses['bound'] = [np.mean(separate_losses['bound'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['bound']), len(e_loaders[0]))]
# separate_losses['kl'] = [np.mean(separate_losses['kl'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['kl']), len(e_loaders[0]))]

# batch_size = len(e_loaders[0])
# for key in ['reconstruct', 'flow', 'latent', 'bound', 'kl']:
#     separate_losses[key] = [
#         np.mean(separate_losses[key][i:i + batch_size], axis=0)
#         for i in range(0, len(separate_losses[key]), batch_size)]


'''---------- PLOTTING PERFORMANCE ----------'''
#autoencoder performance - remember to change part of net_init to include autoencoder and not encoder
plots.plot_performance(
    'Loss',
    net.losses[1][1:],
    plots_dir=plots_directory,
    train=net.losses[0][1:],
    save_name='NF_perfomance.png'
)

# plot_performance_NF(
#     'Loss',
#     np.array(separate_losses),
#     plots_dir=plots_directory,
#     save_name='NF_allloss.png'
# )

#encoder performance - remember to change part of net_init to include encoder and not autoencoder
# plots.plot_performance(
#     'Loss',
#     net.losses[1][1:],
#     plots_dir=plots_directory,
#     train=net.losses[0][1:],
#     save_name='e_perfomance.png'
# )

#decoder performance
plots.plot_performance(
    'Loss',
    decoder.losses[1][1:],
    plots_dir=plots_directory,
    train=decoder.losses[0][1:],
    save_name='NF_d_performance'
)

# # plotting comparison between parametrs
comparison_plot_NF(
    data1,
    param_names,
    log_params,
    dir_name=plots_directory,
)

# plotting parameter distributions
# all_param_samples = distribution_plot_NF(
#     data1,
#     param_names,
#     log_params,
#     dir_name=plots_directory,
# )

all_param_samples = distribution_plot_NF(
    data =specific_data,
    param_names=param_names,
    log_params=log_params,
    dir_name=os.path.join(plots_directory,'distributions/'),
    xspec_data=xspec_data
)

# single reconstructions using samples from all_param_samples
recon_plot_NF(
    data1,
    all_param_samples,
    decoder.net,
    net,
    dir_name=plots_directory+'reconstructions/',
)

# # posterior predictive plots using 500 samples per reconstruction
post_pred_samples = post_pred_plot_NF(
    data1,
    decoder.net,
    net,
    dir_name=plots_directory+'reconstructions/',
    n_samples=500
)

# # distribution plots with samples taken for posterior predictive plots shown
# distribution_plot_NF(
#     data1,
#     param_names,
#     log_params,
#     in_param_samples=post_pred_samples,
#     dir_name=os.path.join(plots_directory,'distributions')
# )

# # latent space scatter plots
# latent_space_scatter_NF(
#     data1,
#     log_params,
#     param_names,
#     dir_name=plots_directory,
# )

# # overall parameter distributions across many spectra
# param_pair_plot_NF(
#     data,
#     log_params,
#     param_names,
#     dir_name=plots_directory
# )

# get test_set from specific spectrum MCMC chain
# net.net.checkpoints[-1].log_prob(data['targets']).mean()
# npe.flow is 
# npe_levels, npe_coverages = expected_coverage_mc(npe.flow, testset, device='cuda')

'''--------- PLOTTING PARAMETER PAIRS ----------'''
# param_pair_axes0 = plot_param_pairs(
#     data=data1['latent'][0],
#     plots_dir=plots_directory,
#     save_name = 'latent_space_0',
#     log_params=log_params,
#     param_names=param_names,
#     colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
#     scatter_colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
#     plot_hist=False
# )

# dist_targs = data1['targets'][0][0]
# dist_targ_errs = data1['targets'][0][1]
# targ_samples = [10**np.random.normal(loc=np.log10(targ), scale=(1/np.log(10))*(targ_err/targ), size=1000) if param_num in log_params
#                         else np.random.normal(loc=targ, scale=targ_err, size=1000)
#                         for param_num, (targ, targ_err) in enumerate(zip(dist_targs, dist_targ_errs))]
# param_pair_data0=np.array(targ_samples)
# ranges = [None] * param_pair_data0.shape[0]
# # Plot scatter plots & histograms
# for i, (axes_row, y_data, y_range) in enumerate(zip(param_pair_axes0, param_pair_data0, ranges)):
#         for j, (axis, x_data, x_range) in enumerate(zip(axes_row, param_pair_data0, ranges)):
#             # if i == j:
#             #     _plot_histogram(x_data, axis, log=i in log_params, data_range=x_range, colour='grey')
#             #     axis.tick_params(labelleft=False, left=False)
#             if j < i:
#                 axis.scatter(
#                     x_data[:1000],
#                     y_data[:1000],
#                     s=20,
#                     alpha=0.2,
#                     color='grey'
#                 )
#             else:
#                 axis.set_visible(False)
# plt.savefig(plots_directory+'latent_space_0.png', dpi=600)

# param_pair_axes1 = plot_param_pairs(
#     data=data1['latent'][1],
#     plots_dir=plots_directory,
#     save_name = 'latent_space_1',
#     log_params=log_params,
#     param_names=param_names,
#     colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
#     scatter_colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
#     plot_hist=False
# )
# dist_targs = data1['targets'][1][0]
# dist_targ_errs = data1['targets'][1][1]
# targ_samples = [10**np.random.normal(loc=np.log10(targ), scale=(1/np.log(10))*(targ_err/targ), size=1000) if param_num in log_params
#                         else np.random.normal(loc=targ, scale=targ_err, size=1000)
#                         for param_num, (targ, targ_err) in enumerate(zip(dist_targs, dist_targ_errs))]
# param_pair_data1=np.array(targ_samples)
# ranges = [None] * param_pair_data1.shape[0]
# # Plot scatter plots & histograms
# for i, (axes_row, y_data, y_range) in enumerate(zip(param_pair_axes1, param_pair_data1, ranges)):
#         for j, (axis, x_data, x_range) in enumerate(zip(axes_row, param_pair_data1, ranges)):
#             if i == j:
#                 _plot_histogram(x_data, axis, log=i in log_params, data_range=x_range, colour='grey')
#                 axis.tick_params(labelleft=False, left=False)
#             elif j < i:
#                 axis.scatter(
#                     x_data[:1000],
#                     y_data[:1000],
#                     s=20,
#                     alpha=0.2,
#                     color='grey'
#                 )
#             else:
#                 axis.set_visible(False)
# plt.savefig(plots_directory+'latent_space_1.png', dpi=600)

# param_pair_axes2 = plot_param_pairs(
#     data=data1['latent'][2],
#     plots_dir=plots_directory,
#     save_name = 'latent_space_2',
#     log_params=log_params,
#     param_names=param_names,
#     colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
#     scatter_colour=plt.rcParams['axes.prop_cycle'].by_key()['color'][2],
#     plot_hist=False
# )
# dist_targs = data1['targets'][2][0]
# dist_targ_errs = data1['targets'][2][1]
# targ_samples = [10**np.random.normal(loc=np.log10(targ), scale=(1/np.log(10))*(targ_err/targ), size=1000) if param_num in log_params
#                         else np.random.normal(loc=targ, scale=targ_err, size=1000)
#                         for param_num, (targ, targ_err) in enumerate(zip(dist_targs, dist_targ_errs))]
# param_pair_data2=np.array(targ_samples)
# ranges = [None] * param_pair_data2.shape[0]
# # Plot scatter plots & histograms
# for i, (axes_row, y_data, y_range) in enumerate(zip(param_pair_axes2, param_pair_data2, ranges)):
#         for j, (axis, x_data, x_range) in enumerate(zip(axes_row, param_pair_data2, ranges)):
#             if i == j:
#                 _plot_histogram(x_data, axis, log=i in log_params, data_range=x_range, colour='grey')
#                 axis.tick_params(labelleft=False, left=False)
#             elif j < i:
#                 axis.scatter(
#                     x_data[:1000],
#                     y_data[:1000],
#                     s=20,
#                     alpha=0.2,
#                     color='grey'
#                 )
#             else:
#                 axis.set_visible(False)
# plt.savefig(plots_directory+'latent_space_2.png', dpi=600)

# param_pair_axes = plot_param_pairs(
#     data=np.array(data['targets'][:,0,:]),
#     plots_dir=plots_directory,
#     save_name='param_pair_plot',
#     log_params=log_params,
#     param_names=param_names,
#     colour='#4cb555',
#     scatter_colour='#4cb555',
#     alpha=0.7,
# )

# # dist_targ_errs = data['targets'][:,1,:]
# # targ_samples = [10**np.random.normal(loc=np.log10(targ), scale=(1/np.log(10))*(targ_err/targ), size=1000) if param_num in log_params
# #                         else np.random.normal(loc=targ, scale=targ_err, size=1000)
# #                         for param_num, (targ, targ_err) in enumerate(zip(dist_targs, dist_targ_errs))]
# param_pair_data=np.squeeze(data['latent']).swapaxes(0,1)
# ranges = [None] * param_pair_data.shape[0]
# # Plot scatter plots & histograms
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
#                     alpha=0.2,
#                     color='#8445cc'
#                 )
#             else:
#                 axis.set_visible(False)

# plt.savefig(plots_directory+'param_pair_plot.png', dpi=600)

data['latent'] = np.squeeze(data['latent'])
# pyxspec_tests(data)
