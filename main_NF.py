# from NFautoencoder import NFautoencoder, net_init, GaussianNLLLoss, MSELoss
# import pandas as pd
# from netloader_tests import TestConfig, gen_indexes, mod_network
import xspec
from fspnet.spectrum_fit import init 

import numpy as np
import os
import pickle
import random
import torch
import sciplots
from astropy.io import fits
import re

import netloader.networks as nets
from netloader import transforms
from netloader.utils.utils import get_device

from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import lampe

from fspnet.utils import plots
from fspnet.utils.utils import open_config
from fspnet.utils.data import SpectrumDataset, loader_init
from fspnet.spectrum_fit import pyxspec_tests

import matplotlib.pyplot as plt

from VAE_plots import comparison_plot, recon_plot, post_pred_plot, latent_corner_plot, rec_2d_plot, param_pairs_plot 
from VAE_plots import performance_plot, post_pred_plot_xspec, coverage_plot
from my_utils.misc_utils import sample, get_energy_widths

import numpy as np
from numpy import ndarray
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from time import time
from typing import Any
from fspnet.spectrum_fit import AutoencoderNet, Tensor

import torch

from fspnet.utils.data import SpectrumDataset
from fspnet.utils.utils import open_config
from netloader import loss_funcs, transforms
from netloader.network import Network
import netloader.networks as nets
from netloader.utils.utils import progress_bar, save_name

plt.style.use(["science", "grid", 'no-latex'])

# # gaussian loss for variational autoencoder with uncertainties
def gaussian_loss(predictions, target):
    return nn.GaussianNLLLoss()(predictions[:,0], target[:,0], target[:,1]**2) #(spectra)(B,240), target[:,1]**2 (B,2,240) (uncertainty)) - for variational with uncertaimties??

#MSE loss for variational autoencoder
def mse_loss(predictions, target):
    return nn.MSELoss()(predictions[:,0], target[:,0]) #(spectra))(B,240)

class GaussianNLLLoss(loss_funcs.BaseLoss):
    """
    Gaussian negative log likelihood loss function
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        *args
            Optional arguments to be passed to GaussianNLLLoss
        **kwargs
            Optional keyword arguments to be passed to GaussianNLLLoss
        """
        super().__init__(nn.GaussianNLLLoss, *args, **kwargs)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._loss_func = nn.GaussianNLLLoss(*self._args, **self._kwargs)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self._loss_func(output[:, 0], target[:, 0], target[:, 1] ** 2)

class MSELoss(loss_funcs.MSELoss):
    """
    Mean Squared Error (MSE) loss function
    """
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self._loss_func(output[:, 0], target[:, 0])

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
            data=transform(datasets[1].spectra),
            mean=False
        ))

        param_transform = transforms.MultiTransform([
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=0, idxs=log_params),
            transforms.Log(idxs=log_params),
        ])
        param_transform.transforms.append(transforms.Normalise(
            data=param_transform(datasets[1].params),
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
            overwrite=True,
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
        )

        decoder.transforms['inputs'] = param_transform
        decoder.loss_func =  GaussianNLLLoss() # gaussian_loss #  #  #changes loss to gaussian loss - can look into other typs of loss

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

        # chooses to train NF_autoencoder
        net = NFautoencoder(
            e_save_num,
            states_dir,
            NFautoencoderNetwork(net, decoder.net, name=encoder_name),
            learning_rate=learning_rate,
            description=description,
            verbose='full',
            transform=transform,
            latent_transform=param_transform,
        )

        # chooses to just train encoder
        # net = nets.NormFlowEncoder(
        #     e_save_num,
        #     states_dir,
        #     net,
        #     learning_rate=[learning_rate]*2,
        #     description=description,
        #     transform=param_transform,
        #     verbose='full'
        # )

        #Loss function settings for autoencoder
        net.reconstruct_func =   GaussianNLLLoss() #  gaussian_loss   #
        net.latent_func =  MSELoss()   # mse_loss    #  
        net.latent_loss = 0 #3.0e-1
        net.flowlossweight = 1 #1e-1 #3.0e-1
        net.reconstruct_loss = 1 #1e-3 #4.0e-1
        net.kl_loss = 0 #
        net.bound_loss = 0 # 3e-1

        # net.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #included scheduler to implement minimum learning rate
        #     net.optimiser,
        #     factor=0.5,
        #     min_lr=1e-5,
        # )

        net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate*0.1)
        net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-6)

    # removed for fake data
    for dataset in datasets:
        # for with uncertainties
        dataset.spectra, dataset.uncertainty = transform(
            dataset.spectra,
            uncertainty=dataset.uncertainty,
        )
        dataset.params, dataset.param_uncertainty = param_transform(
            dataset.params,
            uncertainty=dataset. param_uncertainty,
        )

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

class NFautoencoder(nets.Autoencoder):
    def __init__(self, save_num, states_dir, net, overwrite=True, mix_precision = False, learning_rate = 0.001, description = '', verbose = 'epoch', transform = None, latent_transform = None, in_transform = None):
        super().__init__(save_num=save_num,
                         states_dir=states_dir,
                         net=net,
                         overwrite=overwrite,
                         mix_precision=mix_precision,
                         learning_rate=learning_rate,
                         description=description,
                         verbose=verbose,
                         transform=transform,
                         latent_transform=latent_transform)
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

        # Define a dictionary for loss components
        loss_components = { #torch.log1p() in recon? - might be unstable trianing for negative values
            'reconstruct':  self.reconstruct_func(output, in_data ),
            'flow': -1 * self.net.checkpoints[-1].log_prob(target).mean(),
            'latent':  self.latent_func(latent, target), #if self.latent_loss and latent is not None else None,
            'bound':  torch.mean(torch.cat((
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

        for key, weight in enumerate([self.reconstruct_loss, self.flowlossweight, self.latent_loss]):
            if separate_loss[key] is not None:
                separate_loss[key] *= weight

        # Compute the total loss
        loss = torch.sum(torch.stack(separate_loss))

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
        # for nonvariational wihtout uncertainties?? - uses checkpoints
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

    def training(self, epochs: int, loaders: tuple[DataLoader, DataLoader]) -> None:
        """
        Trains & validates the network for each epoch

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network up to
        loaders : tuple[DataLoader, DataLoader]
            Train and validation data loaders
        """
        t_initial: float
        final_loss: float

        losses =[]

        for i in range(len(self.losses[1])):
            losses.append(float(np.mean(self.losses[1][i-10:i])))

        # Train for each epoch
        for i in range(self._epoch, epochs):
            t_initial = time()

            # Train network
            self.train(True)
            self.losses[0].append(self._train_val(loaders[0]))

            # Validate network
            self.train(False)
            self.losses[1].append(self._train_val(loaders[1]))
            self._update_scheduler(metrics=self.losses[1][-1])

            # Save training progress
            self._update_epoch()
            self.save()

            if self._verbose in ('full', 'epoch'):
                print(f'Epoch [{self._epoch}/{epochs}]\t'
                      f'Training loss: {self.losses[0][-1]:.3e}\t'
                      f'Validation loss: {self.losses[1][-1]:.3e}\t'
                      f'Time: {time() - t_initial:.1f}')
            elif self._verbose == 'progress':
                progress_bar(
                    i,
                    epochs,
                    text=f'Epoch [{self._epoch}/{epochs}]\t'
                         f'Training loss: {self.losses[0][-1]:.3e}\t'
                         f'Validation loss: {self.losses[1][-1]:.3e}\t'
                         f'Time: {time() - t_initial:.1f}',
                )

            losses.append(float(np.mean(self.losses[1][-10:]))) # averages over 10 last values

            # End plateaued networks early
            if (net._epoch > net.scheduler.patience * 2 and
                net._epoch > 300 + net.scheduler.patience * 2 and # added for after synthetic train
                losses[-net.scheduler.patience * 2] < losses[-1]):

                print('Trial plateaued, ending early...')
                break

        self.train(False)
        final_loss = self._train_val(loaders[1])
        print(f'\nFinal validation loss: {final_loss:.3e}')

def NF_train(decoder, net, e_loaders, d_loaders, num_d_epochs, num_e_epochs, real_epochs, learning_rate):
    """
    Trains the normalising flow network.

    Parameters
    ----------
    decoder :
        The decoder model.
    net :
        The normalising flow network.
    e_loaders :
        The encoder data loaders. - in this case is just real data loaders
    d_loaders :
        The decoder data loaders. - in this case is just synthetic data loaders
    num_d_epochs : int
        The maximum number of epochs to train the decoder.
    num_e_epochs : int
        The maximum number of epochs to train the entire network end to end on synthetic data.
    real_epochs : int
        The maximum number of epochs to train the entire network end to end on real data.
    learning_rate : float
        The learning rate for the optimizer.
    """
    # train decoder
    decoder.training(num_d_epochs, d_loaders) 

    # # #fix decoder's weights so they dont change while training the encoder
    net.net.net[1] = decoder.net
    net.net.net[1].requires_grad_(False)

    #setting up autoencoder optimiser correctly - likely to not be needed
    if net.epochs==0:
        net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
        net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, min_lr=1e-6,)

    # train autoencoder on synthetic
    net.training(num_e_epochs, d_loaders)

    '''uncoment this to train only first few layers of encoder - check which layers are indexed''' 
    # net.net.net[0].net[2:].requires_grad_(False)
    # net.net.net[0].net[12].requires_grad_(True)
    ''' uncomment this to use unsupervised training'''
    # net.latent_loss = 0   # for unsupervised
    # net.flowlossweight = 0
    '''train on real'''
    # rest optimiser and train autoencoder on real
    if net.get_epochs() == num_e_epochs:
        net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate*0.1)
        net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-6)
    net.training(num_e_epochs+real_epochs, e_loaders)

#--- making predictions
def NF_predict(net, e_dataset, d_dataset, e_loaders, d_loaders, names, object_names, pred_savename, predict_for_synthetic=False):
    """
    Makes and saves predictions from the normalising flow network.
    Parameters
    ----------
    net :
        The normalising flow network.
    e_dataset :
        The encoder dataset. - in this case is just real dataset
    d_dataset :
        The decoder dataset. - in this case is just synthetic dataset
    e_loaders :
        The encoder dataloaders. - in this case is just real dataloaders
    d_loaders :
        The decoder dataloaders. - in this case is just synthetic dataloaders
    names : list
        List of object names for specific predictions.
    object_names : list
        List of full object names for specific predictions.
    pred_savename : str
        Name to save the prediction files as.
    predict_for_synthetic : bool
        Whether to make predictions for synthetic data or not.
    Returns
    -------
    tuple[dict, dict]
        Tuple containing all data and specific data dictionaries.
    """

    synth_str = '_synthetic_' if predict_for_synthetic else ''

    if predict_for_synthetic:    # specifying loader, based on synthetic predictions or real
        pred_loader = d_loaders
        pred_dataset = d_dataset
    else:
        pred_loader = e_loaders
        pred_dataset = e_dataset


    net_transforms = net.transforms.copy()  # save transforms
    for key in net.transforms:      # clear transforms
        net.transforms[key] = None

    # makes predictions (transformed)
    # train_data = net.predict(pred_loader[0], num_samples=1000, inputs=True)
    val_data = net.predict(pred_loader[1], num_samples=5000, inputs=True)
    data_idxs = np.arange(len(e_dataset))
    specific_subset = Subset(e_dataset, data_idxs[np.isin(e_dataset.names, names)].tolist())
    specific_loader = DataLoader(specific_subset, batch_size=64, shuffle=False)
    specific_data = net.predict(specific_loader, num_samples=5000, inputs=True)
    
    # untransforms data
    for pred_data in [specific_data, val_data]:
        for key, transform in net_transforms.items():
            if transform is None:
                continue
            if key == 'inputs':
                pred_data[key] = np.stack(transform(pred_data[key][:,0], back=True,
                                                uncertainty=pred_data[key][:,1]), axis=1)
            elif key == 'targets':
                pred_data[key] = np.stack(transform(pred_data[key], back=True,
                                                uncertainty=pred_dataset.param_uncertainty[np.isin(pred_dataset.names, pred_data['ids'])].numpy()), axis=1)
            else:
                pred_data[key] = transform(pred_data[key], back=True)

    if 'latent' not in specific_data and 'distributions' in specific_data:
        specific_data['latent']=specific_data['distributions']
        val_data['latent']=val_data['distributions']
        specific_data['preds']=specific_data['inputs']
        val_data['preds']=val_data['inputs']
    
    # add object names to specific data
    name_to_object = dict(zip(names, object_names))
    new_object_names = [name_to_object[name] for name in specific_data['ids']]
    specific_data['object']=new_object_names
    
    # reset transforms 
    net.transforms = net_transforms

    with open('/Users/work/Projects/FSPNet/predictions/specific_'+pred_savename+synth_str+'.pickle', 'wb') as file:
        pickle.dump(specific_data, file)
    with open('/Users/work/Projects/FSPNet/predictions/val_'+pred_savename+synth_str+'.pickle', 'wb') as file:
        pickle.dump(val_data, file)
    
    return val_data, specific_data

def NF_load_preds(pred_savename, predict_for_synthetic=False):
    """
    Loads predictions and data from pickle files.
    Parameters
    ----------
    pred_savename : str
        Name of the prediction files to load.
    predict_for_synthetic : bool
        Whether to load synthetic predictions or not.
    Returns
    -------
    tuple[dict, dict]
        Tuple containing validation and specific data dictionaries.
    """
    synth_str = '_synthetic_' if predict_for_synthetic else ''
    with open('/Users/work/Projects/FSPNet/predictions/specific_'+pred_savename+synth_str+'.pickle', 'rb') as file:
        specific_data = pickle.load(file)
    with open('/Users/work/Projects/FSPNet/predictions/val_'+pred_savename+synth_str+'.pickle', 'rb') as file:
        val_data = pickle.load(file)

    if 'latent' not in specific_data and 'distributions' in specific_data:
        specific_data['latent']=specific_data['distributions']
        val_data['latent']=val_data['distributions']
        specific_data['preds']=specific_data['inputs']
        val_data['preds']=val_data['inputs']

    return val_data, specific_data

# loading in xspec MCMC predictions
def load_xspec_preds(specific_data):
    """
    Loads xspec MCMC predictions from a pickle file.
    Parameters
    ----------
    specific_data : dict
        Dictionary containing specific data with 'object' key for ordering.
    Returns
    -------
    xspec_data : dict
        Dictionary containing ordered xspec predictions.
    """
        # note:
        # xspec_preds is with default values and fitting with 1000 iterations before chain
        # xspec_preds1 is with precalulated values and fitting with 1000 iterations before chain
    with open('/Users/work/Projects/FSPNet/predictions/xspec_preds1.pickle', 'rb') as file:
        xspec_data_unordered = pickle.load(file)

    # making the same order as specific_data
    xspec_lookup = {obj: i for i, obj in enumerate(xspec_data_unordered['object'])}     # Build a lookup dictionary for xspec objects
    xspec_indices = [xspec_lookup[obj] for obj in specific_data['object']]              # Get indices in the order of specific_data['object']   
    xspec_data = {                                                                      # Reorder xspec_data to match specific_data order
        'ids': [xspec_data_unordered['id'][i] for i in xspec_indices],
        'object': [xspec_data_unordered['object'][i] for i in xspec_indices],
        'posteriors': [xspec_data_unordered['posteriors'][i] for i in xspec_indices],
        'xspec_recon': [xspec_data_unordered['xspec_recon'][i] for i in xspec_indices],
        'chain_time': [xspec_data_unordered['chain_time'][i] for i in xspec_indices]
    }
    # taking 5000 uniformly random distributed data points from last half of the posterior samples
    new_posteriors = np.array([[random.sample(list(xspec_data['posteriors'] [spec_num][param_num][len(xspec_data['posteriors'][spec_num][param_num])//2:]), 5000) 
                    for param_num in range(len(xspec_data['posteriors'][0]))] 
                    for spec_num in range(len(xspec_data['posteriors']))])
    xspec_data['posteriors']=new_posteriors

    return xspec_data

def main(num_d_epochs=150, num_e_epochs=150, real_epochs=150, learning_rate=0.001, train=False, predict=False, predict_for_synthetic=False, specific=True):

    #initialise data loaders and networks
    e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init()
    
    # saves name of predictions as encoder name_decoder name
    synth_plot_str = 'synthetic'  if predict_for_synthetic else 'real'
    pred_savename = os.path.basename(net.save_path)[:-4]+' '+os.path.basename(decoder.save_path)[:-4] #'Encoder NF0_2' #
    plots_directory = '/Users/work/Projects/FSPNet/plots/'+pred_savename+'/'+synth_plot_str+'_preds/'
    os.makedirs(plots_directory, exist_ok=True)
    os.makedirs(plots_directory+'reconstructions/', exist_ok=True)
    os.makedirs(plots_directory+'distributions/', exist_ok=True)
    os.makedirs(plots_directory+'comparisons/', exist_ok=True)

    # for getting specific spectra
    names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp','js_ni0103010102_0mpu7_goddard_GTI0.jsgrp','js_ni1014010102_0mpu7_goddard_GTI30.jsgrp','js_ni1050360115_0mpu7_goddard_GTI9.jsgrp','js_ni1100320119_0mpu7_goddard_GTI26.jsgrp','js_ni1200120203_0mpu7_goddard_GTI0.jsgrp','js_ni1200120203_0mpu7_goddard_GTI10.jsgrp','js_ni1200120203_0mpu7_goddard_GTI11.jsgrp','js_ni1200120203_0mpu7_goddard_GTI13.jsgrp','js_ni1200120203_0mpu7_goddard_GTI1.jsgrp','js_ni1200120203_0mpu7_goddard_GTI3.jsgrp','js_ni1200120203_0mpu7_goddard_GTI4.jsgrp','js_ni1200120203_0mpu7_goddard_GTI5.jsgrp','js_ni1200120203_0mpu7_goddard_GTI6.jsgrp','js_ni1200120203_0mpu7_goddard_GTI7.jsgrp','js_ni1200120203_0mpu7_goddard_GTI8.jsgrp','js_ni1200120203_0mpu7_goddard_GTI9.jsgrp']
    object_names=['Cyg X-1 (2017)','GRS 1915+105','LMC X-3','MAXI J1535-571','Cyg X-1 (2018)','MAXI J1820 0','MAXI J1820 10','MAXI J1820 11','MAXI J1820 13','MAXI J1820 1','MAXI J1820 3','MAXI J1820 4','MAXI J1820 5','MAXI J1820 6','MAXI J1820 7','MAXI J1820 8','MAXI J1820 9']

    if train:
        NF_train(decoder, net, e_dataset, d_dataset, e_loaders, d_loaders, num_d_epochs, num_e_epochs, real_epochs, learning_rate)
        val_data, specific_data = NF_predict(net, e_dataset, d_dataset, e_loaders, d_loaders, names, object_names, predict_for_synthetic)
    elif predict:
        val_data, specific_data = NF_predict(net, e_dataset, d_dataset, e_loaders, d_loaders, names, object_names, predict_for_synthetic)
    else:
        val_data, specific_data = NF_load_preds(pred_savename, predict_for_synthetic)

    # overwrite specific_data as none if not plotting specific spectra
    specific_data = specific_data if specific else None

    xspec_data = load_xspec_preds(specific_data)

    '''---------- PLOTTING ----------'''
    # autoencoder performance - remember to change part of net_init to correspond to encoder only vs autoencoder
    plots.plot_performance(
        'Loss',
        net.losses[1][1:],
        plots_dir=plots_directory,
        train=net.losses[0][1:],
        save_name='NF_perfomance.png'
    )

    # plot separate losses
    # gets losses from the seperate losses attribute of the net class
    # final_synth_epoch = 121
    # final_real_epoch = len(net.losses[1])-final_synth_epoch
    # synth_cut = final_synth_epoch*len(d_loaders[0])
    # synth_loss ={}
    # real_loss ={}
    # total_loss = {}
    # for key in ['reconstruct', 'flow', 'latent', 'bound', 'kl']:
    #     synth_loss[key] = np.array(net.separate_losses[key])[:synth_cut].reshape(final_synth_epoch, -1)
    #     real_loss[key] = np.array(net.separate_losses[key])[synth_cut:synth_cut+final_real_epoch*len(e_loaders[0])].reshape(final_real_epoch, -1)
    #     synth_loss[key]=np.mean(synth_loss[key], axis=-1)
    #     real_loss[key]=np.mean(real_loss[key], axis=-1)
    #     total_loss[key] = np.concatenate((synth_loss[key], real_loss[key]), axis=0)
    #     total_loss[key]=synth_loss[key]

    # separate losses performance plot
    # performance_plot(
    #     'Loss',
    #     {key: value.tolist() for key, value in total_loss.items()},
    #     plots_dir=plots_directory,
    #     save_name='NF_allloss.png'
    # )

    # decoder performance
    plots.plot_performance(
        'Loss',
        decoder.losses[1][1:],
        plots_dir=plots_directory,
        train=decoder.losses[0][1:],
        save_name='NF_d_performance'
    )

    # plotting comparison between parameters
    # to color data by certain parameters
    kT = list(val_data['targets'][:,0,3])
    nH = list(val_data['targets'][:,0,0])
    gamma = list(val_data['targets'][:,0,1])
    fsc = list(val_data['targets'][:,0,2])
    nH_errors = list(val_data['targets'][:,1,0])
    det_nums=[]
    for spectrum in val_data['ids']:
        with fits.open('/Users/work/Projects/FSPNet/data/spectra/'+spectrum) as file:
            spectrum_info = file[1].header
        det_nums.append(int(re.search(r'_d(\d+)', spectrum_info['RESPFILE']).group(1)))
    det_nums = np.array(det_nums)[:,np.newaxis]
    widths = get_energy_widths()[np.newaxis]
    total_counts=np.sum(val_data['inputs'][:,0,:]*det_nums*widths, axis=-1)

    comparison_plot(
        val_data,
        specific_data=specific_data,
        log_colour_map=True,
        colour_map=kT,
        colour_map_label='kT (keV)',
        dir_name=os.path.join(plots_directory, 'comparisons/'),
        n_points=50,
        num_dist_specs=250
    )

    coverage_plot(
        dataset=d_dataset, 
        loaders=d_loaders,
        network = net,
        dir_name=plots_directory,
        pred_savename=pred_savename,
    )

    all_param_samples = sample(specific_data if specific else val_data,
                            num_specs=len(specific_data['ids']),
                            num_samples=1,
                            spec_scroll=SPEC_SCROLL)

    # single reconstructions using samples from all_param_samples
    recon_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = val_data,
        specific_data = specific_data,
        all_param_samples = all_param_samples,
        data_dir = '/Users/work/Projects/FSPNet/data/spectra.pickle' if predict_for_synthetic else '/Users/work/Projects/FSPNet/data/spectra/',
        spec_scroll=SPEC_SCROLL
        )

    # posterior predictive plots using 100 samples per reconstruction
    post_pred_samples = post_pred_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = val_data,
        specific_data = specific_data,
    )

    # posterior predictive plots only using xspec to make reconstructions - not decoder
    post_pred_plot_xspec(
        dir_name = plots_directory+'reconstructions/',
        data = val_data,
        specific_data = specific_data,
        post_pred_samples=post_pred_samples
    )

    # latent space corner plot
    latent_corner_plot(
        dir_name = plots_directory+'distributions/',
        data=val_data,
        xspec_data=xspec_data,
        specific_data=specific_data,
    )

    # scatter plot across all target parameters in dataset
    # param_pairs_plot(
    #     data=val_data,
    #     dir_name=plots_directory,
    # )

    # 2D reconstruction plots - still working on this
    # rec_2d_plot(
    #     decoder=decoder.net,
    #     network=net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data=data,
    # )

    '''---------- pyxspec tests ----------'''
    # import xspec
    # xspec.Xset.chatter = 0
    # xspec.Xset.logChatter = 0

    # val_data['latent'] = val_data['latent'][:,0,:] # #np.median(val_data['latent'], axis = 1)
    # val_data['targets'] = val_data['targets'][:,0,:]
    # # for i in range(5):
    # pyxspec_tests(val_data)

    # making predictions for timing purposes
    # for i in range(5):
    #     print('1000 samples:')
    #     net.predict(e_loaders[1], num_samples=1000, inputs=True)
    #     print('1 sample:')
    #     net.predict(e_loaders[1], num_samples=1, inputs=True)

if __name__ == '__main__':
    # settings
    num_d_epochs = 400
    num_e_epochs = 121
    real_epochs = 216
    learning_rate = 1.0e-3 # in config: 1e-4
    train = False
    predict = False
    predict_for_synthetic = False
    specific = True
    SPEC_SCROLL = 0

    main(num_d_epochs, 
         num_e_epochs, 
         real_epochs, 
         learning_rate, 
         train, 
         predict, 
         predict_for_synthetic, 
         specific)
