from fspnet.spectrum_fit import init
from fspnet.spectrum_fit import AutoencoderNet 

import os
import pickle
from typing import Any

import torch
import random
import numpy as np
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
from fspnet.spectrum_fit import pyxspec_tests

import matplotlib.pyplot as plt

from VAE_plots import comparison_plot, recon_plot, post_pred_plot, latent_corner_plot, rec_2d_plot, param_pairs_plot
from VAE_plots import performance_plot
from my_utils.misc_utils import sample

plt.style.use(["science", "grid", 'no-latex'])

#gaussian loss for variational autoencoder with uncertainties
def gaussian_loss(predictions, target):
    return nn.GaussianNLLLoss()(predictions[:,0], target[:,0], target[:,1]**2) #(spectra)(B,240), target[:,1]**2 (B,2,240) (uncertainty)) - for variational with uncertaimties??

#MSE loss for variational autoencoder
def mse_loss(predictions, target):
    return nn.MSELoss()(predictions[:,0], target[:,0]) #(spectra))(B,240)

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
            mean=False,
        ))

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
        # transform = None
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

        # chooses to train NF_autoencoder
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

        #Loss function settings for autoencoder
        net.reconstruct_func = gaussian_loss   
        net.latent_func = mse_loss           
        net.latent_loss = 1 #3.0e-1
        net.flowlossweight = 1 #1e-1 #3.0e-1
        net.reconstruct_loss = 1 #1e-3 #4.0e-1
        net.kl_loss = 0 #
        net.bound_loss = 0 # 3e-1

        net.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #included scheduler to implement minimum learning rate
            net.optimiser,
            factor=0.5,
            min_lr=1e-5,
        )

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

        # for without uncertainties
        # dataset.spectra = transform(dataset.spectra)
        # dataset.params = param_transform(dataset.params)


        # for fake spectra
        # dataset = transform(
        #     dataset)
        # dataset = param_transform(
        #     datasets
        # )

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
    # loader = DataLoader(dataset, batch_size=60, shuffle=False
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
        
        # Define a dictionary for loss components
        loss_components = { #torch.log1p() in recon? - might be unstable trianing for negative values
            'reconstruct': self.reconstruct_loss * self.reconstruct_func(output, in_data ),
            'flow': -1 * self.flowlossweight * self.net.checkpoints[-1].log_prob(target).mean(),
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
# from netloader.layers.flows import SplineFlow

# settings
MAJOR = 26
MINOR = 20
TICK = 16
num_epochs = 50
real_epochs = 50
learning_rate = 1.0e-3 #in config: 1e-4
predict = False
predict_for_synthetic = True
plot_synthetic = True
plot_specific = False
SPEC_SCROLL=0


#initialise data loaders and networks
e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init()

# saves name of predictions as encoder name_decoder name
pred_savename = os.path.basename(net.save_path)[:-4]+' '+os.path.basename(decoder.save_path)[:-4]
plots_directory = '/Users/astroai/Projects/FSPNet/plots/'+pred_savename+'/'
os.makedirs(plots_directory, exist_ok=True)
os.makedirs(plots_directory+'reconstructions/', exist_ok=True)
os.makedirs(plots_directory+'distributions/', exist_ok=True)

# train decoder
decoder.training(num_epochs, d_loaders) 

# #fix decoder's weights so they dont change while training the encoder
net.net.net[1].requires_grad_(False)

#setting up autoencoder optimiser correctly - likely to not be needed
net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-6,)

# train autoencoder
net.training(num_epochs, d_loaders)

# train only first few layers of encoder - check layers 
# net.net.net[0].net[2:].requires_grad_(False) 
# net.net.net[0].net[12].requires_grad_(True) 

# reset optimiser
# net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate*0.1)
# net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-6,)

# net.training(num_epochs+real_epochs, e_loaders)

#--- making predictions
if predict:
    # save transforms
    transform = net.transforms['inputs']
    param_transform = net.transforms['targets']
    # # clear transforms
    net.transforms['inputs'] = None
    net.transforms['targets'] = None


    if predict_for_synthetic:
        pred_loader = d_loaders
        pred_dataset = d_dataset
    else:
        pred_loader = e_loaders
        pred_dataset = e_dataset

    data = net.predict(pred_loader[-1], num_samples=1, input_=True)
    data1 = net.predict(pred_loader[-1], num_samples=3000, input_=True)

    names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp', 
            'js_ni0103010102_0mpu7_goddard_GTI0.jsgrp',
            'js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
            'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp',
            'js_ni1100320119_0mpu7_goddard_GTI26.jsgrp']
    object_names=['Cyg X-1 (2017)',
              'GRS 1915+105',
              'LMC X-3',
              'MAXI J1535-571',
              'Cyg X-1 (2018)']
    train_data = net.predict(e_loaders[0], num_samples=3000, input_=True)
    val_data = net.predict(e_loaders[1], num_samples=3000, input_=True)
    train_idxs = np.isin(train_data['ids'], names)
    val_idxs = np.isin(val_data['ids'], names)
    # # Then with the idxs, you can get the posteriors (or whatever else you want from the predicitons) by:
    targets = np.concat((train_data['targets'][train_idxs],val_data['targets'][val_idxs]), axis=0)
    latent = np.concat((train_data['latent'][train_idxs],val_data['latent'][val_idxs]), axis=0)
    inputs = np.concat((train_data['inputs'][train_idxs], val_data['inputs'][val_idxs]), axis=0)
    preds = np.concat((train_data['preds'][train_idxs], val_data['preds'][val_idxs]), axis=0)
    new_names = np.concat((train_data['ids'][train_idxs], val_data['ids'][val_idxs]), axis=0)

    # Reorder object_names to match new names
    name_to_object = dict(zip(names, object_names))
    new_object_names = [name_to_object[name] for name in new_names]

    specific_data = {
        'id': new_names,
        'targets': targets,
        'latent': latent,
        'inputs': inputs,
        'preds': preds,
        'object': new_object_names
    }

    param_uncertainties = pred_dataset.param_uncertainty[np.isin(pred_dataset.names, data['ids'])]  # get uncertainties in 'ground truth' parameters
    param_uncertainties1 = pred_dataset.param_uncertainty[np.isin(pred_dataset.names, data1['ids'])]

    # orders param_uncertainties
    specific_param_uncertainties = []
    for i in range(len(specific_data['id'])):
        specific_param_uncertainties.append(e_dataset.param_uncertainty[np.isin(e_dataset.names, specific_data['id'][i])])
    specific_param_uncertainties = np.squeeze(np.array(specific_param_uncertainties))

    # untransforms and stacks uncertainties
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

    # reset transforms 
    net.transforms['inputs'] = transform
    net.transforms['targets'] = param_transform

    # saves predictions to pickle file TRANSFORMED PARAM_UNCERTAINTIES
    with open('/Users/astroai/Projects/FSPNet/predictions/preds1_'+pred_savename+'.pickle', 'wb') as file:
        pickle.dump(data1, file)
    with open('/Users/astroai/Projects/FSPNet/predictions/preds_'+pred_savename+'.pickle', 'wb') as file:
        pickle.dump(data, file)
    with open('/Users/astroai/Projects/FSPNet/predictions/preds_specific_'+pred_savename+'.pickle', 'wb') as file:
        pickle.dump(specific_data, file)

else:
    # loads predictions from saved pickle file
    with open('/Users/astroai/Projects/FSPNet/predictions/preds1_'+pred_savename+'.pickle', 'rb') as file:
            data1 =pickle.load(file)
    with open('/Users/astroai/Projects/FSPNet/predictions/preds_'+pred_savename+'.pickle', 'rb') as file:
            data = pickle.load(file)
    with open('/Users/astroai/Projects/FSPNet/predictions/preds_specific_'+pred_savename+'.pickle', 'rb') as file:
        specific_data = pickle.load(file)

# loading in xspec MCMC predictions
with open('/Users/astroai/Projects/FSPNet/predictions/xspec_preds1.pickle', 'rb') as file:
    xspec_data_unordered = pickle.load(file)

    # note:
    # xspec_preds is with default values and fitting with 1000 iterations before chain
    # xspec_preds1 is with precalulated values and fitting with 1000 iterations before chain

    # note: shapes:
    #   id: number of specific spectra (5)
    #   object: number of specific spectra (5)
    #   posteriors: number of specific spectra (5), number of parameters, number of samples
    #   xspec_recon: number of specific spectra (5), 2 (indices: 0 is spectral energies, 1 is the spectra recontructions)
    #   chain_time: number of specific spectra (5)

# shows the order of the spectra - numbers corresponds to xspec_data and their position corresponds to specific_data
xspec_indices = []
for specific_object in specific_data['object']:
    for i, xspec_object in enumerate(xspec_data_unordered['object']):
        if specific_object==xspec_object:
            xspec_indices.append(i)

# making sure xspec_data order matches specific_data order
xspec_data = {
    'id': [xspec_data_unordered['id'][i] for i in xspec_indices],
    'object': [xspec_data_unordered['object'][i] for i in xspec_indices],
    'posteriors': [xspec_data_unordered['posteriors'][i] for i in xspec_indices],
    'xspec_recon': [xspec_data_unordered['xspec_recon'][i] for i in xspec_indices],
    'chain_time': [xspec_data_unordered['chain_time'][i] for i in xspec_indices]
}

# taking ~3000 randomly distributed data points from last half of the posterior samples
new_posteriors = np.array([[random.sample(list(xspec_data['posteriors'] [spec_num][param_num][len(xspec_data['posteriors'][spec_num][param_num])//2:]), 3000) 
                   for param_num in range(len(xspec_data['posteriors'][0]))] 
                   for spec_num in range(len(xspec_data['posteriors']))])

xspec_data['posteriors']=new_posteriors

# xspec_data['id'] = np.stack((xspec_data0['id'], xspec_data1['id'], xspec_data2['id'], xspec_data3['id'], xspec_data4['id']), axis=0)   
# # note: reconstruction lists are all different lengths so we keep as a list
# xspec_data['xspec_recon'] = [xspec_data0['xspec_recon'], xspec_data1['xspec_recon'], xspec_data2['xspec_recon'], xspec_data3['xspec_recon'], xspec_data4['xspec_recon']]
# # note: xspec_data['posteriors'] shape = number of spectra, number of parameters, number of samples
# xspec_data['posteriors'] = np.stack((xspec_data0['posteriors'], xspec_data1['posteriors'], xspec_data2['posteriors'], xspec_data3['posteriors'], xspec_data4['posteriors']), axis=0)

# gets losses from the seperate losses attribute of the net class
separate_losses = net.separate_losses # shape (number of iterations ((dataset/batch)*epochs), number of loss terms)

# # averaging loss over batch size
separate_losses['reconstruct'] = [np.mean(separate_losses['reconstruct'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['reconstruct']), len(e_loaders[0]))]
separate_losses['flow'] = [np.mean(separate_losses['flow'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['flow']), len(e_loaders[0]))]  
separate_losses['latent'] = [np.mean(separate_losses['latent'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['latent']), len(e_loaders[0]))]
separate_losses['bound'] = [np.mean(separate_losses['bound'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['bound']), len(e_loaders[0]))]
separate_losses['kl'] = [np.mean(separate_losses['kl'][i:i+len(e_loaders[0])], axis=0) for i in range(0, len(separate_losses['kl']), len(e_loaders[0]))]


'''---------- PLOTTING PERFORMANCE ----------'''

# autoencoder performance - remember to change part of net_init to include autoencoder and not encoder
plots.plot_performance(
    'Loss',
    net.losses[1][1:],
    plots_dir=plots_directory,
    train=net.losses[0][1:],
    save_name='NF_perfomance.png'
)

performance_plot(
    'Loss',
    separate_losses,
    plots_dir=plots_directory,
    save_name='NF_allloss.png'
)

# # decoder performance
plots.plot_performance(
    'Loss',
    decoder.losses[1][1:],
    plots_dir=plots_directory,
    train=decoder.losses[0][1:],
    save_name='NF_d_performance'
)

if plot_synthetic:
    # plotting comparison between parametrs
    comparison_plot(
        data1,
        dir_name=plots_directory,
    )

    all_param_samples = sample(data1,
                            num_specs=3,
                            num_samples=1,
                            spec_scroll=SPEC_SCROLL)

    # # single reconstructions using samples from all_param_samples
    recon_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = data1,
        all_param_samples = all_param_samples,
        data_dir = '/Users/astroai/Projects/FSPNet/data/spectra.pickle', # for synthetic data,
        spec_scroll=SPEC_SCROLL
        )

    # # posterior predictive plots using 500 samples per reconstruction
    post_pred_samples = post_pred_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = data1,
    )

    # # latent space corner plot
    latent_corner_plot(
        dir_name = plots_directory+'distributions/',
        data=data1,
        )

    # scatter plot across all parameters in dataset
    param_pairs_plot(
        data=data,
        dir_name=plots_directory+'distributions/',
    )

elif plot_specific:
     # # plotting comparison between parametrs
    comparison_plot(
        data1,
        dir_name=plots_directory,
        specific_data=specific_data
    )

    all_param_samples = sample(specific_data,
                            num_specs=3,
                            num_samples=1,
                            spec_scroll=SPEC_SCROLL)

    # # single reconstructions using samples from all_param_samples
    recon_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        specific_data = specific_data,
        data = data1,
        all_param_samples = all_param_samples
        )

    # # posterior predictive plots using 500 samples per reconstruction
    post_pred_samples = post_pred_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = data1,
        specific_data = specific_data
    )

    # # latent space corner plot
    latent_corner_plot(
        dir_name = plots_directory+'distributions/',
        specific_data = specific_data,
        data=data1,
        xspec_data=xspec_data
        )

    # scatter plot across all parameters in dataset
    param_pairs_plot(
        data=data,
        dir_name=plots_directory+'distributions/',
    )

else:
    # # plotting comparison between parametrs
    comparison_plot(
        data1,
        dir_name=plots_directory,
    )

    all_param_samples = sample(data1,
                            num_specs=3,
                            num_samples=1,
                            spec_scroll=SPEC_SCROLL)

    # # single reconstructions using samples from all_param_samples
    recon_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = data1,
        all_param_samples = all_param_samples
        )

    # # posterior predictive plots using 500 samples per reconstruction
    post_pred_samples = post_pred_plot(
        decoder.net,
        net,
        dir_name = plots_directory+'reconstructions/',
        data = data1,
    )

    # # latent space corner plot
    latent_corner_plot(
        dir_name = plots_directory+'distributions/',
        data=data1,
        )

    # scatter plot across all parameters in dataset
    param_pairs_plot(
        data=data,
        dir_name=plots_directory+'distributions/',
    )


'''---------- pyxspec tests ----------'''

# xspec.Xset.chatter = 0
# xspec.Xset.logChatter = 0

# data['latent'] = np.squeeze(data['latent'])
# data['targets'] = data['targets'][:,0,:]
# pyxspec_tests(data)
