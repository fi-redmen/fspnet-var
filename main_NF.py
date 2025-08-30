import xspec
from fspnet.spectrum_fit import init, AutoencoderNet 

import numpy as np
import os
import pickle
from typing import Any
import random
import torch

import netloader.networks as nets
from netloader.network import Network
from netloader import transforms, loss_funcs
from netloader.utils.utils import save_name, get_device, progress_bar

from torch.utils.data import DataLoader
from torch import nn, optim, Tensor
from numpy import ndarray
from time import time
import lampe

from fspnet.utils import plots
from fspnet.utils.utils import open_config
from fspnet.utils.data import SpectrumDataset, loader_init
from fspnet.spectrum_fit import pyxspec_tests

import matplotlib.pyplot as plt

from VAE_plots import comparison_plot, recon_plot, post_pred_plot, rec_2d_plot, param_pairs_plot 
from VAE_plots import performance_plot, post_pred_plot_xspec, coverage_plot
from my_utils.misc_utils import sample

# from netloader.layers.flows

plt.style.use(["science", "grid", 'no-latex'])

# # gaussian loss for variational autoencoder with uncertainties
# def gaussian_loss(predictions, target):
#     return nn.GaussianNLLLoss()(predictions[:,0], target[:,0], target[:,1]**2) #(spectra)(B,240), target[:,1]**2 (B,2,240) (uncertainty)) - for variational with uncertaimties??

# #MSE loss for variational autoencoder
# def mse_loss(predictions, target):
#     return nn.MSELoss()(predictions[:,0], target[:,0]) #(spectra))(B,240)

class MSELoss(loss_funcs.MSELoss):
    """
    Mean Squared Error (MSE) loss function
    """
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self._loss_func(output[:, 0], target[:, 0])


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
            overwrite=True,
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
        )

        decoder.transforms['inputs'] = param_transform
        decoder.loss_func =  GaussianNLLLoss() # gaussian_loss  #  #changes loss to gaussian loss - can look into other typs of loss

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
        net = Network(
            encoder_name,
            networks_dir,
            list((datasets[0][0][2][0].shape,
            datasets[0][0][1].shape)),
            list(datasets[0][0][1].shape)
        )

        # chooses to train NF_autoencoder
        # net = NFautoencoder(
        #     e_save_num,
        #     states_dir,
        #     NFautoencoderNetwork(net, decoder.net, name=encoder_name),
        #     learning_rate=learning_rate,
        #     description=description,
        #     verbose='epoch',
        #     transform=transform,
        #     latent_transform=param_transform,
        # )
        
        #

        #Loss function settings for autoencoder
        net.reconstruct_func =  GaussianNLLLoss() # gaussian_loss   #
        net.latent_func = MSELoss()   # mse_loss    #   
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
num_d_epochs = 400
num_e_epochs = 400
real_epochs = 400
learning_rate = 1.0e-3 #in config: 1e-4
predict = False
predict_for_synthetic = False
plot_synthetic = False
plot_specific = True
plot_obsid = False
SPEC_SCROLL=0

#initialise data loaders and networks
e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init()

# saves name of predictions as encoder name_decoder name
synth_plot_str = 'synth' if plot_synthetic else 'real'
pred_savename = os.path.basename(net.save_path)[:-4]+' '+os.path.basename(decoder.save_path)[:-4]
plots_directory = '/Users/work/Projects/FSPNet/plots/'+pred_savename+'/'+synth_plot_str+'_preds/'
os.makedirs(plots_directory, exist_ok=True)
os.makedirs(plots_directory+'reconstructions/', exist_ok=True)
os.makedirs(plots_directory+'distributions/', exist_ok=True)

# train decoder
# decoder.training(num_d_epochs, d_loaders) 

# # #fix decoder's weights so they dont change while training the encoder
# net.net.net[1] = decoder.net
# net.net.net[1].requires_grad_(False)

# #setting up autoencoder optimiser correctly - likely to not be needed
# net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
# net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, min_lr=1e-6,)

# train autoencoder
net.training(num_e_epochs, d_loaders)

# train only first few layers of encoder - check layers 
# net.net.net[0].net[2:].requires_grad_(False) 
# net.net.net[0].net[12].requires_grad_(True) 
# net.latent_loss = 0   # for unsupervised
# net.flowlossweight = 0

'''uncomment this for real'''
# rest optimeser and train autoencoder on real
# net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate*0.1)
# net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, factor=0.5, min_lr=1e-6)
# net.training(num_e_epochs+real_epochs, e_loaders)



synth_str = '_synthetic_' if predict_for_synthetic else ''

#--- making predictions
# save transforms
transform = net.transforms['inputs']
param_transform = net.transforms['targets']
# # # clear transforms
net.transforms['inputs'] = None
net.transforms['targets'] = None
# net.transforms['latent'] = None

if predict_for_synthetic:
    pred_loader = d_loaders
    pred_dataset = d_dataset
else:
    pred_loader = e_loaders
    pred_dataset = e_dataset

if predict:
    train_data = net.predict(pred_loader[0], num_samples=1000, inputs=True)
    val_data = net.predict(pred_loader[1], num_samples=1000, inputs=True)

    with open('/Users/work/Projects/FSPNet/predictions/train_'+pred_savename+synth_str+'.pickle', 'wb') as file:
        pickle.dump(train_data, file)
    with open('/Users/work/Projects/FSPNet/predictions/val_'+pred_savename+synth_str+'.pickle', 'wb') as file:
        pickle.dump(val_data, file)
    
    # data = net.predict(pred_loader[-1], num_samples=1, input_=True)
    # data1 = net.predict(pred_loader[-1], num_samples=3000, input_=True)

else:
    with open('/Users/work/Projects/FSPNet/predictions/train_'+pred_savename+synth_str+'.pickle', 'rb') as file:
        train_data = pickle.load(file)
    with open('/Users/work/Projects/FSPNet/predictions/val_'+pred_savename+synth_str+'.pickle', 'rb') as file:
        val_data = pickle.load(file)

# getting specific spectra
names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp', 
        'js_ni0103010102_0mpu7_goddard_GTI0.jsgrp',
        'js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
        'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp',
        'js_ni1100320119_0mpu7_goddard_GTI26.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI0.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI10.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI11.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI13.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI1.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI3.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI4.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI5.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI6.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI7.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI8.jsgrp',
        'js_ni1200120203_0mpu7_goddard_GTI9.jsgrp']
object_names=['Cyg X-1 (2017)',
        'GRS 1915+105',
        'LMC X-3',
        'MAXI J1535-571',
        'Cyg X-1 (2018)',
        'MAXI J1820 0',
        'MAXI J1820 10',
        'MAXI J1820 11',
        'MAXI J1820 13',
        'MAXI J1820 1',
        'MAXI J1820 3',
        'MAXI J1820 4',
        'MAXI J1820 5',
        'MAXI J1820 6',
        'MAXI J1820 7',
        'MAXI J1820 8',
        'MAXI J1820 9']

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
    'ids': new_names,
    'targets': targets,
    'latent': latent,
    'inputs': inputs,
    'preds': preds,
    'object': new_object_names
}

# saves untransformed specific
with open('/Users/work/Projects/FSPNet/predictions/specific_untrans_'+pred_savename+synth_str+'.pickle', 'wb') as file:
    pickle.dump(specific_data, file)

# getting parameter uncertainties from datasets
train_param_uncertainties = pred_dataset.param_uncertainty[np.isin(pred_dataset.names, train_data['ids'])]  # get uncertainties in 'ground truth' parameters
val_param_uncertainties = pred_dataset.param_uncertainty[np.isin(pred_dataset.names, val_data['ids'])]
specific_param_uncertainties = []
for i in range(len(specific_data['ids'])):
    specific_param_uncertainties.append(e_dataset.param_uncertainty[np.isin(e_dataset.names, specific_data['ids'][i])])
specific_param_uncertainties = np.squeeze(np.array(specific_param_uncertainties))

# untransforms and stacks uncertainties
train_data['targets'] = np.stack(param_transform(train_data['targets'], back=True,
                                            uncertainty=train_param_uncertainties), axis=1)
train_data['inputs'] = np.stack(transform(train_data['inputs'][:,0], back=True,
                                        uncertainty=train_data['inputs'][:,1]), axis=1)
val_data['targets'] = np.stack(param_transform(val_data['targets'], back=True,
                                            uncertainty=val_param_uncertainties), axis=1)
val_data['inputs'] = np.stack(transform(val_data['inputs'][:,0], back=True,
                                        uncertainty=val_data['inputs'][:,1]), axis=1)
# specific_data['targets'] = np.stack(param_transform(specific_data['targets'], back=True,
#                                             uncertainty=specific_param_uncertainties), axis=1)
# specific_data['inputs'] = np.stack(transform(specific_data['inputs'][:,0], back=True,
#                                             uncertainty=specific_data['inputs'][:,1]), axis=1)

# reset transforms 
net.transforms['inputs'] = transform
net.transforms['targets'] = param_transform

with open('/Users/work/Projects/FSPNet/predictions/specific_'+pred_savename+synth_str+'.pickle', 'wb') as file:
    pickle.dump(specific_data, file)
#setting data1 and data from val_data - and saving as this is transformed data
data1 = val_data
with open('/Users/work/Projects/FSPNet/predictions/data1_'+pred_savename+synth_str+'.pickle', 'wb') as file:
    pickle.dump(data1, file)


# loading in xspec MCMC predictions
    # note:
    # xspec_preds is with default values and fitting with 1000 iterations before chain
    # xspec_preds1 is with precalulated values and fitting with 1000 iterations before chain

    # note: shapes:
    #   ids: number of specific spectra (5)
    #   object: number of specific spectra (5)
    #   posteriors: number of specific spectra (5), number of parameters, number of samples
    #   xspec_recon: number of specific spectra (5), 2 (indices: 0 is spectral energies, 1 is the spectra recontructions)
    #   chain_time: number of specific spectra (5)

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

# gets losses from the seperate losses attribute of the net class
separate_losses = net.separate_losses.copy()   # shape: (number of iterations ((dataset/batch)*epochs), number of loss terms)
split_idx = num_e_epochs                # averaging loss over batch size
# batches_per_epoch_e = 1000              # Number of batches per epoch for each phase
# batches_per_epoch_real = 60
# for key in ['reconstruct', 'flow', 'latent']:
#     losses = separate_losses[key]
#     averaged_losses = []
#     # First num_e_epochs: average over 1000 spectra per batch
#     for i in range(0, split_idx * batches_per_epoch_e, batches_per_epoch_e):
#         averaged_losses.append(np.mean(losses[i:i + batches_per_epoch_e], axis=0))
#     # Next real_epochs: average over 60 spectra per batch
#     for i in range(split_idx * batches_per_epoch_e, len(losses), batches_per_epoch_real):
#         averaged_losses.append(np.mean(losses[i:i + batches_per_epoch_real], axis=0))
#     separate_losses[key] = averaged_losses

# batch_size=60
# for key in ['reconstruct', 'flow', 'latent']:
#     losses = separate_losses[key]
#     averaged_losses = []
#     # First num_e_epochs: average over 1000 spectra per batch
#     for i in range(0, num_e_epochs+real_epochs, batch_size):
#         averaged_losses.append(np.mean(losses[i:i+batch_size], axis=0))
#     # Next real_epochs: average over 60 spectra per batch

    # separate_losses[key] = averaged_losses

final_synth_epoch = 121
final_real_epoch = len(net.losses[1])-final_synth_epoch

synth_cut = final_synth_epoch*len(d_loaders[0])

synth_loss ={}
real_loss ={}
total_loss = {}

for key in ['reconstruct', 'flow', 'latent', 'bound', 'kl']:
    synth_loss[key] = np.array(net.separate_losses[key])[:synth_cut].reshape(final_synth_epoch, -1)
    real_loss[key] = np.array(net.separate_losses[key])[synth_cut:synth_cut+final_real_epoch*len(e_loaders[0])].reshape(final_real_epoch, -1)

    synth_loss[key]=np.mean(synth_loss[key], axis=-1)
    real_loss[key]=np.mean(real_loss[key], axis=-1)

    total_loss[key] = np.concatenate((synth_loss[key], real_loss[key]), axis=0)

'''---------- PLOTTING PERFORMANCE ----------'''

# autoencoder performance - remember to change part of net_init to include autoencoder and not encoder
plots.plot_performance(
    'Loss',
    net.losses[1][1:],
    plots_dir=plots_directory,
    train=net.losses[0][1:],
    save_name='NF_perfomance.png'
)

# performance_plot(
#     'Loss',
#     {key: value.tolist() for key, value in total_loss.items()},
#     plots_dir=plots_directory,
#     save_name='NF_allloss.png'
# )

# # decoder performance
plots.plot_performance(
    'Loss',
    decoder.losses[1][1:],
    plots_dir=plots_directory,
    train=decoder.losses[0][1:],
    save_name='NF_d_performance'
)

total_counts=[np.sum(data1['inputs'][i,0,:]) for i in range(len(data1['inputs']))]
kT = list(data1['targets'][:,0,3])
nH = list(data1['targets'][:,0,0])
gamma = list(data1['targets'][:,0,1])
fsc = list(data1['targets'][:,0,2])
nH_errors = list(data1['targets'][:,1,0])
gamma_errors = list(data1['targets'][:,1,1])
fsc_errors = list(data1['targets'][:,1,2])
kT_errors = list(data1['targets'][:,1,3])
N_errors = list(data1['targets'][:,1,4])

if plot_synthetic:
    # plotting comparison between parameters
    comparison_plot(
        data1,
        log_colour_map=True,
        colour_map=total_counts,
        colour_map_label='total count rate',
        dir_name=plots_directory,
        n_points=50,
        num_dist_specs=250
    )

    coverage_plot(
        dataset=d_dataset, 
        loaders=d_loaders,
        network = net,
        dir_name=plots_directory
    )

    # all_param_samples = sample(data1,
    #                         num_specs=3,
    #                         num_samples=1,
    #                         spec_scroll=SPEC_SCROLL)

    # # single reconstructions using samples from all_param_samples
    # recon_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = data1,
    #     all_param_samples = all_param_samples,
    #     data_dir = '/Users/work/Projects/FSPNet/data/spectra.pickle', # for synthetic data,
    #     spec_scroll=SPEC_SCROLL
    #     )

    # # posterior predictive plots using 500 samples per reconstruction
    # post_pred_samples = post_pred_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = data1,
    # )

    # latent space corner plot
    # latent_corner_plot(
    #     dir_name = plots_directory+'distributions/',
    #     data=data1,
    #     )

    # # scatter plot across all parameters in dataset
    # param_pairs_plot(
    #     data=data,
    #     dir_name=plots_directory+'distributions/',
    # )

#     # rec_2d_plot(
#     #     decoder=decoder.net,
#     #     network=net,
#     #     dir_name = plots_directory+'reconstructions/',
#     #     data=data,
#     # )

# if plot_specific:
#      # # plotting comparison between parametrs
#     # comparison_plot(
#     #     data1,
#     #     dir_name=plots_directory,
#     #     specific_data=specific_data
#     # )

#     all_param_samples = sample(specific_data,
#                             num_specs=len(specific_data['ids']),
#                             num_samples=1,
#                             spec_scroll=SPEC_SCROLL)
    
    # comparison_plot(
    #     data1,
    #     log_colour_map=True,
    #     colour_map=data1['targets'][:,1:,],
    #     colour_map_label='Xspec 1 sigma error',
    #     dir_name=plots_directory,
    #     num_dist_specs=250,
    #     n_points=50,
    # )

    # comparison_plot(
    #     data1,
    #     log_colour_map=True,
    #     colour_map=kT,
    #     colour_map_label='kT (keV)',
    #     dir_name=plots_directory,
    #     # specific_data=specific_data,
    #     num_dist_specs=250,
    #     n_points=50,
    # )

    # comparison_plot(
    #     data1,
    #     log_colour_map=True,
    #     colour_map=total_counts,
    #     colour_map_label='Total count rate',
    #     dir_name=plots_directory,
    #     num_dist_specs=250,
    #     n_points=50,
    # )

    # # # coverage_plot(
    # # #     dataset=e_dataset, 
    # # #     loaders=e_loaders,
    # # #     network = net,
    # # #     dir_name=plots_directory
    # # # )

    # comparison_plot(
    #     data1,
    #     log_colour_map=True,
    #     colour_map=gamma,
    #     colour_map_label='Gamma',
    #     dir_name=plots_directory,
    #     num_dist_specs=250,
    #     n_points=50,
    # )

    # comparison_plot(
    #     data1,
    #     log_colour_map=True,
    #     colour_map=fsc,
    #     colour_map_label='$f_{sc}$',
    #     dir_name=plots_directory,
    #     num_dist_specs=250,
    #     n_points=50,
    # )

    # comparison_plot(
    #     data1,
    #     log_colour_map=True,
    #     colour_map=nH,
    #     colour_map_label='$N_H$',
    #     dir_name=plots_directory,
    #     num_dist_specs=250,
    #     n_points=50,
    # )

#     # single reconstructions using samples from all_param_samples
#     recon_plot(
#         decoder.net,
#         net,
#         dir_name = plots_directory+'reconstructions/',
#         specific_data = specific_data,
#         data = data1,
#         all_param_samples = all_param_samples
#         )

#     # # posterior predictive plots using 500 samples per reconstructio    
#     # post_pred_samples = post_pred_plot(
#     #     decoder.net,
#     #     net,
#     #     dir_name = plots_directory+'reconstructions/',
#     #     data = data1,
#     #     specific_data = specific_data
#     # )

    # post_pred_plot_xspec(
    #     dir_name = plots_directory+'reconstructions/',
    #     data = data1,
    #     specific_data = specific_data
    # )

#     # # latent space corner plot
#     latent_corner_plot(
#         dir_name = plots_directory+'distributions/',
#         specific_data = specific_data,
#         data=data1,
#         xspec_data=xspec_data,
#         in_param_samples=all_param_samples,
#         min_quant=0.0005,
#         max_quant=0.9995,
#         )

#     # scatter plot across all parameters in dataset
#     # param_pairs_plot(
#     #     data=data,
#     #     dir_name=plots_directory+'distributions/',
#     # )

# # elif plot_obsid:
# #     # comparison_plot(
# #     #     data1,
# #     #     dir_name=plots_directory,
# #     #     specific_data=obsid_data
# #     # )

# #     all_param_samples = sample(obsid_data,
# #                             num_specs=len(obsid_data['id']),
# #                             num_samples=1,
# #                             spec_scroll=SPEC_SCROLL)

#     # # # single reconstructions using samples from all_param_samples
#     # recon_plot(
#     #     decoder.net,
#     #     net,
#     #     dir_name = plots_directory+'reconstructions/',
#     #     data = obsid_data,
#     #     num_specs=len(obsid_data['id']),
#     #     all_param_samples = all_param_samples
#     #     )

#     # # # posterior predictive plots using 500 samples per reconstruction
#     # post_pred_samples = post_pred_plot(
#     #     decoder.net,
#     #     net,
#     #     num_specs=len(obsid_data['id']),
#     #     dir_name = plots_directory+'reconstructions/',
#     #     data = obsid_data,
#     # )

#     # # # # latent space corner plot
#     # latent_corner_plot(
#     #     dir_name = plots_directory+'distributions/',
#     #     data=obsid_data,
#     #     num_specs=len(obsid_data['id']),
#     #     in_param_samples= all_param_samples,
#     #     )

# else:
#     # # plotting comparison between parametrs
#     # comparison_plot(
#     #     data1,
#     #     dir_name=plots_directory,
#     # )

#     comparison_plot(
#         data1,
#         log_colour_map=True,
#         colour_map=kT,
#         colour_map_label='kT (keV)',
#         dir_name=plots_directory,
#         num_dist_specs=250,
#         n_points=50,
#     )

    # all_param_samples = sample(data1,
    #                         num_specs=3,
    #                         num_samples=1,
    #                         spec_scroll=SPEC_SCROLL)

    # # # single reconstructions using samples from all_param_samples
    # recon_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = data1,
    #     all_param_samples = all_param_samples
    #     )

    # # # posterior predictive plots using 500 samples per reconstruction
    # post_pred_samples = post_pred_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = data1,
    # )

    # # # latent space corner plot
    # latent_corner_plot(
    #     dir_name = plots_directory+'distributions/',
    #     data=data1,
    #     )

    # # scatter plot across all parameters in dataset
    # param_pairs_plot(
    #     data=data,
    #     dir_name=plots_directory+'distributions/',
    # )




'''---------- pyxspec tests ----------'''
# import xspec
# xspec.Xset.chatter = 0
# xspec.Xset.logChatter = 0

# data1['latent'] = np.median(data1['latent'], axis = 1) #data1['latent'][:,0,:] #
# data1['targets'] = data1['targets'][:,0,:]
# # for i in range(5):
# pyxspec_tests(data1)
