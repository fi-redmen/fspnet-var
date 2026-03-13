# from NFautoencoder import NFautoencoder, net_init, GaussianNLLLoss, MSELoss
# import pandas as pd
# from netloader_tests import TestConfig, gen_indexes, mod_network
# from networkx import config

import xspec
import os
import pickle
import random
import sciplots
import re
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Subset
from typing import Any
from astropy.io import fits

import netloader.transforms as transforms
import netloader.networks as nets
from netloader.network import Network
from netloader.utils.utils import save_name, get_device

from fspnet.utils import plots
from fspnet.utils.utils import open_config
from fspnet.utils.data import SpectrumDataset, loader_init
from fspnet.spectrum_fit import pyxspec_tests

import utils.plots_var as plots_var
from utils.misc_utils import sample, get_energy_widths

from NF_autoencoder import NFautoencoder, NFautoencoderNetwork, NFdecoder, GaussianNLLLoss, MSELoss

plt.style.use(["science", "grid", 'no-latex'])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        decoder = NFdecoder(
            d_save_num,
            states_dir,
            decoder,
            overwrite=True,
            learning_rate=learning_rate,
            description=description,
            verbose='full',
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

        # to train NF_autoencoder
        net = NFautoencoder(
            save_num=e_save_num,
            states_dir=states_dir,
            net=NFautoencoderNetwork(net, decoder.net, name=encoder_name),
            learning_rate=learning_rate,
            description=description,
            verbose='full',
            transform=transform,
            latent_transform=param_transform,
        )

        #Loss function settings for autoencoder
        net.reconstruct_func = GaussianNLLLoss() #  gaussian_loss   #
        net.latent_func = MSELoss()   # mse_loss    #  
        # net.latent_loss = 0 #3.0e-1
        # net.flowlossweight = 1 #1e-1 #3.0e-1
        # net.reconstruct_loss = 1 #1e-3 #4.0e-1
        # net.kl_loss = 0 #
        # net.bound_loss = 0 # 3e-1

        net.set_loss_weights(bound=0,
                             kl=0,
                             latent=0,
                             flow=0,
                             reconstruct=1)

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

def NF_train(cycle_num: int | None = 0,
             config: str = './config.yaml'):
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

    if isinstance(config, str):
        _, config = open_config('spectrum-fit',config)

    # train settings - consistent throughout function
    n_epochs = config['training']['epochs']
    learning_rate = config['training']['learning-rate']
    # root state name of encoder
    if cycle_num: root_encoder_name = str(config['training']['encoder-save']) + '_test_' + str(cycle_num)
    else: root_encoder_name = str(config['training']['encoder-save'])

    # load and save names for synthetic training
    config['training']['encoder-load'] = 0
    config['training']['decoder-load'] = 0
    config['training']['encoder-save'] = root_encoder_name + '_synth'
    config['training']['decoder-save'] = str(1) + '_test_' + str(cycle_num)
    #initialise data loaders and networks for synthetic training
    _, _, e_loaders, d_loaders, decoder, net = init(config)
    
    '''---------- DECODER TRAINING ----------'''
    #setting up decoder optimiser
    decoder.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
    decoder.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, min_lr=1e-8,)
    # train decoder on synthetic
    print('training decoder...')
    # decoder.training(n_epochs, d_loaders) 
    print('decoder trained!')

    #fix decoder's weights so they dont change while training the encoder
    net.net.net[1] = decoder.net
    net.net.net[1].requires_grad_(False)

    '''---------- ENCODER TRAINING SYNTHETIC ----------'''
    #setting up autoencoder optimiser
    if net._epoch==0:
        net.optimiser = optim.AdamW(net.net.parameters(), lr=learning_rate)
        net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(net.optimiser, min_lr=1e-6,)
    # train autoencoder on synthetic
    print('training encoder on synthetic...')
    net.training(n_epochs, d_loaders)
    print('encoder trained on synthetic!')

    # to train only first few layers of encoder - check which layers are indexed
    # net.net.net[0].net[2:].requires_grad_(False)
    # uncomment this to use unsupervised training
    # net.latent_loss = 0   # for unsupervised
    # net.flowlossweight = 0

    '''---------- ENCODER TRANSFER LEARNING ----------'''
    # change load and save names for transfer learning
    config['training']['encoder-load'] = root_encoder_name+'_synth'
    config['training']['encoder-save'] = root_encoder_name+'_synth_real'
    #re-initialise networks
    _, _, _, _, _, trans_net = init(config)
    # keep using old decoder and ensure gradient is still frozen
    trans_net.net.net[1] = decoder.net
    trans_net.net.net[1].requires_grad_(False)
    # resetting autoencoder optimiser
    if trans_net.get_epochs() == n_epochs:
        trans_net.optimiser = optim.AdamW(trans_net.net.parameters(), lr=5e-6)
        trans_net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(trans_net.optimiser, factor=0.5, min_lr=1e-8)
    # training auteoncoder on real
    print('transfer learning encoder to real...')
    trans_net.training(n_epochs*2, e_loaders)
    print('transfer learning complete!')

    '''---------- ENCODER TRAINING REAL ONLY ----------'''
    config['training']['encoder-load'] = 0
    config['training']['encoder-save'] = root_encoder_name+'_real'
    #initialise new networks
    _, _, _, _, _, real_net = init(config)
    # keep using old decoder and ensure gradient is still frozen
    real_net.net.net[1] = decoder.net
    real_net.net.net[1].requires_grad_(False)
    # resetting autoencoder optimiser
    if real_net._epoch==0:
        real_net.optimiser = optim.AdamW(real_net.net.parameters(), lr=learning_rate)
        real_net.scheduler = optim.lr_scheduler.ReduceLROnPlateau(real_net.optimiser, min_lr=1e-8)
    # training auteoncoder on real
    print('training encoder only on real...')
    real_net.training(n_epochs, e_loaders)
    print('training encoder only on real learning complete!')

def NF_predict(net, 
               e_dataset, d_dataset, 
               e_loaders, d_loaders, 
               names, object_names, pred_savename,
               config: str = './config.yaml'):
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

    if isinstance(config, str):
        _, config = open_config('spectrum-fit',config)
    
    root_encoder_name = str(1) + '_test_' + str(cycle_num + 1)
        
    net_transforms = net.transforms.copy()  # save transforms
    for key in net.transforms:      # clear transforms
        net.transforms[key] = None

    net._verbose = 'full'   # allow progress bar to print while network is predicting

    '''---------- SYNTHETIC PREDICTIONS ----------'''
    # loads synthetic network to make prediction
    if 'real' in'real' in config['training']['encoder-save']:
        config['training']['encoder-load'] = ''.join(config['training']['ecnoder-save'].split('_')[:-1])
    _, _, _, _, _, net = init(config)
    # makes synthetic predictions (transformed)
    val_data_synth = net.predict(d_loaders[1], num_samples=5000, inputs=True)

    '''---------- TRANSFER PREDICTIONS ----------'''
    # sets up for real predictions
    pred_dataset = e_dataset
    if 'real' not in 'real' in config['training']['encoder-save']:
        config['training']['encoder-load'] = config['training']['encoder-save']+'_real'
    _, _, _, _, _, net = init(config)
    # makes transfer predictions (transformed)
    val_data_real = net.predict(e_loaders[1], num_samples=5000, inputs=True)
    data_idxs = np.arange(len(e_dataset))
    specific_subset = Subset(e_dataset, data_idxs[np.isin(e_dataset.names, names)].tolist())
    specific_loader = DataLoader(specific_subset, batch_size=64, shuffle=False)
    specific_data = net.predict(specific_loader, num_samples=5000, inputs=True)
    
    # untransforms transfer predictions
    for pred_data in [specific_data, val_data_real]:
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
        specific_data['preds']=specific_data['inputs']
    if 'latent' not in specific_data and 'distributions' in val_data_real:
        val_data_real['latent']=val_data_real['distributions']
        val_data_real['preds']=val_data_real['inputs']
    
    # add object names to specific data
    name_to_object = dict(zip(names, object_names))
    new_object_names = [name_to_object[name] for name in specific_data['ids']]
    specific_data['object']=new_object_names
    
    # reset transforms 
    net.transforms = net_transforms

    with open(os.path.join(ROOT,'predictions/val_'+pred_savename+'_synth.pickle'), 'wb') as file:
        pickle.dump(val_data_synth, file)
    with open(os.path.join(ROOT,'predictions/specific_'+pred_savename+'.pickle'), 'wb') as file:
        pickle.dump(specific_data, file)
    with open(os.path.join(ROOT,'predictions/val_'+pred_savename+'.pickle'), 'wb') as file:
        pickle.dump(val_data_real, file)
    
    return val_data_real, specific_data, val_data_synth

def NF_load_preds(pred_savename):
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
    with open(os.path.join(ROOT,'predictions/specific_'+pred_savename+'.pickle'), 'rb') as file:
        specific_data = pickle.load(file)
    with open(os.path.join(ROOT,'predictions/val_'+pred_savename+'.pickle'), 'rb') as file:
        val_data = pickle.load(file)
    with open(os.path.join(ROOT,'predictions/val_'+pred_savename+'_synth.pickle'), 'rb') as file:
        val_data_synth = pickle.load(file)

    if 'latent' not in specific_data and 'distributions' in specific_data:
        specific_data['latent']=specific_data['distributions']
        specific_data['preds']=specific_data['inputs']
        val_data['latent']=val_data['distributions']
        val_data['preds']=val_data['inputs']
        val_data_synth['latent']=val_data_synth['distributions']
        val_data_synth['preds']=val_data_synth['inputs']

    return val_data, specific_data, val_data_synth

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
    with open(os.path.join(ROOT,'predictions/xspec_preds.pickle'), 'rb') as file:
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

def main(train=False, predict=False, specific=True,
         num_cycles=1):

    '''---------- TRAINING ----------'''
    if train:
        if num_cycles>1:
            for cycle_num in range(1,num_cycles+1):
                print(f'Cycle {cycle_num}/{num_cycles} initialized.')
                NF_train(cycle_num)
                print(f'Cycle {cycle_num}/{num_cycles} completed.')
        else: 
            NF_train()

    else: 
        e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init()

    '''---------- PREDICTING ----------'''
    # # save name of predictions = encoder_name ecoder_name
    # pred_savename = os.path.basename(net.save_path)[:-4]+' '+os.path.basename(decoder.save_path)[:-4] #'Encoder NF0_2' #
    # plots_directory = os.path.join(ROOT,'plots',pred_savename,'_preds/')
    # os.makedirs(plots_directory, exist_ok=True)
    # os.makedirs(plots_directory+'reconstructions/', exist_ok=True)
    # os.makedirs(plots_directory+'distributions/', exist_ok=True)
    # os.makedirs(plots_directory+'comparisons/', exist_ok=True)

    # # for getting specific spectra
    # names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp','js_ni0103010102_0mpu7_goddard_GTI0.jsgrp','js_ni1014010102_0mpu7_goddard_GTI30.jsgrp','js_ni1050360115_0mpu7_goddard_GTI9.jsgrp','js_ni1100320119_0mpu7_goddard_GTI26.jsgrp','js_ni1200120203_0mpu7_goddard_GTI0.jsgrp','js_ni1200120203_0mpu7_goddard_GTI10.jsgrp','js_ni1200120203_0mpu7_goddard_GTI11.jsgrp','js_ni1200120203_0mpu7_goddard_GTI13.jsgrp','js_ni1200120203_0mpu7_goddard_GTI1.jsgrp','js_ni1200120203_0mpu7_goddard_GTI3.jsgrp','js_ni1200120203_0mpu7_goddard_GTI4.jsgrp','js_ni1200120203_0mpu7_goddard_GTI5.jsgrp','js_ni1200120203_0mpu7_goddard_GTI6.jsgrp','js_ni1200120203_0mpu7_goddard_GTI7.jsgrp','js_ni1200120203_0mpu7_goddard_GTI8.jsgrp','js_ni1200120203_0mpu7_goddard_GTI9.jsgrp']
    # object_names=['Cyg X-1 (2017)','GRS 1915+105','LMC X-3','MAXI J1535-571','Cyg X-1 (2018)','MAXI J1820 0','MAXI J1820 10','MAXI J1820 11','MAXI J1820 13','MAXI J1820 1','MAXI J1820 3','MAXI J1820 4','MAXI J1820 5','MAXI J1820 6','MAXI J1820 7','MAXI J1820 8','MAXI J1820 9']

    # if predict:
    #     val_data, specific_data, val_data_synth = NF_predict(net, 
    #                                          e_dataset, d_dataset, 
    #                                          e_loaders, d_loaders, 
    #                                          names, object_names, pred_savename)
    # elif os.path.exists(pred_savename):
    #     val_data, specific_data, val_data_synth = NF_load_preds(pred_savename)
    # else:
    #     print('Predictions have not been made yet!')

    # overwrite specific_data as none if not plotting specific spectra
    # specific_data = specific_data if specific else None
    # load xspec predictions in same order as specific data
    # xspec_data = load_xspec_preds(specific_data) 

    '''---------- PLOTTING ----------'''

    # plot decoder performance
    # plots.plot_performance(
    #     'Loss',
    #     decoder.losses[1][1:],
    #     plots_dir= config['output']['plots-directory'],
    #     train=decoder.losses[0][1:],
    #     save_name='dec_performance')

    # plot autoencoder performance - remember to change part of net_init to correspond to encoder only vs autoencoder
    # plots.plot_performance(
    #     'Loss',
    #     net.losses[1][1:],
    #     plots_dir=config['output']['plots-directory'],
    #     train=net.losses[0][1:],
    #     save_name='net_perfomance.png')

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
    # plots_var.performance_plot(
    #     'Loss',
    #     {key: value.tolist() for key, value in total_loss.items()},
    #     plots_dir=plots_directory,
    #     save_name='NF_allloss.png')

    # # plotting comparison between parameters
    # # to color data by certain parameters
    # kT = val_data['targets'][:,0,3]
    # Norm = val_data['targets'][:,0,4]
    # nH = list(val_data['targets'][:,0,0])
    # gamma = list(val_data['targets'][:,0,1])
    # fsc = list(val_data['targets'][:,0,2])
    # det_nums=[]
    # for spectrum in val_data['ids']:
    #     with fits.open(os.path.join(ROOT,'data','spectra',spectrum)) as file:
    #         spectrum_info = file[1].header
    #     det_nums.append(int(re.search(r'_d(\d+)', spectrum_info['RESPFILE']).group(1)))
    # det_nums = np.array(det_nums)[:,np.newaxis]
    # widths = get_energy_widths()[np.newaxis]
    # total_counts=np.sum(val_data['inputs'][:,0,:]*det_nums*widths, axis=-1)

    # plots_var.comparison_plot(
    #     val_data,
    #     specific_data=specific_data,
    #     log_colour_map=True,
    #     colour_map=total_counts,
    #     colour_map_label='Total Count Rate',
    #     dir_name=os.path.join(plots_directory, 'comparisons/'),
    #     n_points=50,
    #     num_dist_specs=250)

    # plots_var.coverage_plot(
    #     dataset=d_dataset, 
    #     loaders=d_loaders,
    #     network = net,
    #     dir_name=plots_directory,
    #     pred_savename=pred_savename)

    # all_param_samples = sample(specific_data if specific else val_data,
    #                         num_specs=len(specific_data['ids']),
    #                         num_samples=1)

    # # # # single reconstructions using samples from all_param_samples
    # plots_var.recon_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = val_data,
    #     specific_data = specific_data,
    #     all_param_samples = all_param_samples,
    #     data_dir = os.path.join(ROOT,'data','spectra.pickle') if predict_for_synthetic else os.path.join(ROOT,'data','spectra/'))

    # # # # # # posterior predictive plots using 100 samples per reconstruction
    # post_pred_samples = plots_var.post_pred_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = val_data,
    #     specific_data = specific_data,
    #     data_dir = os.path.join(ROOT,'data','spectra.pickle') if predict_for_synthetic else os.path.join(ROOT,'data','spectra/'))

    # # # # # # posterior predictive plots only using xspec to make reconstructions - not decoder
    # plots_var.post_pred_plot_xspec(
    #     dir_name = plots_directory+'reconstructions/',
    #     data = val_data,
    #     specific_data = specific_data,
    #     post_pred_samples=post_pred_samples,
    #     data_dir = os.path.join(ROOT,'data','spectra.pickle') if predict_for_synthetic else os.path.join(ROOT,'data','spectra/'))

    # # # # # latent space corner plot
    # plots_var.latent_corner_plot(
    #     dir_name = plots_directory+'distributions/',
    #     data=val_data,
    #     xspec_data=xspec_data,
    #     specific_data=specific_data,
    #     in_param_samples=all_param_samples)

    # scatter plot across all target parameters in dataset
    # plots_var.param_pairs_plot(
    #     data=val_data,
    #     dir_name=plots_directory)

    # # 2D reconstruction plots
    # plots_var.rec_2d_plot(
    #     plot_dir = plots_directory+'reconstructions/',
    #     data=val_data)

    # # # Gamma Vs scattered fraction coloured by state
    # plots_var.labels_plot(data=val_data,
    #                       plot_dir=plots_directory,
    #                       save_name='labels_plot.png')
    
    # # residuals vs total count rate
    # plots_var.resid_params_plot(
    #     data=val_data.copy(),
    #     x_param=total_counts,
    #     x_param_name='Total Count Rate',
    #     plot_dir=plots_directory+'comparisons/',
    #     save_name='Param_resids_tcr.png')

    # residuals vs normalisation/disk temperature
    # plots_var.resid_params_plot(
    #     data=val_data.copy(),
    #     x_param=Norm/fsc,
    #     x_param_name='N/fsc',
    #     plot_dir=plots_directory+'comparisons/',
    #     save_name = 'Param_Resids_NkT.png')

    '''---------- pyxspec tests ----------'''
    # import xspec
    # xspec.Xset.chatter = 0
    # xspec.Xset.logChatter = 0

    # val_data['latent'] = val_data['latent'][:,0,:] # #np.median(val_data['latent'], axis = 1)
    # val_data['targets'] = val_data['targets'][:,0,:]
    # pyxspec_tests(val_data)

    # making predictions for timing purposes
    # for i in range(5):
    #     print('1000 samples:')
    #     net.predict(e_loaders[1], num_samples=1000, inputs=True)
    #     print('1 sample:')
    #     net.predict(e_loaders[1], num_samples=1, inputs=True)

if __name__ == '__main__':
    # settings
    train = True
    predict = False

    main(train=train,
         predict=predict,
         num_cycles=1)