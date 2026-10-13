# from NFautoencoder import NFautoencoder, net_init, GaussianNLLLoss, MSELoss
# import pandas as pd
# from netloader_tests import TestConfig, gen_indexes, mod_network
# from networkx import config

import os
from typing import Any

import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import netloader.architectures as archs
from netloader import transforms
from netloader.network import Network
from netloader.data import loader_init
from netloader.utils import save_name, get_device
from fspnet.utils.data import SpectrumDataset
from fspnet.utils.utils import open_config
from torch.utils.data import DataLoader

from fspnetvar.utils.misc_utils import ROOT
from fspnetvar.NF_autoencoder import (
    NFautoencoder,
    NFautoencoderNetwork,
    NFdecoder,
    GaussianNLLLoss,
    MSELoss,
)

plt.style.use(["science", "grid", 'no-latex'])

def net_init(
        datasets: tuple[SpectrumDataset, SpectrumDataset],
        config: str | dict[str, Any] = './config.yaml',
) -> tuple[archs.BaseArchitecture, archs.BaseArchitecture]:
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
    tuple[BaseArchitecture, BaseArchitecture]
        Constructed decoder and autoencoder
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', os.path.join(ROOT, config))

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
    device = 'cpu'

    print('device:', device)

    if d_load_num:
        decoder = archs.load_net(d_load_num, states_dir, decoder_name, weights_only=False)
        decoder.description = description
        decoder.save_path = save_name(d_save_num, states_dir, decoder_name)
        transform = decoder.transforms['targets']
        param_transform = decoder.transforms['inputs']
    else:

        transform = transforms.MultiTransform(
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=-1),
            transforms.Log(),
        )
        transform.transforms.append(transforms.Normalise(
            data=transform(datasets[1].spectra),
            mean=False
        ))

        param_transform = transforms.MultiTransform(
            transforms.NumpyTensor(),
            transforms.MinClamp(dim=0, idxs=log_params),
            transforms.Log(idxs=log_params),
        )
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
            verbose='epoch',
            transform=transform,
            scheduler_kwargs={'min_lr': 1e-8}
        )

        decoder.transforms['inputs'] = param_transform
        decoder.loss_func =  GaussianNLLLoss() # gaussian_loss #  #  #changes loss to gaussian loss - can look into other typs of loss

    if e_load_num:
        net = archs.load_net(e_load_num, states_dir, encoder_name, weights_only=False)
        net.description = description
        net.save_path = save_name(e_save_num, states_dir, encoder_name)
    else:
        net = Network(
            encoder_name,
            networks_dir,
            list(datasets[0][0][2].shape),
            list(datasets[0][0][1].shape),
        )

        # to train autoencoder_NF
        net = NFautoencoder(
            save_num=e_save_num,
            states_dir=states_dir,
            net=NFautoencoderNetwork(encoder_name, net, decoder.net),
            overwrite=True,
            learning_rate=learning_rate,
            description=description,
            verbose='epoch',
            transform=transform,
            latent_transform=param_transform,
            scheduler_kwargs={'min_lr': 1e-6}
        )

        # to train encoder only - or could just set reconstruct loss = 0
        # net = archs.NormFlowEncoder(
        #     save_num=e_save_num,
        #     states_dir=states_dir,
        #     net=net,
        #     overwrite=True,
        #     learning_rate=learning_rate,
        #     description=description,
        #     verbose='epoch',
        #     transform=param_transform,
        #     in_transform=transform,
        #     scheduler_kwargs={'min_lr': 1e-6}
        # )

        #Loss function settings for autoencoder
        net.reconstruct_func = GaussianNLLLoss() #  gaussian_loss   #
        net.latent_func = MSELoss()   # mse_loss    #

        net.set_loss_weights(
            bound=0,
            kl=0,
            latent=1,
            flow=1,
            reconstruct=1,
        )

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
        archs.BaseArchitecture,
        archs.BaseArchitecture]:
    """
    Initialises the network and dataloaders

    Parameters
    ----------
    config : dictionary | string, default = './config.yaml'
        Configuration dictionary or path to the configuration dictionary

    Returns
    -------
    tuple[tuple[Dataloader, Dataloader], tuple[Dataloader, Dataloader], BaseArchitecture, BaseArchitecture]
        Train & validation dataloaders for decoder and autoencoder, decoder, and autoencoder
    """
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', os.path.join(ROOT, config))

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

    # Convert old index format to new
    if net.idxs is not None and len(net.idxs) == len(e_dataset):
        net.idxs = net.idxs[:-max(int(len(e_dataset) * val_frac), 1)]

    if decoder.idxs is not None and len(decoder.idxs) == len(d_dataset):
        decoder.idxs = decoder.idxs[:-max(int(len(d_dataset) * val_frac), 1)]

    # Initialise datasets
    e_loaders = loader_init(
        e_dataset,
        batch_size=batch_size,
        ratios=(1 - val_frac, val_frac) if net.idxs is None or len(net.idxs) == 0 else (1,),
        idxs=net.idxs if net.idxs is not None and len(net.idxs) else None,
    )
    d_loaders = loader_init(
        d_dataset,
        batch_size=batch_size,
        ratios=(1 - val_frac, val_frac) if decoder.idxs is None or len(decoder.idxs) == 0 else (1,),
        idxs=decoder.idxs if decoder.idxs is not None and len(decoder.idxs) else None,
    )
    net.idxs = np.array(e_loaders[0].dataset.indices)
    decoder.idxs = np.array(d_loaders[0].dataset.indices)
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
        _, config = open_config('spectrum-fit', os.path.join(ROOT, config))

    # train settings - consistent throughout function
    n_epochs = config['training']['epochs']
    learning_rate = config['training']['learning-rate']

    # root state name of encoder
    if cycle_num:
        root_encoder_name = str(config['training']['encoder-save']) + '_test_' + str(cycle_num)
    else:
        root_encoder_name = str(config['training']['encoder-save'])

    # load and save names for synthetic training
    # config['training']['encoder-load'] = 0
    # config['training']['decoder-load'] = 0
    config['training']['encoder-save'] = root_encoder_name + '_synth'
    config['training']['decoder-save'] = str(1) + str(cycle_num)
    #initialise data loaders and networks for synthetic training
    e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init(config)

    '''---------- DECODER TRAINING ----------'''
    # train decoder on synthetic
    print('training decoder...')
    # decoder.training(n_epochs, d_loaders)
    print('decoder trained!')

    '''---------- ENCODER TRAINING SYNTHETIC ----------'''
    # train autoencoder on synthetic
    print('training encoder on synthetic...')
    net.training(n_epochs, d_loaders)
    print('encoder trained on synthetic!')

    # uncomment this to use unsupervised training
    # net.latent_loss = 0   # for unsupervised
    # net.flowlossweight = 0

    '''---------- ENCODER TRANSFER LEARNING ----------'''
    # change load and save names for transfer learning
    net.set_save_path(net.get_save_path().replace('.pth', '_real.pth'), overwrite=True)

    # resetting autoencoder optimiser
    if net.get_epochs() == n_epochs:
        net.optimiser = net.init_optimiser(net.get_param_groups(5e-6), lr=5e-6)
        net.scheduler = net.init_scheduler(net.optimiser, factor=0.5, min_lr=1e-8)

    # training autoencoder on real
    print('transfer learning encoder to real...')
    net.training(n_epochs*2, e_loaders)
    print('transfer learning complete!')

    '''---------- ENCODER TRAINING REAL ONLY ----------'''
    config['training']['encoder-load'] = 0
    config['training']['encoder-save'] = root_encoder_name+'_real'

    #initialise new networks
    for dataset in (e_dataset, d_dataset):
        # for with uncertainties
        dataset.spectra, dataset.uncertainty = net.transforms['inputs'](
            dataset.spectra,
            back=True,
            uncertainty=dataset.uncertainty,
        )
        dataset.params, dataset.param_uncertainty = net.transforms['targets'](
            dataset.params,
            back=True,
            uncertainty=dataset. param_uncertainty,
        )

    real_net = net_init((e_dataset, d_dataset), config)[1]

    # keep using old decoder and ensure gradient is still frozen
    real_net.net.layers[1] = decoder.net

    # training auteoncoder on real
    print('training encoder only on real...')
    real_net.training(n_epochs, e_loaders)
    print('training encoder only on real learning complete!')


def main(train=False,
         num_cycles=1):

    '''---------- TRAINING ----------'''
    if num_cycles>1:
        for cycle_num in range(1,num_cycles+1):
            print(f'Cycle {cycle_num}/{num_cycles} initialized.')
            NF_train(cycle_num)
            print(f'Cycle {cycle_num}/{num_cycles} completed.')
    else:
        NF_train()


if __name__ == '__main__':
    # settings
    main(num_cycles=1)
