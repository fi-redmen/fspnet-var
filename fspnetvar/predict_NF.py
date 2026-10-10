import os
import pickle

import numpy as np
from fspnet.utils.utils import open_config
from torch.utils.data import DataLoader, Subset

from fspnetvar.train_NF import init
from fspnetvar.utils.misc_utils import ROOT


def NF_predict(load_name, dec_load_name, names, object_names,
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
    predict_for_synthetic : bool
        Whether to make predictions for synthetic data or not.
    Returns
    -------
    tuple[dict, dict]
        Tuple containing all data and specific data dictionaries.
    """

    if isinstance(config, str):
        _, config = open_config('spectrum-fit', os.path.join(ROOT, config))

    savedir = ROOT+'/predictions/'+config['output']['network-states-directory'].split('/')[-2 if config['output']['network-states-directory'].endswith('/') else -1]

    # loads network for predictions
    config['training']['encoder-load'] = load_name
    config['training']['decoder-load'] = dec_load_name
    e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init(config)
    # only predict on synthetic for the synthetically trained model
    if 'real' in load_name:
        pred_dataset = e_dataset
        pred_loaders = e_loaders
    else:
        pred_dataset = d_dataset
        pred_loaders = d_loaders

    # save and clear transforms
    net_transforms = net.transforms.copy()  # save transforms
    for key in net.transforms:      # clear transforms
        net.transforms[key] = None

    net._verbose = 'full'   # allow progress bar to print while network is predicting

    # makes transfer predictions (transformed)
    val_data = net.predict(pred_loaders[1], num_samples=5000, inputs=True)
    data_idxs = np.arange(len(e_dataset))
    specific_subset = Subset(e_dataset, data_idxs[np.isin(e_dataset.names, names)].tolist())
    specific_loader = DataLoader(specific_subset, batch_size=64, shuffle=False)
    specific_data = net.predict(specific_loader, num_samples=5000, inputs=True)

    # untransforms transefer predictions
    for (pred_data, dataset) in zip([specific_data, val_data], [e_dataset, pred_dataset]):
        for key, transform in net_transforms.items():
            if transform is None:
                continue
            if key == 'inputs':
                pred_data[key] = np.stack(transform(pred_data[key][:,0], back=True,
                                                uncertainty=pred_data[key][:,1]), axis=1)
            elif key == 'targets':
                idxs = []
                for pred_id in pred_data['ids']:
                    for j, name in enumerate(dataset.names):
                        if pred_id==name:
                            idxs.append(j)
                pred_data[key] = np.stack(transform(pred_data[key], back=True,
                                            uncertainty=dataset.param_uncertainty[idxs].numpy()), axis=1)
            else:
                pred_data[key] = transform(pred_data[key], back=True)

    # reset transforms
    net.transforms = net_transforms

    # process and save specific data if we are predicting with real data
    if 'latent' not in specific_data and 'distributions' in specific_data:
        specific_data['latent']=specific_data['distributions']
        specific_data['preds']=specific_data['inputs']

    # add object names to specific data
    name_to_object = dict(zip(names, object_names))
    new_object_names = [name_to_object[name] for name in specific_data['ids']]
    specific_data['object']=new_object_names


    if not os.path.exists(os.path.join(savedir)):
        os.makedirs(os.path.join(savedir))

    with open(os.path.join(savedir,'specific_'+load_name+'.pickle'), 'wb') as file:
        pickle.dump(specific_data, file)

    # process and save validation data
    if 'latent' not in val_data and 'distributions' in val_data:
        val_data['latent']=val_data['distributions']
        val_data['preds']=val_data['inputs']

    with open(os.path.join(savedir,'val_'+load_name+'.pickle'), 'wb') as file:
        pickle.dump(val_data, file)

    return val_data, specific_data


def main():

    numcycles=1

    for i in [1]: #range(numcycles*3):
        if i < numcycles:
            load_name = '1_test_' + str(i+1)+'_synth_real'
            dec_load_name = load_name.split('_')[0]+str(i+1)
        elif i < numcycles*2:
            load_name ='1_test_' + str(i-numcycles+1)+'_real'
            dec_load_name = load_name.split('_')[0]+str(i-numcycles+1)
        else:
            load_name =  '1_test_' + str(i-2*numcycles+1)+'_synth'
            dec_load_name = load_name.split('_')[0]+str(i-2*numcycles+1)

        print('predicting for load_name: ', load_name, '...')
        NF_predict(
        load_name,
        dec_load_name,
        names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp','js_ni0103010102_0mpu7_goddard_GTI0.jsgrp','js_ni1014010102_0mpu7_goddard_GTI30.jsgrp','js_ni1050360115_0mpu7_goddard_GTI9.jsgrp','js_ni1100320119_0mpu7_goddard_GTI26.jsgrp','js_ni1200120203_0mpu7_goddard_GTI0.jsgrp','js_ni1200120203_0mpu7_goddard_GTI10.jsgrp','js_ni1200120203_0mpu7_goddard_GTI11.jsgrp','js_ni1200120203_0mpu7_goddard_GTI13.jsgrp','js_ni1200120203_0mpu7_goddard_GTI1.jsgrp','js_ni1200120203_0mpu7_goddard_GTI3.jsgrp','js_ni1200120203_0mpu7_goddard_GTI4.jsgrp','js_ni1200120203_0mpu7_goddard_GTI5.jsgrp','js_ni1200120203_0mpu7_goddard_GTI6.jsgrp','js_ni1200120203_0mpu7_goddard_GTI7.jsgrp','js_ni1200120203_0mpu7_goddard_GTI8.jsgrp','js_ni1200120203_0mpu7_goddard_GTI9.jsgrp'],
        object_names=['Cyg X-1 (2017)','GRS 1915+105','LMC X-3','MAXI J1535-571','Cyg X-1 (2018)','MAXI J1820 0','MAXI J1820 10','MAXI J1820 11','MAXI J1820 13','MAXI J1820 1','MAXI J1820 3','MAXI J1820 4','MAXI J1820 5','MAXI J1820 6','MAXI J1820 7','MAXI J1820 8','MAXI J1820 9'],
        )

        # val_data, specific_data = NF_predict(
        # load_name,
        # dec_load_name,
        # names = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp','js_ni0103010102_0mpu7_goddard_GTI0.jsgrp','js_ni1014010102_0mpu7_goddard_GTI30.jsgrp','js_ni1050360115_0mpu7_goddard_GTI9.jsgrp','js_ni1100320119_0mpu7_goddard_GTI26.jsgrp','js_ni1200120203_0mpu7_goddard_GTI0.jsgrp','js_ni1200120203_0mpu7_goddard_GTI10.jsgrp','js_ni1200120203_0mpu7_goddard_GTI11.jsgrp','js_ni1200120203_0mpu7_goddard_GTI13.jsgrp','js_ni1200120203_0mpu7_goddard_GTI1.jsgrp','js_ni1200120203_0mpu7_goddard_GTI3.jsgrp','js_ni1200120203_0mpu7_goddard_GTI4.jsgrp','js_ni1200120203_0mpu7_goddard_GTI5.jsgrp','js_ni1200120203_0mpu7_goddard_GTI6.jsgrp','js_ni1200120203_0mpu7_goddard_GTI7.jsgrp','js_ni1200120203_0mpu7_goddard_GTI8.jsgrp','js_ni1200120203_0mpu7_goddard_GTI9.jsgrp'],
        # object_names=['Cyg X-1 (2017)','GRS 1915+105','LMC X-3','MAXI J1535-571','Cyg X-1 (2018)','MAXI J1820 0','MAXI J1820 10','MAXI J1820 11','MAXI J1820 13','MAXI J1820 1','MAXI J1820 3','MAXI J1820 4','MAXI J1820 5','MAXI J1820 6','MAXI J1820 7','MAXI J1820 8','MAXI J1820 9'],
        # )

        print('prediction ', load_name, 'complete!')

if __name__ == '__main__':
    main()
