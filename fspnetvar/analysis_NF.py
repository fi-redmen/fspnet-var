from fspnet.utils import plots
from fspnet.utils.utils import open_config
from utils.analysis_utils import pyxspec_tests

from train_NF import init

import pickle
import random
from matplotlib import pyplot as plt
import os
import numpy as np
import xspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.style.use(["science", "grid", 'no-latex'])

def NF_load_preds(pred_savename, mode):
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
    with open(os.path.join(ROOT,'predictions/'+mode+'/specific_'+pred_savename+'.pickle'), 'rb') as file:
        specific_data = pickle.load(file)
    with open(os.path.join(ROOT,'predictions/'+mode+'/val_'+pred_savename+'.pickle'), 'rb') as file:
        val_data = pickle.load(file)
    if 'latent' not in specific_data and 'distributions' in specific_data:
        specific_data['latent']=specific_data['distributions']
        specific_data['preds']=specific_data['inputs']
        val_data['latent']=val_data['distributions']
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
    with open(os.path.join(ROOT,'predictions/xspec_preds1.pickle'), 'rb') as file:
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

def analysis_NF(config: str = './config.yaml',
                mode = 'unsupervised',
                load_name = '1_test_' + str(1)+'_synth',
                dec_load_name = '11'):

    '''----------- LOAD NETWORK & DATA ---------'''
    if isinstance(config, str):
        _, config = open_config('spectrum-fit', config)

    # loads the networks back in to access losses and for reconstructions
    config['training']['encoder-load'] = load_name
    config['training']['decoder-load'] = dec_load_name
    e_dataset, d_dataset, e_loaders, d_loaders, decoder, net = init(config)

    '''---------- PLOTTING PERFORMANCE ----------'''
    plots_directory = ROOT+'/plots/'+mode+'/'+load_name+'/'
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)

    # plot decoder performance
    plots.plot_performance(
        'Loss',
        decoder.losses[1][1:],
        plots_dir= plots_directory,
        train=decoder.losses[0][1:],
        save_name='dec_performance')

    # make separete losses dictionary
    separate_losses_train = {}
    if 'reconstruct' in net.losses[0][0].keys():
        separate_losses_train['reconstruct'] = [net.losses[0][i]['reconstruct'] for i in range(len(net.losses[0]))]
    if 'flow' in net.losses[0][0].keys():
        separate_losses_train['flow'] = [net.losses[0][i]['flow'] for i in range(len(net.losses[0]))]
    separate_losses_train['total'] = [net.losses[0][i]['total'] for i in range(len(net.losses[0]))]

    separate_losses_val = {}
    if 'reconstruct' in net.losses[1][0].keys():
        separate_losses_val['reconstruct'] = [net.losses[1][i]['reconstruct'] for i in range(len(net.losses[1]))]
    if 'flow' in net.losses[1][0].keys():
        separate_losses_val['flow'] = [net.losses[1][i]['flow'] for i in range(len(net.losses[1]))]
    separate_losses_val['total'] = [net.losses[1][i]['total'] for i in range(len(net.losses[1]))]   

    # plot autoencoder performance - remember to change part of net_init to correspond to encoder only vs autoencoder
    # plots.plot_performance(
    #     'Loss',
    #     net.losses[1][1:],
    #     plots_dir=config['output']['plots-directory'],
    #     train=net.losses[0][1:],
    #     save_name='net_perfomance.png')

    # separate losses performance plot
    # plots_var.performance_plot(
    #     'Loss',
    #     {key: value for key, value in separate_losses_train.items()},
    #     {key: value for key, value in separate_losses_val.items()},
    #     plots_dir=plots_directory,
    #     save_name='net_performance_sep.png')
    
    # '''---------- LOADING DATA ----------'''

    val_data, specific_data = NF_load_preds(load_name, mode)

    if 'real' in load_name:
        pred_dataset = e_dataset
        pred_loaders = e_loaders
    else:
        pred_dataset = d_dataset
        pred_loaders = d_loaders

    # loading xspec (default) data
    xspec_data = load_xspec_preds(specific_data)

    # '''---------- PLOTTING RESULTS ----------'''

    # # # plotting comparison between parameters
    # # # to color data by certain parameters
    # kT = val_data['targets'][:,0,3]
    # Norm = val_data['targets'][:,0,4]
    # nH = list(val_data['targets'][:,0,0])
    # gamma = list(val_data['targets'][:,0,1])
    # fsc = list(val_data['targets'][:,0,2])
    # det_nums=[]

    # if 'real' in load_name:
    #     for spectrum in val_data['ids']:
    #         with fits.open(os.path.join(ROOT,'data','spectra',spectrum)) as file:
    #             spectrum_info = file[1].header

    #         det_nums.append(int(re.search(r'_d(\d+)', spectrum_info['RESPFILE']).group(1)))

    # else:
    #     with open(os.path.join(ROOT,'data','synth_spectra_clean.pickle'), 'rb') as file:
    #         synthetic_data = pickle.load(file)
    #     det_nums = [int(re.search(r'_d(\d+)', synthetic_data['info'][spectrum].response[5:]).group(1)) for spectrum in val_data['ids']]

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
    #     dataset=pred_dataset,
    #     loaders=pred_loaders,
    #     network = net,
    #     dir_name=plots_directory,
    #     pred_savename=load_name,
    #     coverage_dir='./coverages/'+mode,
    #     overwrite=False)

    # all_param_samples = sample(specific_data if specific_data else val_data,
    #                         num_specs=len(specific_data['ids']),
    #                         num_samples=1)

    # # # single reconstructions using samples from all_param_samples
    # plots_var.recon_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = val_data,
    #     specific_data = specific_data,
    #     all_param_samples = all_param_samples,
    #     data_dir = os.path.join(ROOT,'data','spectra/'))

    # # # # posterior predictive plots using 100 samples per reconstruction
    # post_pred_samples = plots_var.post_pred_plot(
    #     decoder.net,
    #     net,
    #     dir_name = plots_directory+'reconstructions/',
    #     data = val_data,
    #     specific_data = specific_data,
    #     data_dir = os.path.join(ROOT,'data','spectra/'))

    # # # # # # posterior predictive plots only using xspec to make reconstructions - not decoder
    # plots_var.post_pred_plot_xspec(
    #     dir_name = plots_directory+'reconstructions/',
    #     data = val_data,
    #     specific_data = specific_data,
    #     post_pred_samples=post_pred_samples,
    #     data_dir = os.path.join(ROOT,'data','spectra/'))

    # # # # # latent space corner plot
    # plots_var.latent_corner_plot(
    #     dir_name = plots_directory+'distributions/',
    #     data=val_data,
    #     xspec_data=xspec_data,
    #     specific_data=specific_data,
    #     in_param_samples=all_param_samples)

    # # # scatter plot across all target parameters in dataset
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

    # # residuals vs normalisation/disk temperature
    # plots_var.resid_params_plot(
    #     data=val_data.copy(),
    #     x_param=Norm/fsc,
    #     x_param_name='N/fsc',
    #     plot_dir=plots_directory+'comparisons/',
    #     save_name = 'Param_Resids_NkT.png')

    '''---------- pyxspec tests ----------'''
    os.chdir(ROOT)
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0

    val_data['latent'] = val_data['latent'][:,0,:] # #np.median(val_data['latent'], axis = 1)
    val_data['targets'] = val_data['targets'][:,0,:]
    pyxspec_tests(val_data, config=ROOT+'/config.yaml')

    # print('###--- pyxspec tests results for ', load_name, '---###')
    # print('mean:', np.mean(data_[:, -1].astype(float)))
    # print('std:', np.std(data_[:, -1].astype(float)))
    # print('min quantile:', np.quantile(data_[:, -1].astype(float), 0.157))
    # print('max quantile:', np.quantile(data_[:, -1].astype(float), 0.843))
    # print('median:', np.median(data_[:, -1].astype(float)))
    # print('###------------------------------'+'-'*len(load_name)+'---###')

    # making predictions for timing purposes
    time1000s = []
    times1s = []
    for i in range(5):
        print('1000 samples:')
        _, time1000 = net.predict(e_loaders[1], num_samples=1000, inputs=True, ret_time=True)
        time1000s.append(time1000)
        print('1 sample:')
        _, time1 = net.predict(e_loaders[1], num_samples=1, inputs=True, ret_time=True)
        times1s.append(time1)

    # pyxspec_results = {'mean': np.mean(data_[:, -1].astype(float)), 
    #                    'std': np.std(data_[:, -1].astype(float)), 
    #                    'median': np.median(data_[:, -1].astype(float)),
    #                    'min_quantile': np.quantile(data_[:, -1].astype(float), 0.157),
    #                    'max_quantile': np.quantile(data_[:, -1].astype(float), 0.843),
    #                    'time1000': str(np.mean(time1000s))+r'$\pm$'+str(np.std(time1000s)),
    #                    'time1': str(np.mean(times1s))+r'$\pm$'+str(np.std(times1s))}

    # if not os.path.exists(ROOT+'/pyxspec_tests/'+mode+'/'):
    #     os.mkdir(ROOT+'/pyxspec_tests/'+mode+'/')
    # with open(ROOT+'/pyxspec_tests/'+mode+'/'+load_name+'.pkl', 'wb') as f:
    #     pickle.dump(pyxspec_results, f)

    # return pyxspec_results

def main():

    mode = 'supervised_5'

    numcycles=1

    for i in [0,1,2]:#range(numcycles*3):
        if i < numcycles:
            load_name = '1_test_' + str(i+1)+'_synth_real'
            dec_load_name = load_name.split('_')[0]+str(i+1)
        elif i < numcycles*2:
            load_name ='1_test_' + str(i-numcycles+1)+'_real'
            dec_load_name = load_name.split('_')[0]+str(i-numcycles+1)
        else:
            load_name =  '1_test_' + str(i-2*numcycles+1)+'_synth'
            dec_load_name = load_name.split('_')[0]+str(i-2*numcycles+1)


        print(f'analysing for load_name : {load_name}...')

        if os.path.exists('./pyxspec_tests/'+mode+'/all_results.pkl'):
            with open('./pyxspec_tests/'+mode+'/all_results.pkl', 'rb') as f:
                pyxspecs = pickle.load(f)
        else:
            pyxspecs = {'results': [],
                         'name': []}
            
        if load_name in pyxspecs['name']:
            idx = load_name.index(load_name)
            pyxspecs['results'].remove(pyxspecs['results'][idx])
            pyxspecs['name'].remove(load_name)

        pyxspecs['name'].append(load_name)
        pyxspecs['results'].append(analysis_NF(
            config='./config.yaml',
            load_name=load_name,
            dec_load_name=dec_load_name,
            mode=mode
        ))

        print('analysis ', load_name, 'complete!')

        with open('./pyxspec_tests/'+mode+'/all_results.pkl', 'wb') as f:
            pickle.dump(pyxspecs, f)

if __name__ == '__main__':
    main()