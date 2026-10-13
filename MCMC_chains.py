# Imports
import xspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
import pickle
import time

from fspnet.utils.utils import open_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(ROOT)

def run_MCMC_chains():
    # load spectra names, object names and targets
    with open(os.path.join(ROOT,'FSPNet/predictions/supervised_4/specific_1_test_1_synth.pickle'), 'rb') as file:
        pickle_data = pickle.load(file)

    _, config = open_config('spectrum-fit',ROOT+'/FSPNet/config.yaml')

    model_name = config['model']['custom-model-name']
    model_dir = config['model']['model-directory']

    spectra = pickle_data['ids']
    object_names = pickle_data['object']
    targets = pickle_data['targets']

    print('ROOT:', ROOT)

    # loads simplcutx to be used
    # xspec.AllModels.lmod(model_name, model_dir)
    xspec.AllModels.tclLoad(ROOT+'/Xspec_models/simplcutx/libjscutx.dylib')

    data_dir=os.path.join(ROOT,'FSPNet/data/spectra/')

    # to put all data in one file
    all_dists = []
    all_xspec_recons = []
    all_chain_times = []

    for spec_num in range(len(spectra)):
        spectrum=spectra[spec_num]
        object_name = object_names[spec_num]

        # Load the spectrum information
        with fits.open(data_dir + spectrum) as file:
            spectrum_info = file[1].header
        det_num = int(spectrum_info['RESPFILE'][7:9])

        # create settings for fakeit based on the spectrum we use
        fake_base = xspec.FakeitSettings(
        response=data_dir + spectrum_info['RESPFILE'],
        arf=data_dir + spectrum_info['ANCRFILE'],
        background=data_dir + spectrum_info['BACKFILE'],
        exposure=spectrum_info['EXPOSURE'])

        # loading the data
        os.chdir(data_dir)
        xspec.AllData.clear()
        s1 = xspec.Spectrum(os.path.join(data_dir, spectrum))
        xspec.AllData.ignore("0.-0.3 10.-**")
        xspec.AllData.ignore("bad")

        # settings for plotting data in xspec
        xspec.AllModels.setEnergies("0.003  300. 1000 log")
        xspec.Plot.xAxis="keV"
        xspec.Plot.background = True
        xspec.AllModels.systematic = 0.
        xspec.Fit.statMethod = 'pgstat'
        xspec.Xset.abund = "wilm"
        xspec.Plot("data")

        # fitting to data
        pars = list(np.concat([targets[spec_num][0][:3], [0.0, 100.0], targets[spec_num][0][3:]]))
        xspec_model = xspec.Model("tbabs(simplcutx(ezdiskbb))")
        #setting the parameters individually
        xspec_model.TBabs.nH.values = str(targets[spec_num,0,0])+',,5.0e-3,5.0e-3,75,75'
        xspec_model.simplcutx.Gamma.values = str(targets[spec_num,0,1])+',,1.3,1.3,4,4'
        xspec_model.simplcutx.FracSctr = str(targets[spec_num,0,2])+',,1.0e-3,1.0e-3,1,1'
        xspec_model.ezdiskbb.T_max = str(targets[spec_num,0,3])+',,2.5e-2,2.5e-2,4,4'
        xspec_model.ezdiskbb.norm = str(targets[spec_num,0,4])+',,1.0e-2,1.0e-2,1.0e+10,1.0e-10'

        xspec_model.setPars(pars)
        xspec_model.simplcutx.ReflFrac.frozen = True
        xspec_model.simplcutx.kT_e.frozen = True
        xspec.Fit.renorm()
        xspec.Fit.nIterations = 1000
        xspec.Fit.perform()
        xspec.Plot("data", "model")

        # getting data and model from xspec plots plots
        data_energy = xspec.Plot.x()
        data = np.array(xspec.Plot.y())/det_num
        model_energy = xspec.Plot.x()
        model = np.array(xspec.Plot.model())/det_num

        # plotting our data and model
        plt.cla()
        plt.plot(data_energy, data, linestyle='None', marker='o', color='r', markersize=2, label='Real Data')
        plt.plot(model_energy, model, linestyle='-', color='k', label='Model')
        plt.xlabel('Energy (keV)')
        plt.ylabel('cts cm$^{-2} s^{-1} keV^{-1}$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(ROOT,'fspnet/plots/chain_fits/recon_'+object_name+'.png'))

        # Creating an MCMC chain
        os.chdir(os.path.join(ROOT,'fspnet/chains/'))
        xspec.AllChains.clear()
        c = xspec.Chain("chain"+str(spec_num)+".fits")

        # set chain settings
        c.walkers = 20
        c.burn = 20000
        c.runLength = 200000

        # start and time chain
        start = time.time()
        c.run() 
        stop = time.time()
        chain_time = (stop-start)
        print('Time:', chain_time, 's')

        # save chains
        xspec.Plot("chain", "1")
        nH_dist = np.array(xspec.Plot.y())
        xspec.Plot("chain", "2")
        Gamma_dist = np.array(xspec.Plot.y())
        xspec.Plot("chain", "3")
        f_sc_dist = np.array(xspec.Plot.y())
        xspec.Plot("chain", "6")
        kT_dist = np.array(xspec.Plot.y())
        xspec.Plot("chain", "7")
        norm_dist = np.array(xspec.Plot.y())

        dists = np.array([nH_dist, Gamma_dist, f_sc_dist, kT_dist, norm_dist])

        # reconstruction given from the true posteriors
        xspec_recon = [model_energy, model]
        # true posteriors are our distributions
        
        all_dists.append(dists)
        all_xspec_recons.append(xspec_recon)
        all_chain_times.append(chain_time)

    data = {
            'id': spectra,
            'object': object_names,
            'posteriors': all_dists,
            'xspec_recon': all_xspec_recons,
            'chain_time': all_chain_times
        }

    # Save the data to respective file
    with open(os.path.join(ROOT,'fspnet/predictions/xspec_preds1.pickle'), 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    run_MCMC_chains()

# note: shapes:
#   id: number of specific spectra (5)
#   object: number of specific spectra (5)
#   posteriors: number of specific spectra (5), number of parameters, number of samples
#   xspec_recon: number of spectra (5), 2 (indices: 0 is spectral energies, 1 is the spectra recontructions)
#   chain_time: number of specific spectra (5)