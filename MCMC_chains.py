# Imports
import xspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
import pickle
import time

from fspnet.utils.utils import open_config

# dates for Cyg X-1 observations
# 57934.6648264
# 58223.8182995

# settings
# spectra = ['js_ni0100320101_0mpu7_goddard_GTI0.jsgrp', 
#            'js_ni0103010102_0mpu7_goddard_GTI0.jsgrp',
#            'js_ni1014010102_0mpu7_goddard_GTI30.jsgrp',
#            'js_ni1050360115_0mpu7_goddard_GTI9.jsgrp',
#            'js_ni1100320119_0mpu7_goddard_GTI26.jsgrp'
#            'js_ni1200120203_0mpu7_goddard_GTI0.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI10.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI11.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI13.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI1.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI3.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI4.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI5.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI6.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI7.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI8.jsgrp',
#            'js_ni1200120203_0mpu7_goddard_GTI9.jsgrp',]

# object_names=['Cyg X-1 (2017)',
#             'GRS 1915+105',
#             'LMC X-3',
#             'MAXI J1535-571',
#             'Cyg X-1 (2018)'
#             'MAXI J1820 0',
#             'MAXI J1820 10',
#             'MAXI J1820 11',
#             'MAXI J1820 13',
#             'MAXI J1820 1',
#             'MAXI J1820 3',
#             'MAXI J1820 4',
#             'MAXI J1820 5',
#             'MAXI J1820 6',
#             'MAXI J1820 7',
#             'MAXI J1820 8',
#             'MAXI J1820 9']


# targets = np.array([
#         [[6.41600004e-01, 2.02710009e+00, 2.89700013e-01, 2.42299992e-01,
#          5.36813698e+04],
#         [6.39999891e-03, 6.09999988e-03, 2.90000020e-03, 1.49999990e-03,
#          1.78342395e+03]],                                                  #Cyg X-1 (2017)
        
#         [[6.59300029e+00, 4.00000000e+00, 1.00000000e-03, 1.90190000e+00,
#          1.05888011e+01],
#         [6.58999979e-02, 4.00000000e+00, 5.01999967e-02, 2.30999999e-02,
#          5.03900051e-01]],                                                  #GRS 1915+105
    
#         [[3.99999999e-01, 3.63059998e+00, 9.95000005e-01, 6.91600025e-01,
#          3.82468004e+01],
#         [4.00000019e-03, 1.61799997e-01, 6.21000007e-02, 1.93000007e-02,
#          3.70679998e+00]],                                                  #LMC X-3

#         [[3.44460018e+00, 1.29999995e+00, 2.60000018e-02, 1.20379997e+00,
#          8.27809875e+02],
#         [3.44000012e-02, 1.25000002e-02, 1.00000005e-03, 2.40000035e-03,
#          8.27809906e+00]],                                                  #MAXI J1535-571       

#         [[7.36299983e-01, 2.63450003e+00, 4.03600002e-01, 4.09999993e-01,
#          3.03864082e+04],
#         [7.40000000e-03, 6.30000001e-03, 4.60000010e-03, 1.00000005e-03,
#          3.56234894e+02]],                                                   #Cyg X-1 (2018)
         
#          ])                                 

# load spectra names, object names and targets
with open('/Users/astroai/Projects/FSPNet/predictions/specific_Encoder NF2_5 Decoder NF0_4.pickle', 'rb') as file:
    pickle_data = pickle.load(file)

spectra = pickle_data['ids']
object_names = pickle_data['object']
targets = pickle_data['targets']


# loads simplcutx to be used
xspec.AllModels.lmod("simplcutx", "/Users/astroai/Downloads/simplcutx/")

data_dir='/Users/astroai/Projects/FSPNet/data/spectra/'

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
    plt.savefig('/Users/astroai/Projects/FSPNet/plots/chain_fits/recon_'+object_name)

    # Creating an MCMC chain
    os.chdir('/Users/astroai/Projects/FSPNet/chains/')
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
    print(chain_time)

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
    # param_names = np.array(['N_H', 'f_sc_dist', 'Gamma_dist', 'kT_dist', 'norm_dist'])

    # plt.plot(x, y, linestyle='-', color='k', label='Chain 1')
    # fig, axes = plt.subplot_mosaic('aabbcc\ndddeee', constrained_layout=True)

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
with open('/Users/astroai/Projects/FSPNet/predictions/xspec_preds1.pickle', 'wb') as file:
    pickle.dump(data, file)

# note:
# xspec_preds is with fiducial values and fitting with 1000 iterations before chain
# xspec_preds1 is with precalulated values and fitting with 1000 iterations before chain

# note: shapes:
#   id: number of specific spectra (5)
#   object: number of specific spectra (5)
#   posteriors: number of specific spectra (5), number of parameters, number of samples
#   xspec_recon: number of spectra (5), 2 (indices: 0 is spectral energies, 1 is the spectra recontructions)
#   chain_time: number of specific spectra (5)