# Fast Spectra Predictor Network-Variational (FSPNet-Var)
FSPNet-Var is a variational autoencoder network, building on the [FSPNet](https://github.com/EthanTreg/Spectrum-Machine-Learning) model. A normalising flow is included in between the encoder and latent space to enable complex posterior distributions on inferred physical parameters in the latest model. This significantly outperforms the vanilla variational autoencoder. 

The research paper can be found on [Arxiv](https://arxiv.org/abs/2601.07440)

## Requirements
- Install dependencies:
  ```pip install -r requirements.txt```
- PyXspec dependency:
  Xspec from [HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/xspec/index.html) provided by NASA

## Data compatibility

## File Descriptions
**fspnetvar**
* **main_NF**: Trains variational autoencoder with normalizing flow network, makes predictions, and plots results
* **main_VAE**: Trains vanilla variational autoencoder network, makes predictions, and plots results
* **MCMC_chains**: Runs MCMC chains for certain spectra using pyxspec

* **Utils**
  * **misc_utils**: Miscellaneous functions
  * **plot_utils**: Functions used in plots_var
  * **plots_var**: Plotting functions
  * **xspec_utils**: Functions which use PyXspec

**network_configs** 
Neural network configurations
