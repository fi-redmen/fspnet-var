
## Requirements
- Install dependencies:
  ```pip install -r requirements.txt```
- PyXspec dependency:
  Xspec from [HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/xspec/index.html) provided by NASA

## Data compatibility

## File Descriptions
**Main_NF**: Trains variational autoencoder with normalizing flow network network, makes predictions, and calls functions from VAE_plots for plotting

**Main_VAE**: Trains vanilla variational autoencoder network, makes predictions, and calls functions from VAE_plots for plotting - outdateed

**VAE_plots**: Contains functions used for plotting

**MCMC_chains**: Runs MCMC chains for certain spectra using pyxspec

**xspec_funcs**: Functions that use xspec

**network_configs**: Neural network configurations
