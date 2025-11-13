## Code used for variational autoencoder with normalising flow for spectral fitting of black hole X-ray binaries to rapidly obtain full posterior distributins of spectral parameters
**Main_NF**: Trains variational autoencoder with normalizing flow network network, makes predictions, and calls functions from VAE_plots for plotting

**Main_VAE**: Trains vanilla variational autoencoder network, makes predictions, and calls functions from VAE_plots for plotting

**VAE_plots**: Contains functions used for plotting

**MCMC_chains**: Runs MCMC chains for certain spectra using pyxspec

**xspec_funcs**: functions that use xspec
