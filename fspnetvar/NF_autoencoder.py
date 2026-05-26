
from netloader.network import Network
import netloader.networks as nets
from netloader.utils.utils import progress_bar
from netloader import loss_funcs
from netloader.utils.types import TensorListLike, LossCT
from netloader.data import DataList
from netloader import models

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, nn
from torch.optim.optimizer import ParamsT
from torch.utils.data import DataLoader
from time import time
from typing import Any, cast

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


class MSELoss(loss_funcs.MSELoss):
    """
    Mean Squared Error (MSE) loss function
    """
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return self._loss_func(output[:, 0], target[:, 0])

class NFautoencoder(nets.Autoencoder):
    def __init__(
            self,
            save_num,
            states_dir,
            net,
            overwrite=True,
            mix_precision = False,
            learning_rate = 0.001,
            description = '',
            verbose = 'full',
            transform = None,
            latent_transform = None,
            optimiser_kwargs: dict[str, Any] | None = None,
            scheduler_kwargs: dict[str, Any] | None = None):
        super().__init__(
            save_num=save_num,
            states_dir=states_dir,
            net=net,
            overwrite=overwrite,
            mix_precision=mix_precision,
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            latent_transform=latent_transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self._start_epoch = 0
        self.separate_losses = {
            'reconstruct': [],
            'flow': [],
            'latent': []
        }
        self._loss_weights = {
            'flow': 1,
            'reconstruct': 1,
            'latent': 1,
            'bound': 1,
            'kl': 1
        }

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'separate_losses': self.separate_losses,
            'start_epoch': self._start_epoch,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.separate_losses = state['separate_losses']
        self._start_epoch = state['start_epoch']
        self._loss_weights = state.get('loss_weights', {
            'reconstruct': state.get('reconstruct_loss', 1),
            'latent': state.get('latent_loss', 1),
            'bound': state.get('bound_loss', 1),
            'flow': state.get('flowlossweight', 1),
            'kl': state.get('kl_loss', 1),
        })

    def _loss_tensor(self, in_data: TensorListLike, target: TensorListLike, _: Any) -> dict[str, Tensor]:
        """
        Calculates the loss from the autoencoder's predictions.

        Parameters
        ----------
        in_data : TensorListLike
            Input high dimensional data of shape (N, ...) and type float, where N is the batch size
        target : TensorListLike
            Latent target low dimensional data of shape (N, ...) and type float

        Returns
        -------
        dict[str, Tensor]
            Loss function terms from the autoencoder's predictions
        """
        loss: dict[str, Tensor] = {}
        latent: Tensor | None = None
        bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
        output: Tensor = self.net(in_data)

        if self.net.checkpoints and isinstance(self.net.checkpoints[-1], DataList):
            raise ValueError(f'Autoencoder networks cannot have multiple latent space tensors '
                            f'({len(self.net.checkpoints[-1])})')
        if self.net.checkpoints:
            latent = cast(Tensor, self.net.checkpoints[-1])

        if self.get_loss_weights('reconstruct'):
            loss['reconstruct'] = self.reconstruct_func(output, in_data)

        if self.get_loss_weights('latent') and latent is not None:
            loss['latent'] = self.latent_func(latent, target)

        if self.get_loss_weights('bound') and latent is not None:
            loss['bound'] = torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        if self.get_loss_weights('flow'):
            loss['flow'] = -1 * self.net.checkpoints[-1].log_prob(target).mean()

        if self.get_loss_weights('kl'):
            loss['kl'] = self.net.kl_loss
        return loss

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

        return (
            self.net(data).detach().cpu().numpy(),
            self.net.checkpoints[-1].sample([num_samples]).swapaxes(0,1).detach().cpu().numpy(),
            data.detach().cpu().numpy(),
        )

    def get_param_groups(self, learning_rate: float | tuple[float, ...] | None) -> ParamsT:
        return [
            {'params': self.net[0].parameters(), 'lr': learning_rate},
            {'params': self.net[1].parameters(), 'lr': 0},
        ]

    def training(self, epochs: int, loaders: tuple[DataLoader[Any], DataLoader[Any]]) -> None:
        """
        Trains & validates the network for each epoch.

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network up to
        loaders : tuple[DataLoader[Any], DataLoader[Any]]
            Train and validation data loaders
        """
        i: int
        t_initial: float
        loss: LossCT

        # Train for each epoch
        for i in range(self._epoch, epochs):
            t_initial = time()

            # Train network
            self.train(True)
            self.losses[0].append(self._train_val(loaders[0]))

            # Validate network
            self.train(False)
            self.losses[1].append(self._train_val(loaders[1]))
            self._step(
                False,
                i,
                cast(dict, self.losses[1][-1])['total'] if isinstance(self.losses[1][-1], dict) else
                self.losses[1][-1],
            )

            # Save training progress
            self._update_epoch()
            self.save()
            self._epoch_print(i, epochs, time() - t_initial)

            # End plateaued networks early
            if (self._epoch > self._start_epoch + self.scheduler.patience * 2 and
                self.losses[1][-self.scheduler.patience * 2] < self.losses[1][-1]):

                print('Trial plateaued, ending early...')
                break

        self.train(False)
        self._plot_active = False
        self._start_epoch = self._epoch
        loss = self._train_val(loaders[1])
        print(f"\nFinal validation loss: "
              f"{cast(dict, loss)['total'] if isinstance(loss, dict) else loss:.3e}")


class NFautoencoderNetwork(models.MultiNetwork):
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
        self.checkpoints = []
        x = self.net[0](x)
        self.checkpoints.append(x)
        x = x.rsample([1])[0]
        return self.net[1](x)   # for non variational


class NFdecoder(nets.Decoder):
    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | Network,
            overwrite = False,
            mix_precision = False,
            learning_rate = 1e-3,
            description = '',
            verbose = 'full',
            transform = None,
            in_transform = None,
            optimiser_kwargs: dict[str, Any] | None = None,
            scheduler_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(
            save_num,
            states_dir,
            net,
            overwrite=overwrite,
            mix_precision=mix_precision,
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=in_transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self._start_epoch = 0
        self.loss_func = MSELoss()

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'loss_func': self.loss_func,
            'start_epoch': self._start_epoch}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.loss_func = state['loss_func']
        self._start_epoch = state['start_epoch']

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

        losses=[]

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
                        f'Training: {self.losses[0][-1]:.3e}\t'
                        f'Validation: {self.losses[1][-1]:.3e}\t'
                        f'Time: {time() - t_initial:.1f}',
                )

            losses.append(float(np.mean(self.losses[1][-10:]))) # averages loss over 10 last values

            # End plateaued networks early
            if (self._epoch > self._start_epoch + self.scheduler.patience * 2 and
                self.losses[1][-self.scheduler.patience * 2] < self.losses[1][-1]):

                print('Trial plateaued, ending early...')
                break

        self.train(False)
        final_loss = self._train_val(loaders[1])
        self._start_epoch = self._epoch
        print(f'\nFinal validation loss: {final_loss:.3e}')
