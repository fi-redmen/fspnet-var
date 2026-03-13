
from netloader.network import Network
import netloader.networks as nets
from netloader.utils.utils import progress_bar
from netloader import loss_funcs
from netloader.utils.types import TensorListLike, LossCT, TensorLossCT
from netloader.data import DataList

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from time import time
from typing import Any, cast
from warnings import warn

from fspnet.spectrum_fit import AutoencoderNet

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
    def __init__(self,
                 save_num,
                 states_dir,
                 net,
                 overwrite=True,
                 mix_precision = False,
                 learning_rate = 0.001,
                 description = '',
                 verbose = 'full',
                 transform = None,
                 latent_transform = None):
        super().__init__(save_num=save_num,
                         states_dir=states_dir,
                         net=net,
                         overwrite=overwrite,
                         mix_precision=mix_precision,
                         learning_rate=learning_rate,
                         description=description,
                         verbose=verbose,
                         transform=transform,
                         latent_transform=latent_transform)
        self._start_epoch = 0
        self.flowlossweight = 0.5
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
            'flowlossweight': self.flowlossweight,
            'separate_losses': self.separate_losses,
            'start_epoch': self._start_epoch,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.flowlossweight = state['flowlossweight']
        self.separate_losses = state['separate_losses']
        self._start_epoch = state['start_epoch']
        self._loss_weights = state.get('loss_weights', {
            'reconstruct': state.get('reconstruct_loss', 1),
            'latent': state.get('latent_loss', 1),
            'bound': state.get('bound_loss', 1),
            'kl': state.get('kl_loss', 1),
        })

    def _loss(self, in_data: TensorListLike, target: TensorListLike) -> LossCT:
        """
        Returns the loss as a float & updates network weights if training.

        Parameters
        ----------
        in_data : TensorListLike
            Input data of shape (N,...) and type float, where N is the number of elements
        target : TensorListLike
            Target data of shape (N,...) and type float

        Returns
        -------
        LossCT
            Loss or dictionary of losses which can be summed to get the total loss
        """
        key: str
        value: Tensor
        loss: TensorLossCT

        with torch.autocast(
                enabled=self._half,
                dtype=torch.bfloat16 if self._device == torch.device('cpu') else torch.float16,
                device_type=self._device.type):
            try:
                loss = self._loss_func(in_data, target)
                warn(
                    '_loss_func is deprecated, please use _loss_tensor instead',
                    DeprecationWarning,
                    stacklevel=2,
                )
            except DeprecationWarning:
                loss = self._loss_tensor(in_data, target)

        if isinstance(loss, dict) and 'total' not in loss:
            loss['total'] = self._loss_total(loss)

        self._update(loss['total'] if isinstance(loss, dict) else loss)

        if isinstance(loss, dict):
            return {key: value.item() for key, value in loss.items()}  # type: ignore[return-value]
        return loss.item()  # type: ignore[return-value]

    def _loss_tensor(self, in_data: TensorListLike, target: TensorListLike) -> dict[str, Tensor]:
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

    # def _loss(self, in_data: Tensor, target: Tensor) -> float:
    #     """
    #     Calculates the loss from the autoencoder's predictions

    #     Parameters
    #     ----------
    #     in_data : (N,...) Tensor
    #         Input high dimensional data of batch size N and the remaining dimensions depend on the
    #         network used
    #     target : (N,...) Tensor
    #         Latent target low dimensional data of batch size N and the remaining dimensions depend
    #         on the network used

    #     Returns
    #     -------
    #     float
    #         Loss from the autoencoder's predictions'
    #     """
    #     loss: Tensor
    #     latent: Tensor | None = None
    #     bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
    #     output: Tensor = self.net(in_data) #,target (for inheriting class)

    #     if self.net.checkpoints:
    #         latent = self.net.checkpoints[-1].sample([1])[0]

        # Define a dictionary for loss components
        # loss_components = { #torch.log1p() in recon? - might be unstable trianing for negative values
        #     'reconstruct':  self.reconstruct_func(output, in_data ),
        #     'flow': -1 * self.net.checkpoints[-1].log_prob(target).mean(),
        #     'latent':  self.latent_func(latent, target), #if self.latent_loss and latent is not None else None,
            # 'bound':  torch.mean(torch.cat((
            #     (bounds[0] - latent) ** 2 * (latent < bounds[0]),
            #     (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            # ))) if self.bound_loss and latent is not None else None,
            # 'kl': self.kl_loss * self.net.kl_loss if self.kl_loss else None,
        # }

        # Iterate through the dictionary to process each loss component
        # separate_loss = []
        # for key, loss_value in loss_components.items():
        #     if loss_value is not None:  # Only process non-None losses
        #         separate_loss.append(loss_value)
        #         if self._train_state:
        #             self.separate_losses[key].append(loss_value.clone().item())
        
        # for key, weight in enumerate([self.reconstruct_loss, self.flowlossweight, self.latent_loss]):
        #     if separate_loss[key] is not None:
        #         separate_loss[key] *= weight

        # Compute the total loss
        # loss = torch.sum(torch.stack(separate_loss))

        # appends each loss component to the separate losses dictionary - for saving separate losses
        # if self._train_state:
        #     self.separate_losses["reconstruct"].append(self.reconstruct_func(output, in_data))
        #     # self.separate_losses["flow"].append(-1 * self.net.checkpoints[-1].log_prob(target).mean())
        #     self.separate_losses["latent"].append(self.latent_func(latent, target))

        # # sums each loss component with their respective weights - for updating network
        # loss = torch.sum(torch.stack([self.reconstruct_loss * self.reconstruct_func(output, in_data),
        #     # self.flowlossweight * -1 * self.net.checkpoints[-1].log_prob(target).mean(),
        #     self.latent_loss * self.latent_func(latent, target)]))

        # self._update(loss)
        # return loss.item()

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
        loss: LossCT

        # losses=[]
        # for i in range(len(self.losses[1])):
        #     losses.append(float(np.mean(self.losses[1][i-10:i])))

        # Train for each epoch
        for i in range(self._epoch, epochs):
            t_initial = time()

            # Train network
            self.train(True)
            self.losses[0].append(self._train_val(loaders[0]))

            # Validate network
            # self.train(False)
            # self.losses[1].append(self._train_val(loaders[1]))
            # self._update_scheduler(metrics=self.losses[1][-1])

            # Validate network
            self.train(False)
            self.losses[1].append(self._train_val(loaders[1]))
            self._update_scheduler(
                metrics=cast(dict, self.losses[1][-1])['total']
                if isinstance(self.losses[1][-1], dict) else self.losses[1][-1],
            )

            # Save training progress
            self._update_epoch()
            self.save()
            self._epoch_print(i, epochs, time() - t_initial)

            # if self._verbose in ('full', 'epoch'):
            #     print(f'Epoch [{self._epoch}/{epochs}]\t'
            #           f'Training loss: {self.losses[0][-1]:.3e}\t'
            #           f'Validation loss: {self.losses[1][-1]:.3e}\t'
            #           f'Time: {time() - t_initial:.1f}')
            # elif self._verbose == 'progress':
            #     progress_bar(
            #         i,
            #         epochs,
            #         text=f'Epoch [{self._epoch}/{epochs}]\t'
            #              f'Training: {self.losses[0][-1]:.3e}\t'
            #              f'Validation: {self.losses[1][-1]:.3e}\t'
            #              f'Time: {time() - t_initial:.1f}',
            #     )

            # losses.append(float(np.mean(self.losses[1][-10:]))) # averages loss over 10 last values

            # End plateaued networks early
            if (self._epoch > self._start_epoch + self.scheduler.patience * 2 and
                self.losses[1][-self.scheduler.patience * 2] < self.losses[1][-1]):

                print('Trial plateaued, ending early...')
                break

        self.train(False)
        final_loss = self._train_val(loaders[1])
        print(f'\nFinal validation loss: {final_loss:.3e}')

        self._start_epoch = self._epoch


class NFautoencoderNetwork(AutoencoderNet):
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
        x = self.net[0](x)
        self.checkpoints.append(x)
        x = x.sample([1])
        x = x[0]

        # if hasattr(self.net[0], 'kl_loss'): - don't need kl loss?
        #     self.kl_loss = self.net[0].kl_loss

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
    in_transform = None) -> None:
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
            in_transform=in_transform)
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
        print(f'\nFinal validation loss: {final_loss:.3e}')

        self._start_epoch = self._epoch


