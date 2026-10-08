
from netloader.network import Network
import netloader.architectures as archs
from netloader.utils.utils import progress_bar
from netloader import loss_funcs, models
from netloader.transforms import BaseTransform
from warnings import warn
from netloader.data import Data, DataList, data_collation
from netloader.utils.types import (
    TensorLike,
    NDArrayLike,
    TensorListLike,
    LossCT,
)

import torch
from torch import Tensor, nn
from torch.optim.optimizer import ParamsT
from torch.utils.data import DataLoader
import numpy as np
from numpy import ndarray
from time import time
from typing import Any, cast
from itertools import repeat


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
        return self._loss_func(output, target)

class NFautoencoder(archs.Autoencoder):
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


    # def _loss(self, in_data: TensorListLike, target: TensorListLike, extra: Any) -> LossCT:
    #     """
    #     Returns the loss as a float & updates network weights if training.

    #     Parameters
    #     ----------
    #     in_data : TensorListLike
    #         Input data of shape (N,...) and type float, where N is the number of elements
    #     target : TensorListLike
    #         Target data of shape (N,...) and type float
    #     extra : Any
    #         Extra data from the data loader to pass to the loss function

    #     Returns
    #     -------
    #     LossCT
    #         Loss or dictionary of losses which can be summed to get the total loss
    #     """
    #     key: str
    #     value: Tensor
    #     loss: TensorLossCT

    #     with torch.autocast(
    #             enabled=self._half,
    #             dtype=torch.bfloat16 if self._device == torch.device('cpu') else torch.float16,
    #             device_type=self._device.type):
    #         try:
    #             loss = self._loss_func(in_data, target)
    #             warn(
    #                 '_loss_func is deprecated, please use _loss_tensor instead',
    #                 DeprecationWarning,
    #                 stacklevel=2,
    #             )
    #         except DeprecationWarning:
    #             try:
    #                 loss = self._loss_tensor(in_data, target, extra)
    #             except TypeError:
    #                 warn(
    #                     '_loss_tensor without extra parameter is deprecated, please update '
    #                     'the method to include the extra parameter',
    #                     DeprecationWarning,
    #                     stacklevel=2,
    #                 )
    #                 loss = self._loss_tensor(in_data, target)

    #     if isinstance(loss, dict) and 'total' not in loss:
    #         loss['total'] = self._loss_total(loss)

    #     self._update(loss['total'] if isinstance(loss, dict) else loss)

    #     if isinstance(loss, dict):
    #         return {key: value.item() for key, value in loss.items()}  # type: ignore[return-value]
    #     return loss.item()  # type: ignore[return-value]

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
            loss['flow'] = -1 * self.net.checkpoints[-2].log_prob(target).mean()

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
            self.net.checkpoints[-2].sample([num_samples]).swapaxes(0,1).detach().cpu().numpy(),
            data.detach().cpu().numpy(),
        )

    def predict(
            self,
            loader: DataLoader[Any],
            *,
            inputs: bool = False,
            path: str = '',
            ret_time = False,
            **kwargs: Any) -> dict[str, NDArrayLike]:
        """
        Generates predictions for the network and can save to a file.

        Parameters
        ----------
        loader : DataLoader[Any]
            Data loader to generate predictions for
        inputs : bool, Optional
            If the input data should be returned and saved, default = False
        path : str, Optional
            Path as pkl file to save the predictions if they should be saved
        **kwargs
            Optional keyword arguments to pass to batch_predict

        Returns
        -------
        dict[str, NDArrayLike]
            Prediction IDs, Optional inputs, target values, and predicted values of shape (N,...)
            and type float for dataset of size N
        """
        t_initial: float = time()
        key: str
        ids: tuple[str, ...] | ndarray | Tensor
        low_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        high_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        data: list[list[NDArrayLike | None]] = []
        data_: dict[str, NDArrayLike] = {}
        transform: list[BaseTransform] | BaseTransform | None
        transforms: dict[str, list[BaseTransform] | BaseTransform | None] = {
            key: transform for key, transform in self.transforms.items()
            if inputs or key != 'inputs'
        }
        datum: NDArrayLike
        target: TensorLike
        in_data: TensorLike
        self.train(False)

        if 'input_' in kwargs:
            warn(
                'input_ keyword argument is deprecated, please use inputs instead',
                DeprecationWarning,
                stacklevel=2,
            )
            inputs = kwargs.pop('input_')

        # Generate predictions
        with torch.no_grad(), torch.autocast(
                enabled=self._half,
                device_type=self._device.type,
                dtype=torch.float32):
            for i, (ids, low_dim, high_dim, *_) in enumerate(loader):
                in_data, target = self._data_loader_translation(
                    data_collation(low_dim, data_field=True),
                    data_collation(high_dim, data_field=True),
                )
                data.append(cast(list[NDArrayLike | None], [
                    ids.numpy() if isinstance(ids, Tensor) else np.array(ids),
                    *([in_data.numpy()] if inputs else []),
                    target.numpy(),
                    *self.batch_predict(
                        (in_data if isinstance(in_data, Tensor) else
                        cast(DataList[Tensor], data_collation(
                            cast(list[Data] | list[DataList[Tensor]], [in_data]),
                            data_field=False,
                        ))).to(self._device),
                        **kwargs,
                    ),
                ]))
                self._predict_print(i, time() - t_initial, loader)

        # Transforms all data and saves it to a dictionary
        for (key, transform), datum_ in zip(transforms.items(), zip(*data)):
            if datum_[0] is None:
                continue

            # Concatenate values
            datum = data_collation(list(datum_), data_field=True)

            if isinstance(datum, DataList) and transform:
                data_[key] = datum.apply(transform, back=True).numpy()
                data_[key] = DataList(
                    [trans(val, back=True) for val, trans in zip(
                        datum,
                        transform if isinstance(transform, list) else repeat(transform),
                    )],
                )
            elif isinstance(transform, BaseTransform):
                assert not isinstance(datum, DataList)
                data_[key] = transform(datum, back=True)
            else:
                if isinstance(transform, list):
                    self._logger.warning(f'List of transforms requires corresponding data with key '
                                         f'({key}) to be a DataList, data will not be '
                                         f'untransformed')
                data_[key] = datum

        self._save_predictions(path, data_)
        if ret_time == False:
            return data_
        else:
            return data_, time() - t_initial

    def get_param_groups(self, learning_rate: float | tuple[float, ...] | None) -> ParamsT:
        return [
            {'params': self.net[0].parameters(), 'lr': learning_rate},
            {'params': self.net[1].parameters(), 'lr': 0},
        ]

    def _train_val(self, loader: DataLoader[Any]) -> LossCT:
        """
        Trains the network for one epoch.

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader that contains data to train

        Returns
        -------
        LossCT
            Average loss value or dictionary of average loss values for the epoch
        """
        i: int
        value: float
        t_initial: float
        key: str
        loss: LossCT | None = None
        low_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        high_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        extra: list[Any]
        batch_loss: LossCT
        target: TensorListLike
        in_data: TensorListLike

        with torch.set_grad_enabled(self._train_state):
            for i, (_, low_dim, high_dim, *extra) in enumerate(loader):
                t_initial = time()
                in_data, target = cast(
                    tuple[TensorListLike, TensorListLike],
                    self._data_loader_translation(
                        data_collation(low_dim, data_field=False).to(self._device),
                        data_collation(high_dim, data_field=False).to(self._device),
                    ),
                )

                try:
                    batch_loss = self._loss(in_data, target, extra[0] if extra else None)
                except TypeError:
                    warn(
                        '_loss without extra parameter is deprecated, please update the '
                        'method to include the extra parameter',
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    batch_loss = self._loss(in_data, target, None)

                if isinstance(batch_loss, dict) and loss:
                    for key, value in batch_loss.items():
                        loss[key] += value
                elif isinstance(batch_loss, float) and loss:
                    loss += batch_loss
                else:
                    loss = batch_loss

                if self._train_state:
                    self._step(
                        True,
                        self._epoch + (i + 1) / len(loader),
                        batch_loss['total'] if isinstance(batch_loss, dict) else batch_loss,
                    )

                self._batch_print(i, time() - t_initial, loader, batch_loss)

        assert loss

        if isinstance(loss, dict):
            return {key: value / len(loader) for key, value in loss.items()}
        return loss / len(loader)

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

        self._start_epoch = self._epoch

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
            avg_over = 2
            patience_factor=1
            if (self._epoch > self._start_epoch + self.scheduler.patience*patience_factor + avg_over):
                threshold_losses = [np.mean([np.array(self.losses[1])[-self.scheduler.patience*patience_factor-i][key]
                                             for i in range(avg_over)])
                                             for key in self.losses[1][0].keys() if key!='total']
                current_losses =  [np.mean([np.array(self.losses[1])[-i][key]
                                            for i in range(avg_over)])
                                            for key in self.losses[1][0].keys() if key!='total']
                if all(c > t for c, t in zip(current_losses, threshold_losses)):
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
        self.checkpoints.append(x)
        return self.net[1](x)   # for non variational


class NFdecoder(archs.Decoder):
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
            if (self._epoch > self._start_epoch + self.scheduler.patience*3 + 10):
                threshold_loss = np.mean(self.losses[1][-self.scheduler.patience*3-10:-self.scheduler.patience*3])
                current_loss =  np.mean(self.losses[1][-10:])
                if current_loss > threshold_loss:
                    print('Trial plateaued, ending early...')
                    break
            # if self._epoch > self._start_epoch + self.scheduler.patience * 1 + 10:
            #     threshold_loss = np.mean(self.losses[1][-self.scheduler.patience*1-10:-self.scheduler.patience*5])
            #     current_loss =  np.mean(self.losses[1][-10:])
            #     if current_loss > threshold_loss:
            #         print('Trial plateaued, ending early...')
            #         break

        self.train(False)
        final_loss = self._train_val(loaders[1])
        self._start_epoch = self._epoch
        print(f'\nFinal validation loss: {final_loss:.3e}')
