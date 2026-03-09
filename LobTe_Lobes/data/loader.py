"""
File: loader.py
Author: Ariel Hernán Curiale
Github: https://gitlab.com/Curiale
Description: This module provides a fast implementation of the torch loader
using a weak shuffling strategy
"""

import math
import numpy as np

from torch.utils.data import BatchSampler


class RandomBatchSampler(BatchSampler):
    """Sampling the dataset using a week shuffling strategy. This class sample
    sequential batches randomly.
    """

    def __init__(
        self,
        x_data,
        y_data,
        batch_size: int,
        drop_last: bool,
        boostrap=False,
        nboost_size=None,
        shift_range=0,
        scale_range=1,
        std_noise=0.5,
        rate_noise=0,
    ):
        """Given a dataset this class provides sequential batches of batch_size
        randomly

        Parameters
        ----------
        dataset : pytorch Dataset
        batch_size : int

        """
        self.x_data = x_data
        self.axis = [ax for ax in range(x_data.ndim)]
        self.y_data = y_data
        self.boostrap = boostrap
        self.batch_size = batch_size
        self.n_data = len(x_data)
        self.nboost_size = batch_size if nboost_size is None else nboost_size
        self.drop_last = drop_last

        self.std_noise = std_noise
        self.rate_noise = rate_noise

        if drop_last:
            self.n_batches = int(self.n_data / batch_size)
        else:
            self.n_batches = math.ceil(self.n_data / batch_size)
        self.reset()

        self.augmentation_init(shift_range, scale_range)

    def augmentation_init(self, shift_range, scale_range):
        # Used for data augmentation
        if isinstance(shift_range, list) or isinstance(shift_range, tuple):
            min_shift = shift_range[0]
            max_shift = shift_range[1]
        else:
            min_shift = -shift_range
            max_shift = shift_range

        if (max_shift - min_shift) > 0:
            self.shift_range = (min_shift, max_shift)
        else:
            self.shift_range = None

        min_scale = 1 / scale_range
        max_scale = scale_range
        if isinstance(scale_range, list) or isinstance(scale_range, tuple):
            min_scale = scale_range[0]
            max_scale = scale_range[1]

        if (max_scale - min_scale) > 0:
            self.scale_range = (min_scale, max_scale)
        else:
            self.scale_range = None

    def reset(self):
        self.batch_ids = np.random.permutation(self.n_batches)

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for i in self.batch_ids:
            if i == self.n_batches - 1 and not self.drop_last:
                id_start = (self.n_batches - 1) * self.batch_size
                id_end = self.n_data
            else:
                id_start = i * self.batch_size
                id_end = (i + 1) * self.batch_size

            ids = np.arange(id_start, id_end)
            if self.boostrap:
                ids = np.random.choice(
                    ids, size=self.nboost_size, replace=True
                )

            idx = ids.tolist()

            x_batch = self.x_data[idx]
            y_batch = {k: self.y_data[k][idx] for k in self.y_data}

            if self.shift_range is not None:
                # Shift the data
                random_shift = self.shift_range[0] + (
                    self.shift_range[1] - self.shift_range[0]
                ) * np.random.rand(*x_batch.shape[:2]).astype(x_batch.dtype)
                x_batch = x_batch + np.expand_dims(
                    random_shift, axis=self.axis[2:]
                )

            if self.scale_range is not None:
                # Scale the data
                random_scale = self.scale_range[0] + (
                    self.scale_range[1] - self.scale_range[0]
                ) * np.random.rand(*x_batch.shape[:2]).astype(x_batch.dtype)
                x_batch = x_batch * np.expand_dims(
                    random_scale, axis=self.axis[2:]
                )

            if self.rate_noise > 0:
                # Add noise to the data
                samples = np.prod(x_batch.shape)
                size = int(round(samples * self.rate_noise))
                idx_noise = np.random.randint(0, samples, size=size)
                snoise = np.random.normal(
                    loc=0, scale=self.std_noise, size=size
                )
                noise = np.zeros(samples, dtype=x_batch.dtype)
                noise[idx_noise] = snoise
                noise = noise.reshape(x_batch.shape)

                x_batch = x_batch + noise

            yield ids, x_batch, y_batch
