
from pydpf.base import Module
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple
import torch
from torch import Tensor
from pydpf.utils import multiple_unsqueeze


class Distribution(Module, metaclass=ABCMeta):
    """Base class for implementations of common distributions.
    This is to provide a convenient API, similar to torch.distribution.
    If a user wants to create a custom distribution it will almost always be easier to subclass Module
    and manually implement sample and log_density methods rather than to subclass Distribution.

    Distribution samples have the following dimension order: Batch X Samples X Data

    Data is always a single dimension and is inferred from the distribution parameters, which must not be batched.

    Batch can be any number of batch dimensions, it is inferred from the conditioning variables. For unconditional distributions the batch size
    is always 0.

    Samples can be any number of dimensions, it is manually supplied when calling Distribution.sample(). When calling Distribution.log_density() it
    is inferred as the dimensions of the supplied sample that aren't Batch or Data.

    Log_density returns the log density of each datum in a sample in a batch. Rather than reducing over the sample.
    """

    conditional = False

    def check_sample(self, sample):
        """Check that the sample matches the defined dimension of the distribution."""
        if sample.size(-1) != self.dim:
            raise ValueError(f"Sample must have dimension equal to dimensionality of distribution, found {sample.size(-1)} and {self.dim}")
        if sample.device != sample.device:
            raise ValueError(f"Sample must have device equal to device of distribution parameters, found {sample.device} and {self.device}")

    @staticmethod
    def get_batch_size(size: Tuple[int, ...], data_dims:int) -> Tuple[int, ...]:
        """Get the size of a tensor excluding the last (data_size) dimensions.

        Parameters
        ----------
        size : Tuple[int}
            The size to extract the batch dimensions from.
        data_dims : int
            The number of non-batch dimensions to extract.

        Returns
        -------
        batch_size : Tuple[int]
            The extracted batch dimensions.
        """
        return tuple(list(size)[:-data_dims])

    @staticmethod
    def _unsqueeze_to_size(parameter: Tensor, sample: Union[Tensor, int], data_dims: int = 1) -> Tensor:
        """Unsqueeze a tensor to the same number of dimensions as another.

        Parameters
        ----------
        parameter: Tensor
            The tenosr to unsqueeze.
        sample: Tensor
            The tensor of the size parameter is to be unsqueezed to.
        data_dims: data_dims
            The number of dimensions from the last to unsqueeze at.

        Returns
        -------
        unsqueezed_parameter: Tensor
            parameter unsqueezed to the required size.
        """
        if isinstance(sample, Tensor):
            new_dim = sample.dim()
        else:
            new_dim = sample
        if new_dim - parameter.dim() == 0:
            return parameter
        return multiple_unsqueeze(parameter, new_dim - parameter.dim(), -data_dims-1)

    def __init__(self, generator:torch.Generator, *args, **kwargs) -> None:
        super().__init__()
        self.dim = None
        self.generator = generator
        self.device = self.generator.device
        self.dim = 0

    def __call__(self, *args, **kwargs) -> None:
        # Do not implement a forward method for a Distribution
        pass

    def forward(self, *args, **kwargs) -> None:
        """Do not implement a forward method for a Distribution"""
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError('Sampling not implemented for this distribution')

    @abstractmethod
    def log_density(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError('Density/Mass function not implemented for this distribution')