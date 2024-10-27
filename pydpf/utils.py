import torch
from typing import Tuple
from types import FunctionType
from functools import update_wrapper

from IPython.extensions.autoreload import update_function
from joblib.externals.cloudpickle import instance


def batched_select(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Batched analog to tensor[index].
    Use to select along the mth dimension of tensor.
    index has m dimensions , tensor has n dimensions.
    m <= n
    The size of the every dimension but the last of index must be the same as the corresponding dimension in tensor.

    Parameters
    ----------
    tensor : A1 x A2 X ... Am x B1 x B2 X ... X Bn torch.Tensor
        tensor to select from
    index : A1 x A2 x A3 X ... X Am X D torch.LongTensor
        tensor of indices to select from tensor A1.

    Returns
    -------
    output : A1 x A2 X ... X Am X D X B2 X ... X Bn torch.Tensor
    """
    if tensor.dim() == 3 and index.dim() == 2:
        #Special case common case for efficiency
        return torch.gather(input = tensor, index = index.unsqueeze(-1).expand(-1, -1, tensor.size(2)), dim=1)
    elif tensor.dim() == index.dim():
        return torch.gather(input=tensor, index=index, dim=-1)
    elif tensor.dim() > index.dim():
        index_size = index.size()
        index_dim = index.dim()
        index = index.view((*index_size, *tuple([1 for _ in range(tensor.dim() - index_dim)])))
        index = index.expand(*index_size,*tuple([tensor.size(i) for i in range(index_dim, tensor.dim())]))
        return torch.gather(input =tensor, index=index, dim=index_dim-1)
    raise ValueError('index cannot have more dimensions than tensor')


def normalise(tensor: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalise a log-space tensor to magnitude 1 along a specified dimension.
    Also return the norm.

    Parameters
    ----------
    tensor : torch.Tensor
        tensor to normalise

    dim : int
     dimension to normalise along

    Returns
    -------
    norm_tensor: torch.Tensor
        nomalised tensor

    norm: torch.Tensor
        magnitude of tensor

    """
    norm = torch.logsumexp(tensor, dim=dim, keepdim=True)
    return tensor - norm, norm

def multiple_unsqueeze(tensor: torch.Tensor, n: int, dim: int = -1) -> torch.Tensor:
    """
    Unsqueeze multiple times at the same dimension.
    Equivalent to:

    for i in range(n):
        tensor = tensor.unsqueeze(d)
    return tensor

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to unsqueeze
    n : int
        Number of times to unsqueeze the tensor
    dim : int
        Dimension to unsqueeze the tensor at

    Returns
    -------
    output : torch.Tensor
        Unsqueezed tensor
    """
    if n == 0:
        return tensor
    return tensor[(slice(None),) * dim + (None, ) * n]

class doc_function():
    """
        Reflection hack to allow functions that only define a docstring.
    """

    def __init__(self, fun):
        """
            Mark an overriding function as docstring only. I.e. the function will default to it's parent's implementation at runtime.
        """
        self.fun = fun
        self.name = fun.__name__
        update_wrapper(self, fun)

    def update_function(self, instance, owner, overrides : FunctionType):
        setattr(owner, self.name, overrides)
        if instance is not None:
            return getattr(instance, self.name)
        #For static methods
        return getattr(owner, self.name)

    def __get__(self, instance, owner):
        if owner is None:
            raise RuntimeError(f'Attempting to use doc_function outside of a class.')
        bases = owner.__mro__
        for base in bases[1:]:
            try:
                overrides = base.__dict__[self.name]
                return self.update_function(instance, owner, overrides)
            except KeyError:
                pass
        raise AttributeError(f'Base classes have no attribute {self.name}.')