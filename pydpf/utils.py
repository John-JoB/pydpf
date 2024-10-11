import torch
from typing import Tuple


def batched_select(tensor: torch.Tensor, index: torch.LongTensor) -> torch.Tensor:
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