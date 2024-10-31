from typing import Union, Callable, Any, Tuple
from torch import Tensor

type Resampler = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
type Aggregation = Callable[[Tensor, Tensor, Tensor, Tensor, int], Tensor]
type ImportanceKernelLikelihood = Callable[[Tensor, Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor]]
type ImportanceSamplerLikelihood = Callable[[int, Tensor], Tuple[Tensor, Tensor, Tensor]]
type ImportanceKernel = Callable[[Tensor, Tensor, Tensor, int], Tuple[Tensor, Tensor]]
type ImportanceSampler= Callable[[int, Tensor], Tuple[Tensor, Tensor]]