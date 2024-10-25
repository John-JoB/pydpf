import torch
from torch.nn import Module as TorchModule
from abc import ABCMeta
from typing import Any, Callable, Tuple
import functools
from torch import Tensor


class Module(TorchModule, metaclass=ABCMeta):
    """
    Base class for all modules in pydpf.
    Includes an update method that should be called after a gradient update to update quantities derived from parameters.
    This provided to work around pytorch insisting on gradient updates being in place.
    We provide two new function decorators, 'constrained_parameter' and 'cached_property'.
    Both are used to store functions of module parameters that are expensive to compute so it is undesirable to recalculate them everytime
    they are used.

    @cached_property is used to store any intermediate value, for example the inverse of a covariance matrix. Gradient is freely passed through
    the computation of a cached_property.

    @constrained_parameter should be used to impose constraints only, the underlying data is modified inplace and without gradient. This
    provides similar functionality to pytorch's Paramatarisations API but simpler.
    """

    def __init__(self):
        self.cached_properties = {}
        self.constrained_parameters = {}
        for attr, v in self.__class__.__dict__.items():
            if isinstance(v, cached_property):
                self.cached_properties[attr] = v
            if isinstance(v, constrained_parameter):
                self.constrained_parameters[attr] = v
        super().__init__()


    def __setattr__(self, key: str, value: Any) -> None:
        #To be safe if a cached_property is set after object initialisation
        #Not sure how this would come about
        if isinstance(value, cached_property):
            self.cached_properties[key] = value
        if isinstance(value, constrained_parameter):
            self.constrained_parameters[key] = value
        super().__setattr__(key, value)


    def update(self):
        for child in self.children():
            if isinstance(child, Module):
                child.update()
        for name, property in self.cached_properties.items():
            property._update()
        for name, property in self.constrained_parameters.items():
            property._update(self)


class constrained_parameter:
    def __init__(self, function: Callable[[Module], Tuple[Tensor, Tensor]]):
        self.function = function
        functools.update_wrapper(self, function)
        self.value = None

    def __get__(self, instance: Module, owner: Any) -> Tensor:
        if self.value is None:
            self._update(instance)
        return self.value

    def _update(self, instance: Module) -> None:
        with torch.no_grad():
            d = self.function(instance)
            d[0].data = d[1].data
        self.value = d[0]


class cached_property:
    def __init__(self, function: Callable):
        self.function = function
        functools.update_wrapper(self, function)
        self.value = None

    def __get__(self, instance, owner):
        if self.value is None:
            self.value = self.function(instance)
        return self.value

    def _update(self):
        self.value = None