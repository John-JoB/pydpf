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
    These provide similar functionality to pytorch's parameterization API, but with simpler to use for code that has a lot of custom modules.

    @cached_property is used to store any intermediate value, for example the inverse of a covariance matrix. Gradient is freely passed through
    the computation of a cached_property.

    @constrained_parameter should be used to impose constraints only, the underlying data is modified inplace and without gradient.

    Notes
    -----
    A use case not covered by @cached_property and @constrained_parameter, is if the wrapped function should be computed out-of-place, such as to
    allow gradient tracking, but appear as if the change is made in-place. I.e. we want to modify a parameter from a function declared outside the
    module to which it belongs, and pass gradient through this map. This is the intended use case of the torch.parameterization API so use that
    instead.
    """

    def __init__(self):
        self.updatable = True
        self.cached_properties = {}
        self.constrained_parameters = {}
        #Need to iterate over __class__.__dict__ rather than dir(self) to by pass getattr()
        for attr, v in self.__class__.__dict__.items():
            if isinstance(v, cached_property):
                self.cached_properties[attr] = v
            if isinstance(v, constrained_parameter):
                self.constrained_parameters[attr] = v
        self.disallow_set_values = False
        super().__init__()


    def __setattr__(self, key: str, value: Any) -> None:

        if key.endswith('_value') and self.disallow_set_values:
            raise AttributeError('Attributes ending with _value are reserved')
        # To be safe if a cached_property is set after object initialisation
        # Not sure how this would come about
        if isinstance(value, cached_property):
            self.cached_properties[key] = value
        if isinstance(value, constrained_parameter):
            self.constrained_parameters[key] = value
        super().__setattr__(key, value)

    def _update(self):
        for name, property in self.cached_properties.items():
            property._update(self)
        for name, property in self.constrained_parameters.items():
            property._update(self)

    def update(self):
        """
        Update all constrained_parameters and cached_properties belonging to this Module.
        """
        self._update()
        for child in self.modules():
            if isinstance(child, Module):
                child._update()

class constrained_parameter:


    def __init__(self, function: Callable[[Module], Tuple[Tensor, Tensor]]):
        """
            Wrapper for constraining parameters.
            The wrapped function must belong to a Module and should take only a reference to its parent Module and return a reference to the original
            parameter and a tensor containing the new value.

            constrained_parameter applies the change in-place; the underlying data of the parameter is modified. Necessarily, therefore this is
            done without gradient tracking.

            The constraint is applied on calling Module.update(), so you should do this after every gradient update to preserve the constraint.

            Notes
            -----
            constrained_parameters can be applied to any parameter accessible from a Module including as attributes of a child Module.
            This is safe as the parameter is changed in-place.
        """
        self.function = function
        functools.update_wrapper(self, function)
        self.value_name = f'{self.function.__name__}_value'

    def __get__(self, instance: Module, owner: Any) -> Tensor:
        if not hasattr(instance, self.value_name):
            self._update(instance)
        return getattr(instance, self.value_name)

    def _update(self, instance: Module) -> None:
        #Changing tensors in place inside inference mode can cause runtime errors.
        @torch.inference_mode(mode=False)
        def f():
            with torch.no_grad():
                d = self.function(instance)
                d[0].data = d[1]
            return d[0]

        v = f()
        instance.disallow_set_values = False
        setattr(instance, self.value_name, v)
        instance.disallow_set_values = True

    def __set__(self):
        raise AttributeError("Cannot directly set a constrained parameter.")


class cached_property:
    def __init__(self, function: Callable):
        """
            Wrapper for caching functions of parameters.
            The wrapped function must belong to a Module and should take only a reference to its parent Module and tensor containing the new value
            that will be stored.

            cached_property applies its map out-of-place, creating a new tensor. Gradient tracking is permitted through the map.

            The cached_property is calculated lazily; when it is first accessed it is calculated and cached.
            Calling Module.update() resets the value so that it will be recomputed on next access.
        """
        self.function = function
        functools.update_wrapper(self, function)
        self.value_name = f'{self.function.__name__}_value'

    def __get__(self, instance, owner):
        if not hasattr(instance, self.value_name):
            v = self.function(instance)
            instance.disallow_set_values = False
            setattr(instance, self.value_name, v)
            instance.disallow_set_values = True
        return getattr(instance, self.value_name)

    def _update(self, instance):
        if hasattr(instance, self.value_name):
            delattr(instance, self.value_name)

    def __set__(self, instance):
        raise AttributeError("Cannot directly set a cached property, update the underlying data instead.")