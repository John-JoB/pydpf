from torch.nn import Module as TorchModule
from abc import ABCMeta

class Module(TorchModule, metaclass=ABCMeta):
    """
    Base class for all modules in pydpf.
    Includes an update module that should be called after a gradient update to update quantities derived from parameters
    """

    def __init__(self):
        super().__init__()

    def update(self):
        for child in self.children():
            if isinstance(child, Module):
                child.update()
        pass