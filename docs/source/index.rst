PyDPF
=====

Installation
============
PyDPF can be be installed by running:

.. code-block::
    pip install pydpf

PyDPF basics
============

Modules
-------
PyDPF includes it’s own Module class that extends torch.nn.Module that we find useful
in defining custom parameterised probability distributions. We include the following two
Property-like environments.
cached property: Used to cache the results of functions of the parameters. For example if we
need to repeatedly use the matrix inverse of a parameter. Gradients can be passed through
the transform that creates the cached_property. Cached properties can be stacked.
constrained parameter: Used to constrain parameters. constrained_parameter applies a
transform in-place from an unconstrained parameter to a parameter satisfying the required
constraint. Because this is performed in-place, gradient tracking is not supported. This is
intended to be used to guard against parameters entering disallowed regions, for example
keeping a variance positive. It is expected that the transform, should in most cases, make at
most a small adjustment to the parameter and there should be a region for which the transform
leaves the parameter unchanged that includes the parameter at convergence. Because the
modifications are made in-place constrained_parameter objects cannot depend on other
constrained_parameter objects nor cached_property objects. If it is desired to constrain
functions of the parameters rather than the parameters themselves, we point the reader towards
PyTorch’s parametrize API which provides similar functionality but out-of-place.
We provide the following minimal example that shows how one might implement a PyDPF
module that evaluates the probability density function of a Gaussian where the mean and
variance are parameters.

.. code-block::
    class GaussianDensity(pydpf.Module):

        log_2pi = log(2*pi)

        def __init__(self, initial_mean, initial_variance):
            super().__init__()
            self.mean = torch.nn.parameter.Parameter(torch.tensor(initial_mean))
            self.variance_data = torch.nn.parameter.Parameter(torch.tensor(initial_variance))
            print(self.variance)

        @pydpf.constrained_parameter
        def variance(self):
            #Return references to the parameter we want to modify in place and
            #to a tensor containing the new value
            return self.variance_data, torch.abs(self.variance_data)

        @pydpf.cached_property
            def inverse_variance(self):
            return 1/self.variance

        @pydpf.cached_property
        def log_variance(self):
            return torch.log(self.variance)

        def log_density(self, input):
            sqd_residual = (input - self.mean)**2
            return -(sqd_residual*self.inverse_variance + self.log_2pi + self.log_variance)/2

PyTorch does not provide a way to detect optimiser steps so we have to manually update the
model by calling .update() on the highest level Module in a model any time the parameters may
have changed. For this reason, if any sub-modules in a model have either a cached_property
or a constrained_parameter the top-level module should be a pydpf.Module, however any
sub-module can be a torch.nn.Module without consequence.

PyDPF data categories
---------------------
The intended usage for PyDPF is for the user to create their own models and algorithms, as
pydpf.Module objects, and define custom functions to interface with those that the package
provides. Therefore, we need a schema for the different variables that can be passed from
the base filtering algorithms to user defined functions. We define this schema below. PyDPF
assumes a batch-sequential paradigm, with additional dimensions for each draw, known in
the SMC literature as a particle, in a sample and the dimensionality of the distribution.
Tensors handled and returned by PyDPF functions may be at most of size (T × B × K × Di)
corresponding to (time-step × batch × particle × intrinsic-dimension). Frequently, one or
more of these dimensions will not be present, in which case ordering is maintained. When
we pass the data as arguments to user-defined functions we index a single time-step so the
data-types and dimensions given in Table 1 are as the user-defined functions will receive. In
PyDPF all arguments are passed by keyword so unneeded arguments received from a PyDPF
call to a user-defined function can neatly be grouped into a **dictionary object.

state - The particle estimates of the latent state of the state-space system at the current time-step - Tensor (B × K × Ds)

weight - The log weights of the particles, entries aligned to state - Tensor (B × K)

prev_state - The particle estimates of the latent state of the state-space system at the previous time-step - Tensor (B × K × Ds)

observation - The observations of the state-space system at the current time-step - Tensor (B × Do)

control - Control actions or other exogenous variables at the current time-step - Tensor (B × Dc)

time - The time the current time-step occurs at - Tensor (B)

prev_time - The time of the previous time-step - Tensor (B)

series_metadata - Exogenous variables that are constant for a given trajectory - Tensor (B × Dm)

t - The index of the time-step - Integer


PyDPF deserialisation and data loading
--------------------------------------

When passed to a PyDPF filtering algorithm all the data should be supplied as torch Tensors
in the format given in Table 1. prev_state, prev_time and t are calculated automatically
so don’t need to be passed, and all categories apart from series_metadata should have a
time-step dimension.
However, for convenience we provide methods to load data from files, obeying a certain format,
into a map-style torch.utils.data.Dataset object and therefore be accessed easily from a
torch.utils.data.DataLoader. We allow one of two data storage formats, either storing
the entire dataset in a single .csv file, or storing each trajectory in separate files {1.csv,
2.csv, ..., T.csv} in a dedicated directory. The .csv files are formed of headed columns
there must be at least one observation column, with state, time, and control columns
being optional. As all the data categories, apart from time, are vector valued there can be
multiple columns for each category. For the single-file format there must be additionally a
series_id column that will be used to index each trajectory, for the multiple file format the
series_id is encoded in the file name.
The data category series_metadata exists to store exogenous variables that the trajectories
might depend on, but are constant over a trajectory. These are to be stored in a separate
.csv indexed by a series_id column.
Given a file in the required format, loading a dataset is simple: call pydpf.StateSpaceDataset
with the data’s path, the column labels and the device to store data retrieved by the data
loader. When initialising the data loader, it is crucial that the argument collate_fn is set to
dataset.collate where dataset is the dataset passed to the data loader. PyTorch’s default
collate function will not return the data in a format that obeys PyDPF conventions. When
looping over the data loader, data is returned as tuple in the ordering state - observation -
time - control - series_metadata with only the field that exist being returned.

.. code-block::
    dataset = pydpf.StateSpaceDataset(data_path=data_path,
                                        series_id_column='series_id',
                                        state_prefix='state',
                                        observation_prefix='observation',
                                        device='cpu')
    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=10,
                                                shuffle=True,
                                                collate_fn=dataset.collate)

Reproducibility
---------------
PyTorch does not provide the fine-grained tracking of pseudo-random state that competing
numerical libraries such as JAX do. Our solution, that we use for all
built-in implementations with pseudo-random operations, and that we recommend the user
adopt for all their extensions, is to initialise a random generator per Module that is used to
control all random operations used within that Module.

Some torch CUDA operations are non-deterministic by default, see the documentation of
torch.use_deterministic_algorithms8 for detail ( https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html ). This non-determinacy is at the or-
der of precision over a single operation, but our tests showed it can result in a significant variance over the the course of a full forward pass. We provide a context manager
pydpf.utils.set_deterministic_mode that set the environment variable
"CUBLAS_WORKSPACE_CONFIG" = ":4096:8" and calls
torch.use_deterministic_algorithms(True) before reverting to default settings on context
exit. Under this context we expect an increase in the time and memory costs for CUDA
operations compared to the non-deterministic implementations.
Note, however, that several of the implemented DPFs rely on the torch.cumsum() operation
that is never guaranteed to be deterministic. Despite this, in our experiments we observe
that the results are consistent on our set-up. Furthermore, PyTorch does not guarantee reproducibility across differing hardware, PyTorch versions or versions of upstream dependencies
such as CUDA. For this reason we repeat all our experiments across several random seeds to
mitigate some of this unavoidable variance.

Documentation Index
===================

.. toctree::
   :maxdepth: 4

   pydpf
