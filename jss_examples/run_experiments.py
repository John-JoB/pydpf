#!/usr/bin/env python
"""Master script for the pydpf JSS supplementary material.

This single, self-contained file reproduces every experiment that is otherwise
spread across the notebooks/scripts in the ``jss_examples`` sub-folders:

    * kalman        -- "Linear Gaussian / Comparison with the Kalman filter"
    * proposal      -- "Linear Gaussian / Learning proposal parameters"
    * sv_filtering  -- "Stochastic Volatility / Performing filtering given a fully specified model"
    * sv_single     -- "Stochastic Volatility / Unsupervised learning of a single parameter"
    * sv_multiple   -- "Stochastic Volatility / Unsupervised learning of multiple parameters"
    * maze          -- "Deep mind maze / Deep Learning"
    * example_usage -- "Stochastic Volatility / example_usage.py" (a short demonstration)

All of the auxiliary modules that used to live next to the notebooks
(``lg_model``, ``sv_model``, ``sv_training_loop``, ``dm_model``,
``dm_neural_networks``, ``dm_training`` and the various ``*_setup`` scripts) have
been folded into this one file so that it can be submitted as a single script.

Data is read from a single central folder ``jss_examples/data`` and every result
is written to ``jss_examples/results``.  Both folders are created and populated
on demand: simulated data sets are regenerated (or copied from the original
sub-folders if already present), and the maze data set is downloaded if missing.

Run everything with::

    python run_experiments.py

Run a subset of experiments and/or a subset of methods, e.g. only the
stop-gradient and soft DPFs of the two stochastic-volatility filtering tasks::

    python run_experiments.py --experiment sv_filtering sv_single --models "Soft" "Stop-Gradient"

See ``python run_experiments.py --help`` for the full list of options.
"""

import argparse
import datetime
import math
import os
import pathlib
import shutil
import time
import traceback
from copy import deepcopy, copy
from math import ceil, sqrt

import warnings
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

_original = warnings.formatwarning

def _orange(message, category, filename, lineno, line=None):
    return ORANGE + _original(message, category, filename, lineno, line) + RESET

warnings.formatwarning = _orange



# Avoid the libiomp double-load crash that can occur on Windows + MKL/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.init import (
    _calculate_fan_in_and_fan_out,
    kaiming_uniform_,
    uniform_,
)

import pydpf

# --------------------------------------------------------------------------- #
#  Paths / constants                                                          #
# --------------------------------------------------------------------------- #

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# The differentiable particle filters available in (most) experiments.
DPF_METHODS = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient",
               "Optimal Transport", "Kernel"]


# --------------------------------------------------------------------------- #
#  Shared helpers (previously duplicated across the *_setup / training files)  #
# --------------------------------------------------------------------------- #

def make_new_csv(rows, columns, name, string_columns=(), overwrite=False):
    """Create an empty results table (indexed by ``method``).

    Numeric columns are left empty (NaN); any column named in ``string_columns``
    is initialised with blank strings so that it is an object/string column
    rather than a float column. ``overwrite=True`` recreates the file even if it
    already exists.

    Note: a CSV cannot store dtypes, and a default ``pd.read_csv`` parses empty
    cells as NaN, so a *completely empty* template still loads as all-float. Once
    at least one real value is written to the string column (or when reading with
    ``keep_default_na=False``) the column loads with string dtype.
    """
    if name.exists() and not overwrite:
        return
    df = pd.DataFrame(index=pd.Index(rows, name="method"), columns=columns)
    for col in string_columns:
        df[col] = ""
    df.to_csv(name)


def _get_split_amounts(split, data_length):
    """Convert a (train, val, test) ratio into integer counts that sum to N."""
    split_sum = sum(split)
    s = [0] * 3
    s[0] = int(split[0] * data_length / split_sum)
    s[1] = int(split[1] * data_length / split_sum)
    s[2] = data_length - s[0] - s[1]
    if s[0] < 1:
        raise ValueError("Trying to assign too small a fraction to the train set")
    if s[1] < 1:
        raise ValueError("Trying to assign too small a fraction to the validation set")
    if s[2] < 1:
        raise ValueError("Trying to assign too small a fraction to the test set")
    return s


def fractional_diff_exp(a, b):
    """|1 - exp(b - a)|, used to compare (log) likelihood factors."""
    return torch.abs(1 - torch.exp(b - a))


def sync(device):
    """Synchronise CUDA so that timings are accurate; a no-op on CPU."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def build_dpf(method, SSM, generator, *, soft_softness=None, ot_regularisation=0.5,
              ot_clip=None, kernel_factory=None):
    """Single factory replacing the near-identical ``get_DPF`` helpers.

    ``kernel_factory`` is a callable ``generator -> kernel`` used only for the
    ``"Kernel"`` method, because the kernel differs (dimension / learnability)
    between experiments.
    """
    if method == "DPF":
        return pydpf.DPF(SSM=SSM, resampling_generator=generator)
    if method == "Soft":
        kw = {} if soft_softness is None else {"softness": soft_softness}
        return pydpf.SoftDPF(SSM=SSM, resampling_generator=generator, **kw)
    if method == "Stop-Gradient":
        return pydpf.StopGradientDPF(SSM=SSM, resampling_generator=generator)
    if method == "Marginal Stop-Gradient":
        return pydpf.MarginalStopGradientDPF(SSM=SSM, resampling_generator=generator)
    if method == "Optimal Transport":
        kw = {"regularisation": ot_regularisation}
        if ot_clip is not None:
            kw["transport_gradient_clip"] = ot_clip
        return pydpf.OptimalTransportDPF(SSM=SSM, **kw)
    if method == "Kernel":
        if kernel_factory is None:
            raise ValueError("A kernel_factory is required to build a KernelDPF")
        return pydpf.KernelDPF(SSM=SSM, kernel=kernel_factory(generator))
    raise ValueError("method should be one of the allowed options")


# ===========================================================================
#  Linear-Gaussian model components  (was lg_model.py)
# ===========================================================================

class GaussianDynamic(pydpf.Module):
    def __new__(cls, dx: int, generator):
        device = generator.device
        dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1)
                                            - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        dynamic_offset = torch.zeros(dx, device=device)
        return pydpf.LinearGaussian(weight=dynamic_matrix, bias=dynamic_offset,
                                    cholesky_covariance=torch.eye(dx, device=device),
                                    generator=generator)


class GaussianObservation(pydpf.Module):
    def __new__(cls, dx: int, dy: int, generator):
        device = generator.device
        observation_matrix = torch.zeros((dy, dx), device=device)
        for i in range(dy):
            observation_matrix[i, i] = 1
        observation_offset = torch.zeros(dy, device=device)
        return pydpf.LinearGaussian(weight=observation_matrix, bias=observation_offset,
                                    cholesky_covariance=torch.eye(dy, device=device),
                                    generator=generator)


class GaussianPrior(pydpf.Module):
    def __new__(cls, dx: int, generator):
        device = generator.device
        return pydpf.MultivariateGaussian(torch.zeros(dx, device=device),
                                          torch.eye(dx, device=device), generator=generator)


class GaussianOptimalProposal(pydpf.Module):
    def __init__(self, dx: int, dy: int, generator):
        super().__init__()
        device = generator.device
        covariance = torch.eye(dx, device=device)
        self.dx = dx
        self.dy = dy
        for i in range(dy):
            covariance[i, i] = .5
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1)
                                                - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device),
                                               cholesky_covariance=torch.sqrt(covariance),
                                               generator=generator)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:, :, :self.dy] = (mean[:, :, :self.dy] + observation.unsqueeze(1)) / 2
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:, :, :self.dy] = (mean[:, :, :self.dy] + observation.unsqueeze(1)) / 2
        sample = state - mean
        return self.dist.log_density(sample)


class GaussianLearnedProposal(pydpf.Module):
    def __init__(self, dx: int, dy: int, generator):
        super().__init__()
        device = generator.device
        cov = torch.nn.Parameter(torch.eye(dx, device=device))
        self.dx = dx
        self.dy = dy
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1)
                                                - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.x_weight = torch.nn.Parameter(torch.ones(dx, device=device))
        self.y_weight = torch.nn.Parameter(torch.zeros(dy, device=device))
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device),
                                               cholesky_covariance=cov, generator=generator,
                                               diagonal_cov=True)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1) * self.x_weight
        mean[:, :, :self.dy] = mean[:, :, :self.dy] + observation.unsqueeze(1) * self.y_weight
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1) * self.x_weight
        mean[:, :, :self.dy] = mean[:, :, :self.dy] + observation.unsqueeze(1) * self.y_weight
        sample = state - mean
        return self.dist.log_density(sample)


def lg_make_components(dx, dy, generator, proposal=None):
    """Build the linear-Gaussian SSM components.

    ``proposal`` is ``None`` (no proposal), ``"optimal"`` or ``"learned"``.
    """
    dynamic_model = GaussianDynamic(dx, generator)
    observation_model = GaussianObservation(dx, dy, generator)
    prior_model = GaussianPrior(dx, generator)
    if proposal is None:
        return prior_model, dynamic_model, observation_model
    if proposal == "optimal":
        proposal_model = GaussianOptimalProposal(dx, dy, generator)
    elif proposal == "learned":
        proposal_model = GaussianLearnedProposal(dx, dy, generator)
    else:
        raise ValueError("proposal must be None, 'optimal' or 'learned'")
    return prior_model, dynamic_model, observation_model, proposal_model


# ===========================================================================
#  Stochastic-volatility model components  (was sv_model.py)
# ===========================================================================

class StochasticVolatility_Prior(pydpf.Module):
    @pydpf.cached_property
    def sd(self):
        i1 = torch.ones((1, 1), device=self.alpha.device)
        return torch.sqrt(i1 * (self.sigma ** 2 / (1 - self.alpha ** 2)))

    @pydpf.constrained_parameter
    def alpha(self):
        return self.alpha_, torch.clip(self.alpha_, 1e-2, 1 - 1e-2)

    def __init__(self, sigma, alpha, generator):
        super().__init__()
        self.sigma = sigma
        self.alpha_ = alpha
        i1 = torch.ones((1, 1), device=generator.device)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(1, device=generator.device),
                                               cholesky_covariance=i1, generator=generator)

    def sample(self, batch_size: int, n_particles: int, **data):
        return self.dist.sample(sample_size=(batch_size, n_particles)) * self.sd

    def log_density(self, state, **data):
        return self.dist.log_density(sample=state / self.sd) - torch.log(self.sd)


class StochasticVolatility_Dynamic(pydpf.Module):
    def __new__(cls, sigma, alpha, generator):
        return pydpf.LinearGaussian(weight=alpha, bias=torch.zeros(1, device=generator.device),
                                    cholesky_covariance=sigma, generator=generator)


class StochasticVolatility_Observation(pydpf.Module):
    @pydpf.constrained_parameter
    def beta(self):
        return self.beta_, torch.clip(self.beta_, 1e-3)

    def __init__(self, beta, generator):
        super().__init__()
        self.beta_ = beta
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(1, device=generator.device),
                                               cholesky_covariance=torch.ones((1, 1), device=generator.device),
                                               generator=generator)

    def sample(self, state, **data):
        sample = self.dist.sample((state.size(0), state.size(1)))
        return sample * torch.exp(state / 2) * self.beta

    def fitness(self, observation, state, **data):
        sd = torch.exp(state / 2) * self.beta
        # No convenient way to disallow very small volatilities; clip for stability.
        sd = torch.clip(sd, 1e-7)
        return self.dist.log_density(observation.unsqueeze(1) / sd) - torch.log(sd).squeeze()


def sv_make_SSM(sigma, alpha, beta, device, generator=None):
    if generator is None:
        generator = torch.Generator(device).manual_seed(0)
    return pydpf.FilteringModel(prior_model=StochasticVolatility_Prior(sigma, alpha, generator),
                                dynamic_model=StochasticVolatility_Dynamic(sigma, alpha, generator),
                                observation_model=StochasticVolatility_Observation(beta, generator))


# ===========================================================================
#  Deep-mind maze neural networks  (was dm_neural_networks.py)
#
#  The authors thank Xiongjie Chen for kindly providing the code for his paper
#  'Normalizing Flow-based Differentiable Particle Filters' which the networks
#  below are heavily based on.
# ===========================================================================

class SeedableConv2D(torch.nn.Conv2d):
    """Conv2d that can be initialised from a torch.Generator."""

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(self.bias, -bound, bound, generator=generator)


class SeedableLinear(torch.nn.Linear):
    """Linear that can be initialised from a torch.Generator."""

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_(self.bias, -bound, bound, generator=generator)


class SeedableConvTranspose2D(torch.nn.ConvTranspose2d):
    """ConvTranspose2d that can be initialised from a torch.Generator."""

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(self.bias, -bound, bound, generator=generator)


class SeedableDropoutNd(pydpf.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False,
                 generator: torch.Generator = torch.default_generator) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
        self.generator = generator

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class SeedableDropout(SeedableDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        mask = (torch.rand(input.size(), device=input.device, dtype=torch.float32,
                           generator=self.generator) > self.p).to(input.dtype)
        return input.multiply_(mask) if self.inplace else input * mask


class SeedableDropout2d(SeedableDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        if input.dim() != 4:
            raise ValueError("SeedableDropout2d only supports Batch,Channel,Height,Width tensors")
        mask = pydpf.multiple_unsqueeze(
            (torch.rand((input.size(0), input.size(1)), device=input.device, dtype=torch.float32,
                        generator=self.generator) > self.p).to(input.dtype), 2, -1)
        return input.multiply_(mask) if self.inplace else input * mask


class FCNN(pydpf.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, generator):
        super().__init__()
        self.network = torch.nn.Sequential(
            SeedableLinear(in_dim, hidden_dim, generator=generator, device=generator.device),
            torch.nn.Tanh(),
            SeedableLinear(hidden_dim, hidden_dim, generator=generator, device=generator.device),
            torch.nn.Tanh(),
            SeedableLinear(hidden_dim, out_dim, generator=generator, device=generator.device),
        )

    def forward(self, x):
        return self.network(x)


class ObservationEncoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio=0.7, generator=torch.default_generator):
        return torch.nn.Sequential(  # input: 3*24*24
            SeedableConv2D(3, 16, kernel_size=4, stride=2, padding=1, bias=False, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            SeedableConv2D(16, 32, kernel_size=4, stride=2, padding=1, bias=False, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            SeedableConv2D(32, 64, kernel_size=4, stride=2, padding=1, bias=False, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
            torch.nn.Flatten(),
            SeedableDropout(p=1 - dropout_keep_ratio, generator=generator),
            SeedableLinear(64 * 3 * 3, hidden_size, generator=generator, device=generator.device),
        )


class ObservationDecoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio=0.7, generator=torch.default_generator):
        return torch.nn.Sequential(
            SeedableLinear(hidden_size, 3 * 3 * 64, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.Unflatten(-1, (64, 3, 3)),
            SeedableConvTranspose2D(64, 32, kernel_size=4, padding=1, stride=2, bias=False, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            SeedableConvTranspose2D(32, 16, kernel_size=4, padding=1, stride=2, bias=False, generator=generator, device=generator.device),
            SeedableDropout2d(p=1 - dropout_keep_ratio, generator=generator),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            SeedableConvTranspose2D(16, 3, kernel_size=4, padding=1, stride=2, bias=False, generator=generator, device=generator.device),
            SeedableDropout2d(p=1 - dropout_keep_ratio, generator=generator),
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid(),
        )


class StateEncoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio=0.7, generator=torch.default_generator):
        return torch.nn.Sequential(
            SeedableLinear(4, 16, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableLinear(16, 32, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableLinear(32, 64, generator=generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableDropout(p=1 - dropout_keep_ratio, generator=generator),
            SeedableLinear(64, hidden_size, generator=generator, device=generator.device),
        )


class RealNVP_cond(pydpf.Module):
    def __init__(self, dim, hidden_dim=8, base_network=FCNN, condition_on_dim=None,
                 generator=torch.default_generator, zero_i=False):
        super().__init__()
        self.dim = dim
        self.condition_on_dim = condition_on_dim
        self.t1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, generator)
        self.t2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim, generator)
        self.generator = generator
        if zero_i:
            self.zero_initialization()

    def zero_initialization(self, std=0.01):
        for layer in self.t1.network:
            if layer.__class__.__name__ == "SeedableLinear":
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__ == "SeedableLinear":
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)

    def forward(self, x, condition_on):
        lower, upper = x[..., :self.dim // 2], x[..., self.dim // 2:]
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        upper = t1_transformed + upper
        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        lower = t2_transformed + lower
        z = torch.cat([lower, upper], dim=-1)
        return z, 0

    def inverse(self, z, condition_on):
        lower, upper = z[..., :self.dim // 2], z[..., self.dim // 2:]
        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        lower = lower - t2_transformed
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        upper = upper - t1_transformed
        x = torch.cat([lower, upper], dim=-1)
        return x, 0


class NormalizingFlowModel_cond(pydpf.Module):
    def __init__(self, prior, flows, device="cuda:0"):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = torch.nn.ModuleList(flows).to(self.device)

    def forward(self, x, condition_on):
        b, m, d = x.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition_on)
            log_det += ld
        return x, log_det

    def inverse(self, z, condition_on):
        b, m, d = z.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition_on)
            log_det += ld
        return z, log_det

    def log_density(self, x, condition_on):
        z, log_det = self.forward(x, condition_on)
        return self.prior.log_density(z) + log_det

    def sample(self, sample_size, condition_on):
        z = self.prior.sample(sample_size, device=self.device)
        return self.inverse(z, condition_on)[0]


# ===========================================================================
#  Deep-mind maze model components  (was dm_model.py)
# ===========================================================================

class MazePrior(pydpf.Module):
    def __init__(self, width, height, generator):
        super().__init__()
        self.size_tensor = torch.tensor([width, height, torch.pi * 2], device=generator.device)
        self.generator = generator

    def sample(self, n_particles, batch_size, **data):
        return (torch.rand((batch_size, n_particles, 3), generator=self.generator,
                           device=self.generator.device) - 0.5) * self.size_tensor[None, None, :]


class MazeDynamic(pydpf.Module):
    def __init__(self, generator, cov):
        super().__init__()
        self.generator = generator
        self.dist = pydpf.MultivariateGaussian(torch.zeros(3, device=generator.device), cov,
                                               generator=generator, diagonal_cov=True)

    def deteriministic_action(self, prev_state, control):
        angle_i = prev_state[:, :, 2]
        c = torch.cos(angle_i)
        s = torch.sin(angle_i)
        rotation_matrix = torch.stack([torch.stack([c, -s], dim=-1),
                                       torch.stack([s, c], dim=-1)], dim=-2)
        new_pos = torch.einsum('bnij, bj -> bni', rotation_matrix, control[:, :2]) + prev_state[:, :, :2]
        new_angle = prev_state[:, :, 2:3] + control[:, None, 2:3]
        return torch.concat([new_pos, new_angle], dim=-1)

    def sample(self, prev_state, control, **data):
        return self.deteriministic_action(prev_state, control) + self.dist.sample(
            sample_size=(prev_state.shape[0], prev_state.shape[1]))

    def log_density(self, prev_state, control, state, **data):
        return self.dist.log_density(state - self.deteriministic_action(prev_state, control))


class MazeObservation(pydpf.Module):
    def __init__(self, flow_model, encoder, decoder, state_encoder, device=torch.device("cpu")):
        super().__init__()
        self.flow_model = flow_model
        self.encoder = encoder
        self.scaling_tensor = torch.tensor([[[1., 1., torch.pi]]], device=device)
        self.decoder = decoder
        self.state_encoder = state_encoder

    def fitness(self, state, observation, t, **data):
        b, n, _ = state.shape
        c = torch.cos(state[:, :, 2:3])
        s = torch.sin(state[:, :, 2:3])
        encoded_state = self.state_encoder(torch.concat([state[:, :, :2], c, s], dim=-1))
        return self.flow_model.log_density(observation.unsqueeze(1).expand(-1, n, -1), encoded_state)


def bind_angle(angle):
    """Wrap an angle to (-pi, pi]; used in maze training/evaluation."""
    bound_angle = torch.remainder(angle, 2 * torch.pi)
    return torch.where(bound_angle > torch.pi, bound_angle - 2 * torch.pi, bound_angle)


# ===========================================================================
#  Stochastic-volatility / maze training loops  (was sv_training_loop.py,
#  dm_training.py).  They differ enough (image autoencoder, angle losses,
#  controls) that they are kept separate, but share ``_get_split_amounts``.
# ===========================================================================

def train_sv(dpf, opt, dataset, epochs, n_particles, batch_size, split_size,
             likelihood_scaling=1., data_loading_generator=torch.default_generator,
             gradient_regulariser=None, target="MSE", time_extent=None, lr_scheduler=None):
    """Generic SSM training loop for the stochastic-volatility experiments."""
    batch_size = list(batch_size)
    aggregation_function = {"MSE": pydpf.MSE_Loss(), "ELBO": pydpf.ElBO_Loss()}

    data_length = len(dataset)
    split = _get_split_amounts(split_size, data_length)
    train_set, validation_set, test_set = torch.utils.data.random_split(
        dataset, split, generator=data_loading_generator)
    for k, subset in enumerate((train_set, validation_set, test_set)):
        if batch_size[k] == -1 or batch_size[k] > len(subset):
            batch_size[k] = len(subset)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size[0], shuffle=True,
                                               generator=data_loading_generator, collate_fn=dataset.collate)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size[1], shuffle=False,
                                                    generator=data_loading_generator, collate_fn=dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size[2], shuffle=False,
                                              generator=data_loading_generator, collate_fn=dataset.collate)
    best_eval = torch.inf
    dpf.update()
    best_dict = deepcopy(dpf.state_dict())
    if time_extent is None:
        time_extent = dataset.observation.size(0) - 1
    for epoch in range(epochs):
        train_loss = []
        total_size = 0
        dpf.train()
        for state, observation in train_loader:
            dpf.update()
            opt.zero_grad()
            loss = dpf(n_particles[0], time_extent, aggregation_function, observation=observation,
                       ground_truth=state, gradient_regulariser=gradient_regulariser)
            loss = torch.mean(loss["ELBO"]) * likelihood_scaling + (1 - likelihood_scaling) * torch.mean(loss["MSE"])
            loss.backward()
            for p in dpf.parameters():
                torch.clamp_(p.grad, -1., 1.)
            train_loss.append(loss.item() * state.size(1))
            opt.step()
            total_size += state.size(1)
        train_loss = np.sum(np.array(train_loss)) / total_size
        if lr_scheduler is not None:
            lr_scheduler.step()
        dpf.update()
        dpf.eval()
        with torch.inference_mode():
            total_size = 0
            validation_MSE = []
            validation_ELBO = []
            for state, observation in validation_loader:
                loss = dpf(n_particles[1], time_extent, aggregation_function, observation=observation,
                           ground_truth=state)
                validation_MSE.append(torch.mean(loss["MSE"]).item() * state.size(1))
                validation_ELBO.append(torch.sum(loss["ELBO"]).item() * state.size(1))
                total_size += state.size(1)
            validation_MSE = np.sum(np.array(validation_MSE)) / total_size
            validation_ELBO = np.sum(np.array(validation_ELBO)) / total_size

        if np.isnan(validation_MSE) or np.isnan(validation_ELBO):
            dpf.load_state_dict(best_dict)
            continue
        if target == "MSE":
            if validation_MSE < best_eval and not np.isnan(validation_MSE):
                best_eval = validation_MSE
                best_dict = deepcopy(dpf.state_dict())
        else:
            if validation_ELBO < best_eval and not np.isnan(validation_ELBO):
                best_eval = validation_ELBO
                best_dict = deepcopy(dpf.state_dict())

        print(f"epoch {epoch + 1}/{epochs}, train loss: {train_loss}, "
              f"validation MSE: {validation_MSE}, validation ELBO: {-validation_ELBO}")
    total_size = 0
    with torch.inference_mode():
        test_MSE = []
        test_ELBO = []
        dpf.load_state_dict(best_dict)
        for state, observation in test_loader:
            loss = dpf(n_particles[1], time_extent, aggregation_function, observation=observation,
                       ground_truth=state)
            test_MSE.append(torch.mean(loss["MSE"]).item() * state.size(1))
            test_ELBO.append(torch.sum(loss["ELBO"]).item() * state.size(1))
            total_size += state.size(1)
    test_MSE = np.sum(np.array(test_MSE)) / total_size
    test_ELBO = np.sum(np.array(test_ELBO)) / total_size
    print("")
    print(f"test MSE: {test_MSE}, test ELBO: {-test_ELBO}")
    return test_MSE, -test_ELBO


def train_maze(dpf, opt, dataset, epochs, n_particles, batch_size, split_size,
               scalings=(1., 1., 1.), data_loading_generator=torch.default_generator,
               gradient_regulariser=None, target="MSE", time_extent=None, lr_scheduler=None,
               pre_train_epochs=0, device=torch.device("cuda:0"), state_scaling=1000.):
    """Training loop for the deep-mind maze experiment (images + controls)."""
    batch_size = list(batch_size)
    position_scaling = torch.tensor([[[state_scaling, state_scaling]]], device=device)

    aggregation_function = {"Mean Pose": pydpf.FilteringMean(
        lambda state: torch.concat([state[..., :2], torch.sin(state[..., 2:3]),
                                    torch.cos(state[..., 2:3])], dim=-1)), "ELBO": pydpf.ElBO_Loss()}
    validation_aggregation_function = {"Mean Pose": pydpf.FilteringMean(
        lambda state: torch.concat([state[..., :2], bind_angle(state[..., 2:3])], dim=-1)),
        "ELBO": pydpf.ElBO_Loss()}
    data_length = len(dataset)
    split = _get_split_amounts(split_size, data_length)
    train_set, validation_set, test_set = torch.utils.data.random_split(
        dataset, split, generator=data_loading_generator)
    for k, subset in enumerate((train_set, validation_set, test_set)):
        if batch_size[k] == -1 or batch_size[k] > len(subset):
            batch_size[k] = len(subset)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size[0], shuffle=True,
                                               generator=data_loading_generator, collate_fn=dataset.collate)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size[1], shuffle=False,
                                                    generator=data_loading_generator, collate_fn=dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size[2], shuffle=False,
                                              generator=data_loading_generator, collate_fn=dataset.collate)
    best_eval = torch.inf
    best_dict = None
    encoder = dpf.SSM.observation_model.encoder
    decoder = dpf.SSM.observation_model.decoder

    if time_extent is None:
        time_extent = dataset.observation.size(0) - 1

    for epoch in range(pre_train_epochs):
        dpf.train()
        train_loss = 0
        total_size = 0
        for state, observation, control in train_loader:
            dpf.update()
            opt.zero_grad()
            observation = observation.to(device).reshape(observation.size(0) * observation.size(1), 3, 24, 24)
            encoded_obs = encoder(observation)
            decoded_obs = decoder(encoded_obs)
            AE_loss = torch.mean((decoded_obs - observation) ** 2) * scalings[2]
            AE_loss.backward()
            opt.step()
            train_loss += AE_loss.item() * observation.size(1)
            total_size += observation.size(1)
        print(f"Auto_encoder training loss = {train_loss / total_size}")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = []
        total_size = 0
        dpf.train()
        for state, observation, control in train_loader:
            dpf.update()
            opt.zero_grad()
            batch_size_now = observation.size(1)
            observation = observation.to(device).reshape(observation.size(0) * observation.size(1), 3, 24, 24)
            state = state.to(device)
            control = control.to(device)
            encoded_obs = encoder(observation)
            decoded_obs = decoder(encoded_obs)
            AE_loss = torch.mean((decoded_obs - observation) ** 2)
            outputs = dpf(n_particles[0], time_extent, aggregation_function,
                          observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(),
                          ground_truth=state, control=control, gradient_regulariser=gradient_regulariser)
            cos_loss = torch.mean((outputs["Mean Pose"][:, :, 3] - torch.cos(state[:, :, 2])) ** 2)
            sin_loss = torch.mean((outputs["Mean Pose"][:, :, 2] - torch.sin(state[:, :, 2])) ** 2)
            angle_loss = cos_loss + sin_loss
            position_loss = torch.mean(torch.sum((outputs["Mean Pose"][:, :, :2] - state[:, :, :2]) ** 2, dim=-1))
            loss = scalings[0] * position_loss + scalings[1] * angle_loss + AE_loss * scalings[2]
            loss.backward()
            train_loss.append(loss.item() * state.size(1))
            opt.step()
            total_size += state.size(1)
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_loss = np.sum(np.array(train_loss)) / total_size
        dpf.update()
        dpf.eval()
        with torch.inference_mode():
            total_size = 0
            validation_Pos_MSE = []
            validation_Angle_MSE = []
            for state, observation, control in validation_loader:
                batch_size_now = observation.size(1)
                observation = observation.to(device).reshape(observation.size(0) * observation.size(1), 3, 24, 24)
                state = state.to(device)
                control = control.to(device)
                encoded_obs = encoder(observation)
                outputs = dpf(n_particles[1], time_extent, validation_aggregation_function,
                              observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(),
                              ground_truth=state, control=control)
                validation_Pos_MSE.append(torch.mean(torch.sum(
                    ((outputs["Mean Pose"][-1, :, :2] - state[-1, :, :2]) * position_scaling) ** 2, dim=-1)).item() * state.size(1))
                validation_Angle_MSE.append(torch.mean(bind_angle(
                    bind_angle(outputs["Mean Pose"][:, :, 2]) - bind_angle(state[:, :, 2])) ** 2).item() * state.size(1))
                total_size += state.size(1)
            validation_Pos_MSE = np.sum(np.array(validation_Pos_MSE)) / total_size
            validation_Angle_MSE = np.sum(np.array(validation_Angle_MSE)) / total_size
            if validation_Pos_MSE < best_eval:
                best_eval = validation_Pos_MSE
                best_dict = deepcopy(dpf.state_dict())

        print(f"epoch {epoch + 1}/{epochs}, train loss: {train_loss}, "
              f"validation position RMSE: {np.sqrt(validation_Pos_MSE)}, "
              f"validation angle RMSE: {np.sqrt(validation_Angle_MSE)}")
    total_size = 0
    dpf.load_state_dict(best_dict)

    with torch.inference_mode():
        test_Pos_MSE = []
        test_angle_MSE = []
        for state, observation, control in test_loader:
            batch_size_now = observation.size(1)
            observation = observation.to(device).reshape(observation.size(0) * observation.size(1), 3, 24, 24)
            state = state.to(device)
            control = control.to(device)
            encoded_obs = encoder(observation)
            outputs = dpf(n_particles[1], time_extent, validation_aggregation_function,
                          observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(),
                          ground_truth=state, control=control)
            test_Pos_MSE.append(torch.mean(torch.sum(
                ((outputs["Mean Pose"][-1, :, :2] - state[-1, :, :2]) * position_scaling) ** 2, dim=-1)).item() * state.size(1))
            test_angle_MSE.append(torch.mean(bind_angle(
                bind_angle(outputs["Mean Pose"][:, :, 2]) - bind_angle(state[:outputs["Mean Pose"].size(0), :, 2])) ** 2).item() * state.size(1))
            total_size += state.size(1)
    test_Pos_MSE = np.sum(np.array(test_Pos_MSE)) / total_size
    test_angle_MSE = np.sum(np.array(test_angle_MSE)) / total_size
    print("")
    print(f"test position RMSE: {np.sqrt(test_Pos_MSE)}, test angle RMSE: {np.sqrt(test_angle_MSE)}")
    print(f"Final time = {time.time() - start_time}")
    return test_Pos_MSE, test_angle_MSE


# --------------------------------------------------------------------------- #
#  Run configuration                                                          #
# --------------------------------------------------------------------------- #

class RunConfig:
    """Bundle of run-wide settings threaded through every experiment."""

    def __init__(self, device, data_dir, results_dir, smoke=False,
                 dx=25, dy=1, alpha=0.91, beta=0.5, sigma=1.0, batch_size=128,
                 maze_deterministic=("deterministic",), maze_repeats=1, delete_raw=False,
                 overwrite_results=False):
        self.device = torch.device(device)
        self.data_dir = pathlib.Path(data_dir)
        self.results_dir = pathlib.Path(results_dir)
        self.smoke = smoke
        self.overwrite_results = overwrite_results
        self.dx = dx
        self.dy = dy
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.batch_size = batch_size
        self.maze_deterministic = maze_deterministic
        self.maze_repeats = maze_repeats
        self.delete_raw = delete_raw

    def pick(self, paper, smoke):
        """Return the reduced value in --smoke mode, otherwise the paper value."""
        return smoke if self.smoke else paper


def select_methods(requested, available):
    """Intersect a user request with the methods a given experiment supports."""
    if not requested:
        return list(available)
    chosen = [m for m in available if m in requested]
    for m in requested:
        if m not in available:
            print(f"  (note: '{m}' is not a method of this experiment, skipping)")
    if not chosen:
        print("  (no requested methods apply to this experiment, skipping)")
    return chosen


# --------------------------------------------------------------------------- #
#  Data preparation -- everything ends up in the single central data folder.  #
# --------------------------------------------------------------------------- #

def _ensure_data_file(central_path, generate_fn, description, regenerate=True):
    if not regenerate and central_path.exists():
        print(f"  data ready: {central_path.name}")
        return central_path
    if generate_fn is None:
        raise FileNotFoundError(
            f"Required data file '{central_path}' for {description} is missing and "
            f"cannot be generated automatically. Please place it in {central_path.parent}.")
    print(f"  generating {description} -> {central_path.name}")
    generate_fn(central_path)
    return central_path


def prepare_lg_data(cfg):
    name = f"dx={cfg.dx}-dy={cfg.dy}.csv"
    central = cfg.data_dir / name

    def generate(path):
        gen = torch.Generator(device=cfg.device).manual_seed(0)
        prior, dynamic, observation = lg_make_components(cfg.dx, cfg.dy, gen)
        SSM = pydpf.FilteringModel(prior_model=prior, dynamic_model=dynamic, observation_model=observation)
        n_traj = 2000
        pydpf.simulate_and_save(path, SSM=SSM, time_extent=1000, n_trajectories=n_traj,
                                batch_size=100, device=cfg.device, bypass_ask=True)

    return _ensure_data_file(
        central,
        generate,
        "linear-Gaussian data")


def prepare_sv_data(cfg):
    name = f"alpha={cfg.alpha}-beta={cfg.beta}-sigma={cfg.sigma}.csv"
    central = cfg.data_dir / name

    def generate(path):
        gen = torch.Generator(device=cfg.device).manual_seed(0)
        alpha = torch.tensor([[cfg.alpha]], device=cfg.device)
        beta = torch.tensor([cfg.beta], device=cfg.device)
        sigma = torch.tensor([[cfg.sigma]], device=cfg.device)
        SSM = sv_make_SSM(sigma, alpha, beta, cfg.device, generator=gen)
        n_traj = 500
        pydpf.simulate_and_save(path, SSM=SSM, time_extent=1000, n_trajectories=n_traj,
                                batch_size=cfg.batch_size, device=cfg.device, bypass_ask=True)

    return _ensure_data_file(
        central,
        generate,
        "stochastic-volatility data")


def prepare_sv_test_trajectory(cfg):
    central = cfg.data_dir / "test_trajectory.csv"
    return _ensure_data_file(
        central,
        None,
        "the fixed SV test trajectory (download 'test_trajectory.csv' from the pydpf "
        "GitHub repository, https://github.com/John-JoB/pydpf)",
        regenerate = False)


def _download_maze_dataset(folder_path):
    """Download + unpack the raw maze data (mirrors dm_setup.download_dataset)."""
    import requests  # imported lazily; only needed when the maze data is missing
    from tqdm import tqdm
    import zipfile

    data_url = "https://depositonce.tu-berlin.de/bitstreams/fe02c1e0-64d9-4a92-ac4d-a8a0ef455c8f/download"
    download_path = folder_path / "raw_zip.zip"
    with requests.get(data_url, stream=True) as r:
        total = int(r.headers.get("content-length", 0))
        with open(download_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                                  desc="Downloading data") as bar:
            for chunk in r.iter_content(chunk_size=1024):
                bar.update(f.write(chunk))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)
    unzipped_path = folder_path / "data"
    (unzipped_path / "100s/nav03_test.npz").rename(folder_path / "maze_data_raw.npz")
    (unzipped_path / "100s/nav03_train.npz").rename(folder_path / "maze_data_raw_2.npz")
    shutil.rmtree(unzipped_path)
    download_path.unlink()


def _maze_bind_angle(angle):
    """Angle wrapping used while building the maze data set (matches dm_setup)."""
    out = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)
    return torch.where(out < -torch.pi, out + 2 * torch.pi, out)


def prepare_maze_data(cfg):
    central = cfg.data_dir / "maze_data.csv"

    def generate(path):
        device = cfg.device
        data_raw_1 = cfg.data_dir / "maze_data_raw.npz"
        data_raw_2 = cfg.data_dir / "maze_data_raw_2.npz"
        if not (data_raw_1.exists() and data_raw_2.exists()):
            _download_maze_dataset(cfg.data_dir)

        def create_actions_and_modify_state(pos):
            pos[:, 2] = pos[:, 2] * np.pi / 180
            pos[:, 2] = _maze_bind_angle(pos[:, 2])
            diffs = torch.empty_like(pos)
            diffs[1:, :] = pos[1:, :] - pos[:-1, :]
            diffs[::100, :] = 0
            angles = torch.roll(pos[:, 2], 1, 0)
            c = torch.cos(angles)
            s = torch.sin(angles)
            rotation_matrix = torch.stack([torch.stack([c, s], dim=-1),
                                           torch.stack([-s, c], dim=-1)], dim=-2)
            actions = torch.empty_like(pos)
            actions[:, :2] = (rotation_matrix @ diffs[:, :2].unsqueeze(-1)).squeeze(-1)
            actions[:, 2] = _maze_bind_angle(diffs[:, 2])
            return actions

        def create_observations(obs):
            new_o = torch.zeros([obs.shape[0], 24, 24, 3], device=obs.device, dtype=torch.uint8)
            rng = np.random.Generator(np.random.PCG64(seed=0))
            for i in range(obs.shape[0]):
                offsets = rng.integers(low=0, high=9, size=2)
                new_o[i] = obs[i, offsets[0]:offsets[0] + 24, offsets[1]:offsets[1] + 24, :3]
            new_o = new_o.to(dtype=torch.float16)
            new_o = torch.round(torch.clip(new_o, 0, 255)).to(dtype=torch.uint8)
            new_o = new_o.permute(0, 3, 1, 2)
            return new_o.flatten(start_dim=1)

        def create_df(data, label):
            data = data.cpu().numpy()
            df = pd.DataFrame(data, columns=[f"{label}_{i + 1}" for i in range(data.shape[1])])
            df["series_id"] = np.arange(2000).repeat(100)
            return df

        data1 = dict(np.load(data_raw_1, allow_pickle=True))
        data2 = dict(np.load(data_raw_2, allow_pickle=True))
        state = torch.tensor(np.concatenate((data1["pose"], data2["pose"]), axis=0),
                             device=device, dtype=torch.float32)
        observation = torch.tensor(np.concatenate((data1["rgbd"], data2["rgbd"]), axis=0),
                                   device=device, dtype=torch.uint8)
        actions = create_actions_and_modify_state(state)
        control_df = create_df(actions, "control")
        observation_df = create_df(create_observations(observation), "observation")
        state_df = create_df(state, "state")
        observation_df.drop(columns=["series_id"], inplace=True)
        control_df.drop(columns=["series_id"], inplace=True)
        total_df = pd.merge(control_df, state_df, left_index=True, right_index=True)
        total_df = pd.merge(total_df, observation_df, left_index=True, right_index=True)
        total_df.to_csv(path, index=False)
        if cfg.delete_raw:
            data_raw_1.unlink()
            data_raw_2.unlink()

    return _ensure_data_file(
        central,
        generate,
        "deep-mind maze data",
        regenerate = False)


# --------------------------------------------------------------------------- #
#  Experiment 1: Linear Gaussian -- comparison with the Kalman filter         #
# --------------------------------------------------------------------------- #

def run_kalman(cfg, requseted_models):
    run_kalman_help(cfg, requseted_models)
    device = cfg.device
    if device.type == "cuda":
        copied_cfg = copy(cfg)
        copied_cfg.device = torch.device("cpu")
        run_kalman_help(copied_cfg, requseted_models)


def run_kalman_help(cfg, requested_models):
    device = cfg.device
    print(f"\n=== Linear Gaussian: comparison with the Kalman filter on {device.type}===")
    data_path = prepare_lg_data(cfg)
    results_file = cfg.results_dir / "Kalman_comparison_results.csv"
    if cfg.smoke:
        results_file = results_file.with_stem(results_file.stem + "_smoke")

    make_new_csv(["Kalman Filter", "PF K = 25", "PF K = 100", "PF K = 1000", "PF K = 10000"],
                 ["Time CPU (s)", "Time GPU (s)", "epsilon x", "epsilon y"],
                 results_file, overwrite=cfg.overwrite_results)

    dx, dy = cfg.dx, cfg.dy
    cuda = cfg.device.type == "cuda"
    device = cfg.device
    all_Ks = [25, 100, 1000, 10000]
    if cfg.smoke:
        all_Ks = [25, 100]
    selected = select_methods([str(k) for k in requested_models] if requested_models else None,
                              [str(k) for k in all_Ks])
    # 'None' is the Kalman-only timing pass; it always runs to populate that row.
    Ks = [None] + [int(k) for k in selected]

    cuda_gen = torch.Generator(device=device).manual_seed(0)
    cpu_gen = torch.Generator().manual_seed(0)
    batch_size = cfg.batch_size

    dataset = pydpf.StateSpaceDataset(data_path=data_path, series_id_column="series_id",
                                      state_prefix="state", observation_prefix="observation", device=device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=dataset.collate, generator=cpu_gen)

    prior, dynamic, observation = lg_make_components(dx, dy, cuda_gen)
    multinomial_resampler = pydpf.MultinomialResampler(cuda_gen)
    SSM = pydpf.FilteringModel(prior_model=prior, dynamic_model=dynamic, observation_model=observation)
    PF = pydpf.ParticleFilter(resampler=multinomial_resampler, SSM=SSM)
    KalmanFilter = pydpf.KalmanFilter(prior_model=prior, dynamic_model=dynamic, observation_model=observation)

    aggregation_function_dict = {"Means": pydpf.FilteringMean(), "Likelihood_factors": pydpf.LogLikelihoodFactors()}

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    for K in Ks:
        print(f"\nRunning with {K} particles")
        size = 0
        state_error = []
        kalman_time = []
        pf_time = []
        likelihood_error = []
        for state, observation_b in tqdm(data_loader):
            with torch.inference_mode():
                size += state.size(1)
                sync(device)
                s_time = time.time()
                kalman_state, kalman_cov, kalman_likelihood = KalmanFilter(observation=observation_b, time_extent=cfg.pick(1000,10))
                sync(device)
                kalman_time.append(time.time() - s_time)
                if K is not None:
                    sync(device)
                    s_time = time.time()
                    outputs = PF(observation=observation_b, n_particles=K,
                                 aggregation_function=aggregation_function_dict, time_extent=cfg.pick(1000,10))
                    sync(device)
                    pf_time.append(time.time() - s_time)
                    state_sq_error = torch.sum((outputs["Means"] - kalman_state) ** 2, dim=-1).mean()
                    state_error.append(state_sq_error.item() * state.size(1))
                    log_abs_likelihood_error = fractional_diff_exp(
                        kalman_likelihood, outputs["Likelihood_factors"].squeeze()).mean()
                    likelihood_error.append(log_abs_likelihood_error.item() * state.size(1))

        results_df = pd.read_csv(results_file, index_col=0)
        if K is not None:
            row_label = f"PF K = {K}"
            row = list(results_df.loc[row_label])
        kalman_row = list(results_df.loc["Kalman Filter"])
        # Ignore the first iteration (CUDA warm-up) and the last (possibly smaller batch).
        denom = max(len(data_loader) - 2, 1)
        if cuda:
            if K is None:
                kalman_row[1] = sum(kalman_time[1:-1]) / denom
                kalman_row[2] = 0.0
                kalman_row[3] = 0.0
            else:
                row[1] = sum(pf_time[1:-1]) / denom
                row[2] = sum(state_error) / size
                row[3] = sum(likelihood_error) / size
        else:
            if K is None:
                kalman_row[0] = sum(kalman_time[1:-1]) / denom
            else:
                row[0] = sum(pf_time[1:-1]) / denom

        if K is not None:
            results_df.loc[row_label] = row
        results_df.loc["Kalman Filter"] = kalman_row
        results_df.to_csv(results_file)
    print(pd.read_csv(results_file, index_col=0))


# --------------------------------------------------------------------------- #
#  Experiment 2: Linear Gaussian -- learning proposal parameters              #
# --------------------------------------------------------------------------- #

def run_proposal(cfg, requested_models):
    print("\n=== Linear Gaussian: learning proposal parameters ===")
    data_path = prepare_lg_data(cfg)
    results_file = cfg.results_dir / "proposal_learning_results.csv"
    if cfg.smoke:
        results_file = results_file.with_stem(results_file.stem + "_smoke")
    make_new_csv(["Bootstrap", "Optimal"] + DPF_METHODS, ["e_x", "e_l", "mean W2", "ELBO"],
                 results_file, overwrite=cfg.overwrite_results)

    dx, dy = cfg.dx, cfg.dy
    device = cfg.device
    batch_size = 32
    n_repeats = cfg.pick(5, 1)
    epochs = cfg.pick(20, 1)
    experiment_list = select_methods(requested_models, ["Bootstrap", "Optimal"] + DPF_METHODS)
    if not experiment_list:
        return

    if dy > dx:
        raise ValueError("The dimension of the observations cannot be more than the dimension of the states.")

    def kernel_factory(generator):
        return pydpf.KernelMixture(
            pydpf.MultivariateGaussian(torch.zeros(25, device=device),
                                       torch.nn.Parameter(torch.eye(25, device=device) * 0.1),
                                       diagonal_cov=True, generator=generator), generator=generator)

    def training_loop(dpf, epochs, train_loader, proposal_model, experiment):
        ELBO_fun = pydpf.ElBO_Loss()
        if experiment == "Kernel":
            opt = torch.optim.SGD(
                [{"params": [dpf.SSM.proposal_model.x_weight], "lr": 0.01},
                 {"params": [dpf.SSM.proposal_model.y_weight, proposal_model.dist.cholesky_covariance], "lr": 0.05},
                 {"params": dpf.resampler.parameters(), "lr": 0.001}], lr=.5, momentum=0.9, nesterov=True)
        elif experiment == "Optimal Transport":
            opt = torch.optim.SGD(
                [{"params": [dpf.SSM.proposal_model.x_weight], "lr": 0.01},
                 {"params": [dpf.SSM.proposal_model.y_weight, proposal_model.dist.cholesky_covariance], "lr": 0.05}],
                lr=.5, momentum=0.9, nesterov=True)
        else:
            opt = torch.optim.SGD(
                [{"params": [dpf.SSM.proposal_model.x_weight], "lr": 0.1},
                 {"params": [dpf.SSM.proposal_model.y_weight, proposal_model.dist.cholesky_covariance], "lr": 0.5}],
                lr=.5, momentum=0.9, nesterov=True)
        opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
        best_validation_loss = torch.inf
        best_dict = deepcopy(dpf.state_dict())
        for epoch in range(epochs):
            dpf.train()
            total_size = 0
            for state, observation in train_loader:
                dpf.update()
                opt.zero_grad()
                ELBO = dpf(cfg.pick(100,2), cfg.pick(100,2), ELBO_fun, observation=observation)
                loss = torch.mean(ELBO)
                loss.backward()
                opt.step()
                total_size += state.size(1)
                opt_scheduler.step()
            dpf.eval()
            dpf.update()
            total_size = 0
            validation_loss = []
            with torch.inference_mode():
                for state, observation in train_loader:
                    ELBO = dpf(cfg.pick(100,2), cfg.pick(100,2), ELBO_fun, observation=observation)
                    validation_loss.append(torch.mean(ELBO).item() * state.size(1))
                    total_size += state.size(1)
                validation_loss = np.sum(np.array(validation_loss)) / total_size
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_dict = deepcopy(dpf.state_dict())
            dpf.load_state_dict(best_dict)

    def test_dpf(dpf, test_loader, KalmanFilter):
        aggregation_fun = {"ELBO": pydpf.ElBO_Loss(), "Filtering Mean": pydpf.FilteringMean(),
                           "Likelihood_factors": pydpf.LogLikelihoodFactors()}
        test_ELBO = []
        epsilon_x = []
        epsilon_l = []
        dpf.update()
        total_size = 0
        with torch.inference_mode():
            for state, observation in test_loader:
                outputs = dpf(n_particles=cfg.pick(100,2), time_extent=cfg.pick(1000,10), aggregation_function=aggregation_fun,
                              observation=observation)
                test_ELBO.append(outputs["ELBO"].sum().item() * state.size(1))
                kalman_state, kalman_cov, kalman_likelihood = KalmanFilter(observation=observation, time_extent=cfg.pick(1000,10))
                epsilon_x.append(torch.sum((outputs["Filtering Mean"] - kalman_state) ** 2, dim=-1).mean().item() * state.size(1))
                log_abs_likelihood_error = fractional_diff_exp(
                    kalman_likelihood, outputs["Likelihood_factors"].squeeze()).mean()
                epsilon_l.append(log_abs_likelihood_error.item() * state.size(1))
                total_size += state.size(1)
        return -sum(test_ELBO) / total_size, sum(epsilon_x) / total_size, sum(epsilon_l) / total_size

    def mean_wass_dist(train_set, x_weight, y_weight, prop_cov):
        state = train_set.state
        observation = train_set.observation
        dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1)
                                            - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        pred_state =  dynamic_matrix@state.unsqueeze(-1)
        pred_state = pred_state[:-1].squeeze()
        observation = observation[1:]
        optimal_x_weight = torch.ones(dx, device=device)
        optimal_x_weight[:dy] = .5
        optimal_cov = torch.ones(dx, device=device)
        for i in range(dy):
            optimal_cov[i] = .5
        a = x_weight - optimal_x_weight
        b = y_weight - .5
        weight_state = a * pred_state
        weight_obs = b * observation
        dif = weight_state
        dif[..., :dy] += weight_obs
        dif_mag = torch.sum(dif ** 2, dim=-1)
        av_mean_div = torch.mean(dif_mag)
        cov_div = torch.sum((optimal_cov + prop_cov - 2 * torch.sqrt(optimal_cov * prop_cov)))
        return av_mean_div + cov_div

    def chain(*its):
        it_list = []
        for it in its:
            it_list += list(it)
        return it_list

    def rotate_range(c_repeat, rel_start, rel_end, repeats, total_elements):
        range_rotation_amount = (total_elements // repeats) * c_repeat
        start = (rel_start + range_rotation_amount) % total_elements
        end = (rel_end + range_rotation_amount) % total_elements
        if end == 0:
            return range(start, total_elements)
        if start > end:
            return chain(range(start, total_elements), range(0, end))
        return range(start, end)

    dataset = pydpf.StateSpaceDataset(data_path=data_path, series_id_column="series_id",
                                      state_prefix="state", observation_prefix="observation", device=device)

    for experiment in experiment_list:
        print(f"\nRunning {experiment}")
        rep_mean_wass_dist = torch.tensor(0., device=device)
        mean_epsilon_l = 0
        mean_epsilon_x = 0
        mean_ELBO = 0
        for repeat in range(n_repeats):
            print(f"Repeat {repeat + 1} of {n_repeats}:")
            cpu_gen = torch.Generator().manual_seed(10 * repeat)
            cuda_gen = torch.Generator(device=device).manual_seed(10 * repeat)
            train_set = dataset.select(rotate_range(repeat, 0, 1000, n_repeats, 2000))
            test_set = dataset.select(rotate_range(repeat, 1000, 1500, n_repeats, 2000))
            #test_set = dataset.select(rotate_range(repeat, 1500, 2000, n_repeats, 2000))
            proposal_kind = "optimal" if experiment == "Optimal" else "learned"
            prior, dynamic, observation, proposal_model = lg_make_components(dx, dy, cuda_gen, proposal_kind)
            if experiment == "Bootstrap":
                SSM = pydpf.FilteringModel(prior_model=prior, dynamic_model=dynamic, observation_model=observation)
                dpf = build_dpf("DPF", SSM, cuda_gen)
                rep_mean_wass_dist += mean_wass_dist(test_set, torch.ones(dx, device=device),
                                                torch.zeros(dy, device=device),
                                                torch.ones(dx, device=device))
            elif experiment == "Optimal":
                SSM = pydpf.FilteringModel(prior_model=prior, dynamic_model=dynamic,
                                           observation_model=observation, proposal_model=proposal_model)
                dpf = build_dpf("DPF", SSM, cuda_gen)
            else:
                trained_model = pydpf.FilteringModel(prior_model=prior, dynamic_model=dynamic,
                                                     observation_model=observation, proposal_model=proposal_model)
                dpf = build_dpf(experiment, trained_model, cuda_gen, ot_regularisation=0.5,
                                kernel_factory=kernel_factory)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                           generator=cpu_gen, collate_fn=dataset.collate)
                training_loop(dpf, epochs, train_loader, proposal_model, experiment)
                cholesky_prop_cov = torch.diag(proposal_model.dist.cholesky_covariance)
                prop_cov = cholesky_prop_cov ** 2
                rep_mean_wass_dist += mean_wass_dist(test_set, proposal_model.x_weight, proposal_model.y_weight, prop_cov)

            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                      generator=cpu_gen, collate_fn=dataset.collate)
            kalman_filter = pydpf.KalmanFilter(prior_model=prior, dynamic_model=dynamic, observation_model=observation)
            ELBO, e_x, e_l = test_dpf(dpf, test_loader, kalman_filter)
            mean_ELBO += ELBO
            mean_epsilon_l += e_l
            mean_epsilon_x += e_x
        rep_mean_wass_dist = sqrt(rep_mean_wass_dist.item() / n_repeats)
        mean_ELBO /= n_repeats
        mean_epsilon_x /= n_repeats
        mean_epsilon_l /= n_repeats
        results_df = pd.read_csv(results_file, index_col=0)
        results_df.loc[experiment] = np.array([mean_epsilon_x, mean_epsilon_l, rep_mean_wass_dist, mean_ELBO])
        results_df.to_csv(results_file)
        print(results_df)


# --------------------------------------------------------------------------- #
#  Experiment 3: SV -- filtering given a fully specified model                #
# --------------------------------------------------------------------------- #

def run_sv_filtering(cfg, requested_models):
    print("\n=== Stochastic Volatility: filtering given a fully specified model ===")
    data_path = prepare_sv_data(cfg)
    results_file = cfg.results_dir / "fully_specified_results.csv"
    if cfg.smoke:
        results_file = results_file.with_stem(results_file.stem + "_smoke")
    make_new_csv(DPF_METHODS, ["e_x", "e_l", "time"], results_file,
                 overwrite=cfg.overwrite_results)


    device = cfg.device
    experiments = select_methods(requested_models, DPF_METHODS)
    if not experiments:
        return
    alpha, beta, sigma = cfg.alpha, cfg.beta, cfg.sigma
    batch_size = cfg.batch_size
    alpha_t = torch.tensor([[alpha]], device=device)
    beta_t = torch.tensor([beta], device=device)
    sigma_t = torch.tensor([[sigma]], device=device)
    dataset = pydpf.StateSpaceDataset(data_path=data_path, series_id_column="series_id",
                                      state_prefix="state", observation_prefix="observation", device=device)

    def kernel_factory(generator):
        return pydpf.KernelMixture(
            pydpf.MultivariateGaussian(torch.zeros(1, device=device), torch.eye(1, device=device) * 0.1,
                                       generator=generator), generator=generator)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    for experiment in experiments:
        print(f"Testing {experiment}")
        rng = torch.Generator(device=device).manual_seed(0)
        cpu_rng = torch.Generator().manual_seed(0)
        size = 0
        pf_time = []
        MSE = []
        likelihood_error = []
        SSM = sv_make_SSM(sigma_t, alpha_t, beta_t, device, generator=rng)
        dpf = build_dpf(experiment, SSM, rng, soft_softness=0.7, ot_regularisation=0.5, ot_clip=1.,
                        kernel_factory=kernel_factory)
        pf = pydpf.DPF(SSM=SSM, resampling_generator=rng, multinomial=True)
        aggregation_function = {"Likelihood": pydpf.LogLikelihoodFactors(), "Filtering mean": pydpf.FilteringMean()}
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                  generator=cpu_rng, collate_fn=dataset.collate)
        for state, observation in tqdm(data_loader):
            with torch.inference_mode():
                size += state.size(1)
                true_outputs = pf(observation=observation, n_particles=10000,
                                  aggregation_function=aggregation_function, time_extent=1000)
                sync(device)
                s_time = time.time()
                outputs = dpf(observation=observation, n_particles=100,
                              aggregation_function=aggregation_function, time_extent=1000)
                sync(device)
                pf_time.append(time.time() - s_time)
                MSE.append(torch.sum((true_outputs["Filtering mean"] - outputs["Filtering mean"]) ** 2,
                                     dim=-1).mean().item() * state.size(1))
                likelihood_error.append(fractional_diff_exp(
                    true_outputs["Likelihood"], outputs["Likelihood"]).mean().item() * state.size(1))

        denom = max(len(data_loader) - 2, 1)
        results = pd.read_csv(results_file, index_col=0)
        results.loc[experiment] = np.array([sum(MSE) / size, sum(likelihood_error) / size,
                                            sum(pf_time[1:-1]) / denom])
        print(results)
        results.to_csv(results_file)


# --------------------------------------------------------------------------- #
#  Experiment 4: SV -- unsupervised learning of a single parameter            #
# --------------------------------------------------------------------------- #

def run_sv_single(cfg, requested_models):
    print("\n=== Stochastic Volatility: unsupervised learning of a single parameter ===")
    prepare_sv_test_trajectory(cfg)
    results_file = cfg.results_dir / "single_parameter_results.csv"
    if cfg.smoke:
        results_file = results_file.with_stem(results_file.stem + "_smoke")
    make_new_csv(DPF_METHODS, ["Forward Time (s)", "Backward Time (s)",
                               "Gradient standard deviation", "alpha error"],
                  results_file, overwrite=cfg.overwrite_results)


    device = cfg.device
    experiments = select_methods(requested_models, DPF_METHODS)
    if not experiments:
        return
    n_repeats = cfg.pick(10, 2)
    training_epochs = cfg.pick(10, 1)
    batch_size_test = cfg.batch_size
    batch_size_train = 32
    true_alpha, true_beta, true_sigma = 0.91, 0.5, 1.
    data_path = cfg.data_dir
    temp_data_path = data_path / "temp.csv"

    def kernel_factory(generator):
        return pydpf.KernelMixture(
            pydpf.MultivariateGaussian(torch.zeros(1, device=device),
                                       torch.nn.Parameter(torch.eye(1, device=device) * 0.1),
                                       generator=generator), generator=generator)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    def test_gradients(experiment):
        rng = torch.Generator(device).manual_seed(0)
        aggregation_function_dict = {"ELBO": pydpf.LogLikelihoodFactors()}
        test_dataset = pydpf.StateSpaceDataset(data_path=data_path / "test_trajectory.csv",
                                               state_prefix="state", device=device)
        gradients = []
        alpha_p = torch.nn.Parameter(torch.tensor([[0.93]], dtype=torch.float32, device=device))
        SSM = sv_make_SSM(torch.tensor([[1.]], device=device), alpha_p,
                          torch.tensor([0.5], device=device), device)
        DPF = build_dpf(experiment, SSM, rng, ot_regularisation=0.5, ot_clip=1.,
                        kernel_factory=kernel_factory)
        forward_time = []
        backward_time = []
        state = test_dataset.state[:, 0:1].expand((101, batch_size_test, 1)).contiguous()
        observation = test_dataset.observation[:, 0:1].expand((101, batch_size_test, 1)).contiguous()
        n_batches = cfg.pick(2560 // batch_size_test, 2)
        for _ in tqdm(range(n_batches)):
            DPF.update()
            sync(device)
            start = time.time()
            outputs = DPF(observation=observation, n_particles=100, ground_truth=state,
                          aggregation_function=aggregation_function_dict, time_extent=100)
            ls = torch.mean(outputs["ELBO"], dim=0)
            loss = ls.mean()
            sync(device)
            forward_time.append(time.time() - start)
            alpha_p.grad = None
            for i in range(len(ls)):
                ls[i].backward(retain_graph=True)
                gradients.append(alpha_p.grad.item())
                alpha_p.grad = None
            sync(device)
            start = time.time()
            loss.backward()
            sync(device)
            backward_time.append(time.time() - start)
        return forward_time, backward_time, gradients

    def test_learning_alpha(experiment):
        alphas = np.empty(n_repeats)
        for n in range(n_repeats):
            rng = torch.Generator(device).manual_seed(n * 10)
            cpu_rng = torch.Generator().manual_seed(n * 10)
            generation_rng = torch.Generator(device).manual_seed(n * 10)
            true_SSM = sv_make_SSM(torch.tensor([[true_sigma]], device=device),
                                   torch.tensor([[true_alpha]], device=device),
                                   torch.tensor([true_beta], device=device), device, generation_rng)
            pydpf.simulate_and_save(temp_data_path, SSM=true_SSM, time_extent=1000,
                                    n_trajectories=cfg.pick(500, 50), batch_size=100, device=device, bypass_ask=True)
            alpha = torch.nn.Parameter(torch.rand((1, 1), device=device, generator=rng), requires_grad=True)
            SSM = sv_make_SSM(torch.tensor([[1.]], device=device), alpha,
                              torch.tensor([0.5], device=device), device, generation_rng)
            dpf = build_dpf(experiment, SSM, rng, ot_regularisation=0.5, ot_clip=1.,
                            kernel_factory=kernel_factory)
            if experiment == "Kernel":
                opt = torch.optim.SGD([{"params": [alpha], "lr": 0.05},
                                       {"params": dpf.resampler.mixture.parameters(), "lr": 0.01}])
            else:
                opt = torch.optim.SGD([{"params": [alpha], "lr": 0.05}])
            opt_schedule = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)
            dataset = pydpf.StateSpaceDataset(temp_data_path, state_prefix="state", device=device)
            _, ELBO = train_sv(dpf, opt, dataset, training_epochs, (100, 100, 100),
                               (batch_size_train, batch_size_test, batch_size_test),
                               (0.5, 0.25, 0.25), 1., cpu_rng, target="ELBO", time_extent=100,
                               lr_scheduler=opt_schedule)
            alphas[n] = alpha
            os.remove(temp_data_path)
        return alphas

    for experiment in experiments:
        print(f"\nRunning {experiment}")
        results = pd.read_csv(results_file, index_col=0)
        ft, bt, grads = test_gradients(experiment)
        alpha_list = test_learning_alpha(experiment)
        denom_f = max(len(ft) - 2, 1)
        denom_b = max(len(bt) - 2, 1)
        results.loc[experiment] = np.array([sum(ft[1:-1]) / denom_f, sum(bt[1:-1]) / denom_b,
                                           np.sqrt(np.var(grads)), np.mean(np.abs(alpha_list - 0.91))])
        print(results)
        results.to_csv(results_file)


# --------------------------------------------------------------------------- #
#  Experiment 5: SV -- unsupervised learning of multiple parameters           #
# --------------------------------------------------------------------------- #

def run_sv_multiple(cfg, requested_models):
    print("\n=== Stochastic Volatility: unsupervised learning of multiple parameters ===")
    results_file = cfg.results_dir / "multiple_parameter_results.csv"
    if cfg.smoke:
        results_file = results_file.with_stem(results_file.stem + "_smoke")
    make_new_csv(DPF_METHODS, ["ELBO", "alpha error", "beta error", "sigma error"],
                 results_file, overwrite=cfg.overwrite_results)

    device = cfg.device
    experiments = select_methods(requested_models, DPF_METHODS)
    if not experiments:
        return
    batch_size_test = cfg.batch_size
    batch_size_train = 32
    true_alpha, true_beta, true_sigma = 0.91, 0.5, 1.
    temp_data_path = cfg.data_dir / "temp.csv"
    n_repeats = cfg.pick(10, 2)
    training_epochs = cfg.pick(20, 1)

    def make_globals(experiment_cuda_rng):
        alpha = torch.nn.Parameter(torch.rand((1, 1), device=device, generator=experiment_cuda_rng), requires_grad=True)
        sigma = torch.nn.Parameter(torch.rand((1, 1), device=device, generator=experiment_cuda_rng) * 5, requires_grad=True)
        beta = torch.nn.Parameter(torch.rand((1,), device=device, generator=experiment_cuda_rng) * 2, requires_grad=True)
        return sv_make_SSM(sigma, alpha, beta, device, experiment_cuda_rng), alpha, beta, sigma

    def kernel_factory(generator):
        kernel = pydpf.StandardGaussian(1, generator, False, True)
        return pydpf.KernelMixture(kernel, generator=generator)

    for experiment in experiments:
        print(f"\nRunning {experiment}")
        ELBOs = np.empty(n_repeats)
        alphas = np.empty(n_repeats)
        betas = np.empty(n_repeats)
        sigmas = np.empty(n_repeats)
        for n in range(n_repeats):
            experiment_cuda_rng = torch.Generator(device).manual_seed(n * 10)
            generation_rng = torch.Generator(device).manual_seed(n * 10)
            experiment_cpu_rng = torch.Generator().manual_seed(n * 10)
            true_SSM = sv_make_SSM(torch.tensor([[true_sigma]], device=device),
                                   torch.tensor([[true_alpha]], device=device),
                                   torch.tensor([true_beta], device=device), device, generation_rng)
            pydpf.simulate_and_save(temp_data_path, SSM=true_SSM, time_extent=1000,
                                    n_trajectories=cfg.pick(500, 50), batch_size=batch_size_test,
                                    device=device, bypass_ask=True)
            SSM, alpha, beta, sigma = make_globals(experiment_cuda_rng)
            dpf = build_dpf(experiment, SSM, experiment_cuda_rng, ot_regularisation=0.5, ot_clip=1.,
                            kernel_factory=kernel_factory)
            if experiment == "Kernel":
                opt = torch.optim.SGD([{"params": [alpha], "lr": 0.05}, {"params": [beta], "lr": 0.1},
                                       {"params": [sigma], "lr": 0.25},
                                       {"params": dpf.resampler.mixture.parameters(), "lr": 0.1}],
                                      lr=0.2, momentum=0.9, nesterov=True)
            else:
                opt = torch.optim.SGD([{"params": [alpha], "lr": 0.05}, {"params": [beta], "lr": 0.1},
                                       {"params": [sigma], "lr": 0.25}], lr=0.2, momentum=0.9, nesterov=True)
            opt_schedule = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)
            dataset = pydpf.StateSpaceDataset(temp_data_path, state_prefix="state", device=device)
            _, ELBO = train_sv(dpf, opt, dataset, training_epochs, (100, 100, 100),
                               (batch_size_train, batch_size_test, batch_size_train),
                               (0.5, 0.25, 0.25), 1., experiment_cpu_rng, target="ELBO", time_extent=100,
                               lr_scheduler=opt_schedule)
            ELBOs[n] = ELBO
            alphas[n] = alpha
            betas[n] = beta
            sigmas[n] = sigma
            os.remove(temp_data_path)
        results = pd.read_csv(results_file, index_col=0)
        results.loc[experiment] = np.array([np.mean(ELBOs), np.mean(np.abs(alphas - 0.91)),
                                            np.mean(np.abs(betas - 0.5)), np.mean(np.abs(sigmas - 1.))])
        results.to_csv(results_file)
        print(results)


# --------------------------------------------------------------------------- #
#  Experiment 6: Deep-mind maze -- deep learning                              #
# --------------------------------------------------------------------------- #

def run_maze(cfg, requested_models):
    print("\n=== Deep-mind maze: deep learning ===")
    data_path = prepare_maze_data(cfg)
    device = cfg.device
    methods = select_methods(requested_models, DPF_METHODS)
    if not methods:
        return

    observation_encoding_size = 128
    state_encoding_size = 64
    scaling = 1000.
    epochs = cfg.pick(100, 1)
    n_repeats = cfg.pick(cfg.maze_repeats, 1)


    def flatten_gens(list_of_gens):
        return [item for gen in list_of_gens for item in gen]

    for det_label in cfg.maze_deterministic:
        deterministic = det_label == "deterministic"
        results_file = ("deep_mind_maze_results.csv" if deterministic
                        else "nondeterministic_deep_mind_maze_results.csv")
        if cfg.smoke:
            results_file= results_file[:-4] + "_smoke.csv"

        result_path = cfg.results_dir / results_file
        make_new_csv(DPF_METHODS, ["Total time (hrs:min:s)", "Test MSE"],
                     result_path, string_columns=["Total time (hrs:min:s)"], overwrite=cfg.overwrite_results)
        print(f"\n--- {'deterministic' if deterministic else 'non-deterministic'} run ---")

        with pydpf.utils.set_deterministic_mode(deterministic, True):
            for DPF_type in methods:
                print(f"\nRunning {DPF_type}")
                total_MSE = 0
                total_time = 0
                for i in range(n_repeats):
                    cuda_gen = torch.Generator(device=device).manual_seed(i * 10)

                    def get_SSM():
                        encoder = ObservationEncoder(observation_encoding_size, generator=cuda_gen, dropout_keep_ratio=0.3)
                        decoder = ObservationDecoder(observation_encoding_size, generator=cuda_gen, dropout_keep_ratio=0.3)
                        state_encoder = StateEncoder(state_encoding_size, generator=cuda_gen, dropout_keep_ratio=0.6)
                        observation_partial_flows = [
                            RealNVP_cond(dim=observation_encoding_size, hidden_dim=observation_encoding_size,
                                         condition_on_dim=state_encoding_size, generator=cuda_gen, zero_i=True),
                            RealNVP_cond(dim=observation_encoding_size, hidden_dim=observation_encoding_size,
                                         condition_on_dim=state_encoding_size, generator=cuda_gen, zero_i=True)]
                        flow_cov = torch.nn.Parameter(torch.eye(observation_encoding_size, device=device) * 1,
                                                      requires_grad=False)
                        observation_flow = NormalizingFlowModel_cond(
                            pydpf.MultivariateGaussian(torch.zeros(observation_encoding_size, device=device),
                                                       cholesky_covariance=flow_cov, diagonal_cov=True,
                                                       generator=cuda_gen), observation_partial_flows, device)
                        observation_model = MazeObservation(observation_flow, encoder, decoder, state_encoder, device=device)
                        dynamic_cov = torch.diag(torch.tensor([30 / scaling, 30 / scaling, 0.1], device=device))
                        dynamic_model = MazeDynamic(cuda_gen, dynamic_cov)
                        prior_model = MazePrior(2 * 1000 / scaling, 1.3 * 1000 / scaling, cuda_gen)
                        encoder_parameters = flatten_gens([encoder.parameters(), state_encoder.parameters(),
                                                           decoder.parameters()])
                        flow_parameters = flatten_gens([observation_flow.parameters(), prior_model.parameters()])
                        SSM = pydpf.FilteringModel(dynamic_model=dynamic_model, prior_model=prior_model,
                                                   observation_model=observation_model)
                        return SSM, encoder_parameters, flow_parameters, [flow_cov]

                    def maze_kernel_factory(generator):
                        Gaussian_kernel = pydpf.StandardGaussian(3, generator, learn_mean=False, learn_cov=True)
                        return pydpf.KernelMixture(kernel=Gaussian_kernel, generator=generator)

                    SSM, encoder_params, flow_params, flow_cov = get_SSM()
                    # The maze Optimal-Transport DPF uses regularisation=1.; the kernel is 3-D.
                    dpf = build_dpf(DPF_type, SSM, cuda_gen, ot_regularisation=1.,
                                    kernel_factory=maze_kernel_factory)
                    dpf.to(device)
                    if DPF_type == "Kernel":
                        opt = torch.optim.AdamW(
                            [{"params": encoder_params, "lr": 0.005},
                             {"params": flow_params, "lr": 0.001},
                             {"params": dpf.resampler.mixture.parameters(), "lr": 0.001, "weight_decay": 0}],
                            weight_decay=1e-3, betas=(0.7, 0.98), eps=1e-9)
                    else:
                        opt = torch.optim.AdamW(
                            [{"params": encoder_params, "lr": 0.005}, {"params": flow_params, "lr": 0.001}],
                            weight_decay=1e-2, betas=(0.8, 0.99), eps=1e-9)
                    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
                    data = pydpf.StateSpaceDataset(data_path=data_path, state_prefix="state",
                                                   control_prefix="control", device=device)
                    data.apply(lambda observation, **d: (observation - torch.mean(observation)) / torch.std(observation),
                               "observation")
                    scaling_tensor = torch.tensor([[[scaling, scaling, 1.]]], device=device)
                    data.apply(lambda state, **d: (state - torch.tensor([[[1000., 650., 0.]]], device=device)) / scaling_tensor,
                               "state")
                    data.apply(lambda control, **d: control / torch.tensor([[[scaling, scaling, 1.]]], device=device),
                               "control")
                    print("Data Loaded")
                    start_time = time.time()
                    test_mse, _ = train_maze(dpf, opt, data, epochs, (100, 100, 100), (64, 64, 64),
                                             (0.45, 0.2, 0.35), (1., 1., 1.),
                                             torch.Generator().manual_seed(i * 10), None, "MSE", 99,
                                             lr_scheduler=opt_scheduler, pre_train_epochs=0, device=device,
                                             state_scaling=scaling)
                    total_MSE += test_mse
                    total_time += time.time() - start_time
                MSE = total_MSE / n_repeats
                runtime = total_time / n_repeats
                results = pd.read_csv(result_path, index_col=0)
                time_col = "Total time (hrs:min:s)"
                results[time_col] = results[time_col].astype(object)
                results.loc[DPF_type] = [str(datetime.timedelta(seconds=runtime)), math.sqrt(MSE)]
                results[time_col] = results[time_col].fillna("")
                print(results)
                results.to_csv(result_path)


# --------------------------------------------------------------------------- #
#  Experiment 7: the short example_usage.py demonstration                     #
#  (kept self-contained; its model classes differ from sv_model on purpose)   #
# --------------------------------------------------------------------------- #

class ExampleSVDynamicModel(pydpf.Module):
    def __init__(self, alpha, sigma, device):
        super().__init__()
        self.alpha_ = torch.nn.Parameter(alpha)
        self.log_sigma = torch.nn.Parameter(torch.log(sigma))
        self.device = device

    @pydpf.constrained_parameter
    def alpha(self):
        return self.alpha_, torch.clip(self.alpha_, 1e-3, 1 - 1e-3)

    @pydpf.cached_property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def sample(self, prev_state, **data):
        noise = self.sigma * torch.normal(0, 1, device=self.device, size=prev_state.size())
        return prev_state * self.alpha + noise


class ExampleSVObservationModel(pydpf.Module):
    def __init__(self, beta, device):
        super().__init__()
        self.log_beta = torch.nn.Parameter(torch.log(beta))
        self.half_log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device)) / 2
        self.device = device

    @pydpf.cached_property
    def beta(self):
        return torch.exp(self.log_beta)

    def fitness(self, state, observation, **data):
        log_root_v = state + self.log_beta
        root_v = torch.exp(log_root_v)
        normalised_obs = observation.unsqueeze(1) / root_v
        return (-log_root_v - (normalised_obs ** 2) / 2 - self.half_log_2pi).squeeze()

    def sample(self, state, **data):
        log_root_v = state + self.log_beta
        root_v = torch.exp(log_root_v)
        return root_v * torch.normal(0, 1, device=self.device, size=state.size())


class ExampleSVPriorModel(pydpf.Module):
    def __init__(self, dynamic_model):
        super().__init__()
        self.device = dynamic_model.device
        self.dyn_mod = dynamic_model

    @pydpf.cached_property
    def sd(self):
        return torch.sqrt(self.dyn_mod.sigma ** 2 / (1 - self.dyn_mod.alpha ** 2))

    def sample(self, batch_size, n_particles, **data):
        return self.sd * torch.normal(0, 1, device=self.device, size=(batch_size, n_particles, 1))


def example_make_SSM(alpha, beta, sigma, device):
    dynamic = ExampleSVDynamicModel(alpha, sigma, device)
    observation = ExampleSVObservationModel(beta, device)
    prior = ExampleSVPriorModel(dynamic)
    return pydpf.FilteringModel(prior_model=prior, dynamic_model=dynamic, observation_model=observation)


def run_example_usage(cfg, empty):
    print("\n=== Stochastic Volatility: example_usage demonstration ===")
    device = cfg.device
    data_path = cfg.data_dir / "example_usage.csv"

    SSM = example_make_SSM(torch.tensor(0.91, device=device), torch.tensor(0.5, device=device),
                           torch.tensor(1., device=device), device)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=100, n_trajectories=200,
                            batch_size=100, device=device, bypass_ask=True)
    learned_SSM = example_make_SSM(torch.tensor(0.6, device=device), torch.tensor(0.2, device=device),
                                   torch.tensor(1.5, device=device), device)
    multinomial_base = pydpf.MultinomialResampler(generator=torch.Generator(device=device))
    soft_resampler = pydpf.SoftResampler(softness=0.7, base_resampler=multinomial_base, device=device)
    DPF = pydpf.ParticleFilter(soft_resampler, learned_SSM)
    full_dataset = pydpf.StateSpaceDataset(data_path, series_id_column="series_id", state_prefix="state",
                                           observation_prefix="observation", device=device)
    train_set, test_set = torch.utils.data.random_split(full_dataset, [0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=full_dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=full_dataset.collate)
    output_function = pydpf.MSE_Loss()
    opt = torch.optim.Adam(DPF.parameters(), lr=0.01)
    n_epochs = cfg.pick(50, 5)

    alpha_error, beta_error, sigma_error = [], [], []
    for e in range(n_epochs):
        train_loss = 0.0
        for state, observation in train_loader:
            opt.zero_grad()
            DPF.update()
            MSE = DPF(n_particles=64, time_extent=100, aggregation_function=output_function,
                      observation=observation, ground_truth=state)
            loss = MSE.mean()
            loss.backward()
            train_loss += loss.item()
            alpha_error.append(torch.abs(learned_SSM.dynamic_model.alpha - 0.91).item())
            beta_error.append(torch.abs(learned_SSM.observation_model.beta - 0.5).item())
            sigma_error.append(torch.abs(learned_SSM.dynamic_model.sigma - 1.).item())
            opt.step()
        if e % 10 == 0:
            print(f"Epoch {e + 1}, loss: {train_loss / len(train_loader)}")

    DPF.update()
    with torch.inference_mode():
        mean_loss = 0.0
        for state, observation in test_loader:
            MSE = DPF(n_particles=64, time_extent=100, aggregation_function=output_function,
                      observation=observation, ground_truth=state)
            mean_loss += MSE.mean().item()

    print(f"Test MSE: {mean_loss / len(test_loader)}")
    print(f"Learned alpha: {learned_SSM.dynamic_model.alpha.item()}")
    print(f"Learned beta: {learned_SSM.observation_model.beta.item()}")
    print(f"Learned sigma: {learned_SSM.dynamic_model.sigma.item()}")
    # Save (rather than show) the convergence plot so the run stays headless.
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.plot(np.array(alpha_error), label="|alpha - 0.91|")
        plt.plot(np.array(beta_error), label="|beta - 0.5|")
        plt.plot(np.array(sigma_error), label="|sigma - 1.0|")
        plt.legend()
        plt.xlabel("optimisation step")
        plt.ylabel("absolute parameter error")
        out = cfg.results_dir / f"example_usage_parameter_errors_{cfg.pick('','smoke')}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"Saved parameter-error plot to {out}")
    except Exception as exc:  # plotting is optional
        print(f"(skipping plot: {exc})")

# --------------------------------------------------------------------------- #
#  Experiment 8: run snippets from Section 7: Advanced usage                  #
# --------------------------------------------------------------------------- #

def wrap_example_multinomial_resampler_with_cache():
    class MultinomialResampler(pydpf.Module):

        def __init__(self, generator: torch.Generator):
            super().__init__()
            self.generator = generator
            self.cache = {}

        def forward(self, state, weight, **data):
            sampled_indices = torch.multinomial(torch.exp(weight),
                                                weight.size(1),
                                                replacement=True,
                                                generator=self.generator).detach()
            self.cache['used_weight'] = weight
            self.cache['sampled_indices'] = sampled_indices
            return (pydpf.batched_select(state, sampled_indices),
                    torch.full(weight.size(), -math.log(weight.size(1)), device=weight.device))
    return MultinomialResampler

def wrap_example_multinomial_resampler_without_cache():
    class MultinomialResampler(pydpf.Module):

        def __init__(self, generator: torch.Generator):
            super().__init__()
            self.generator = generator

        def forward(self, state, weight, **data):
            sampled_indices = torch.multinomial(torch.exp(weight),
                                                weight.size(1),
                                                replacement=True,
                                                generator=self.generator).detach()
            return (pydpf.batched_select(state, sampled_indices),
                    torch.full(weight.size(), -math.log(weight.size(1)), device=weight.device))
    return MultinomialResampler


class BootstrapSISInitialProp(pydpf.Module):

    def __init__(self, prior_model, observation_model):
        super().__init__()
        self.prior_model = prior_model
        self.observation_model = observation_model

    def forward(self, n_particles, observation, **data):
        state = self.prior_model.sample(n_particles=n_particles,
                                        batch_size=observation.size(0),
                                        **data)
        weights = self.observation_model.fitness(state=state,
                                               observation=observation,
                                               **data)
        normalised_weights, norm = pydpf.normalise(weights, dim=-1)
        return state, normalised_weights, norm - math.log(state.size(1))


class BootstrapSISProp(pydpf.Module):

    def __init__(self, dynamic_model, observation_model):
        super().__init__()
        self.dynamic_model = dynamic_model
        self.observation_model = observation_model

    def forward(self, prev_state, prev_weight, observation, **data):
        state = self.dynamic_model.sample(prev_state=prev_state, **data)
        fitness = self.observation_model.fitness(state=state,
                                             observation=observation,
                                             **data)
        normalised_weight, norm = pydpf.normalise(fitness + prev_weight, dim=-1)
        log_likelihood = pydpf.normalise(fitness, dim=-1)[1] - math.log(state.size(1))
        return state, normalised_weight, norm - log_likelihood

def make_filter_for_example_multinomial_resampler_without_cache(SSM, device):
    res = wrap_example_multinomial_resampler_without_cache()(generator=torch.Generator(device=device))
    DPF = pydpf.ParticleFilter(res, SSM)
    return DPF

def make_filter_for_example_multinomial_resampler_with_cache(SSM, device):
    res = wrap_example_multinomial_resampler_with_cache()(generator=torch.Generator(device=device))
    #Need to have cache defined to use compound resamplers in general
    soft_res = pydpf.SoftResampler(softness=0.7, base_resampler=res, device=device)
    DPF = pydpf.ParticleFilter(soft_res, SSM)
    return DPF

def make_filter_for_example_conditional_resample(SSM, device):
    gen = torch.Generator(device=device)
    cond_resampler = pydpf.ConditionalResampler(resampler=pydpf.MultinomialResampler(generator = gen), condition=pydpf.ESS_Condition(threshold=0.7),)
    DPF = pydpf.ParticleFilter(cond_resampler, SSM)
    return DPF

def make_filter_for_example_custom_alg(SSM, device):
    class BootstrapSISInitialProp(pydpf.Module):

        def __init__(self, prior_model, observation_model):
            super().__init__()
            self.prior_model = prior_model
            self.observation_model = observation_model

        def forward(self, n_particles, observation, **data):
            state = self.prior_model.sample(n_particles=n_particles,
                                            batch_size=observation.size(0),
                                            **data)
            weights = self.observation_model.fitness(state=state,
                                                   observation=observation,
                                                   **data)
            normalised_weights, norm = pydpf.normalise(weights, dim=-1)
            return state, normalised_weights, norm - math.log(state.size(1))


    class BootstrapSISProp(pydpf.Module):

        def __init__(self, dynamic_model, observation_model):
            super().__init__()
            self.dynamic_model = dynamic_model
            self.observation_model = observation_model

        def forward(self, prev_state, prev_weight, observation, **data):
            state = self.dynamic_model.sample(prev_state=prev_state, **data)
            fitness = self.observation_model.fitness(state=state,
                                                 observation=observation,
                                                 **data)
            normalised_weight, norm = pydpf.normalise(fitness + prev_weight, dim=-1)
            log_likelihood = pydpf.normalise(fitness, dim=-1)[1] - math.log(state.size(1))
            return state, normalised_weight, norm - log_likelihood

    custom_prior_model = SSM.prior_model
    custom_observation_model = SSM.observation_model
    custom_dynamic_model = SSM.dynamic_model

    custom_initial_proposal = BootstrapSISInitialProp(prior_model=custom_prior_model,
                                                      observation_model=custom_observation_model)
    custom_proposal = BootstrapSISProp(dynamic_model=custom_dynamic_model,
                                       observation_model=custom_observation_model)
    custom_filter = pydpf.SIS(initial_proposal=custom_initial_proposal, proposal=custom_proposal)
    return custom_filter


#Pretty similar to run_example_usage but copied so that run_example_usage is unchanged from the paper
def run_example_usage_generalised(cfg, name, make_filter):
    print(f"\n=== Advanced usage: testing {name} ===")
    device = cfg.device
    data_path = cfg.data_dir / "example_usage.csv"

    SSM = example_make_SSM(torch.tensor(0.91, device=device), torch.tensor(0.5, device=device),
                           torch.tensor(1., device=device), device)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=100, n_trajectories=200,
                            batch_size=100, device=device, bypass_ask=True)
    learned_SSM = example_make_SSM(torch.tensor(0.6, device=device), torch.tensor(0.2, device=device),
                                   torch.tensor(1.5, device=device), device)
    DPF = make_filter(learned_SSM, device)
    full_dataset = pydpf.StateSpaceDataset(data_path, series_id_column="series_id", state_prefix="state",
                                           observation_prefix="observation", device=device)
    train_set, test_set = torch.utils.data.random_split(full_dataset, [0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=full_dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=full_dataset.collate)
    output_function = pydpf.MSE_Loss()
    opt = torch.optim.Adam(DPF.parameters(), lr=0.01)
    n_epochs = cfg.pick(50, 5)

    alpha_error, beta_error, sigma_error = [], [], []
    for e in range(n_epochs):
        train_loss = 0.0
        for state, observation in train_loader:
            opt.zero_grad()
            DPF.update()
            MSE = DPF(n_particles=cfg.pick(64, 8), time_extent=cfg.pick(100, 10), aggregation_function=output_function,
                      observation=observation, ground_truth=state)
            loss = MSE.mean()
            loss.backward()
            train_loss += loss.item()
            alpha_error.append(torch.abs(learned_SSM.dynamic_model.alpha - 0.91).item())
            beta_error.append(torch.abs(learned_SSM.observation_model.beta - 0.5).item())
            sigma_error.append(torch.abs(learned_SSM.dynamic_model.sigma - 1.).item())
            opt.step()
        if e % 10 == 0:
            print(f"Epoch {e + 1}, loss: {train_loss / len(train_loader)}")

    DPF.update()
    with torch.inference_mode():
        mean_loss = 0.0
        for state, observation in test_loader:
            MSE = DPF(n_particles= 64, time_extent=100, aggregation_function=output_function,
                      observation=observation, ground_truth=state)
            mean_loss += MSE.mean().item()

    print(f"Test MSE: {mean_loss / len(test_loader)}")
    print(f"Learned alpha: {learned_SSM.dynamic_model.alpha.item()}")
    print(f"Learned beta: {learned_SSM.observation_model.beta.item()}")
    print(f"Learned sigma: {learned_SSM.dynamic_model.sigma.item()}")
    # Save (rather than show) the convergence plot so the run stays headless.
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        plt.plot(np.array(alpha_error), label="|alpha - 0.91|")
        plt.plot(np.array(beta_error), label="|beta - 0.5|")
        plt.plot(np.array(sigma_error), label="|sigma - 1.0|")
        plt.legend()
        plt.xlabel("optimisation step")
        plt.ylabel("absolute parameter error")
        out = cfg.results_dir / f"{name}_parameter_errors_{cfg.pick('','smoke')}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"Saved parameter-error plot to {out}")
    except Exception as exc:  # plotting is optional
        print(f"(skipping plot: {exc})")


def run_advanced_usage_tests(cfg, empty):
    print(f"\n=== Testing Advanced usage snippets ===")
    run_example_usage_generalised(cfg, "Conditional resampler", make_filter_for_example_conditional_resample)
    run_example_usage_generalised(cfg, "Custom resampler without cache", make_filter_for_example_multinomial_resampler_without_cache)
    run_example_usage_generalised(cfg, "Custom resampler with cache", make_filter_for_example_multinomial_resampler_with_cache)
    run_example_usage_generalised(cfg, "Custom filter", make_filter_for_example_custom_alg)
    print("\n=== Finished testing advanced usage snippets ===")


# --------------------------------------------------------------------------- #
#  Dispatch / CLI                                                             #
# --------------------------------------------------------------------------- #

EXPERIMENTS = {
    "kalman": run_kalman,
    "proposal": run_proposal,
    "sv_filtering": run_sv_filtering,
    "sv_single": run_sv_single,
    "sv_multiple": run_sv_multiple,
    "maze": run_maze,
    "example_usage": run_example_usage,
    "advanced_usage": run_advanced_usage_tests,
}

# Methods (rows) supported by each experiment, for the --help text.
EXPERIMENT_METHODS = {
    "kalman": ["25", "100", "1000", "10000"],
    "proposal": ["Bootstrap", "Optimal"] + DPF_METHODS,
    "sv_filtering": DPF_METHODS,
    "sv_single": DPF_METHODS,
    "sv_multiple": DPF_METHODS,
    "maze": DPF_METHODS,
    "example_usage": [],
    "advanced_usage": []
}

# Run everything by default
DEFAULT_EXPERIMENTS = ["example_usage", "advanced_usage", "kalman", "proposal", "sv_filtering", "sv_single", "sv_multiple", "maze"]


def build_parser():
    epilog = "Methods available per experiment (pass with --models):\n"
    for name in EXPERIMENTS:
        ms = EXPERIMENT_METHODS[name]
        epilog += f"  {name:<14} {', '.join(ms) if ms else '(no method selection)'}\n"
    epilog += ("\nExamples:\n"
               "  python run_experiments.py\n"
               "  python run_experiments.py --experiment sv_filtering --models Soft \"Stop-Gradient\"\n"
               "  python run_experiments.py -e maze --models DPF --device cpu --smoke\n")
    parser = argparse.ArgumentParser(
        description="Run any/all of the pydpf JSS experiments with a single command.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    parser.add_argument("-e", "--experiment", nargs="+", default=None,
                        choices=list(EXPERIMENTS) + ["all"],
                        help="Which experiment(s) to run. Default: all paper experiments ")
    parser.add_argument("-m", "--models", nargs="+", default=None,
                        help="Restrict to these methods/models (names as listed below). "
                             "Default: every method of each selected experiment.")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, cuda:0, ...")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Central data folder.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Central results folder.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny epochs/repeats/particle counts to validate the pipeline quickly.")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only prepare the central data/results folders, then exit.")
    parser.add_argument("--overwrite-results", action="store_true",
                        help="Recreate (blank out) the results CSVs of the selected experiments, "
                             "even if they already exist.")
    # Data-generation parameters (match the original *_setup scripts).
    parser.add_argument("--dx", type=int, default=25, help="Linear-Gaussian state dimension.")
    parser.add_argument("--dy", type=int, default=1, help="Linear-Gaussian observation dimension.")
    parser.add_argument("--alpha", type=float, default=0.91, help="SV generation alpha.")
    parser.add_argument("--beta", type=float, default=0.5, help="SV generation beta.")
    parser.add_argument("--sigma", type=float, default=1.0, help="SV generation sigma.")
    parser.add_argument("--batch-size", type=int, default=128, help="Generation / evaluation batch size.")
    parser.add_argument("--maze-deterministic", choices=["deterministic", "nondeterministic", "both"],
                        default="both", help="Which maze run(s) to perform.")
    parser.add_argument("--maze-repeats", type=int, default=5,
                        help="Number of repeats for the maze experiment (averaged).")
    parser.add_argument("--delete-raw", action="store_false",
                        help="Delete the raw maze archives after building the maze data set.")
    return parser


def resolve_device(name):
    if name == "auto":
        if not torch.cuda.is_available():
            warnings.warn("Warning CUDA not available, defaulting to CPU")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.experiment is None or "all" in args.experiment:
        experiments = list(DEFAULT_EXPERIMENTS)
    else:
        # Preserve a sensible, de-duplicated order.
        experiments = [e for e in EXPERIMENTS if e in args.experiment]

    device = resolve_device(args.device)
    maze_det = {"deterministic": ("deterministic",), "nondeterministic": ("nondeterministic",),
                "both": ("deterministic", "nondeterministic")}[args.maze_deterministic]

    cfg = RunConfig(device=device, data_dir=args.data_dir, results_dir=args.results_dir,
                    smoke=args.smoke, dx=args.dx, dy=args.dy, alpha=args.alpha, beta=args.beta,
                    sigma=args.sigma, batch_size=args.batch_size, maze_deterministic=maze_det,
                    maze_repeats=args.maze_repeats, delete_raw=args.delete_raw,
                    overwrite_results=args.overwrite_results)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {cfg.device}")
    print(f"Data folder:    {cfg.data_dir}")
    print(f"Results folder: {cfg.results_dir}")
    print(f"Experiments:    {', '.join(experiments)}")
    if args.models:
        print(f"Restricted to methods: {', '.join(args.models)}")
    if cfg.smoke:
        print("** SMOKE MODE: reduced epochs/repeats/particles -- results are NOT paper-accurate **")

    if args.setup_only:
        # Prepare only the data needed by the selected experiments.
        if "kalman" in experiments or "proposal" in experiments:
            prepare_lg_data(cfg)
        if "sv_filtering" in experiments:
            prepare_sv_data(cfg)
        if "sv_single" in experiments:
            prepare_sv_test_trajectory(cfg)
        if "maze" in experiments:
            prepare_maze_data(cfg)
        print("Setup complete.")
        return

    failures = []
    for name in experiments:
        try:
            EXPERIMENTS[name](cfg, args.models)
        except Exception:
            # Keep going so that one failing experiment does not abort the rest.
            print(f"\n!! Experiment '{name}' failed:")
            traceback.print_exc()
            failures.append(name)

    if failures:
        print(f"\nFinished, but the following experiment(s) failed: {', '.join(failures)}")
    else:
        print("\nAll requested experiments finished.")


if __name__ == "__main__":
    main()
