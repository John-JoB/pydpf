"""
Python file to contain the functions for performing resampling.

For use with our filtering algorithms the resampling functions must take the particle positions and log normalised weights
and return the resampled positions, resampled normalised weights.

We keep the usual pytorch design pattern of passing parameters/hyperparameters at object creation. I.e. the top-level functions in this file
are functions from the hyperparameters to a Callable with the above specified signature. In most cases the resampling algorithm has no trainable
parameters, so the returned object is a simple python function, but if it does then the object is a Module with the forward() method implemented.
"""
import torch
from torch import Tensor
from typing import Tuple, Any, Optional
from pydpf.utils import batched_select, normalise
from pydpf.base import Module, cached_property
from pydpf.custom_types import WeightedSample
from pydpf import optimal_transport

class MultinomialResampler(Module):

    def __init__(self, generator:torch.Generator):
        """
            Multinomial resampler.


            Parameters
            ----------
            generator: torch.Generator
                The generator to track the random state of the resampling process.
        """

        super().__init__()
        self.cache = {}
        self.generator = generator
        self._need_weight_output = True

    def forward(self, state:Tensor, weight:Tensor, **data) -> WeightedSample:
        with torch.no_grad():
            sampled_indices = torch.multinomial(torch.exp(weight), weight.size(1), replacement=True, generator=self.generator).detach()
            self.cache['used_weight'] = weight
            self.cache['sampled_indices'] = sampled_indices
        if self._need_weight_output:
            return batched_select(state, sampled_indices), torch.zeros_like(weight)
        return batched_select(state, sampled_indices)

class SystematicResampler(Module):
    def __init__(self, generator:torch.Generator):
        """
            Systematic resampler. Equivalent to the Systematic resampler described in Chopin and Papaspiliopoulis 2020.

            Under systematic resampling, the expected number of times a given particle is resampled is the same as for multinomial resampling. But it
            inter-correlates all the particles within a sample so it is difficult to provide the same theoretical guarantees on the asymptotic
            behaviour of filters that use systematic resampling compared to multinomial resampling. However, the stability offered by systematic
            resampling often results in better performance in practice.

            Warnings
            ---------
            Systematic resampling introduces strong dependence between particles and their index. Should the forward kernel be dependent on the
            particle index then the particles should be shuffled after resampling.


            Parameters
            ----------
            generator: torch.Generator
                The generator to track the random state of the resampling process.
        """
        super().__init__()
        self.cache = {}
        self.generator = generator
        self._need_weight_output = True

    def forward(self, state:Tensor, weight:Tensor, **data) -> WeightedSample:
        with torch.no_grad():
            offset = torch.rand((weight.size(0),), device=state.device, generator=self.generator)
            cum_probs = torch.cumsum(torch.exp(weight), dim=1)
            # No index can be above 1. and the last index must be exactly 1.
            # Fix this in case of numerical errors
            cum_probs = torch.where(cum_probs > 1., 1., cum_probs)
            cum_probs[:, -1] = 1.
            resampling_points = torch.arange(weight.size(1), device=state.device) + offset.unsqueeze(1)
            sampled_indices = torch.searchsorted(cum_probs * weight.size(1), resampling_points)
            self.cache['used_weight'] = weight
            self.cache['sampled_indices'] = sampled_indices
        if self._need_weight_output:
            return batched_select(state, sampled_indices), torch.zeros_like(weight)
        return batched_select(state, sampled_indices)


class SoftResampler(Module):
    def __init__(self, softness:float, base_resampler:Module, device:torch.device):
        """
            Module for perfoming soft-resampling, (P. Karkus, D. Hsu and W. S. Lee 'Particle Filter Networks with Application to
            Visual Localization' 2018).

            Soft resampling allows gradients to be passed through resampling by inducing importance weights. This is done by instead drawing the
            resampled particle from an alternative distribution and re-weighting the samples. The chosen alternative distribution is a mixture of
            the target with probability a; and a uniform distribution over the particles, with probability 1-a.

            The softness parameter, a, can be thought of as trading off between unbiased gradients (a = 0) and efficient resampling (a = 1). With
            a > 0, the resampled index depends (randomly) on the previous weights. The contribution to the gradient from this dependence is ignored.

            The underlying resampling algorithm is systematic, see resampling.systematic for details.

            Parameters
            ----------
            softness:  float
                The trade-off parameter between
            generator: torch.Generator
                The generator to track the random state of the resampling process.
        """
        super().__init__()
        self.cache = {}
        self.softness = softness
        if softness < 0 or softness >= 1:
            raise ValueError(f'Softness {softness} is out of range, must be in [0,1)')
        self.log_softness = torch.log(torch.tensor(softness, device = device))
        self.neg_log_softness = torch.log(torch.tensor(1-softness ,device = device))
        self.resampler = base_resampler
        self._need_weight_output = True

    def forward(self, state:Tensor, weight:Tensor, **data) -> WeightedSample:
        soft_weight = torch.logaddexp(weight + self.log_softness, self.neg_log_softness - torch.log(torch.tensor(weight.size(1), device=state.device)))
        state, output_weights = self.resampler(state, soft_weight)
        self.cache = self.resampler.cache
        if self._need_weight_output:
            return state, batched_select(weight, self.cache['sampled_indices']) - batched_select(self.cache['used_weight'], self.cache['sampled_indices'])
        return state


class StopGradientResampler(Module):
    '''
    Module for  stop-gradient resampling, (A. Scibor and F. Wood 'Differentiable Particle Filtering without
    Modifying the Forward Pass' 2021).

    Stop-gradient resampling uses the REINFORCE or score-based Monte-Carlo gradient technique. Unlike soft-resampling REINFORCE is unbiased.
    For numerical stability our implementation attaches the gradients to the log-space weights, rather than the linear-space particles.

    Parameters
    ----------
    generator: torch.Generator
        The generator to track the random state of the resampling process.

    Returns
    -------
    SoftResampler: Resampler
        The systematic resampling function.

    '''
    def __init__(self, base_resampler:Module):
        super().__init__()
        self.cache = {}
        self.base_resampler = base_resampler
        self._need_weight_output = True

    def forward(self, state:Tensor, weight:Tensor, **data) -> WeightedSample:

        if self._need_weight_output and torch.is_grad_enabled():
            self.base_resampler._need_weight_output = False
            state = self.base_resampler(state, weight)
            resampled_weight = batched_select(self.base_resampler.cache['used_weight'], self.base_resampler.cache['sampled_indices'])
            self.cache = self.base_resampler.cache
            return state, resampled_weight - resampled_weight.detach()

        if self._need_weight_output:
            self.base_resampler._need_weight_output = True
            state, new_weight = self.base_resampler(state, weight)
            self.cache = self.base_resampler.cache
            return state, new_weight

        self.base_resampler._need_weight_output = False
        state = self.base_resampler(state, weight)
        self.cache = self.base_resampler.cache
        return state

class OptimalTransportResampler(Module):


    def __init__(self, regularisation: float, decay_rate: float, min_update_size: float, max_iterations: int, transport_gradient_clip: float):
        r"""
            Module for performing optimal transport resampling, (A. Corenflos, J. Thornton, G. Deligiannidis and A. Doucet
            'Differentiable Particle Filtering via Entropy-Regularized Optimal Transport' 2021)

            Optimal transport resampling produces a differentiable deterministic transport map from the proposal distribution to the posterior.
            This is achieved by finding the solution to an entropy regularised Kantorovich optimal transport problem between the two empirical
            distributions. The particles are transformed by the resulting optimal map to obtain a new unweighted approximation of the posterior.

            Our implementation is closely based on the original code of Thornton and Corenflos, the following details being taken from theirs:
            We anneal the regularisation strength over the Sinkhorn iterations.
            We chose the initial strength of the regularisation parameter to be equal to maximum of the per-dimension standard deviations
            of the particle positions.
            For numerical stability we cap the magnitude of the contribution to the gradient due to the transport matrix.

            Warnings
            --------
            Optimal transport resampling places particles in new positions on $\mathbb{R}^n$, so it cannot directly be applied when some component of
            the state space is discrete/categorical.

            Optimal transport resampling results in biased (but asymptotically consistent) estimates of all non-affine functions of the latent state.
            Including the likelihood. The authors of the proposing paper investigate this effect and find it sufficiently small to ignore. See their
            paper for details.


            Parameters
            ----------
            regularisation: float
                The maximum strength of the entropy regularisation, in our implementation regularisation automatically chosen per sample and
                 annealed.
            step_size: float
                The factor by which to decrease the entropy regularisation per Sinkhorn loop.
            min_update_size: float
                The size of update to the transport potentials below which iteration should stop.
            max_iterations: int
                The maximum number iterations of the Sinkhorn loop, before stopping. Regardless of convergence.
            transport_gradient_clip: float
                The maximum per-element gradient of the transport matrix that should be passed. Higher valued gradients will be clipped to this value.

            Returns
            -------
            OTResampler: Resampler
                The optimal transport resampling function.

            """
        super().__init__()
        self.cache = {}
        self.regularisation = regularisation
        self.decay_rate = decay_rate
        self.min_update_size = min_update_size
        self.max_iterations = max_iterations
        self._need_weight_output = True

        class OTGradientWrapper(torch.autograd.Function):
            """
            Optimal transport gradient can suffer from numerical instability.
            Clip the gradient of the loss wrt the transport matrix to some user specified value.
            This is done in Corenflos and Thornton's original implementation.
            """

            @staticmethod
            def forward(ctx: Any, transport_matrix: Tensor):
                return transport_matrix

            @staticmethod
            def backward(ctx: Any, dtransport) -> Any:
                return torch.clip(dtransport, -transport_gradient_clip, transport_gradient_clip)

        self.gradient_wrapper = OTGradientWrapper

    @staticmethod
    def diameter(x: Tensor):
        """
        Calculates the diameter of the data.
        The diameter is defined as the maximum of the standard deviation across a sample across data dimensions.

        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
        diameter: Tensor
            The diameter of the data per batch.
        """
        diameter_x = torch.amax(x.std(dim=1, unbiased=False), dim=-1, keepdim=True)
        return torch.where(torch.eq(diameter_x, 0.), 1., diameter_x)

    @staticmethod
    def extent(x: Tensor):
        return torch.amax(x, dim=(1,2)) - torch.amin(x, dim=(1,2))

    @staticmethod
    def get_sinkhorn_inputs_OT(Nk, log_weight: Tensor, x_t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get the inputs to the Sinkhorn algorithm as used for OT resampling

        Parameters
        -----------
        log_weights: (B,N) Tensor
            The particle weights

        N: int
            Number of particles

        x_t: (B,N,D) Tensor
            The particle state

        Returns
        -------------
        log_uniform_weights: (B,N) Tensor
            A tensor of log(1/N)

        cost_matrix: (B, N, N) Tensor
            The auto-distance matrix of scaled_x_t under the 2-Norm.

        scale_x: (B, N, D) Tensor
            The amount the particles where scaled by in calculating the cost matrix.
        """
        log_uniform_weight = torch.log(torch.ones((log_weight.size(0), Nk), device=log_weight.device) / Nk)
        centred_x_t = x_t - torch.mean(x_t, dim=1, keepdim=True).detach()
        scale_x = OptimalTransportResampler.diameter(x_t).detach()
        scaled_x_t = centred_x_t / scale_x.unsqueeze(2)
        cost_matrix = torch.cdist(scaled_x_t, scaled_x_t, 2) ** 2
        extent = OptimalTransportResampler.extent(scaled_x_t)
        return log_uniform_weight, cost_matrix, extent


    def forward(self, state: Tensor, weight: Tensor, **data) -> WeightedSample:
        N = state.size(1)
        log_b, cost, extent = self.get_sinkhorn_inputs_OT(N, weight, state)
        f, g, epsilon_used = optimal_transport.sinkhorn_loop(weight, log_b, cost, self.regularisation, self.min_update_size, self.max_iterations, extent.reshape(-1, 1, 1), self.decay_rate)
        transport = optimal_transport.get_transport_from_potentials(weight, log_b, cost, f, g, self.regularisation)
        transport = self.gradient_wrapper.apply(transport)
        self.cache['used_weight'] = weight
        if self._need_weight_output:
            return optimal_transport.apply_transport(state, transport, N), torch.zeros_like(weight)
        return optimal_transport.apply_transport(state, transport, N)

class KernelResampler(Module):


    def __init__(self, kernel):
        """
            Returns a function for performing differentiable kernel resampling (Younis and Sudderth 'Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes' 2024).

            Notes
            -----
            Unlike the majority of implemented resampling schemes this function returns a Module object so that learnable parameters may be properly registered.
            In this case the final returned field is always a tensor containing a single zero.

            Parameters
            ----------
            kernel: Distribution
                The kernel to convolve over the particles to form the KDE sampling distribution.
            generator: torch.Generator
                The generator to track the random state of the resampling process.

            Returns
            -------
            kernel_resampler: Resampler:
                A Module whose forward method implements kernel resampling.
        """
        super().__init__()
        self.mixture = kernel
        self.cache = {}
        self._need_weight_output = True

    def forward(self, state: Tensor, weight: Tensor, **data) -> WeightedSample:
        new_state = self.mixture.sample(state, weight, sample_size=(state.size(1),))
        self.cache = self.mixture.resampler.cache
        # Save computation if gradient is not required
        if torch.is_grad_enabled() and self._need_weight_output:
            density = self.mixture.log_density(new_state, state, weight)
            new_weight = density - density.detach()
            return new_state, new_weight

        if self._need_weight_output:
            return new_state, torch.zeros_like(weight)

        return new_state