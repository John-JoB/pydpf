'''
Python file to contain the functions for performing resampling.

For use with our filtering algorithms the resampling functions must take the particle positions and log normalised weights
and return the resampled positions, resampled normalised weights and some other tensor, which can be used to report on intermediates
of the resampling process. Usually this is the resampled indices.

We keep the usual pytorch design pattern of passing parameters/hyperparameters at object creation. I.e. the top-level functions in this file
are functions from the hyperparameters to a Callable with the above specified signature. In most cases the resampling algorithm has no trainable
parameters, so the returned object is a simple python function, but if it does then the object is a Module with the forward() method implemented.
'''

import torch
from torch import Tensor
from typing import Tuple, Any, Callable
from .utils import batched_select
from .distributions import KernelMixture, Distribution

def multinomial(generator: torch.Generator) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    '''
    Returns a function to perform multinomial resampling.

    Each particle is redrawn as independent samples from a categorical distribution with probabilities specified by the log-weights.

    Parameters
    ----------
    generator: torch.Generator
        The generator to track the random state of the resampling process.

    Returns
    -------
    MultinomialResampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The multinomial resampling function.
    '''
    def _multinomial(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            sampled_indices = torch.multinomial(torch.exp(weights), weights.size(1), replacement=True, generator=generator).detach()
        return batched_select(state, sampled_indices), torch.zeros_like(weights), sampled_indices
    return _multinomial

def systematic(generator: torch.Generator) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    '''
    Returns a function to perform multinomial resampling as described in An Introduction to Sequential Monte Carlo (Chopin and Papaspiliopoulos 2020).

    Under systematic resampling, the expected number of times a given particle is resampled is the same as for multi-nomial resampling. But it
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

    Returns
    -------
    SystematicResampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The systematic resampling function.
    '''
    def _systematic(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            offset = torch.rand((weights.size(0),), device=state.device, generator=generator)
            cum_probs = torch.cumsum(torch.exp(weights), dim= 1)
            #No index can be above 1. and the last index must be exactly 1.
            #Fix this in case of numerical errors
            cum_probs = torch.where(cum_probs > 1., 1., cum_probs)
            cum_probs[:,-1] = 1.
            resampling_points = torch.arange(weights.size(1), device=state.device) + offset.unsqueeze(1)
            sampled_indices = torch.searchsorted(cum_probs * weights.size(1), resampling_points)
        return batched_select(state, sampled_indices), torch.zeros_like(weights), sampled_indices
    return _systematic


def soft(softness: float, generator: torch.Generator):
    '''
    Returns a function for perfoming soft-resampling, (P. Karkus, D. Hsu and W. S. Lee 'Particle Filter Networks with Application to
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

    Returns
    -------
    SoftResampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The systematic resampling function.

    '''
    if softness < 0 or softness >= 1:
        raise ValueError(f'Softness {softness} is out of range, must be in [0,1)')
    log_softness = torch.log(torch.tensor([softness]))
    neg_log_softness = torch.log(torch.tensor([1 - softness]))
    _systematic = systematic(generator)
    def _soft(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        nonlocal log_softness, neg_log_softness
        log_softness = log_softness.to(device=state.device)
        neg_log_softness = neg_log_softness.to(device=state.device)
        soft_weights = torch.logaddexp(weights + log_softness, neg_log_softness - torch.log(torch.tensor(weights.size(1), device = state.device)))
        state, _, sampled_indices = _systematic(state, soft_weights)
        return state, weights - soft_weights, sampled_indices
    return _soft


def soft_multinomial(softness: float, generator: torch.Generator):
    '''

    Parameters
    ----------
    softness
    generator

    Returns
    -------

    '''
    if softness < 0 or softness >= 1:
        raise ValueError(f'Softness {softness} is out of range, must be in [0,1)')
    log_softness = torch.log(torch.tensor([softness]))
    neg_log_softness = torch.log(torch.tensor([1 - softness]))
    _multinomial = multinomial(generator=generator)
    def _soft_systematic(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        nonlocal log_softness, neg_log_softness
        log_softness = log_softness.to(device = state.device)
        neg_log_softness = neg_log_softness.to(device = state.device)
        soft_weights = torch.logaddexp(weights + log_softness, neg_log_softness - torch.log(torch.tensor(weights.size(1), device = state.device)))
        state, _, sampled_indices = _multinomial(state, soft_weights)
        return state, weights - soft_weights, sampled_indices
    return _soft_systematic


def stop_gradient(generator: torch.Generator):
    '''
    Returns a function for perfoming stop-gradient resampling, (A. Scibor and F. Wood 'Differentiable Particle Filtering without
    Modifying the Forward Pass' 2021).

    Stop-gradient resampling uses the REINFORCE or score-based Monte-Carlo gradient technique. Unlike soft-resampling REINFORCE is unbiased.
    For numerical stability our implementation attaches the gradients to the log-space weights, rather than the linear-space particles.

    Parameters
    ----------
    generator: torch.Generator
        The generator to track the random state of the resampling process.

    Returns
    -------
    SoftResampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The systematic resampling function.

    '''
    _systematic = systematic(generator=generator)
    def _stop_gradient(state: Tensor, weight: Tensor):
        state, no_grad_weights, sampled_indices = _systematic(state, weight)
        #Save computation if gradient is not required
        if torch.is_grad_enabled():
            resampled_weights = batched_select(weight, sampled_indices)
            return state, resampled_weights - resampled_weights.detach(), sampled_indices
        else:
            return state, no_grad_weights, sampled_indices
    return _stop_gradient

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

def get_sinkhorn_inputs_OT(Nk, log_weights: Tensor, x_t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
    log_uniform_weights = torch.log(torch.ones((log_weights.size(0), Nk), device=log_weights.device) / Nk)
    centred_x_t = x_t - torch.mean(x_t, dim=1, keepdim=True).detach()
    scale_x = diameter(x_t).detach()
    scaled_x_t = centred_x_t / scale_x.unsqueeze(2)
    cost_matrix = torch.cdist(scaled_x_t, scaled_x_t, 2) ** 2
    return log_uniform_weights, cost_matrix, scale_x


def get_transport_from_potentials(log_a: Tensor, log_b: Tensor, cost: Tensor, f: Tensor, g: Tensor, epsilon: float) -> Tensor:
    """
    Calculates the transport matrix from the Sinkhorn potentials

    Parameters
    ------------

    log_a: (B,M) Tensor
            log of the weights of the proposal distribution

    log_b: (B,N) Tensor
        log of the weights of the target distribution

    cost: (B,M,N) Tensor
        The per unit cost of transporting mass from the proposal to the target

    f: (B,M) pt.Tensor
            Potential on the proposal

    g: (B,N) pt.Tensor
        Potential on the target

    epsilon: float
        Regularising parameter

    Returns
    ---------

    T: (B,M,N)
        The transport matrix
    """
    log_prefactor = log_b.unsqueeze(1) + log_a.unsqueeze(2)
    # Outer sum of f and g
    f_ = torch.unsqueeze(f, 2)
    g_ = torch.unsqueeze(g, 1)
    exponent = (f_ + g_ - cost) / epsilon
    log_transportation_matrix = log_prefactor + exponent
    return torch.exp(log_transportation_matrix)


def apply_transport(x_t: Tensor, transport: Tensor, N: int) -> Tensor:
    """
    Apply a transport matrix to a vector of particles

    Parameters
    -------------
    x_t: (B,N,D) Tensor
        Particle locations to be transported

    transport: (B,M,N) Tensor
        The transport matrix

    N: int
        Number of particles

    """
    return (N * torch.transpose(transport, 1, 2)) @ x_t


def opt_potential(log_a: Tensor, c_potential: Tensor, cost: Tensor, epsilon: Tensor) -> Tensor:
    """
        Calculates the update in the Sinkhorn loop for distribution b (either proposal or target)

        Parameters
        -----------
        log_a: (B,N) Tensor
            log of the weights of distribution a

        c_potential: (B, N) Tensor
            the current potential of distribution a

        cost: (B,N,M) Tensor
            The per unit cost of transporting mass from distribution a to distribution b

        epsilon: float
            Regularising parameter

        Returns
        -----------
        n_potential: (B, M) pt.Tensor
            The updated potential of distribution b


    """
    temp = log_a.unsqueeze(2) + (c_potential.unsqueeze(2) - cost) / epsilon
    temp = torch.logsumexp(temp, dim=1)
    return -epsilon.squeeze(2) * temp

def sinkhorn_loop(log_a: Tensor, log_b: Tensor, cost: Tensor, epsilon: float, threshold: float, max_iter: int, diam: Tensor, rate: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
        Calculates the Sinkhorn potentials for entropy regularised optimal transport between two atomic distributions via the Sinkhorn algorithm

        Parameters
        ---------------
        log_a: (B,M) Tensor
            log of the weights of the proposal distribution

        log_b: (B,N) Tensor
            log of the weights of the target distribution

        cost: (B,M,N) Tensor
            The per unit cost of transporting mass from the proposal to the target

        epsilon: float
            Regularising parameter

        threshold: float
            The difference in iteratations below which to halt and return

        max_iter: int
            The maximum amount of iterations to run regardless of whether the threshold is hit

        diam: Tensor
            The diameter of the data, used to

        Returns
        ---------------

        f: (B,M) Tensor
            Potential on the proposal

        g: (B,N) Tensor
            Potential on the target

        Notes
        -----------
        Due to convergening to a point, this implementation only retains the gradient at the last step
    """
    device = log_a.device
    i = 1
    f_i = torch.zeros_like(log_a, device=device)
    g_i = torch.zeros_like(log_b, device=device)
    cost_T = torch.transpose(cost, 1, 2)
    epsilon_now = torch.clip(diam ** 2, max=epsilon)
    continue_criterion = torch.ones((f_i.size(0),), device=device, dtype=torch.bool).unsqueeze(1)

    def stop_criterion(i_, continue_criterion_):
        return i_ < max_iter and torch.any(continue_criterion_)

    #Point convergence, the gradient due to the last step can be substituted for the gradient of the whole loop.
    with torch.no_grad():
        while stop_criterion(i, continue_criterion):
            f_u = torch.where(continue_criterion, (f_i + opt_potential(log_b, g_i, cost_T, epsilon_now)) / 2, f_i)
            g_u = torch.where(continue_criterion, (g_i + opt_potential(log_a, f_i, cost, epsilon_now)) / 2, g_i)
            update_size = torch.maximum(torch.abs(f_u - f_i), torch.abs(g_u - g_i))
            update_size = torch.max(update_size, dim=1)[0]
            continue_criterion = torch.logical_or(update_size > threshold, epsilon_now.squeeze() > epsilon).unsqueeze(1)
            epsilon_now = torch.clip(rate * epsilon_now, epsilon)
            f_i = f_u
            g_i = g_u
            i += 1
    f_i = f_i.clone().detach()
    g_i = g_i.clone().detach()
    epsilon_now = epsilon_now.clone().detach()
    f = opt_potential(log_b, g_i, cost_T, epsilon_now)
    g = opt_potential(log_a, f_i, cost, epsilon_now)
    return f, g, epsilon_now


def optimal_transport(regularisation: float, step_size: float, min_update_size: float, max_iterations: int, transport_gradient_clip: float) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    '''
    Returns a function for perfoming optimal transport resampling, (A. Corenflos, J. Thornton, G. Deligiannidis and A. Doucet
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
    OTResampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
        The optimal transport resampling function.

    '''

    class OTGradientWrapper(torch.autograd.Function):
        '''
        Optimal transport gradient can suffer from numerical instability.
        Clip the gradient of the loss wrt the transport matrix to some user specified value.
        This is done in Corenflos and Thornton's original implementation.
        '''
        @staticmethod
        def forward(ctx: Any, transport_matrix: Tensor):
            return transport_matrix

        def backward(ctx: Any, dtransport) -> Any:
            return torch.clip(dtransport, -transport_gradient_clip, transport_gradient_clip)

    def _optimal_transport(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        N = state.size(1)
        log_b, cost, diam = get_sinkhorn_inputs_OT(N, weights, state)
        f, g, epsilon_used = sinkhorn_loop(weights, log_b, cost, regularisation, min_update_size, max_iterations, diam.reshape(-1, 1, 1), step_size)
        transport = get_transport_from_potentials(weights, log_b, cost, f, g, epsilon_used)
        transport = OTGradientWrapper.apply(transport)
        return apply_transport(state, transport, N), torch.zeros_like(weights), transport
    return _optimal_transport


def kernel_resampling(kernel: Distribution, generator: torch.Generator):
    mixture = KernelMixture(kernel, gradient_estimator='none',generator=generator)
    def kernel_resampling_(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        new_state = mixture.sample(state, weights, sample_size=state.size(1))
        # Save computation if gradient is not required
        if torch.is_grad_enabled():
            density = mixture.log_density(new_state, state, weights)
            new_weights = density - density.detach()
        else:
            new_weights = torch.zeros_like(weights)
        return new_state, new_weights, None

    return kernel_resampling_

