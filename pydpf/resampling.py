import torch
from typing import Tuple, Any

from .utils import batched_select

def multinomial(state: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        sampled_indices = torch.multinomial(torch.exp(weights), weights.size(1), replacement=True).detach()
    return batched_select(state, sampled_indices), torch.zeros_like(weights), sampled_indices

def systematic(state: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        offset = torch.rand((weights.size(0),), device=state.device)
        cum_probs = torch.cumsum(torch.exp(weights), dim= 1)
        #No index can be above 1. and the last index must be exactly 1.
        #Fix this in case of numerical errors
        cum_probs = torch.where(cum_probs > 1., 1., cum_probs)
        cum_probs[:,-1] = 1.
        resampling_points = torch.arange(weights.size(1), device=state.device) + offset.unsqueeze(1)
        sampled_indices = torch.searchsorted(cum_probs * weights.size(1), resampling_points)
    return batched_select(state, sampled_indices), torch.zeros_like(weights), sampled_indices


def soft(softness):
    if softness < 0 or softness >= 1:
        raise ValueError(f'Softness {softness} is out of range, must be in [0,1)')
    log_softness = torch.log(torch.tensor([softness]))
    neg_log_softness = torch.log(torch.tensor([1 - softness]))
    def _soft_systematic(state: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal log_softness, neg_log_softness
        log_softness = log_softness.to(device=state.device)
        neg_log_softness = neg_log_softness.to(device=state.device)
        soft_weights = torch.logaddexp(weights + log_softness, neg_log_softness - torch.log(torch.tensor(weights.size(1), device = state.device)))
        state, _, sampled_indices = systematic(state, soft_weights)
        return state, weights - soft_weights, sampled_indices
    return _soft_systematic

def soft_multinomial(softness):
    if softness < 0 or softness >= 1:
        raise ValueError(f'Softness {softness} is out of range, must be in [0,1)')
    log_softness = torch.log(torch.tensor([softness]))
    neg_log_softness = torch.log(torch.tensor([1 - softness]))
    def _soft_systematic(state: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal log_softness, neg_log_softness
        log_softness = log_softness.to(device = state.device)
        neg_log_softness = neg_log_softness.to(device = state.device)
        soft_weights = torch.logaddexp(weights + log_softness, neg_log_softness - torch.log(torch.tensor(weights.size(1), device = state.device)))
        state, _, sampled_indices = multinomial(state, soft_weights)
        return state, weights - soft_weights, sampled_indices
    return _soft_systematic

def stop_gradient(state: torch.Tensor, weight: torch.Tensor):
    state, _, sampled_indices = systematic(state, weight)
    resampled_weights = batched_select(weight, sampled_indices)
    return state, resampled_weights - resampled_weights.detach(), sampled_indices

def diameter(x: torch.Tensor):
    diameter_x = torch.amax(x.std(dim=1, unbiased=False), dim=-1, keepdim=True)
    return torch.where(torch.eq(diameter_x, 0.), 1., diameter_x).detach()

def get_sinkhorn_inputs_OT(Nk, log_weights: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        The auto-distance matrix of scaled_x_t under the 2-Norm
    """
    log_uniform_weights = torch.log(torch.ones((log_weights.size(0), Nk), device=log_weights.device) / Nk)
    centred_x_t = x_t - torch.mean(x_t, dim=1, keepdim=True).detach()
    scale_x = diameter(x_t)
    scaled_x_t = centred_x_t / scale_x.unsqueeze(2)
    cost_matrix = torch.cdist(scaled_x_t, scaled_x_t, 2) ** 2
    return log_uniform_weights, cost_matrix, scale_x


def get_transport_from_potentials(log_a: torch.Tensor, log_b: torch.Tensor, cost: torch.Tensor, f: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
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


def apply_transport(x_t: torch.Tensor, transport: torch.Tensor, N: int) -> torch.Tensor:
    """
    Apply a transport matrix to a vector of particles

    Parameters
    -------------
    x_t: (B,N,D) Tensor
        Particle state to be transported

    transport: (B,M,N) Tensor
        The transport matrix

    N: int
        Number of particles

    """
    return N * torch.einsum('bji, bjd -> bid', transport, x_t)


def opt_potential(log_a: torch.Tensor, c_potential: torch.Tensor, cost: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
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

def sinkhorn_loop(log_a: torch.Tensor, log_b: torch.Tensor, cost: torch.Tensor, epsilon: float, threshold: float, max_iter: int, diam: torch.Tensor, rate: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    cost_T = torch.einsum('bij -> bji', cost)
    epsilon_now = torch.clip(diam ** 2, max=epsilon)
    continue_criterion = torch.ones((f_i.size(0),), device=device, dtype=torch.bool).unsqueeze(1)

    def stop_criterion(i_, continue_criterion_):
        return i_ < max_iter and torch.any(continue_criterion_)

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
    epsilon_now = epsilon_now.clone()
    f = opt_potential(log_b, g_i, cost_T, epsilon_now)
    g = opt_potential(log_a, f_i, cost, epsilon_now)
    return f, g, epsilon_now


def optimal_transport(epsilon, threshold: float, max_iter: int, rate: float, transport_gradient_clip: float):
    class OTGradientWrapper(torch.autograd.Function):
        '''
        Optimal transport gradient can suffer from numerical instability.
        Add clip the gradient of the loss wrt the transport matrix to some user specified value.
        This is done in Corenflos and Thornton's original implementation.
        '''
        @staticmethod
        def forward(ctx: Any, transport_matrix):
            return transport_matrix

        def backward(ctx: Any, dtransport) -> Any:
            return torch.clip(dtransport, -transport_gradient_clip, transport_gradient_clip)

    def _optimal_transport(state: torch.Tensor, weights: torch.Tensor):
        N = state.size(1)
        log_b, cost, diam = get_sinkhorn_inputs_OT(N, weights, state)
        diam = state.amax(dim=(1,2)) - state.amin(dim=(1,2))
        f, g, epsilon_used = sinkhorn_loop(weights, log_b, cost, epsilon, threshold, max_iter, diam.reshape(-1, 1, 1), rate)
        transport = get_transport_from_potentials(weights, log_b, cost, f, g, epsilon_used)
        transport = OTGradientWrapper.apply(transport)
        return apply_transport(state, transport, N), torch.zeros_like(weights), transport
    return _optimal_transport

