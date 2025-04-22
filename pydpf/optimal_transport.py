import torch
from torch import Tensor
from typing import Tuple

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
    epsilon_now = torch.clip(diam ** 2, min=epsilon)
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
            epsilon_now = torch.clip(rate * epsilon_now, min=epsilon)
            f_i = f_u
            g_i = g_u
            i += 1
    f_i = f_i.clone().detach()
    g_i = g_i.clone().detach()
    epsilon_now = epsilon_now.clone().detach()
    f = opt_potential(log_b, g_i, cost_T, epsilon_now)
    g = opt_potential(log_a, f_i, cost, epsilon_now)
    return f, g, epsilon_now