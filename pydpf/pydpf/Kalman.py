"""Python module that implements the Kalman filter using the Distributions submodule"""
import torch
from torch import Tensor
from .base import Module
from .distributions import LinearGaussian, MultivariateGaussian


class KalmanFilter(Module):
    """The standard Kalman filter for exact filtering of linear and Gaussian SSMs.

    Parameters
    ----------
    prior_model: distributions.MultivariateGaussian
        The prior model
    dynamic_model: distributions.LinearGaussian
        The dynamic model
    observation_model: distributions.MultivariateGaussian
        The observation model

    Notes
    -----
    See [1]_.

    References
    ----------
    .. [1] Kalman RE (1960). “A new approach to linear filtering and prediction problems.” Transactions of the ASME–Journal of Basic Engineering, 82(Series D), 35–45.
    """

    half_log_2pi = (1 / 2) * torch.log(torch.tensor(2 * torch.pi))

    def __init__(self, prior_model:MultivariateGaussian = None, dynamic_model:LinearGaussian=None, observation_model:LinearGaussian=None):
        super().__init__()
        self.prior_model = prior_model
        self.dynamic_model = dynamic_model
        self.observation_model = observation_model

    @staticmethod
    def _get_time_data(t: int, **data: dict, ) -> dict:
        time_dict = {k: v[t] for k, v in data.items() if k != 'series_metadata' and k != 'state' and v is not None}
        if data['time'] is not None and t > 0:
            time_dict['prev_time'] = data['time'][t - 1]
        if data['series_metadata'] is not None:
            time_dict['series_metadata'] = data['series_metadata']
        return time_dict

    @staticmethod
    def _Gaussian_log_density(mean, covariance, sample) -> Tensor:
        prefactor = -sample.size(-1) * KalmanFilter.half_log_2pi - torch.slogdet(covariance)[1]/2
        residuals = sample - mean
        exponent = (-1/2) * (residuals.unsqueeze(-2) @  torch.linalg.inv_ex(covariance)[0] @ residuals.unsqueeze(-1)).reshape(-1)
        return prefactor + exponent

    @staticmethod
    def _Bayes_update(prior_mean, prior_covariance, observation, observation_weight, observation_bias, observation_kernel_covariance):
        cov_weight = prior_covariance @ observation_weight.T
        inverse_term = torch.linalg.inv_ex(observation_weight @ cov_weight + observation_kernel_covariance)[0]
        weight_inverse_term_weight = observation_weight.T @ inverse_term @ observation_weight
        posterior_covariance = prior_covariance @ (torch.eye(prior_mean.size(-1), device = prior_mean.device) -  weight_inverse_term_weight @ prior_covariance)
        mean_term_1 = (torch.eye(prior_mean.size(-1), device = prior_mean.device) - prior_covariance @ weight_inverse_term_weight).unsqueeze(0) @ prior_mean.unsqueeze(-1)
        mean_term_2 = (cov_weight @ inverse_term).unsqueeze(0) @ (observation - observation_bias.unsqueeze(0)).unsqueeze(-1)
        return (mean_term_1 + mean_term_2).squeeze(-1), posterior_covariance

    @staticmethod
    def _propagate(prev_mean, prev_covariance, kernel_weight, kernel_bias, kernel_covariance):
        new_mean = (kernel_weight.unsqueeze(0) @ prev_mean.unsqueeze(-1)).squeeze(-1) + kernel_bias.unsqueeze(0)
        new_covariance = (kernel_weight @ prev_covariance @ kernel_weight.T) + kernel_covariance
        return new_mean, new_covariance


    def forward(self, time_extent: int, observation: Tensor):
        r"""Returns the filtering posterior distributions :math:`p(x_{t}\mid y_{0:t})` from Kalman filtering.


        Parameters
        ----------
        time_extent: int
            The maximum time-step to run to, including time 0, the filter will calculate {time_extent + 1} filtering distributions.
        observation: Tensor
            The observations of the hidden variable system.

        Returns
        -------
        posterior_means: Tensor
            The means of the Gaussian filtering posteriors.
        posterior_covariances: Tensor
            The covariances of the Gaussian filtering posteriors.
        likelihood_factors: Tensor
            The likelihood factors of the Gaussian filtering posteriors. Equal to :math:`p(y_{t}\mid y_{0:t-1})`.
        """
        prior_mean = self.prior_model.mean
        prior_covariance = self.prior_model.cholesky_covariance @ self.prior_model.cholesky_covariance.T

        dynamic_weight = self.dynamic_model.mean_fun.weight
        dynamic_bias = self.dynamic_model.mean_fun.bias
        dynamic_covariance = self.dynamic_model.dist.cholesky_covariance @ self.dynamic_model.dist.cholesky_covariance.T

        observation_weight = self.observation_model.mean_fun.weight
        observation_bias = self.observation_model.mean_fun.bias
        observation_kernel_covariance = self.observation_model.dist.cholesky_covariance @ self.observation_model.dist.cholesky_covariance.T

        posterior_means = torch.empty((time_extent+1, observation.size(1), prior_mean.size(-1)), device = observation.device, dtype=observation.dtype)
        posterior_covariances = torch.empty((time_extent+1, observation.size(1), prior_mean.size(-1), prior_mean.size(-1)), device = observation.device, dtype=observation.dtype)
        likelihood_factors = torch.empty((time_extent+1, observation.size(1)), device = observation.device, dtype=observation.dtype)
        observation_mean, observation_covariance = self._propagate(prior_mean.unsqueeze(0), prior_covariance, observation_weight, observation_bias, observation_kernel_covariance)
        posterior_means[0], posterior_covariances[0] = self._Bayes_update(prior_mean.unsqueeze(0), prior_covariance, observation[0], observation_weight, observation_bias, observation_kernel_covariance)
        likelihood_factors[0] = self._Gaussian_log_density(observation_mean, observation_covariance, observation[0])

        for t in range(time_extent):
            predictive_mean, predictive_covariance = self._propagate(posterior_means[t].clone(), posterior_covariances[t].clone(), dynamic_weight, dynamic_bias, dynamic_covariance)
            observation_mean, observation_covariance = self._propagate(predictive_mean, predictive_covariance, observation_weight, observation_bias, observation_kernel_covariance)
            likelihood_factors[t+1] = self._Gaussian_log_density(observation_mean, observation_covariance, observation[t+1])
            posterior_means[t+1], posterior_covariances[t+1] = self._Bayes_update(predictive_mean, predictive_covariance, observation[t+1], observation_weight, observation_bias, observation_kernel_covariance)

        return posterior_means, posterior_covariances, likelihood_factors