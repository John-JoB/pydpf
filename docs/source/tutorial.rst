========
Tutorial
========

This is a basic tutorial to take a user through using our package to define a state-space model, generate a synthetic dataset, and train a dpf to filter the generated observations using supervised learning.
This tutorial is also available in notebook form in our `github repository <https://github.com/John-JoB/pydpf/blob/main/tutorial_notebooks/pydpf-tutorial.ipynb>`__.

.. contents::
    :local:
    :depth: 2


The State Space Model
=====================

We start with defining a state space model in this case we take a very simple non-linear example with one dimensional state and observations.

.. math::
    \begin{gathered}
        x_{t} = \alpha x_{t-1} + \sigma q_{t},\\
        y_{t} = \beta \exp \left( \frac{x_{t}}{2} \right) r_{t},\\
        x_{0} \sim \mathcal{N}\left(0, \frac{\sigma^{2}}{1 - \alpha^{2}}\right)
    \end{gathered}

where :math:`x_{t},y_{t}` are the state and observations at time :math:`t` respectively, :math:`q_{t},r_{t}` are independent draws from a standard uni-variate Gaussian and :math:`\alpha,\beta,\sigma` are unknown parameters.

This model is a simplified discrete time-stochastic volatility model as studied in [1]_, where the observations are a sequence of returns of a financial asset and the latent state are the associated log volatilities.
:math:`\beta` represents the square-root of the long term volatility of the system and so is restricted to be positive. :math:`\alpha` encodes the tendency for the volatility to decay back to its long term value,
therefore we expect :math:`0<\alpha<1`. Finally, :math:`\sigma^{2}` is the volatility of the log-volatility.

Implementing the Model Components
=================================

In this example we are going to use the well known bootstrap particle filter [2]_, where the particles are proposed from the SSMs dynamic kernel. In this case neither the density of the prior model, nor the dynamic kernel are required to calculate the
importance weights, so we don't need to implement them. We start by implementing the dynamic model, the inputted ``alpha`` and ``sigma`` are registered as torch module parameters. We decide to manually force :math:`\alpha` to be within the required
range, with a small additional disallowed region for stability, with the PyDPF ``@constrained_parameter`` decorator that will clip the value of ``alpha`` if it exceeds the allowed range. It is natural to force the volatility to be positive by
parameterising it via its log. We use a ``@cached_property`` decorator to efficiently recover :math:`\sigma`.

.. code-block:: python

    class SVDynamicModel(pydpf.Module):

        def __init__(self, alpha, sigma, device):
            super().__init__()
            self.alpha_ = torch.nn.Parameter(alpha)
            self.log_sigma = torch.nn.Parameter(torch.log(sigma))
            self.device = device

        @pydpf.constrained_parameter
        def alpha(self):
            return self.alpha_, torch.clip(self.alpha_, 1e-3, 1-1e-3)

        @pydpf.cached_property
        def sigma(self):
            return torch.exp(self.log_sigma)

        def sample(self, prev_state, **data):
            state_size = prev_state.size()
            noise = self.sigma * torch.normal(0, 1, device=self.device, size=state_size)
            return prev_state * self.alpha + noise

Next we define the observation model. During filtering we will only need to evaluate the model's log density, not sample from it. However, we are going to generate data so we additionally define a sampling routine.

.. code-block:: python

    class SVObservationModel(pydpf.Module):

        def __init__(self, beta, device):
            super().__init__()
            self.log_beta = torch.nn.Parameter(torch.log(beta))
            self.half_log_2pi = torch.log(torch.tensor(2*torch.pi, device = device))/2
            self.device = device

        @pydpf.cached_property
        def beta(self):
            return torch.exp(self.log_beta)

        #Note: the evaluation function for the observation model is called 'score'
        #rather than 'log_density' as there is no requirement for this to be a
        #valid Markov kernel, and frequently for DPFs it isn't
        def score(self, state, observation, **data):
            log_root_v = state + self.log_beta
            root_v = torch.exp(log_root_v)
            #Observations are independent of the particle so have one less
            #dimension than the particle dependent state, we unsqueeze
            #this dimension to broadcast over the particles.
            normalised_obs = observation.unsqueeze(1) / root_v
            return (-log_root_v - (normalised_obs**2)/2 - self.half_log_2pi).squeeze()

        def sample(self, state, **data):
            log_root_v = state + self.log_beta
            root_v = torch.exp(log_root_v)
            state_size = state.size()
            return root_v * torch.normal(0, 1, device=self.device, size=state_size)

Finally we define the prior model, as the prior and dynamic model share parameters we pass the dynamic model to the prior models ``__init__`` function and make sure to not corrupt the computation graph by creating duplicate parameters.

.. code-block:: python

    class SVPriorModel(pydpf.Module):

        def __init__(self, dynamic_model):
            super().__init__()
            self.device = dynamic_model.device
            self.dyn_mod = dynamic_model

        @pydpf.cached_property
        def sd(self):
            return torch.sqrt(self.dyn_mod.sigma**2 / (1-self.dyn_mod.alpha**2))

        def sample(self, batch_size, n_particles, **data):
            state_size = (batch_size, n_particles, 1)
            return self.sd * torch.normal(0, 1, device=self.device, size=state_size)

Now we have our model components we can group them together into a neat SSM object.

.. code-block:: python

    def make_SSM(alpha, beta, sigma, device):
        dynamic = SVDynamicModel(alpha, sigma, device)
        observation = SVObservationModel(beta, device)
        prior = SVPriorModel(dynamic)
        return pydpf.FilteringModel(prior_model=prior,
                                    dynamic_model=dynamic,
                                    observation_model=observation)

Generating Synthetic Data
=========================

From a given state-space model it is simple in PyDPF to simulate a large amount of trajectories and save them to file. Here we choose to simulate 1000 trajectories of 1000 time-steps from our stochastic volatility model with
:math:`\alpha=0.91, \beta=0.5, \sigma=1`. We simulate trajectories in, parallelised if using CUDA, batches of 100.


.. code-block:: python

    SSM = make_SSM(torch.tensor(0.91, device = device),
               torch.tensor(0.5, device = device),
               torch.tensor(1., device = device),
               device)
    #data_path must have a .csv extension
    pydpf.simulate_and_save(data_path,
                            SSM = SSM,
                            time_extent = 100,
                            n_trajectories = 200,
                            batch_size = 100,
                            device = device)


Defining a DPF
==============

Now we have an SSM, and before we can learn a filter we have to specify the filter's functional form. In this example we use the simple pseudo-differentiable filter of [3]_ by modifying the basic particle filtering algorithm with their soft
resampling procedure with a base non-differentiable multinomial resampler. We also assume that the true parameters :math:`\alpha,\beta,\sigma` are unknown and initialise the algorithm with the guesses `\alpha=0.6, \beta=0.2, \sigma=1.5`.

.. code-block:: python

    learned_SSM = make_SSM(torch.tensor(0.6, device = device),
                       torch.tensor(0.2, device = device),
                       torch.tensor(1.5, device = device),
                       device)
    #The generator parameter is a torch RNG generator.
    #Used to track the random state if reproducibility is required.
    multinomial_base = pydpf.MultinomialResampler(generator = torch.Generator(device=device))
    soft_resampler = pydpf.SoftResampler(softness = 0.7,
                                         base_resampler = multinomial_base,
                                         device = device)
    DPF = pydpf.ParticleFilter(soft_resampler, learned_SSM)

Note, whilst I have explicitly illustrated the procedure of defining a DPF from a resampler and an SSM, all implemented DPFs in PyDPF have convenience aliases that avoid explicitly creating the SSM. Moreover, not all implemented DPFs can be
derived from the basic particle filter by only swapping out the resampling algorithm. For example the above code block can be replaced by:

.. code-block:: python

    learned_SSM = make_SSM(torch.tensor(0.6, device = device),
                       torch.tensor(0.2, device = device),
                       torch.tensor(1.5, device = device),
                       device)
    DPF = pydpf.SoftDPF(SSM = learned_SSM,
                        softness = 0.7,
                        resampling_generator = torch.Generator(device=device),
                        multinomial = True
                        )

Loading the Data
================
The default PyDPF data format is to save the dataset to a single csv. As it is assumed that all data in each datum category (see our :ref:`pydpf-basics` guide for information on the data categories) has the same dimension as all other data in
that category we store the data in csv files. Each dimension of all data categories is stored in its own column where the prefix denotes the category and the column ordering denotes the ordering of the dimensions.
There is also a series id column that denotes the trajectory each row belongs to. The only data categories that are relevant to our model are ``state`` and ``observation``, so the generated dataset has the following header:

============= =========== =================
``series_id`` ``state_1`` ``observation_1``
============= =========== =================

The prefix labels can be changed, but since we generated our dataset with PyDPF they will have the default values. In this case the device parameter is the device the dataset will be stored on.

.. code-block:: python

   full_dataset = pydpf.StateSpaceDataset(data_path,
                                       series_id_column = "series_id",
                                       state_prefix = "state",
                                       observation_prefix = "observation",
                                       device = device)

In this example, we split our dataset randomly into a train and test set using the usual ``torch`` splitting routine, however PyDPF has custom dataset splitters that offer more fined grain control if required.

.. code-block:: python

    train_set, test_set = torch.utils.data.random_split(full_dataset,
                                                        [0.5, 0.5])

As ``pydpf.StateSpaceDataset`` subclasses ``torch.Dataset`` we can use the data loading implementation in ``torch``, however you must set the ``collate_fn`` argument to the collate member of the dataset you are loading from.
Note that this is the full dataset ``pydpf.pydpf.StateSpaceDataset`` object and not any ``torch.Subset``, ``torch.StackDataset``, ``torch.ConcatDataset``, etc. objects created by base ``torch`` operations on the dataset.

.. code-block:: python

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=full_dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=full_dataset.collate)

Obtaining the Correct Outputs
=============================
The filtering algorithm generates many random variables, including the particle locations, their normalised weight, the empirical normalisation factor and any other diagnostic variables.
Most of the time many of these variables are not of interest in their raw form. So, it is therefore efficient to aggregate them into a useful output of a smaller size.
In this example, we are going to train the DPF by minimising the MSE so do not need to retain and return to the user the complete set of particles and weights.
So, in ``PyDPF`` we provide the option to retain and return only a given function of the generated random variables. This is most useful during inference as, with gradient enabled,
many intermediates are retained for the backwards pass. Common aggregation functions such as the ``MSE`` are packaged with ``PyDPF``, see the ``outputs.py`` module in the API reference.
Note the ``aggregation_function`` can either be a callable, in which case the forward pass returns it's results stacked in time; or a dictionary of callables,
in which case the forward pass returns a dictionary of the same keys where each value is the respective results stacked in time.

.. code-block:: python

    output_function = pydpf.MSE_Loss()

Training a DPF
==============
The training procedure is much the same as base ``torch``, and like base ``torch`` we do not implement training loops in ``PyDPF`` instead we leave it up to the users to design a
routine that suits their needs. The only difference to a generic ``torch`` training loop is that it is safest to always call ``DPF.update()`` after any backwards pass or
optimiser step to ensure that the cache is properly invalidated and the computation graph cleared.

.. code-block:: python

    opt = torch.optim.Adam(DPF.parameters(), lr = 0.01)
    n_epochs = 20
    for e in range(n_epochs):
        train_loss = 0.0
        for state, observation in train_loader:
            opt.zero_grad()
            DPF.update()
            MSE = DPF(n_particles= 64,
                      time_extent=100,
                      aggregation_function=output_function,
                      observation=observation,
                      ground_truth=state)
            loss = MSE.mean()
            loss.backward()
            train_loss += loss.item()
            opt.step()
        if e % 10 == 0:
            print(f"Epoch {e+1}, loss: {train_loss/len(train_loader)}")


    DPF.update()
    with torch.inference_mode():
        mean_loss = 0.0
        for state, observation in test_loader:
            outputs = DPF(n_particles= 64,
                          time_extent=100,
                          aggregation_function=output_function,
                          observation=observation,
                          ground_truth=state)
            mean_loss += outputs["MSE"].mean().item()

    print(f"Test MSE: {mean_loss/len(test_loader)}")
    print(f"Learned alpha: {SSM.dynamic_model.alpha.item()}")
    print(f"Learned beta: {SSM.observation_model.beta.item()}")
    print(f"Learned sigma: {SSM.dynamic_model.sigma.item()}")

References
==========

.. [1] Doucet and Johansen (2011), *A tutorial on particle filtering and smoothing: Fifteen years later*
.. [2] Gordon, Salmond and Smith (1993), *Novel Approach to Nonlinear and Non-Gaussian Bayesian State Estimation*
.. [3] Karkus, Hsu and Lee (2018), *Particle Filter Networks with Application to Visual Localization*