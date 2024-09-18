import jax.numpy as jnp
from core.distribution import Gaussian, Uniform, Distribution, GaussianMixtureModel, DistributionKinetic
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance

# from core.potential import QuadraticPotential, GMMPotential

import example_problems.Fokker_Planck.OU as OU

# For every potential, should implement the following functions
# 1. initialize_configuration
# 2. get_potential_fn
# 3. equilibrium
# 4. ground_truth
POTENTIALS = {
    "OU":   OU,
    "GMM":  None,
}


class FokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.rng, rng_init_config = jax.random.split(self.rng)

        # Configurations that determines the potential
        self.initial_configuration = POTENTIALS[self.cfg.pde_instance.potential].initialize_configuration(cfg.pde_instance.domain_dim, rng_init_config)

        # Potential
        self.target_potential = POTENTIALS[self.cfg.pde_instance.potential].get_potential_fn(self.initial_configuration)

        # Equilibrium distribution
        self.equilibrium = POTENTIALS[cfg.pde_instance.potential].equilibrium(self.initial_configuration)

        # Ground truth
        self.ground_truth = POTENTIALS[cfg.pde_instance.potential].ground_truth(self.initial_configuration)

        # Initial distribution
        self.distribution_0 = Gaussian(self.initial_configuration["m_0"], self.initial_configuration["P_0"])
        self.score_0 = self.distribution_0.score
        self.density_0 = self.distribution_0.density
        self.logprob_0 = self.distribution_0.logdensity

    def forward_fn_to_dynamics(self, forward_fn, time_offset=jnp.zeros([])):
        def dynamics(t, z):
            return - forward_fn(t, z) - self.target_potential.gradient(z)

        return dynamics
