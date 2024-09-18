from core.distribution import Distribution
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple
from jax._src.random import KeyArray
from dataclasses import dataclass
from omegaconf import DictConfig
from functools import partial
from utils.plot_utils import plot_velocity
import jax
from jax.experimental.ode import odeint
from core.distribution import Uniform
import warnings

class ProblemInstance:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.instance_name = cfg.pde_instance.name
        self.dim = cfg.pde_instance.domain_dim
        self.diffusion_coefficient = jnp.ones([]) * cfg.pde_instance.diffusion_coefficient
        self.total_evolving_time = jnp.ones([]) * cfg.pde_instance.total_evolving_time
        self.rng = rng
        
        # The following instance attributes should be implemented
        
        # Configurations that lead to an analytical solution
        
        # Analytical solution

        # Distributions for KiNet

        # Distributions for PINN

        # Test data

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        # Should return the test time stamp and the corresponding ground truth
        raise NotImplementedError

    def forward_fn_to_dynamics(self, forward_fn, time_offset=jnp.zeros([])):
        raise NotImplementedError
    
    def score_0(self, xs: jnp.ndarray):
        raise NotImplementedError
    
    def density_0(self, xs: jnp.ndarray):
        raise NotImplementedError

class Method:
    def __init__(self, pde_instance: ProblemInstance, cfg: DictConfig, rng: KeyArray) -> None:
        self.pde_instance = pde_instance
        self.cfg = cfg
        self.rng = rng

    def value_and_grad_fn(self, forward_fn, params, time_interval, rng):
        # the data generating process should be handled within this function
        raise NotImplementedError

    def test_fn(self, forward_fn, params, time_interval, rng):
        pass

    def plot_fn(self, forward_fn, params, time_interval, rng):
        if self.cfg.pde_instance.domain_dim != 2 and self.cfg.pde_instance.domain_dim != 3:
            msg = f"Plotting {self.cfg.pde_instance.domain_dim}D problem is not supported! Only 2D and 3D problems are supported."
            warnings.warn(msg)
            return
        else:
            total_time = len(time_interval["previous"]) * time_interval["previous"][0][-1]
            @jax.jit
            def produce_data():
                z_0T = []
                states_0 = {"z": self.pde_instance.distribution_0.sample(batch_size=200, key=rng)}
                for _params, _time_interval in zip(params["previous"], time_interval["previous"]):
                    dynamics_fn = self.pde_instance.forward_fn_to_dynamics(partial(forward_fn, _params))
                    
                    def ode_func1(states, t):
                        return {"z": dynamics_fn(t, states["z"])}

                    tspace = jnp.linspace(_time_interval[0], _time_interval[-1], num=21)
                    result_forward = odeint(ode_func1, states_0, tspace, atol=1e-5, rtol=1e-5)
                    z_0T.append(result_forward["z"])
                    states_0 = {"z": result_forward["z"][-1]}
                return jnp.concatenate(z_0T, axis=0)

            
            plot_velocity(total_time, produce_data())

    def create_model_fn(self) -> Tuple[nn.Module, Dict]:
        raise NotImplementedError

    def metric_fn(self, forward_fn, params, time_interval, rng):
        pass