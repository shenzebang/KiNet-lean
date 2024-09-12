from api import ProblemInstance
from core.distribution import Uniform, BKW, Gaussian
import jax.numpy as jnp
import jax
import jax.random as random
from functools import partial

def K(t):
    return 1. - jnp.exp(-t/8.)/2.

# =============================================
# Collision kernel

def K_fn(x: jnp.ndarray, y: jnp.ndarray, score_x: jnp.ndarray, score_y: jnp.ndarray, weight_y: jnp.ndarray):
    assert x.ndim == 1
    assert x.shape == y.shape and score_x.shape == score_y.shape and x.shape == score_x.shape
    assert weight_y.ndim == 0
    dx = x - y
    dscore = score_x - score_y
    norm2 = jnp.sum(dx**2, axis=-1)
    A_z = (norm2 * jnp.eye(dx.shape[-1]) - jnp.outer(dx, dx))/16.
    return -jnp.matmul(A_z, dscore) * weight_y

K_fn_vmapy = jax.vmap(K_fn, in_axes=[None, 0, None, 0, 0])

def conv_fn(x: jnp.ndarray, y: jnp.ndarray, score_x: jnp.ndarray, score_y: jnp.ndarray, weight_y: jnp.ndarray):
    K = K_fn_vmapy(x, y, score_x, score_y, weight_y)
    return jnp.mean(K, axis=0)

conv_fn_vmap = jax.vmap(conv_fn, in_axes=[0, None, 0, None, None])
# =============================================


def density_fn(t: jnp.ndarray, x: jnp.ndarray):
    # Compatible with single and batch x inputs
    # t should be zero dimension
    assert t.ndim == 0
    x_norm_2 = jnp.sum(x ** 2, axis=-1)
    K_t = K(t)
    return 1 / 2 / jnp.pi / K_t * jnp.exp(-x_norm_2 / 2. / K_t) * (
                (2. * K_t - 1.) / K_t + (1. - K_t) / 2. / K_t ** 2 * x_norm_2)

def log_density_fn(t: jnp.ndarray, x: jnp.ndarray):
    # Compatible with single and batch x inputs
    assert t.ndim == 0
    x_norm_2 = jnp.sum(x ** 2, axis=-1)
    K_t = K(t)
    return jnp.log(1 / 2 / jnp.pi / K_t) - x_norm_2/2./K_t + jnp.log((2. * K_t - 1.) / K_t + (1. - K_t) / 2. / K_t ** 2 * x_norm_2)

def _sum_log_density_fn(t: jnp.ndarray, x: jnp.ndarray):
    # return jnp.sum(jnp.log(density_fn(t, x, None))) This generates nan when x_norm_2 is large!!!

    # should manually compute the log_density
    log_density = log_density_fn(t, x)
    return jnp.sum(log_density) # reduce to a single scale in the batch case

score_fn = jax.grad(_sum_log_density_fn, argnums=1)

def _velocity_fn(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 and x.ndim == 1
    K_t = K(t)
    x_norm_2 = jnp.sum(x ** 2, axis=-1)
    return - (1-K_t) ** 2 / 32 / K_t ** 2 * (4*K_t - x_norm_2) / (2 * K_t -1 + (1-K_t) * x_norm_2) * x

_velocity_fn_vmap = jax.vmap(_velocity_fn, in_axes=(None, 0))

def velocity_fn(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 
    if x.ndim == 1:
        return _velocity_fn(t, x)
    elif x.ndim == 2:
        return _velocity_fn_vmap(t, x)
    else:
        raise ValueError("x should be either 1D (non-batched) or 2D (batched) array.")


class HomogeneousLandau(ProblemInstance):
    def __init__(self, cfg, rng):
        super(HomogeneousLandau, self).__init__(cfg, rng)
        
        # Configurations that lead to an analytical solution
        distribution_x_0 = Gaussian(jnp.zeros([2]), jnp.sqrt(1./2.) * jnp.eye(2))

        # Analytical solution
        self.density_t = density_fn
        self.velocity_t = velocity_fn
        self.score_t = score_fn

        # Distributions for KiNet
        self.distribution_0 = distribution_x_0
        self.density_0 = partial(self.density_t, jnp.zeros([]))
        self.score_0 = partial(self.score_t, jnp.zeros([]))

        # Distributions for PINN
        effective_domain_dim = cfg.pde_instance.domain_dim # (d for position)
        self.mins = cfg.pde_instance.domain_min * jnp.ones(effective_domain_dim)
        self.maxs = cfg.pde_instance.domain_max * jnp.ones(effective_domain_dim)
        self.domain_area = (cfg.pde_instance.domain_max - cfg.pde_instance.domain_min) ** effective_domain_dim
        # Test data
        self.test_data = self.prepare_test_data()

    def prepare_test_data(self):
        print(f"Using the instance {self.instance_name}. Will use the close-form solution to test accuracy.")
        # x_test = self.distribution_0.sample(self.cfg.test.batch_size, random.PRNGKey(1234))
        side_x = jnp.linspace(self.mins[0], self.maxs[0], 1000)
        side_y = jnp.linspace(self.mins[1], self.maxs[1], 1000)
        X, Y = jnp.meshgrid(side_x, side_y)
        x_test = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        return {"x_T": x_test, }

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        assert ts.ndim == 0 or (ts.ndim == 1 and len(ts) == 1)
        if ts.ndim == 1:
            ts = ts[0]
        
        return density_fn(ts, xs)

    def forward_fn_to_dynamics(self, forward_fn, time_offset=jnp.zeros([])):
        def dynamics(t, x):
            return forward_fn(t, x)

        return dynamics
