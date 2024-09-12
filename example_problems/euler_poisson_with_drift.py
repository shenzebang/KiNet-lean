from api import ProblemInstance
from core.distribution import Uniform, Uniform_over_3d_Ball, DistributionKineticDeterministic
import jax.numpy as jnp
import jax
import jax.random as random

# =============================================
# Coulomb Kernel in 3D!
def K_fn(x: jnp.ndarray, y: jnp.ndarray):
    dx = x - y
    norm2 = jnp.sum(dx ** 2, axis=-1)
    norm = jnp.sqrt(norm2)
    # norm = jnp.maximum(jnp.sqrt(norm2), 1e-4)
    conditions = [
        norm <= 1e-2,
        norm >  1e-2
    ]
    functions = [
        jnp.inf,
        norm,
    ]
    norm_clipped = jnp.piecewise(norm, conditions, functions)
    return dx / norm_clipped ** 3 / 4 / jnp.pi

K_fn_vmapy = jax.vmap(K_fn, in_axes=[None, 0])

def conv_fn(x: jnp.ndarray, y: jnp.ndarray):
    K = K_fn_vmapy(x, y)
    return jnp.mean(K, axis=0)

conv_fn_vmap = jax.vmap(conv_fn, in_axes=[0, None])
# =============================================

t_0 = 5

threshold_0 = (3/4/jnp.pi * t_0) ** (1/3)

def coulomb_potential_uniform_fn(t: jnp.ndarray, xi: jnp.ndarray):
    xi_norm_2 = jnp.sum(xi ** 2)
    xi_norm = jnp.sqrt(xi_norm_2)
    threshold_t = (3/4/jnp.pi * (t+t_0)) ** (1/3)
    conditions = [
        xi_norm <= threshold_t,
        xi_norm > threshold_t
    ]
    functions = [
        (2 * threshold_t ** 2 - xi_norm_2)/6/(t+t_0),
        1 / 4 /jnp.pi / xi_norm
    ]
    return jnp.piecewise(xi_norm, conditions, functions)

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
    if t.ndim == 1:
        t = t[0]
    
    if x.ndim == 2:
        return - 2 * x / 9 / (t + t_0) ** 2 - ground_truth_op_vmapx(t, x)
    elif x.ndim == 1:
        return - 2 * x / 9 / (t + t_0) ** 2 - ground_truth_op_uniform(t, x)
    else:
        raise ValueError("x should be either 1D or 2D array.")

def ground_truth_op_uniform(t: jnp.ndarray, x: jnp.ndarray):
    assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
    if t.ndim == 1:
        t = t[0]

    coulomb_field_uniform = jax.grad(coulomb_potential_uniform_fn, argnums=1)
    return -coulomb_field_uniform(t, x)

ground_truth_op_vmapx = jax.vmap(ground_truth_op_uniform, in_axes=[None, 0])
ground_truth_op_vmapx_vmapt = jax.vmap(ground_truth_op_vmapx, in_axes=[0, None])

# def nabla_phi_0(x: jnp.ndarray):
#     if x.shape[-1] != 3:
#         raise ValueError("x should be of shape [3] or [N, 3]")
#     if x.ndim == 1:
#         return -ground_truth_op_uniform(jnp.zeros([]), x)
#     elif x.ndim == 2: # batched
#         return -ground_truth_op_vmapx(jnp.zeros([]), x)
#     else:
#         raise NotImplementedError
    
# def _mu_0(x: jnp.ndarray):
#     norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1))
#     conditions = [
#         norm <= threshold_0,
#         norm >  threshold_0
#     ]
#     functions = [
#         jnp.ones([]),
#         jnp.zeros([]),
#     ]
#     value = jnp.piecewise(norm, conditions, functions)
#     return value / t_0

# _mu_0_vmapx = jax.vmap(_mu_0, in_axes=[0])

# def mu_0(x: jnp.ndarray):
#     if x.shape[-1] != 3:
#         raise ValueError("x should be of shape [3] or [N, 3]")
#     if x.ndim == 1:
#         return _mu_0(x)
#     elif x.ndim == 2: # batched
#         return _mu_0_vmapx(x)
#     else:
#         raise NotImplementedError
        
# def u_0(x: jnp.ndarray):
#     return x / 3 / t_0

def u_t(t: jnp.ndarray, x: jnp.ndarray):
    return x / 3 / (t_0 + t)

def mu_t(t: jnp.ndarray, x: jnp.ndarray):
    threshold_t = (3/4/jnp.pi * (t+t_0)) ** (1/3)

    norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1))
    conditions = [
        norm <= threshold_t,
        norm >  threshold_t
    ]
    functions = [
        jnp.ones([]),
        jnp.zeros([]),
    ]
    value = jnp.piecewise(norm, conditions, functions)
    return value / (t_0 + t)

def nabla_phi_t(t: jnp.ndarray, x: jnp.ndarray):
    if x.shape[-1] != 3:
        raise ValueError("x should be of shape [3] or [N, 3]")
    if x.ndim == 1:
        return -ground_truth_op_uniform(t, x)
    elif x.ndim == 2: # batched
        return -ground_truth_op_vmapx(t, x)
    else:
        raise NotImplementedError

class EulerPoissonWithDrift(ProblemInstance):
    def __init__(self, cfg, rng):
        super(EulerPoissonWithDrift, self).__init__(cfg, rng)
        
        # Configurations that lead to an analytical solution
        self.drift_term = drift_term
        self.distribution_x_0 = Uniform_over_3d_Ball(threshold_0)


        # Analytical solution
        self.u_0 = lambda x: u_t(jnp.zeros([]), x)
        self.mu_0 = lambda x: mu_t(jnp.zeros([]), x)
        self.nabla_phi_0 = lambda x: nabla_phi_t(jnp.zeros([]), x)

        self.u_t = u_t
        self.mu_t = mu_t
        self.nabla_phi_t = nabla_phi_t

        # Distributions for KiNet
        self.distribution_0 = DistributionKineticDeterministic(self.distribution_x_0, self.u_0)
        self.density_0 = self.distribution_x_0.density
        # self.score_0 is not to be used

        # Distributions for PINN
        effective_domain_dim = cfg.pde_instance.domain_dim # (d for position)
        self.mins = cfg.pde_instance.domain_min * jnp.ones(effective_domain_dim)
        self.maxs = cfg.pde_instance.domain_max * jnp.ones(effective_domain_dim)
        self.domain_area = (cfg.pde_instance.domain_max - cfg.pde_instance.domain_min) ** effective_domain_dim

        self.distribution_t = Uniform(jnp.zeros(1), jnp.ones(1) * cfg.pde_instance.total_evolving_time)
        self.distribution_domain = Uniform(self.mins, self.maxs)

        # Test data
        if self.cfg.pde_instance.perform_test:
            self.test_data = self.prepare_test_data()

    def prepare_test_data(self):
        print(f"Using the instance {self.instance_name}. Will use the close-form solution to test accuracy.")
        x_test = self.distribution_x_0.sample(self.cfg.test.batch_size, random.PRNGKey(1234))
        return {"x_T": x_test, }

    def ground_truth(self, ts: jnp.ndarray, xs: jnp.ndarray):
        assert ts.ndim == 0 or ts.ndim == 1
        if ts.ndim == 0:
            ts = ts * jnp.ones(1)
        
        return ground_truth_op_vmapx_vmapt(ts, xs)
        
    # def ground_truth_t(self, xs: jnp.ndarray, t: jnp.ndarray):
    def forward_fn_to_dynamics(self, forward_fn, time_offset=jnp.zeros([])):
        def dynamics(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = forward_fn(t, x) + self.drift_term(t + time_offset, x)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        return dynamics
