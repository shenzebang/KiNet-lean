import jax.numpy as jnp
from api import ProblemInstance
from core.distribution import Uniform, Gaussian, DistributionKinetic, Uniform_over_3d_Ball
import jax.random as random
import jax
from jax.experimental.ode import odeint
import gc
import warnings

# =============================================
# Flocking Kernel in 3D!
beta = .2
def K_fn(z1: jnp.ndarray, z2: jnp.ndarray, beta: float=.2):
    dz = z1 - z2
    dx, dv = jnp.split(dz, indices_or_sections=2, axis=-1)
    dx_norm2 = jnp.sum(dx ** 2, axis=-1)
    return -dv / (1+dx_norm2) ** beta

K_fn_vmapy = jax.vmap(K_fn, in_axes=[None, 0, None])

def conv_fn(z1: jnp.ndarray, z2: jnp.ndarray, beta: float=.2):
    K = K_fn_vmapy(z1, z2, beta)
    return jnp.mean(K, axis=0)

conv_fn_vmap = jax.vmap(conv_fn, in_axes=[0, None, None])
# =============================================

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    assert x.shape[-1] >= 2
    x1, x2, etc = jnp.split(x, indices_or_sections=[1, 2,], axis=-1)
    return jnp.concatenate([-x2, x1, jnp.zeros_like(etc)], axis=-1)

def void_drift(t: jnp.ndarray, x: jnp.ndarray):
    return jnp.zeros_like(x)

class Flocking(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)

        self.drift_term = drift_term if cfg.pde_instance.with_drift else void_drift
        self.preserve_momentum = not cfg.pde_instance.with_drift
        self.dim = cfg.pde_instance.domain_dim
        self.beta = cfg.pde_instance.beta

        Sigma_x_0 = jnp.eye(self.dim)
        mu_x_0 = jnp.zeros(self.dim)
        distribution_x_0 = Gaussian(mu_x_0, Sigma_x_0)

        Sigma_v_0 = jnp.eye(self.dim)
        mu_v_0 = jnp.zeros(self.dim)
        distribution_v_0 = Gaussian(mu_v_0, Sigma_v_0)

        warnings.warn("Assuming the initial momentum is zero in the Flocking model!")
        self.m_0 =  jnp.zeros(self.dim) if self.preserve_momentum else None

        # Distributions for KiNet
        self.distribution_0 = DistributionKinetic(distribution_x=distribution_x_0, distribution_v=distribution_v_0)
        self.density_0 = self.distribution_0.density
        
        # Distributions for PINN

        # Test data
        if self.cfg.pde_instance.perform_test:
            self.test_data = self.prepare_test_data()

        # self.run_particle_method_baseline()

    def prepare_test_data(self):
        # use particle method to generate the test dataset
        # 1. sample particles from the initial distribution
        z_test = self.distribution_0.sample(self.cfg.test.batch_size, random.PRNGKey(1234))
        # if jax.devices()[0].platform == "gpu":
        z_test = jax.device_put(z_test, jax.devices("cpu")[0])
        # 2. evolve the system to t = self.total_evolving_time

        forward_fn = lambda t, z: conv_fn_vmap(z, z, self.beta)
        velocity = self.forward_fn_to_dynamics(forward_fn)

        states_0 = {
            "z": z_test,
        }

        def ode_func1(states, t):
            return {"z": velocity(t, states["z"])}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.cfg.ODE_tolerance, rtol=self.cfg.ODE_tolerance)
        z_T = result_forward["z"][-1]
        velocity_T = conv_fn_vmap(z_T, z_T, self.beta)

        print(f"preparing the ground truth by running the particle method with {self.cfg.test.batch_size} particles.")
        if jax.devices()[0].platform == "gpu":
            # x_T = jax.device_put(x_T, jax.devices("gpu")[0])
            # v_T = jax.device_put(v_T, jax.devices("gpu")[0])
            z_T = jax.device_put(z_T, jax.devices("gpu")[0])
            velocity_T = jax.device_put(velocity_T, jax.devices("gpu")[0])
        return {"z_T": z_T, "velocity_T": velocity_T,}


    def run_particle_method_baseline(self):
        z_particle = self.distribution_0.sample(self.cfg.baseline.batch_size, random.PRNGKey(12345))
        forward_fn = lambda t, z: conv_fn_vmap(z, z, self.beta)
        velocity = self.forward_fn_to_dynamics(forward_fn)

        states_0 = {"z": z_particle,}

        def ode_func1(states, t):
            return {"z": velocity(t, states["z"])}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.cfg.ODE_tolerance, rtol=self.cfg.ODE_tolerance)
        z_T = result_forward["z"][1]

        l2_error, l2 = 0, 0
        z_trues = jnp.split(self.test_data["z_T"], 10, axis=0)
        for z_true in z_trues:
            acceleration_pred = conv_fn_vmap(z_true, z_T, self.beta)
            acceleration_true = self.ground_truth(z_true)
            l2_error = l2_error + jnp.sum(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
            l2 = l2 + jnp.sum(jnp.sqrt(jnp.sum((acceleration_true) ** 2, axis=-1)))

        relative_l2 = l2_error / l2
        print(f"The particle method baseline with {self.cfg.baseline.batch_size} particles has relative l2 error {relative_l2}.")

    def ground_truth(self, ts: jnp.ndarray, zs: jnp.ndarray):
        # return the ground truth acceleration at t = self.total_evolving_time
        # assert ts.ndim == 0 or ts.ndim == 1
        # if ts.ndim == 0:
        #     ts = ts * jnp.ones(1)
        warnings.warn("In the flocking model, only the ground truth at the terminal time is known!")

        return conv_fn_vmap(zs, self.test_data["z_T"], self.beta)

    def forward_fn_to_dynamics(self, forward_fn, time_offset=jnp.zeros([])):
        def dynamics(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = forward_fn(t, z) + self.drift_term(t + time_offset, x)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        return dynamics
