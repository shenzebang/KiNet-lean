import jax.numpy as jnp
from api import ProblemInstance
from core.distribution import Uniform, Uniform_over_3d_Ball
import jax.random as random
import jax
from jax.experimental.ode import odeint
import gc
import warnings

t_0 = 1

threshold_0 = (3/4/jnp.pi * t_0) ** (1/3)
# =============================================
# Configuration for the initial distribution
# Sigma_x_0 = jnp.diag(jnp.array([1., 1., 1.]))
# mu_x_0 = jnp.array([0., 0., 0.])
# distribution_x_0 = Gaussian(mu_x_0, Sigma_x_0)

distribution_x_0 = Uniform_over_3d_Ball(threshold_0)

# distribution_v_0 = Dirac(jnp.zeros(3))
# warnings.warn("(Euler-Poisson) Currently only supports u(x, 0) is a constant zero function")
def u_0(x: jnp.ndarray):
    return jnp.zeros_like(x)
# =============================================

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

# def K_fn(x: jnp.ndarray, y: jnp.ndarray):
#
#     dx = x - y
#     norm2 = jnp.sum(dx ** 2)
#     norm = jnp.maximum(jnp.sqrt(norm2), 1e-4)
#     # norm2 = jnp.maximum(norm2, 1e-4)
#     # norm2 = jnp.sum(dx ** 2)
#     return dx / norm / norm**2 / 4 / jnp.pi
#     # dx = x - y
#     # norm2 = jnp.sum(dx**2)
#     # norm = jnp.sqrt(norm2)
#     # norm2 = jnp.maximum(norm2, 1e-4)
#     # # norm2 = jnp.sum(dx ** 2)
#     # conditions = [
#     #     norm > 0,
#     #     norm <= 0,
#     # ]
#     # functions = [
#     #     dx / norm / norm2 / 4 / jnp.pi,
#     #     dx / norm / norm2 / 4 / jnp.pi,
#     # ]
#     # return jnp.piecewise(norm, conditions, functions)

K_fn_vmapy = jax.vmap(K_fn, in_axes=[None, 0])

def conv_fn(x: jnp.ndarray, y: jnp.ndarray):
    K = K_fn_vmapy(x, y)
    return jnp.mean(K, axis=0)

conv_fn_vmap = jax.vmap(conv_fn, in_axes=[0, None])
# =============================================

def drift_term(t: jnp.ndarray, x: jnp.ndarray):
    return jnp.zeros([])

class EulerPoisson(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.distribution_x_0 = distribution_x_0
        self.distribution_0 = distribution_x_0
        self.u_0 = u_0

        # domain of interest (d dimensional box)
        effective_domain_dim = cfg.pde_instance.domain_dim # (d for position)
        self.mins = cfg.pde_instance.domain_min * jnp.ones(effective_domain_dim)
        self.maxs = cfg.pde_instance.domain_max * jnp.ones(effective_domain_dim)
        self.domain_area = (cfg.pde_instance.domain_max - cfg.pde_instance.domain_min) ** effective_domain_dim

        self.distribution_t = Uniform(jnp.zeros(1), jnp.ones(1) * cfg.pde_instance.total_evolving_time)
        self.distribution_domain = Uniform(self.mins, self.maxs)

        self.drift_term = self.get_drift_term()
        self.test_data = self.prepare_test_data()

        # self.run_particle_method_baseline()

    def get_drift_term(self):
        return drift_term

    def prepare_test_data(self):
        raise ValueError("This function is decrypted.")
        print(f"Using the instance {self.instance_name}. No closed form solution. "
              f"Use particle method to generate the test dataset.")
        # use particle method to generate the test dataset
        # 1. sample particles from the initial distribution
        x_test = self.distribution_0.sample(self.args.batch_size_test_ref, random.PRNGKey(1234))
        if jax.devices()[0].platform == "gpu":
            x_test = jax.device_put(x_test, jax.devices("gpu")[-1])
        v_test = self.u_0(x_test)
        z_test = jnp.concatenate([x_test, v_test], axis=-1)
        # 2. evolve the system to t = self.total_evolving_time
        def velocity(z: jnp.ndarray):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = conv_fn_vmap(x, x)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        states_0 = {
            "z": z_test,
        }

        def ode_func1(states, t):
            dz = velocity(states["z"])

            return {"z": dz}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.args.ODE_tolerance, rtol=self.args.ODE_tolerance)
        z_T = result_forward["z"][1]
        x_T, v_T = jnp.split(z_T, indices_or_sections=2, axis=-1)

        print(f"preparing the ground truth by running the particle method with {self.args.batch_size_test_ref} particles.")
        if jax.devices()[0].platform == "gpu":
            x_T = jax.device_put(x_T, jax.devices("gpu")[0])
            v_T = jax.device_put(v_T, jax.devices("gpu")[0])
        return {"x_T": x_T, "v_T": v_T}


    def run_particle_method_baseline(self):
        x_particle = self.distribution_0.sample(self.args.batch_size_ref, random.PRNGKey(12345))
        v_particle = self.u_0(x_particle)
        z_particle = jnp.concatenate([x_particle, v_particle], axis=-1)
        def velocity(z: jnp.ndarray):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = conv_fn_vmap(x, x)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        states_0 = {
            "z": z_particle,
        }

        def ode_func1(states, t):
            dz = velocity(states["z"])

            return {"z": dz}

        tspace = jnp.array((0., self.total_evolving_time))
        result_forward = odeint(ode_func1, states_0, tspace, atol=self.args.ODE_tolerance, rtol=self.args.ODE_tolerance)
        z_T = result_forward["z"][1]
        x_T, v_T = jnp.split(z_T, indices_or_sections=2, axis=-1)

        l2_error, l2 = 0, 0
        x_trues = jnp.split(self.test_data["x_T"], 10, axis=0)
        for x_true in x_trues:
            acceleration_pred = conv_fn_vmap(x_true, x_T)
            acceleration_true = self.ground_truth(x_true)
            l2_error = l2_error + jnp.sum(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
            l2 = l2 + jnp.sum(jnp.sqrt(jnp.sum((acceleration_true) ** 2, axis=-1)))

        relative_l2 = l2_error / l2
        print(f"The particle method baseline with {self.args.batch_size_ref} particles has relative l2 error {relative_l2}.")

    def ground_truth(self, xs: jnp.ndarray):
        # return the ground truth acceleration at t = self.total_evolving_time

        return conv_fn_vmap(xs, self.test_data["x_T"])


