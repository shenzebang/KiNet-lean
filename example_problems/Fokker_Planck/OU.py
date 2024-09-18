import jax.numpy as jnp
import jax.random as random
from jax import vmap
from utils.common_utils import v_gaussian_score, v_gaussian_log_density, gaussian_log_density
from functools import partial
from core.potential import QuadraticPotential
# OU process
# dX(t) = -FX(t) dt + \sqrt{L} dW(t)
# Assume F is positive definite
# If X(0) \sim N(m(0), P(0)), we have X(t) \sim N(m(t), P(t))
# where m(t) = exp(-Ft) m(0), P(t) = exp(-Ft)P(0)exp(-Ft) + \int_0^t exp(-F(t-s)) L exp(-F(t-s)) ds

# For the simplicity of computation, let F = USU' be the SVD of F.
# Denote B = U'LU

# To ensure that the coefficient of the Laplacian term in the FPE is 1, L should be 2

def initialize_configuration(domain_dim: int, rng):
    F_scale = 1.
    L_scale = 2
    m_0_scale = 1.
    P_0_scale = 5.

    m_0 = jnp.ones(domain_dim) * m_0_scale
    P_0 = jnp.eye(domain_dim) * P_0_scale
    # F = jnp.eye(domain_dim) * F_scale
    _F = random.normal(random.PRNGKey(2217), (domain_dim, domain_dim + 1))
    F = _F @ _F.transpose() * F_scale
    # _L = jax.random.normal(jax.random.PRNGKey(2219), (domain_dim, domain_dim + 1))
    # L = _L @ _L.transpose() * L_scale
    L = jnp.eye(domain_dim) * L_scale
    U, s, _ = jnp.linalg.svd(F)
    
    return {
        "F": F,
        "L": L,
        "U": U,
        "ss": s + s[:, None],
        "B": U.transpose() @ L @ U,
        "B_0": U.transpose() @ P_0 @ U,
        "s": s,
        "m_0": m_0,
        "P_0": P_0,
    }

@partial(vmap, in_axes=[0, None])
def OU_process(t, configuration):
    exp_t_s = jnp.diag(jnp.exp(- t * configuration["s"]))
    m_t = configuration["U"] @ exp_t_s @ configuration["U"].transpose() @ configuration["m_0"]
    P_t_1 =  exp_t_s @ configuration["B_0"] @ exp_t_s 
    B_S = configuration["B"] / configuration["ss"]
    P_t_2 = B_S - exp_t_s @ B_S @ exp_t_s
    P_t = configuration["U"] @ (P_t_1 + P_t_2) @ configuration["U"].transpose()
    return m_t, P_t


def ground_truth(configuration):
    def ground_truth_fn(ts: jnp.ndarray, xs: jnp.ndarray):
        assert ts.ndim == 1
        assert xs.ndim == 2 or xs.ndim == 3

        mus, Sigmas = OU_process(ts, configuration)
        in_axes = [0, 0, 0] if xs.ndim == 3 else [None, 0, 0]
        v_v_gaussian_score, v_v_gaussian_log_density \
            = vmap(v_gaussian_score, in_axes=in_axes), vmap(v_gaussian_log_density, in_axes=in_axes)

        scores_true = v_v_gaussian_score(xs, Sigmas, mus)
        log_densities_true = v_v_gaussian_log_density(xs, Sigmas, mus)

        return scores_true, log_densities_true
    
    return ground_truth_fn

def get_potential_fn(configuration):
    return QuadraticPotential(jnp.zeros_like(configuration["m_0"]), jnp.linalg.inv(configuration["F"]))


def equilibrium(configuration):
    return None