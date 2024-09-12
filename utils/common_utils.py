import jax
import jax.numpy as jnp
import math
from jax import grad, vjp
from jax.experimental.ode import odeint


def _divergence_fn(f, _x, _v):
    # Hutchinson’s Estimator
    # computes the divergence of net at x with random vector v
    _, u = jax.jvp(f, (_x,), (_v,))
    # print(u.shape, _x.shape, _v.shape)
    return jnp.sum(u * _v)


# f_list = [lambda x: f(x)[i]]

def _divergence_bf_fn(f, _x):
    # brute-force implementation of the divergence operator
    # _x should be a d-dimensional vector
    jacobian = jax.jacfwd(f)
    a = jacobian(_x)
    return jnp.sum(jnp.diag(a))



batch_div_bf_fn = jax.vmap(_divergence_bf_fn, in_axes=[None, 0])

batch_div_fn = jax.vmap(_divergence_fn, in_axes=[None, None, 0])


def divergence_fn(f, _x: jnp.ndarray, _v=None):
    if _v is None:
        if _x.ndim == 1:
            return _divergence_bf_fn(f, _x)
        return batch_div_bf_fn(f, _x)
    else:
        return batch_div_fn(f, _x, _v).mean(axis=0)

def evolve_data_score_logprob(dynamics_fn, time_interval, data, score, logprob):
    states_0 = {
        "z": data,
    }
    if score is not None:
        states_0["xi"] = score 
    
    if logprob is not None:
        states_0["logprob"] = logprob
    
    def ode_func1(states, t):
        bar_f_t_theta = lambda _x: dynamics_fn(t, _x)
        
        update = {"z": bar_f_t_theta(states["z"]), }

        
        def h_t_theta(xi, z):
            div_bar_f_t_theta = lambda _z: divergence_fn(bar_f_t_theta, _z).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(z)
            _, vjp_fn = vjp(bar_f_t_theta, z)
            h2 = - vjp_fn(xi)[0]
            return h1 + h2
        if "xi" in states:
            update["xi"] = h_t_theta(states["xi"], states["z"])

        def dlog_density_func(in_1):
            # in_1 is x
            div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x)
            return -div_bar_f_t_theta(in_1)
        if "logprob" in states:
            update["logprob"] = dlog_density_func(states["z"])

        return update

    result_forward = odeint(ode_func1, states_0, time_interval, atol=1e-5, rtol=1e-5)

    data = result_forward["z"][-1]
    score = result_forward["xi"][-1] if score is not None else None
    logprob = result_forward["logprob"][-1] if logprob is not None else None

    return data, score, logprob




def _gaussian_score(x, cov, mu): # return the score for a given Gaussian(mu, Sigma) at x
    return jax.numpy.linalg.inv(cov) @ (mu - x)

# return the score for a given Gaussians(Sigma, mu) at [x1, ..., xN]
v_gaussian_score = jax.vmap(_gaussian_score, in_axes=[0, None, None])

def _gaussian_log_density(x, cov, mu):
    log_det = jnp.log(jax.numpy.linalg.det(cov * 2 * jnp.pi))
    inv_cov = jax.numpy.linalg.inv(cov)
    quad = jnp.dot(x - mu, inv_cov @ (x - mu))
    return - .5 * (log_det + quad)

v_gaussian_log_density = jax.vmap(_gaussian_log_density, in_axes=[0, None, None])

def gaussian_log_density(x, cov, mu):
    if x.ndim == 2:
        return v_gaussian_log_density(x, cov, mu)
    elif x.ndim == 1:
        return _gaussian_log_density(x, cov, mu)
    else:
        raise ValueError("x should be a either 1D or 2D array")

v_matmul = jax.vmap(jnp.matmul, in_axes=(None, 0))


def volume_nd_ball(d: int):
    k = d // 2
    if d % 2 == 0:
        return jnp.pi ** k / math.factorial(k)
    else:
        return 2 * math.factorial(k) * ((4 * jnp.pi) ** k) / math.factorial(d)

def compute_pytree_norm(pytree):
    pytree_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(pytree)))
    return pytree_norm

@jax.jit
def compute_pytree_difference(pytree1, pytree2):
    pytree_diff = jax.tree_util.tree_map(lambda x, y: x - y, pytree1, pytree2)
    pytree_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(pytree_diff)))
    return pytree_norm

def normalize_grad(pytree, norm):
    return jax.tree_util.tree_map(lambda x: x/norm, pytree)


def estimate_momentum(XV: jnp.ndarray):
        assert XV.ndim == 2 and XV.shape[-1] >= 2
        X, V = jnp.split(XV, indices_or_sections=2, axis=-1)
        return jnp.mean(V, axis=0)
    
def estimate_energy(XV: jnp.ndarray):
    assert XV.ndim == 2 and XV.shape[-1] >= 2
    X, V = jnp.split(XV, indices_or_sections=2, axis=-1)
    return jnp.mean(jnp.sum(V**2, axis=-1), axis=0)