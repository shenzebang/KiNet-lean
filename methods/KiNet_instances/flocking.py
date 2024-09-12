import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import grad, vjp
import jax.numpy as jnp
from utils.common_utils import compute_pytree_norm, estimate_energy, estimate_momentum
from jax.experimental.ode import odeint
from example_problems.flocking_example import Flocking, conv_fn_vmap
from core.model import get_model
import jax.random as random


def value_and_grad_fn(forward_fn, params, data, time_interval, rng, config, pde_instance: Flocking):
    time_interval_current = time_interval["current"]
    time_offset = time_interval["current"][-1] * len(time_interval["previous"])
    # use time_interval in training module
    # unpack the parameters
    ODE_tolerance = config["ODE_tolerance"]
    # T = pde_instance.total_evolving_time
    # unpack the data

    n_train = data["data_initial"].shape[0]
    z_0 = jnp.concatenate([data["data_initial"], data["data_ref"]], axis=0)

    params_flat, params_tree = tree_flatten(params)

    def bar_f_fn(_z, _t, _params):
        forward_fn_params = lambda t, z: forward_fn(_params, t, z)
        dynamics = pde_instance.forward_fn_to_dynamics(forward_fn_params, time_offset)
        return dynamics(_t, _z)

    def f_fn(_z, _t, _params):
        dv_pred = forward_fn(_params, _t, _z)
        return dv_pred

    # compute x(T) by solve IVP (I)
    # ================ Forward ===================
    states_0 = {
        "z": z_0,
        "loss": jnp.zeros([])
    }

    def ode_func1(states, t):
        z = states["z"]
        dz = bar_f_fn(z, t, params)

        def g_t(_z):
            z_train, z_ref = jnp.split(_z, [n_train], axis=0)
            acceleration = f_fn(z_train, t, params)
            return jnp.mean(jnp.sum((acceleration - conv_fn_vmap(z_train, z_ref, pde_instance.beta)) ** 2, axis=(1,)))

        dloss = g_t(z)

        return {
            "z": dz,
            "loss": dloss
        }

    result_forward = odeint(ode_func1, states_0, time_interval_current, atol=ODE_tolerance, rtol=ODE_tolerance)
    loss_f = result_forward["loss"][-1]
    # ================ Forward ===================

    # ================ Backward ==================
    # compute dl/d theta via adjoint method
    states_T = {
        "z": result_forward["z"][-1],
        "a": jnp.zeros_like(z_0),
        "loss": jnp.zeros([]),
        "grad": [jnp.zeros_like(_var) for _var in params_flat],
    }

    def ode_func2(states, t):
        t = time_interval_current[-1] - t
        z = states["z"]

        a = states["a"]


        f_t = lambda _x, _params: f_fn(_x, t, _params)
        bar_f_t = lambda _x, _params: bar_f_fn(_x, t, _params)
        dz = bar_f_t(z, params)

        _, vjp_fx_fn = vjp(lambda _x: bar_f_t(_x, params), z)
        vjp_fx_a = vjp_fx_fn(a)[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(z, _params), params)
        vjp_ftheta_a = vjp_ftheta_fn(a)[0]

        def g_t(_z, _params):
            z_train, z_ref = jnp.split(_z, [n_train], axis=0)
            acceleration = f_t(z_train, _params)
            return jnp.mean(jnp.sum((acceleration - conv_fn_vmap(z_train, z_ref, pde_instance.beta)) ** 2, axis=-1))

        dxg = grad(g_t, argnums=0)
        dthetag = grad(g_t, argnums=1)

        da = - vjp_fx_a - dxg(z, params)
        dloss = g_t(z, params)

        vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
        dthetag_flat, _ = tree_flatten(dthetag(z, params))
        dgrad = [_dgrad1 + _dgrad2 for _dgrad1, _dgrad2
                 in zip(vjp_ftheta_a_flat, dthetag_flat)]

        return {
            "z": -dz,
            "a": -da,
            "loss": dloss,
            "grad": dgrad
        }


    # ================ Backward ==================
    result_backward = odeint(ode_func2, states_T, time_interval_current, atol=ODE_tolerance, rtol=ODE_tolerance)

    grad_T = tree_unflatten(params_tree, [_var[-1] for _var in result_backward["grad"]])
    # x_0_b = result_backward[0][-1]
    # ref_0_b = result_backward[6][-1]
    # xi_0_b = result_backward[3][-1]

    # These quantities are for the purpose of debug
    # error_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(x_0.shape[0], -1) ** 2, axis=(1,)))
    # error_xi = jnp.mean(jnp.sum((xi_0 - xi_0_b).reshape(xi_0.shape[0], -1) ** 2, axis=(1,)))
    # error_ref = jnp.mean(jnp.sum((ref_0 - ref_0_b).reshape(ref_0.shape[0], -1) ** 2, axis=(1,)))
    # loss_b = result_backward[4][-1]

    return {
        "loss": loss_f,
        "grad": grad_T,
        "grad norm": compute_pytree_norm(grad_T),
    }

def test_fn(forward_fn, data, time_interval, pde_instance: Flocking, rng):
    # use time_interval in testing module
    z_ground_truth = pde_instance.test_data["z_T"]
    acceleration_pred = forward_fn(jnp.ones(1) * pde_instance.total_evolving_time, z_ground_truth)
    acceleration_true = pde_instance.test_data["velocity_T"]
    relative_l2 = jnp.mean(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
    relative_l2 = relative_l2 / jnp.mean(jnp.sqrt(jnp.sum(acceleration_true ** 2, axis=-1)))

    return {"relative l2 error": relative_l2}

def create_model_fn(pde_instance: Flocking):
    net = get_model(pde_instance.cfg)
    params = net.init(random.PRNGKey(11), jnp.zeros(1), pde_instance.distribution_0.sample(1, random.PRNGKey(1)))
    return net, params

def functional_decay_fn(data, pde_instance: Flocking, rng):
    assert pde_instance.preserve_momentum
    m_t = estimate_momentum(data["data_initial"])
    energy = estimate_energy(data["data_initial"])
    return {"Lyapunov functional": energy - 2 * jnp.sum(m_t*pde_instance.m_0) + jnp.sum(pde_instance.m_0 ** 2), 
            "momentum preservation: ": jnp.sqrt(jnp.sum((m_t - pde_instance.m_0)**2, axis=-1))}