import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import vjp
import jax.numpy as jnp
from utils.common_utils import divergence_fn, compute_pytree_norm
from jax.experimental.ode import odeint
from utils.plot_utils import plot_velocity
from example_problems.hlandau_example import HomogeneousLandau, conv_fn_vmap
from core.model import get_model
import jax.random as random
import optax
from tqdm import tqdm

def value_and_grad_fn(forward_fn, params, time_interval, data, rng, config, pde_instance: HomogeneousLandau):
    time_interval_current = time_interval["current"]
    time_offset = time_interval["current"][-1] * len(time_interval["previous"])
    
    params_flat, params_tree = tree_flatten(params)
    f_fn = lambda _x, _t, _params: forward_fn(_params, _t, _x)

    # unpack the parameters
    ODE_tolerance = config["ODE_tolerance"]

    # unpack the data
    n_train = data["data_initial"].shape[0]

    x_0 = jnp.concatenate([data["data_initial"], data["data_ref"]], axis=0)

    # xi_0 = pde_instance.score_t(jnp.zeros([]), x_0)
    xi_0 = jnp.concatenate([data["score_initial"], data["score_ref"]], axis=0)

    # weight_0 = pde_instance.density_t(jnp.zeros([]), x_0) / pde_instance.distribution_0.density(x_0)
    weight_train, weight_ref = data["weight_initial"], data["weight_ref"]

    # compute x(T) by solve IVP (I)
    # ================ Forward ===================
    loss_0 = jnp.zeros([])
    states_0 = {
        "x"   : x_0,
        "xi"  : xi_0,
        "loss": loss_0,
    }
    # print(f(x_0[:10], jnp.zeros([]), params))
    def ode_func1(states, t):
        x = states["x"]
        xi = states["xi"]
        f_t_theta = lambda _x: f_fn(_x, t, params)
        dx = f_t_theta(x)

        def h_t_theta(in_1, in_2):
            # dynamics of the score
            # in_1 is xi
            # in_2 is x
            div_f_t_theta = lambda _x: divergence_fn(f_t_theta, _x).sum(axis=0)
            grad_div_fn = jax.grad(div_f_t_theta)
            h1 = - grad_div_fn(in_2)
            _, vjp_fn = vjp(f_t_theta, in_2)
            h2 = - vjp_fn(in_1)[0]
            return h1 + h2

        dxi = h_t_theta(xi, x)

        def g_t(_xi, _x):
            # in_1 is xi
            # in_2 is x
            # split x and xi to train set and reference set
            x_train, x_ref = jnp.split(_x, [n_train], axis=0)
            xi_train, xi_ref = jnp.split(_xi, [n_train], axis=0)
            f_t_theta_x_train = f_t_theta(x_train)
            return jnp.mean(jnp.sum((f_t_theta_x_train - conv_fn_vmap(x_train, x_ref, xi_train, xi_ref, weight_ref)) ** 2, axis=-1) * weight_train)

        dloss = g_t(xi, x)

        # #################################################
        # x_train, x_ref = jnp.split(x, [n_train], axis=0)
        # true_velocity = true_velocity_fn(x_train, t)
        # dx_train, _ = jnp.split(dx, [n_train], axis=0)
        # dloss = jnp.sqrt(jnp.sum((dx_train - true_velocity) ** 2)/jnp.sum(true_velocity ** 2))
        # #################################################
        return {
            "x": dx,
            "xi": dxi,
            "loss": dloss,
        }

    result_forward = odeint(ode_func1, states_0, time_interval_current, atol=ODE_tolerance, rtol=ODE_tolerance)
    x_T = result_forward["x"][1]
    xi_T = result_forward["xi"][1]
    loss_f = result_forward["loss"][1]
    # print(loss_f)
    # ================ Forward ===================

    # ================ Backward ==================
    # compute dl/d theta via adjoint method
    a_T = jnp.zeros_like(x_T)
    b_T = jnp.zeros_like(xi_T)
    grad_T = [jnp.zeros_like(_var) for _var in params_flat]
    loss_T = jnp.zeros([])
    states_T = {
        "x"   : x_T,
        "xi"  : xi_T,
        "a"   : a_T,
        "b"   : b_T,
        "loss": loss_T,
        "grad": grad_T,
    }

        # [x_T, a_T, b_T, xi_T, loss_T, grad_T, ref_T, score_ref_T]

    def ode_func2(states, t):
        t = time_interval_current[-1] - t
        x = states["x"]
        xi = states["xi"]
        a = states["a"]
        b = states["b"]

        f_t = lambda _x, _params: f_fn(_x, t, _params)
        # bar_f_t = lambda _x, _params: bar_f(_x, t, _params)
        dx = f_t(x, params)

        _, vjp_fx_fn = vjp(lambda _x: f_t(_x, params), x)
        vjp_fx_a = vjp_fx_fn(a)[0]
        _, vjp_ftheta_fn = vjp(lambda _params: f_t(x, _params), params)
        vjp_ftheta_a = vjp_ftheta_fn(a)[0]

        def h_t(in_1, in_2, in_3):
            # in_1 is xi
            # in_2 is x
            # in_3 is theta
            f_t_theta = lambda _x: f_t(_x, in_3)
            div_f_t_theta = lambda _x: divergence_fn(f_t_theta, _x).sum(axis=0)
            grad_div_fn = jax.grad(div_f_t_theta)
            h1 = - grad_div_fn(in_2)
            _, vjp_fn = vjp(f_t_theta, in_2)
            h2 = - vjp_fn(in_1)[0]
            return h1 + h2

        dxi = h_t(xi, x, params)

        _, vjp_hxi_fn = vjp(lambda _xi: h_t(_xi, x, params), xi)
        vjp_hxi_b = vjp_hxi_fn(b)[0]
        _, vjp_hx_fn = vjp(lambda _x: h_t(xi, _x, params), x)
        vjp_hx_b = vjp_hx_fn(b)[0]
        _, vjp_htheta_fn = vjp(lambda _params: h_t(xi, x, _params), params)
        vjp_htheta_b = vjp_htheta_fn(b)[0]

        def g_t(_xi, _x, in_3):
            # in_1 is xi
            # in_2 is x
            # in_3 is theta
            # split x and xi to train set and reference set
            x_train, x_ref = jnp.split(_x, [n_train], axis=0)
            xi_train, xi_ref = jnp.split(_xi, [n_train], axis=0)
            f_t_in_2_in_3 = f_t(x_train, in_3)
            return jnp.mean(jnp.sum((f_t_in_2_in_3 - conv_fn_vmap(x_train, x_ref, xi_train, xi_ref, weight_ref)) ** 2, axis=-1) * weight_train)

        dxig = jax.grad(g_t, argnums=0)
        dxg = jax.grad(g_t, argnums=1)
        dthetag = jax.grad(g_t, argnums=2)

        da = - vjp_fx_a - vjp_hx_b - dxg(xi, x, params)
        db = - vjp_hxi_b - dxig(xi, x, params)

        dloss = g_t(xi, x, params)

        vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
        vjp_htheta_b_flat, _ = tree_flatten(vjp_htheta_b)
        dthetag_flat, _ = tree_flatten(dthetag(xi, x, params))
        dgrad = [_dgrad1 + _dgrad2 + _dgrad3 for _dgrad1, _dgrad2, _dgrad3 in
                 zip(vjp_ftheta_a_flat, vjp_htheta_b_flat, dthetag_flat)]
        # dgrad = vjp_ftheta_a + vjp_htheta_b + dthetag(xi, x, params)

        return {
        "x"   : -dx,
        "xi"  : -dxi,
        "a"   : -da,
        "b"   : -db,
        "loss": dloss,
        "grad": dgrad,
        }

    # ================ Backward ==================
    result_backward = odeint(ode_func2, states_T, time_interval_current, atol=ODE_tolerance, rtol=ODE_tolerance)

    grad_T = tree_unflatten(params_tree, [_var[-1] for _var in result_backward["grad"]])

    grad_norm = compute_pytree_norm(grad_T)
    return {"loss": loss_f, 
            "grad": grad_T, 
            "grad norm": grad_norm,
            "ODE error x": jnp.mean(jnp.sum((result_backward["xi"][-1] - states_0["xi"]) ** 2, axis=-1)),
            }


def test_fn(forward_fn, data, time_interval, pde_instance: HomogeneousLandau, rng):
    x_ground_truth = pde_instance.test_data["x_T"]
    # test_time_stamps = jnp.linspace(0, pde_instance.total_evolving_time, 11)
    # test_time_stamps = 
    forward_fn_vmapt = jax.vmap(forward_fn, in_axes=[0, None])
    velocity_fn_vmapt = jax.vmap(pde_instance.velocity_t, in_axes=[0, None])
    conv_pred = forward_fn_vmapt(time_interval["current"][-1] * jnp.ones([1]), x_ground_truth)
    conv_true = velocity_fn_vmapt(time_interval["current"][-1] * (len(time_interval["previous"]) + 1) * jnp.ones([1]), x_ground_truth)
    relative_l2_velocity = jnp.mean(jnp.sqrt(jnp.sum((conv_pred - conv_true) ** 2, axis=-1)), axis=-1)
    relative_l2_velocity = relative_l2_velocity / jnp.mean(jnp.sqrt(jnp.sum(conv_true ** 2, axis=-1)), axis=-1)
    # return {"relative l2 error (average over time)": jnp.mean(relative_l2), "relative l2 error (maximum over time)": jnp.max(relative_l2)}
    
    return {"relative l2 (velocity)": relative_l2_velocity, }
    # TODO: Include the test for density

    T = pde_instance.total_evolving_time


    def bar_f(_x, _t):
        dx = forward_fn(_t, _x)
        return dx
    # prepare the data for testing
    # test_time_stamps = jnp.linspace(0, T, num=10)

    side_x = jnp.linspace(pde_instance.mins[0], pde_instance.maxs[0], 1000)
    side_y = jnp.linspace(pde_instance.mins[1], pde_instance.maxs[1], 1000)
    X, Y = jnp.meshgrid(side_x, side_y)
    data_T = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)


    state_T = [data_T]

    def ode_func_backward(states, t):
        t = T-t
        x = states[0]
        dx = bar_f(x, t)
        return [-dx]

    tspace = jnp.array((0., T))
    result_backward = odeint(ode_func_backward, state_T, tspace, atol=1e-5, rtol=1e-5)
    data_0 = result_backward[0][1]

    density_0 = pde_instance.density_t(jnp.zeros([]), data_0)
    log_density_0 = jnp.log(density_0)
    states_0 = [data_0, log_density_0]

    def ode_func(states, t):
        bar_f_t_theta = lambda _x: bar_f(_x, t)

        x = states[0]
        dx = bar_f(x, t)

        def dlog_density_func(in_1):
            # in_1 is x
            div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x)
            return -div_bar_f_t_theta(in_1)

        d_logdensity = dlog_density_func(x)

        return [dx, d_logdensity]

    tspace = jnp.array((0., T))
    result_forward = odeint(ode_func, states_0, tspace, atol=1e-6, rtol=1e-6)

    # xs = result_forward[0]  # the first axis is time, the second axis is batch, the last axis is problem dimension
    densities = jnp.exp(result_forward[1][1])  # the first axis is time, the second axis is batch

    # v_density_fn = jax.vmap(density_fn, in_axes=[0, 0])
    densities_true = pde_instance.density_t(jnp.ones([]) * T, data_T)

    L2_error = jnp.sqrt(jnp.mean((densities - densities_true)**2))
    L2 = jnp.sqrt(jnp.mean(densities_true ** 2))
    relative_L2 = L2_error/L2

    return {"relative l2 (density)": relative_L2, "relative l2 (velocity)": relative_l2_velocity, }



def create_model_fn(pde_instance: HomogeneousLandau):
    net = get_model(pde_instance.cfg)
    # net = get_model(pde_instance.cfg, DEBUG=True, pde_instance=pde_instance)
    params = net.init(random.PRNGKey(11), jnp.zeros(1), pde_instance.distribution_0.sample(1, random.PRNGKey(1)))
    # print("Pretraining the hypothesis velocity field using the initial data to improve the performance.")
    # params = velocity_field_pretraining(pde_instance, net, params)
    # print("Finished pretraining.")
    return net, params

def velocity_field_pretraining(pde_instance: HomogeneousLandau, net, params):
    # create an optimizer for pretrain
    optimizer = optax.chain(optax.adaptive_grad_clip(1),
                                    optax.add_decayed_weights(1e-3),
                                    optax.sgd(learning_rate=1e-2, momentum=0.9)
                                    )
    opt_state = optimizer.init(params)

    pretrain_steps = 4096
    # pretrain using the initial data
    key_pretrains = random.split(random.PRNGKey(2199), pretrain_steps)


    # create time stamps:
    time_stampes = jnp.linspace(0, pde_instance.total_evolving_time, 128)
    
    def pretrain_loss_fn(params, t, data):
        conv_true_0 = pde_instance.velocity_t(t, data)
        return jnp.mean(jnp.sum((net.apply(params, t, data) - conv_true_0) ** 2, axis=-1))
    
    pretrain_loss_fn = jax.vmap(pretrain_loss_fn, in_axes=[None, 0, None])

    def loss_fn(params, t, data):
        return jnp.mean(pretrain_loss_fn(params, t, data))
    
    grad_fn = jax.grad(loss_fn,)

    def update_fn(rng, params, opt_state):
        # sample from initial distribution
        data_initial = pde_instance.distribution_0.sample(256, key_pretrain)
        grad = grad_fn(params, time_stampes, data_initial)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    update_fn = jax.jit(update_fn)

    for key_pretrain in tqdm(key_pretrains):
        params, opt_state = update_fn(key_pretrain, params, opt_state)
        
    return params