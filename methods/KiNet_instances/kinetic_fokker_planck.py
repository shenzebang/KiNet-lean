import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax import grad, vjp
import jax.numpy as jnp
from utils.common_utils import divergence_fn
from jax.experimental.ode import odeint
from utils.plot_utils import plot_velocity
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from core.model import get_model
from utils.common_utils import compute_pytree_norm
import jax.random as random
from utils.optimizer import get_optimizer
import optax

def value_and_grad_fn(forward_fn, params, time_interval, data, rng, config, pde_instance: KineticFokkerPlanck):
    time_interval_current = time_interval["current"]
    time_offset = time_interval["current"][-1] * len(time_interval["previous"])
    params_flat, params_tree = tree_flatten(params)

    def bar_f_fn(_z, _t, _params):
        forward_fn_params = lambda t, z: forward_fn(_params, t, z)
        dynamics_fn = pde_instance.forward_fn_to_dynamics(forward_fn_params, time_offset)
        return dynamics_fn(_t, _z)

    def f_fn(_z, _t, _params):
        score = forward_fn(_params, _t, _z)
        return score

    # compute x(T) by solve IVP (I)
    # ================ Forward ===================
    states_0 = {
        "z": data["data_initial"],
        "xi": data["score_initial"],
        "loss": jnp.zeros([])
    }

    def ode_func1(states, t):
        f_t_theta = lambda _x: f_fn(_x, t, params)
        bar_f_t_theta = lambda _x: bar_f_fn(_x, t, params)

        def h_t_theta(xi, z):
            div_bar_f_t_theta = lambda _z: divergence_fn(bar_f_t_theta, _z).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(z)
            _, vjp_fn = vjp(bar_f_t_theta, z)
            h2 = - vjp_fn(xi)[0]
            return h1 + h2

        def g_t(xi, z):
            f_t_theta_in_2 = f_t_theta(z)
            score_x, score_v = jnp.split(xi, indices_or_sections=2, axis=-1)
            return jnp.mean(jnp.sum((f_t_theta_in_2 - score_v) ** 2, axis=(1,)))

        return {
            "z": bar_f_t_theta(states["z"]),
            "xi": h_t_theta(states["xi"], states["z"]),
            "loss": g_t(states["xi"], states["z"]),}

    # tspace = jnp.array((0., pde_instance.total_evolving_time))
    result_forward = odeint(ode_func1, states_0, time_interval_current, atol=config["ODE_tolerance"], rtol=config["ODE_tolerance"])
    z_T = result_forward["z"][-1]
    xi_T = result_forward["xi"][-1]
    loss_f = result_forward["loss"][-1]
    # ================ Forward ===================

    # ================ Backward ==================
    # compute dl/d theta via adjoint method
    states_T = {
        "z": z_T,
        "a": jnp.zeros_like(z_T),
        "b": jnp.zeros_like(xi_T),
        "xi": xi_T,
        "loss": jnp.zeros([]),
        "grad": [jnp.zeros_like(_var) for _var in params_flat]
    }

    def ode_func2(states, t):
        t = time_interval_current[-1] - t

        f_t = lambda _x, _params: f_fn(_x, t, _params)
        bar_f_t = lambda _x, _params: bar_f_fn(_x, t, _params)

        _, vjp_fx_fn = vjp(lambda _x: bar_f_t(_x, params), states["z"])
        vjp_fx_a = vjp_fx_fn(states["a"])[0]
        _, vjp_ftheta_fn = vjp(lambda _params: bar_f_t(states["z"], _params), params)
        vjp_ftheta_a = vjp_ftheta_fn(states["a"])[0]

        def h_t(xi, z, theta):
            # in_1 is xi
            # in_2 is z
            # in_3 is theta
            bar_f_t_theta = lambda _z: bar_f_t(_z, theta)
            div_bar_f_t_theta = lambda _z: divergence_fn(bar_f_t_theta, _z).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(z)
            _, vjp_fn = vjp(bar_f_t_theta, z)
            h2 = - vjp_fn(xi)[0]
            return h1 + h2

        _, vjp_hxi_fn = vjp(lambda _xi: h_t(_xi, states["z"], params), states["xi"])
        vjp_hxi_b = vjp_hxi_fn(states["b"])[0]
        _, vjp_hx_fn = vjp(lambda _x: h_t(states["xi"], _x, params), states["z"])
        vjp_hx_b = vjp_hx_fn(states["b"])[0]
        _, vjp_htheta_fn = vjp(lambda _params: h_t(states["xi"], states["z"], _params), params)
        vjp_htheta_b = vjp_htheta_fn(states["b"])[0]

        def g_t(xi, z, theta):
            # in_1 is xi
            # in_2 is z
            # in_3 is theta
            f_t_in_2_in_3 = f_t(z, theta)
            score_x, score_v = jnp.split(xi, indices_or_sections=2, axis=-1)
            return jnp.mean(jnp.sum((f_t_in_2_in_3 - score_v) ** 2, axis=(1,)))

        dxig = grad(g_t, argnums=0)
        dxg = grad(g_t, argnums=1)
        dthetag = grad(g_t, argnums=2)

        da = - vjp_fx_a - vjp_hx_b - dxg(states["xi"], states["z"], params)
        db = - vjp_hxi_b - dxig(states["xi"], states["z"], params)

        vjp_ftheta_a_flat, _ = tree_flatten(vjp_ftheta_a)
        vjp_htheta_b_flat, _ = tree_flatten(vjp_htheta_b)
        dthetag_flat, _ = tree_flatten(dthetag(states["xi"], states["z"], params))
        dgrad = [_dgrad1 + _dgrad2 + _dgrad3 for _dgrad1, _dgrad2, _dgrad3 in
                 zip(vjp_ftheta_a_flat, vjp_htheta_b_flat, dthetag_flat)]

        return {
            "z": -bar_f_t(states["z"], params),
            "a": -da,
            "b": -db,
            "xi": -h_t(states["xi"], states["z"], params),
            "loss": g_t(states["xi"], states["z"], params),
            "grad": dgrad
        }

    # ================ Backward ==================
    result_backward = odeint(ode_func2, states_T, time_interval_current, atol=config["ODE_tolerance"], rtol=config["ODE_tolerance"])

    grad_T = tree_unflatten(params_tree, [_var[-1] for _var in result_backward["grad"]])
    # x_0_b = result_backward[0][-1]
    # ref_0_b = result_backward[6][-1]
    # xi_0_b = result_backward[3][-1]

    # These quantities are for the purpose of debug
    # error_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(x_0.shape[0], -1) ** 2, axis=(1,)))
    # error_xi = jnp.mean(jnp.sum((xi_0 - xi_0_b).reshape(xi_0.shape[0], -1) ** 2, axis=(1,)))
    # error_ref = jnp.mean(jnp.sum((ref_0 - ref_0_b).reshape(ref_0.shape[0], -1) ** 2, axis=(1,)))
    # loss_b = result_backward[4][-1]
    grad_norm = compute_pytree_norm(grad_T)
    return {
        "loss": loss_f,
        "grad": grad_T,
        "grad norm": grad_norm,
        "ODE error x": jnp.mean(jnp.sum((result_backward["z"][-1] - states_0["z"]) ** 2, axis=-1)),
    }

def distance_to_equilibrium(data, pde_instance: KineticFokkerPlanck, rng):
    # compute the KL divergence between the latest variable distribution and the equilibrium
    score_true, logprob_true = pde_instance.equilibrium.score(data["data_initial"]), pde_instance.equilibrium.logdensity(data["data_initial"])
    KL = jnp.mean(data["logprob_initial"] - logprob_true)
    Fisher_information = jnp.mean(jnp.sum((data["score_initial"] - score_true) ** 2, axis=-1))

    return {"KL": KL, "Fisher Information": Fisher_information}

def test_fn(forward_fn, data, time_interval, pde_instance: KineticFokkerPlanck, rng):
    time_offset = time_interval["current"][-1] * len(time_interval["previous"])
    test_time_stamps = jnp.linspace(time_interval["current"][0], time_interval["current"][-1], 11)
    # compute the KL divergence and Fisher-information
    def bar_f(_z, _t):
        dynamics_fn = pde_instance.forward_fn_to_dynamics(forward_fn, time_offset=time_offset)
        return dynamics_fn(_t, _z)

    states_0 = {
        "z"         : data["data_initial"],
        "score"     : data["score_initial"],
        "logprob"   : data["logprob_initial"]
    }

    def ode_func(states, t):
        bar_f_t_theta = lambda _x: bar_f(_x, t)

        def dlog_density_fn(in_1):
            # in_1 is x
            div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x)
            return -div_bar_f_t_theta(in_1)

        def dscore_fn(in_1, in_2):
            # in_1 is score
            # in_2 is x
            div_bar_f_t_theta = lambda _x: divergence_fn(bar_f_t_theta, _x).sum(axis=0)
            grad_div_fn = grad(div_bar_f_t_theta)
            h1 = - grad_div_fn(in_2)
            _, vjp_fn = vjp(bar_f_t_theta, in_2)
            h2 = - vjp_fn(in_1)[0]
            return h1 + h2

        return {
            "z"         : bar_f_t_theta(states["z"]),
            "score"     : dscore_fn(states["score"], states["z"]),
            "logprob"   : dlog_density_fn(states["z"])

        }

    result_forward = odeint(ode_func, states_0, test_time_stamps, atol=1e-6, rtol=1e-6)

    xs = result_forward["z"]  # the first axis is time, the second axis is batch, the last axis is problem dimension
    log_densities = result_forward["logprob"]  # the first axis is time, the second axis is batch
    scores = result_forward["score"]  # the first axis is time, the second axis is batch, the last axis is problem dimension

    scores_true, log_densities_true = pde_instance.ground_truth(test_time_stamps + time_offset, xs)

    KL = jnp.mean(log_densities - log_densities_true, axis=(0, 1))
    Fisher_information = jnp.mean(jnp.sum((scores - scores_true) ** 2, axis=-1), axis=(0, 1))

    return {"KL": KL, "Fisher Information": Fisher_information}

def create_model_fn(pde_instance: KineticFokkerPlanck):
    # net = KiNet(time_embedding_dim=20, append_time=False)
    net = get_model(pde_instance.cfg)
    params = net.init(random.PRNGKey(11), jnp.zeros(1), pde_instance.distribution_0.sample(1, random.PRNGKey(1)))
    # params = net.init(random.PRNGKey(11), jnp.zeros(1),
    #                   jnp.squeeze(pde_instance.distribution_0.sample(1, random.PRNGKey(1))))

    if pde_instance.cfg.train.pretrain:
        print("Pretraining the hypothesis velocity field using the initial data to improve the performance.")
        params = velocity_field_pretraining(pde_instance=pde_instance, net=net, params=params)
        print("Finished pretraining.")

    return net, params

def velocity_field_pretraining(pde_instance: KineticFokkerPlanck, net, params):
    # create an optimizer for pretrain
    # optimizer = optax.chain(optax.adaptive_grad_clip(1),
    #                                 optax.add_decayed_weights(1e-2),
    #                                 optax.sgd(learning_rate=1e-2, momentum=0.9)
    #                                 )
    optimizer = get_optimizer(pde_instance.cfg.train)

    opt_state = optimizer.init(params)

    pretrain_steps = 40_000
    # pretrain using the initial data
    key_pretrains = random.split(random.PRNGKey(2199), pretrain_steps)


    # create time stamps:
    time_stampes = jnp.linspace(0, pde_instance.total_evolving_time, 128)
    
    def pretrain_loss_fn(params, t, data):
        score_xv_true = pde_instance.distribution_0.score(data)
        _, score_v_true = jnp.split(score_xv_true, 2, axis=-1)
        return jnp.mean(jnp.sum((net.apply(params, t, data) - score_v_true) ** 2, axis=-1))
    
    pretrain_loss_fn = jax.vmap(pretrain_loss_fn, in_axes=[None, 0, None])

    def loss_fn(params, t, data):
        return jnp.mean(pretrain_loss_fn(params, t, data))
    
    grad_fn = jax.grad(loss_fn,)

    def update_fn(rng, params, opt_state):
        # sample from initial distribution
        data_initial = pde_instance.distribution_0.sample(256, rng)
        grad = grad_fn(params, time_stampes, data_initial)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    update_fn = jax.jit(update_fn)

    for key_pretrain in key_pretrains:
        params, opt_state = update_fn(key_pretrain, params, opt_state)
        
    return params

        
    
