import jax
from jax import jacrev
import jax.numpy as jnp
from utils.plot_utils import plot_density_2d
from core.distribution import Uniform
from core.model import MLP
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from core.normalizing_flow import RealNVP, MNF
import jax.random as random
import flax.linen as nn
import optax

def value_and_grad_fn(forward_fn, params, data, rng, config, pde_instance: KineticFokkerPlanck):
    weights = config["weights"]
    beta = pde_instance.beta
    Gamma = pde_instance.Gamma
    diffusion_coefficient = pde_instance.beta * pde_instance.Gamma
    # unpack data
    z_initial = data["data_initial"]
    time_train, space_train, = data["data_train"]

    def model_loss(_params):
        def acceleration_fn(x, v):
            return -beta * x - 4 * beta / Gamma * v

        def velocity(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            dx = v
            dv = acceleration_fn(x, v)
            dz = jnp.concatenate([dx, dv], axis=-1)
            return dz

        def div_v_acceleration(t, z):
            # z.shape[-1] = 2d, d is the dimension of the spatial domain
            return -4. * beta / Gamma * z.shape[-1] / 2.

        # @functools.partial(jax.vmap, in_axes=(0, 0))
        def fokker_planck_eq(t, z):
            x, v = jnp.split(z, indices_or_sections=2, axis=-1)
            acceleration = acceleration_fn(x, v)
            u = forward_fn(_params, t, z) # u is a scalar
            
            u_t = jacrev(forward_fn, argnums=1, )(_params, t, z)[0] # use [0] to make u_t a scalar

            u_z_fn = jax.jacrev(forward_fn, argnums=2)
            u_z = u_z_fn(_params, t, z)
            u_x, u_v = jnp.split(u_z, indices_or_sections=2, axis=-1)
            u_x_dot_v = jnp.dot(u_x, v)

            u_v_dot_acceleration = jnp.dot(u_v, acceleration)
            
            u_dot_div_v_acceleration = u * div_v_acceleration(t, z)
            
            jacobian_fn = jax.hessian(forward_fn, argnums=2)
            jacobian_diag = jnp.diag(jacobian_fn(_params, t, z))
            _, jacobian_diag_v = jnp.split(jacobian_diag, indices_or_sections=2, axis=-1)
            u_laplacian_v = jnp.sum(jacobian_diag_v)
            return u_t + u_v_dot_acceleration + u_x_dot_v + u_dot_div_v_acceleration - u_laplacian_v * diffusion_coefficient

        if pde_instance.cfg.neural_network.name == "RealNVP":
            fokker_planck_eq = jax.vmap(fokker_planck_eq, in_axes=(0, 0))
        else:
            fokker_planck_eq = jax.vmap(jax.vmap(fokker_planck_eq, in_axes=(None, 0)), in_axes=(0, None))

        def mass_change_fn(t, z):
            u_t = jacrev(forward_fn, argnums=1, )(_params, t, z)[0]
            return u_t

        mass_change_fn_vmap_z = jax.vmap(mass_change_fn, in_axes=(None, 0))
        def mass_change_t_fn(t, z):
            return jnp.mean(mass_change_fn_vmap_z(t, z))

        mass_change_t_fn_vmap_t = jax.vmap(mass_change_t_fn, in_axes=(0, None))

        
        forward_fn_vmapx = jax.vmap(forward_fn, in_axes=(None, None, 0))
        u_pred_initial = forward_fn_vmapx(_params, jnp.zeros([]), z_initial)
        f_pred_train = fokker_planck_eq(time_train, space_train)
        # mass_change_total = mass_change_t_fn_vmap_t(time_train, space_train)
        
        # loss_u_initial = jnp.mean((u_pred_initial - pde_instance.u_0(z_initial)) ** 2)
        # loss_f_train = jnp.mean((f_pred_train) ** 2)
        # loss_mass_change = jnp.mean(mass_change_total ** 2)

        loss_u_initial = jnp.mean(jnp.abs(u_pred_initial - pde_instance.u_0(z_initial)) ** 1.1)
        loss_f_train = jnp.mean(jnp.abs(f_pred_train) ** 1.1)
        # loss_mass_change = jnp.mean(jnp.abs(mass_change_total) ** 1.1)

        # return loss_u_boundary * weights["weight_boundary"] + loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"]
        loss = loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"]
        # return loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"] + loss_mass_change * weights["mass_change"]
        return loss * pde_instance.domain_area
        # return loss_u_initial * weights["weight_initial"] + loss_f_train * weights["weight_train"], {"loss initial": loss_u_initial, "loss train": loss_f_train}
        # return loss_f_train * weights["weight_train"]
    
    v_g = jax.value_and_grad(model_loss)
    value, grad = v_g(params)
    return {"PINN loss": value, "grad": grad, }


def test_fn(forward_fn, config, pde_instance: KineticFokkerPlanck, rng):
    domain_area = pde_instance.domain_area

    # These functions take a single point (t, x) as input
    rho = forward_fn
    log_rho = lambda t, z: jnp.maximum(jnp.log(forward_fn(t, z)), -100)
    nabla_log_rho = jax.jacrev(log_rho, argnums=1)

    # unpack the test data
    test_time_stamps = pde_instance.test_data[0]
    test_time_stamps = test_time_stamps[:, None] # make sure that t is of size [1]

    # side_x = jnp.linspace(mins[0], maxs[0], 256)
    # side_y = jnp.linspace(mins[1], maxs[1], 256)
    # X, Y = jnp.meshgrid(side_x, side_y)
    # grid_points_test = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    # grid_points_test = jnp.concatenate([grid_points_test, grid_points_test], axis=-1)
    # distribution_0 = Uniform(mins, maxs)
    points_test = pde_instance.distribution_domain.sample(pde_instance.cfg.test.batch_size, rng)

    
    # rho = jax.vmap(rho, in_axes=[None, 0])
    # log_rho = jax.vmap(log_rho, in_axes=[None, 0])
    # nabla_log_rho = jax.vmap(nabla_log_rho, in_axes=[None, 0])

    # densities = []
    # log_densities = []
    # scores = []
    # for i in range(len(test_time_stamps)):
    #     densities.append(rho(test_time_stamps[i], points_test))
    #     log_densities.append(log_rho(test_time_stamps[i], points_test))
    #     scores.append(nabla_log_rho(test_time_stamps[i], points_test))

    # densities = jnp.stack(densities, axis=0)
    # log_densities = jnp.stack(log_densities, axis=0)
    # scores = jnp.stack(scores, axis=0)
    
    rho = jax.vmap(jax.vmap(rho, in_axes=[None, 0]), in_axes=[0, None])
    # log_rho = jax.vmap(jax.vmap(log_rho, in_axes=[None, 0]), in_axes=[0, None])
    nabla_log_rho = jax.vmap(jax.vmap(nabla_log_rho, in_axes=[None, 0]), in_axes=[0, None])

    densities = jnp.maximum(rho(test_time_stamps, points_test), 1e-20)
    log_densities = jnp.log(densities)
    # log_densities = log_rho(test_time_stamps, points_test)
    scores = nabla_log_rho(test_time_stamps, points_test)

    scores_true, log_densities_true = pde_instance.ground_truth(test_time_stamps, points_test)
    

    KL = jnp.mean(densities * (log_densities - log_densities_true)) * domain_area
    L1 = jnp.mean(jnp.abs(densities - jnp.exp(log_densities_true))) * domain_area
    total_mass = jnp.mean(densities) * domain_area
    total_mass_true = jnp.mean(jnp.exp(log_densities_true)) * domain_area

    Fisher_information = jnp.mean(densities * jnp.sum((scores - scores_true) ** 2, axis=-1)) * domain_area

    # print(f"KL {KL: .2f}, L1 {L1: .2f}, Fisher information {Fisher_information: .2f}")
    # print(f"Total mass {total_mass: .2f}, True total mass {total_mass_true: .2f}")
    return {"L1": L1, "KL": KL, "Fisher Information": Fisher_information, "total_mass": total_mass, "total_mass_true": total_mass_true}


def plot_fn(forward_fn, config, pde_instance: KineticFokkerPlanck, rng):
    pass
    # T = KineticFokkerPlanck.total_evolving_time
    # t_part = config["t_part"]
    # for t in range(t_part):
    #     def f(x: jnp.ndarray):
    #         batch_size_x = x.shape[0]
    #         return forward_fn(jnp.ones((batch_size_x, 1)) * T / t_part * t, x)

    #     plot_density_2d(f, config)


class MLPKFPE(nn.Module):
    pde_instance: KineticFokkerPlanck
    DEBUG: bool = False
    
    def setup(self):
        self.time_embedding_dim = self.pde_instance.cfg.neural_network.time_embedding_dim
        self.hidden_dims = [self.pde_instance.cfg.neural_network.hidden_dim] * self.pde_instance.cfg.neural_network.layers
        # self.u = MLP(output_dim=1, time_embedding_dim=self.time_embedding_dim, hidden_dims=self.hidden_dims)
        self.u = create_normalizing_flow_fn(self.pde_instance.logprob_0, dim=self.pde_instance.dim * 2) # 2d for Kinetic problems

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        if t.ndim == 1 and len(t) == 1:
            t = t[0]
        elif t.ndim == 0:
            pass
        else:
            raise ValueError("t should be either a scalar!")
        
        if self.DEBUG:
            return self.pde_instance.u_t(t, x)
        else:
            # if x.ndim == 1: # non-batched
            #     return self.u(t, x)[0]
            # elif x.ndim == 2: # batched
            #     return self.u(t, x)[:, 0]
            # else:
            #     raise ValueError("x should be either 1D or 2D array")
            if x.ndim == 1: # non-batched
                return self.u(t, x)
            elif x.ndim == 2: # batched
                return self.u(t, x)
            else:
                raise ValueError("x should be either 1D or 2D array")


def create_model_fn(pde_instance: KineticFokkerPlanck):
    net = MLPKFPE(pde_instance=pde_instance, DEBUG=False)
    params = net.init(random.PRNGKey(11), jnp.zeros([]), jnp.squeeze(pde_instance.distribution_0.sample(1, random.PRNGKey(1))))
    # set the scaling of the MLP so that the total mass is of the right order.



    print("Pretraining the hypothesis velocity field using the initial data to improve the performance.")
    params = model_pretrain_fn(pde_instance=pde_instance, net=net, params=params)
    print("Finished pretraining.")

    return net, params

def model_pretrain_fn(pde_instance: KineticFokkerPlanck, net, params):
    # create an optimizer for pretrain
    optimizer = optax.chain(optax.clip(1),
                            optax.add_decayed_weights(1e-3),
                            optax.sgd(learning_rate=1e-2, momentum=0.9)
                            )
    opt_state = optimizer.init(params)

    pretrain_steps = 4096
    # pretrain using the initial data
    key_pretrains = random.split(random.PRNGKey(2199), pretrain_steps)


    # create time stamps:
    time_stampes = jnp.linspace(jnp.zeros([1]), jnp.ones([1])*pde_instance.total_evolving_time, 128)
    
    def pretrain_loss_u_fn(params, t, data):
        u_0_true = pde_instance.u_0(data)
        forward_fn_vmapx = jax.vmap(net.apply, in_axes=[None, None, 0])
        u_t_predict = forward_fn_vmapx(params, t, data)
        return jnp.mean((u_t_predict - u_0_true) ** 2)
        
    pretrain_loss_u_fn = jax.vmap(pretrain_loss_u_fn, in_axes=[None, 0, None])

    def loss_fn(params, t, data):
        return jnp.mean(pretrain_loss_u_fn(params, t, data)) * pde_instance.domain_area
    
    grad_fn = jax.grad(loss_fn,)
    
    @jax.jit
    def update_fn(key_pretrain, params, opt_state):
        data_initial = pde_instance.distribution_domain.sample(256, key_pretrain)

        grad = grad_fn(params, time_stampes, data_initial)

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state


    for key_pretrain in key_pretrains:
        params, opt_state = update_fn(key_pretrain, params, opt_state)
        

    return params

def create_normalizing_flow_fn(log_prob_0, dim):
    param_dict = {
        'dim': dim,
        'embed_time_dim': 0,
        'couple_mul': 2,
        'mask_type': 'loop',
        'activation_layer': 'celu',
        'soft_init': 0.,
        'ignore_time': False,
    }
    mnf = MNF(**param_dict)
    return RealNVP(mnf, log_prob_0)