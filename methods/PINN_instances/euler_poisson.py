import jax
from jax import jacrev
import jax.numpy as jnp
from utils.plot_utils import plot_density_2d
from core.model import MLP
from example_problems.euler_poisson_with_drift import EulerPoissonWithDrift
# from core.normalizing_flow import RealNVP, MNF
import jax.random as random
from flax import linen as nn
from typing import List
import optax
from utils.common_utils import compute_pytree_norm

def value_and_grad_fn(forward_fn, params, data, rng, config, pde_instance: EulerPoissonWithDrift):
    weights = config["weights"]
    # unpack data
    space_initial = data["data_initial"]
    time_train, space_train, = data["data_train"]

    def model_loss(_params):
        mu_fn = lambda t, z: forward_fn(_params, t, z, ["mu"])["mu"]
        u_fn = lambda t, z: forward_fn(_params, t, z, ["u"])["u"]
        nabla_phi_fn = lambda t, z: forward_fn(_params, t, z, ["nabla_phi"])["nabla_phi"]
        # ====================================================================================================
        # residual
        def euler_poisson_equation(t, x):
            mu_t = jacrev(mu_fn, argnums=0,)(t, x)[0]

            u_mu_fn = lambda t, x: mu_fn(t, x) * u_fn(t, x)
            jac_u_mu_fn = jax.jacfwd(u_mu_fn, argnums=1)
            div_u_mu = jnp.sum(jnp.diag(jac_u_mu_fn(t, x)))
            term_1 = (mu_t + div_u_mu) ** 2

            u_t = jacrev(u_fn, argnums=0, )(t, x)[0]
            uu_fn = lambda _x: jnp.dot(u_fn(t, _x), u_fn(t, x))
            u_nabla_u_fn = jax.grad(uu_fn)
            term_2 = jnp.sum((u_t + u_nabla_u_fn(x) + nabla_phi_fn(t, x) - pde_instance.drift_term(t, x))**2, axis=-1)

            jac_nabla_phi_fn = jax.jacfwd(nabla_phi_fn, argnums=1)
            laplacian_phi = jnp.sum(jnp.diag(jac_nabla_phi_fn(t, x)))
            term_3 = (laplacian_phi + mu_fn(t, x)) ** 2

            return term_1 + term_2 + term_3

        euler_poisson_equation = jax.vmap(jax.vmap(euler_poisson_equation, in_axes=(None, 0)), in_axes=(0, None))
        residual = euler_poisson_equation(time_train, space_train)
        # ====================================================================================================

        # ====================================================================================================
        # mass change
        def mass_change(t, z):
            u_t = jacrev(mu_fn, argnums=0,)(t, z)[0]
            return u_t

        mass_change = jax.vmap(mass_change, in_axes=(None, 0))
        def mass_change_t(t, z):
            return jnp.mean(mass_change(t, z))

        mass_change_t = jax.vmap(mass_change_t, in_axes=(0, None))
        mass_change_total = mass_change_t(time_train, space_train)
        # ====================================================================================================

        # ====================================================================================================
        # initial loss
        mu_fn_vmapx = jax.vmap(mu_fn, in_axes=(None, 0))
        mu_pred_initial = mu_fn_vmapx(jnp.zeros([]), space_initial)
        loss_mu_initial = jnp.mean((mu_pred_initial - pde_instance.mu_0(space_initial)) ** 2)

        u_fn_vmapx = jax.vmap(u_fn, in_axes=(None, 0))
        u_pred_initial = u_fn_vmapx(jnp.zeros([]), space_initial)
        loss_u_initial = jnp.mean(jnp.sum((u_pred_initial - pde_instance.u_0(space_initial)) ** 2, axis=-1))

        nabla_phi_fn_vmapx = jax.vmap(nabla_phi_fn, in_axes=(None, 0))
        nabla_phi_pred_initial = nabla_phi_fn_vmapx(jnp.zeros([]), space_initial)
        loss_nabla_phi_initial = jnp.mean(jnp.sum((nabla_phi_pred_initial - pde_instance.nabla_phi_0(space_initial)) ** 2, axis=-1))

        # ====================================================================================================

        # ====================================================================================================
        # total loss = (loss of initial condition) + (loss of residual) + (loss of mass change)
        loss_initial = loss_u_initial + loss_mu_initial + loss_nabla_phi_initial
        loss_residual = jnp.mean(residual)
        loss_mass_change = jnp.mean(mass_change_total ** 2)
        # ====================================================================================================

        return loss_initial * weights["weight_initial"] + loss_residual * weights["weight_train"] + loss_mass_change * weights["mass_change"]

    v_g = jax.value_and_grad(model_loss)
    value, grad = v_g(params)

    return {"PINN loss": value, "grad": grad, "grad norm": compute_pytree_norm(grad)}


def test_fn(forward_fn, config, pde_instance: EulerPoissonWithDrift, rng):
    nabla_phi_fn = lambda t, x: forward_fn(t, x, ["nabla_phi"])["nabla_phi"]
    nabla_phi_fn = jax.vmap(nabla_phi_fn, in_axes=[None, 0])
    x_ground_truth = pde_instance.test_data["x_T"]
    acceleration_pred = - nabla_phi_fn(jnp.ones(1) * pde_instance.total_evolving_time, x_ground_truth)
    acceleration_true = pde_instance.ground_truth(jnp.ones(1) * pde_instance.total_evolving_time, x_ground_truth)
    relative_l2 = jnp.mean(jnp.sqrt(jnp.sum((acceleration_pred - acceleration_true) ** 2, axis=-1)))
    relative_l2 = relative_l2 / jnp.mean(jnp.sqrt(jnp.sum(acceleration_true ** 2, axis=-1)))

    return {"relative l2 error": relative_l2}

def plot_fn(forward_fn, config, pde_instance: EulerPoissonWithDrift, rng):
    pass

class MLPEulerPoisson(nn.Module):
    pde_instance: EulerPoissonWithDrift
    # hidden_dims: List[int] 
    # time_embedding_dim: int = 0
    DEBUG: bool = False
    

    def setup(self):
        self.time_embedding_dim = self.pde_instance.cfg.neural_network.time_embedding_dim
        self.hidden_dims = [self.pde_instance.cfg.neural_network.hidden_dim] * self.pde_instance.cfg.neural_network.layers
        self.mu = MLP(output_dim=1, time_embedding_dim=self.time_embedding_dim, hidden_dims=self.hidden_dims)
        self.u = MLP(output_dim=3, time_embedding_dim=self.time_embedding_dim, hidden_dims=self.hidden_dims)
        self.nabla_phi = MLP(output_dim=3, time_embedding_dim=self.time_embedding_dim, hidden_dims=self.hidden_dims)

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, keys: List[str]):
        result = {}
        for key in keys:
            if key == "mu":
                result[key] = self.mu(t, x) if not self.DEBUG else self.pde_instance.mu_t(t, x)
            elif key == "u":
                result[key] = self.u(t, x) if not self.DEBUG else self.pde_instance.u_t(t, x)
                # result[key] = self.u(t, x)
            elif key == "nabla_phi":
                # result[key] = self.nabla_phi(t, x) if not self.DEBUG else self.pde_instance.nabla_phi_t(t, x)
                result[key] = self.nabla_phi(t, x)
            else:
                raise Exception("(PINN) unknown key!")
        return result


def create_model_fn(pde_instance: EulerPoissonWithDrift):
    # net = MLPEulerPoisson(time_embedding_dim=pde_instance.cfg.neural_network.time_embedding_dim,
                        #   hidden_dims=[pde_instance.cfg.neural_network.hidden_dim] * pde_instance.cfg.neural_network.layers)
    net = MLPEulerPoisson(pde_instance=pde_instance, DEBUG=True)

    params = net.init(random.PRNGKey(11), jnp.zeros(1), jnp.squeeze(pde_instance.distribution_0.sample(1, random.PRNGKey(1))), ["mu", "u", "nabla_phi"])

    print("Pretraining the hypothesis velocity field using the initial data to improve the performance.")
    params = model_pretrain_fn(pde_instance=pde_instance, net=net, params=params)
    print("Finished pretraining.")

    return net, params


def model_pretrain_fn(pde_instance: EulerPoissonWithDrift, net, params):
    # create an optimizer for pretrain
    optimizer = optax.chain(optax.clip(1),
                            optax.add_decayed_weights(1e-4),
                            optax.sgd(learning_rate=1e-2, momentum=0.9)
                            )
    opt_state = optimizer.init(params)

    pretrain_steps = 4096
    # pretrain using the initial data
    key_pretrains = random.split(random.PRNGKey(2199), pretrain_steps)


    # create time stamps:
    time_stampes = jnp.linspace(0, pde_instance.total_evolving_time, 128)
    
    def pretrain_loss_mu_fn(params, t, data):
        mu_0_true = pde_instance.mu_0(data)
        mu_t_predict = net.apply(params, t, data, ["mu"])["mu"]
        return jnp.mean(jnp.sum((mu_t_predict - mu_0_true) ** 2, axis=-1))
    
    def pretrain_loss_u_fn(params, t, data):
        u_0_true = pde_instance.u_0(data)
        u_t_predict = net.apply(params, t, data, ["u"])["u"]
        return jnp.mean(jnp.sum((u_t_predict - u_0_true) ** 2, axis=-1))

    def pretrain_loss_nabla_phi_fn(params, t, data):
        nabla_phi_0_true = pde_instance.nabla_phi_0(data)
        nabla_phi_t_predict = net.apply(params, t, data, ["nabla_phi"])["nabla_phi"]
        return jnp.mean(jnp.sum((nabla_phi_t_predict - nabla_phi_0_true) ** 2, axis=-1))
    
    pretrain_loss_mu_fn = jax.vmap(pretrain_loss_mu_fn, in_axes=[None, 0, None])
    pretrain_loss_u_fn = jax.vmap(pretrain_loss_u_fn, in_axes=[None, 0, None])
    pretrain_loss_nabla_phi_fn = jax.vmap(pretrain_loss_nabla_phi_fn, in_axes=[None, 0, None])


    def loss_fn(params, t, data):
        return jnp.mean(pretrain_loss_mu_fn(params, t, data) + pretrain_loss_u_fn(params, t, data) + pretrain_loss_mu_fn(params, t, data))
    
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