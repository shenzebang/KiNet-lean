import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple
from core.normalizing_flow import TimeEmbedding, ActivationFactory, ActivationModule
from example_problems.euler_poisson_with_drift import ground_truth_op_vmapx, ground_truth_op_uniform
# For Debugging KFP
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck, Kinetic_OU_process
from example_problems.hlandau_example import HomogeneousLandau
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
class MLP(nn.Module):
    output_dim: int = 1
    hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    # hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    use_normalization: bool = False
    time_embedding_dim: int = 0

    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.normalizations = [nn.LayerNorm() for _ in list(self.hidden_dims) + [self.output_dim]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=20,)
        # self.act = jax.nn.sigmoid
        # self.act = jax.nn.relu
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        # z can be batched input, but t should be a single input
        assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
        assert x.ndim == 1 or x.ndim == 2

        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        if x.ndim == 2:
            t = jnp.broadcast_to(t, (x.shape[0], t.shape[0]))
        x = jnp.concatenate([t, x], axis=-1)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)

        return x

    # def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
    #     # if t.ndim != 2 or t.shape[1] != 1:
    #     #     raise ValueError("t should be a 2D array and the second dimension should be 1")
    #     #
    #     # if x.ndim != 2:
    #     #     raise ValueError("x should be a 2D array")
    #
    #     if x.ndim == 2 and x.shape[0] != t.shape[0]:
    #         raise ValueError("x.shape[0] should agree with t.shape[0]")
    #
    #     tx = jnp.concatenate([t, x], axis=-1)
    #     y = tx
    #     for i, layer in enumerate(self.layers):
    #         y = layer(y)
    #         if i < len(self.layers) - 1:
    #             y = self.act(y)
    #
    #     return jax.nn.relu(y)

class KiNet(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 2
    time_dim: int = 10
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    activation: str = "tanh"
    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [self.output_dim]]

        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        # self.act = jax.nn.sigmoid
        # self.act = jax.nn.tanh
        self.act = ActivationFactory.create(self.activation)
        # self.act = jax.nn.relu

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)


        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        # t = self.layer_time(t)
        if z.ndim == 2:
            t = jnp.broadcast_to(t, (z.shape[0], t.shape[0]))
        z = jnp.concatenate([t, z], axis=-1)

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = self.act(z)

        return z


class KiNet_Debug_2(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 2
    time_dim: int = 10
    # hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100,)
    def setup(self):
        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [1]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        self.layer_time = nn.Dense(self.time_dim)
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        # t = self.layer_time(t)
        if z.ndim == 2:
            t = jnp.broadcast_to(t, (z.shape[0], t.shape[0]))
        # z = jnp.concatenate([t, z], axis=-1)

        for i, layer in enumerate(self.layers):
            t = layer(t)
            if i < len(self.layers) - 1:
                t = self.act(t)

        return z * t



class ResBlock(nn.Module):
    hidden_dims: int = 64
    hidden_layers: int = 2
    act: str = "relu"

    def setup(self):
        self.layers = [nn.Dense(features=hidden_dim, kernel_init=nn.initializers.xavier_uniform())
                       for hidden_dim in (self.hidden_dims,) * self.hidden_layers]
        self.activation = ActivationFactory.create(self.act)

    def __call__(self, z: jnp.ndarray):
        z_in = z
        for layer in self.layers:
            z = self.activation(layer(z))

        return z + z_in

class ResBlockBottle(nn.Module):
    output_dim: int
    hidden_dims: int = 64
    hidden_layers: int = 2
    act: str = "relu"

    def setup(self):
        self.layers = [nn.Dense(features=hidden_dim, kernel_init=nn.initializers.xavier_uniform())
                       for hidden_dim in (self.hidden_dims,) * self.hidden_layers]
        self.activation = ActivationFactory.create(self.act)
        self.output_layer = nn.Dense(features=self.output_dim, kernel_init=nn.initializers.xavier_uniform())

    def __call__(self, z: jnp.ndarray):
        assert z.shape[-1] == self.output_dim
        z_in = z
        for layer in self.layers:
            z = self.activation(layer(z))
        z = self.activation(self.output_layer(z))

        return z + z_in


class KiNet_ResNet(nn.Module):
    output_dim: int
    time_embedding_dim: int = 0
    use_bottle: bool = False
    resnet_hidden_dims: int = 64
    resnet_layers: int = 2
    n_resblocks: int = 3
    act: str = "relu"

    def setup(self):
        if self.use_bottle:
            res_output_dim = self.output_dim + self.time_embedding_dim if self.time_embedding_dim > 0 else self.output_dim + 1
            self.resblocks = [
                ResBlockBottle(hidden_dims=self.resnet_hidden_dims, hidden_layers=self.resnet_layers, output_dim=res_output_dim, act=self.act)
                for _ in range(self.n_resblocks)]
        else:
            self.dense_input = nn.Dense(features=self.resnet_hidden_dims, kernel_init=nn.initializers.xavier_uniform())
            self.resblocks = [ResBlock(hidden_dims=self.resnet_hidden_dims, hidden_layers=self.resnet_layers, act=self.act)
                          for _ in range(self.n_resblocks)]
        self.dense_output = nn.Dense(features=self.output_dim, kernel_init=nn.initializers.xavier_uniform())
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )

        self.activation = ActivationFactory.create(self.act)

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        # t = self.layer_time(t)
        if z.ndim == 2:
            t = jnp.broadcast_to(t, (z.shape[0], t.shape[0]))
        z = jnp.concatenate([t, z], axis=-1)

        if not self.use_bottle:
            z = self.dense_input(z)

        for resblock in self.resblocks:
            z = resblock(z)

        z = self.dense_output(z)

        return z

class DEBUG_KFP(nn.Module):
    pde_instance: KineticFokkerPlanck

    def setup(self):
        self.layers = nn.Dense(1)

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        mu_t, Sigma_t = Kinetic_OU_process(
                            t,
                            self.pde_instance.initial_configuration,
                            self.pde_instance.beta,
                            self.pde_instance.Gamma
        )
        score = v_gaussian_score(x, Sigma_t, mu_t)
        score_x, score_v = jnp.split(score, indices_or_sections=2, axis=-1)
        return score_v
    
class DEBUG_EP(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 2
    time_dim: int = 10
    # hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100,)
    def setup(self):
        self.layers = [nn.Dense(dim_out) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        self.layer_time = nn.Dense(self.time_dim)
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        assert t.ndim == 0 or t.ndim == 1
        if t.ndim == 1:
            t = t[0]
        # x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        if x.ndim == 2:
            return ground_truth_op_vmapx(t, x)
        elif x.ndim == 1:
            return ground_truth_op_uniform(t, x)
        else:
            raise ValueError("xs should be of size either (batch, dim) or (dim).")
    
class DEBUG_HLANDAU(nn.Module):
    pde_instance: HomogeneousLandau

    def setup(self):
        self.layers = nn.Dense(1)

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        assert t.ndim == 0 or t.ndim == 1
        if t.ndim == 1:
            t = t[0]

        return self.pde_instance.velocity_t(t, x)

def get_model(cfg, DEBUG=False, pde_instance=None):
    if cfg.neural_network.n_resblocks > 0:
        # use resnet
        raise NotImplementedError
    else:
        # do not use resnet
        if not DEBUG:
            model = KiNet(output_dim=cfg.pde_instance.domain_dim,
                          time_embedding_dim=cfg.neural_network.time_embedding_dim,
                          hidden_dims=[cfg.neural_network.hidden_dim] * cfg.neural_network.layers,
                          activation=cfg.neural_network.activation
                          )
        else:
        # Use the ground truth for debugging
            if cfg.pde_instance.name == "Kinetic-Fokker-Planck":
                model = DEBUG_KFP(pde_instance)

            elif cfg.pde_instance.name == "Euler-Poisson":
                model = DEBUG_EP(output_dim=cfg.pde_instance.domain_dim,
                          time_embedding_dim=cfg.neural_network.time_embedding_dim,
                          hidden_dims=[cfg.neural_network.hidden_dim] * cfg.neural_network.layers)
            elif cfg.pde_instance.name == "Homogeneous-Landau":
                model = DEBUG_HLANDAU(pde_instance)

        return model

class KiNet_Landau(nn.Module):
    time_embedding_dim: int = 0
    output_dim: int = 1
    time_dim: int = 10
    # hidden_dims: Tuple[int] = (20, 20, 20, 20, 20, 20, 20, 20,)
    hidden_dims: Tuple[int] = (30, 30, 30, 30, 30, 30, 30, 30,)
    # hidden_dims: Tuple[int] = (50, 50, 50, 50, 50, 50, 50, 50,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100, 100, 100, 100, 100,)
    # hidden_dims: Tuple[int] = (100, 100, 100, 100,)
    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [self.output_dim]]

        if self.time_embedding_dim > 0:
            self.time_embedding = TimeEmbedding(dim=self.time_embedding_dim, )
        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh
        # self.act = jax.nn.relu

    def __call__(self, t: jnp.ndarray, z: jnp.ndarray):
        assert t.ndim == 0 or (t.ndim == 1 and len(t) == 1)
        z_in = z

        if t.ndim == 0:
            t = jnp.ones(1) * t

        if self.time_embedding_dim > 0:
            t = self.time_embedding(t)

        # t = self.layer_time(t)
        if z.ndim == 2:
            t = jnp.broadcast_to(t, (z.shape[0], t.shape[0]))
        z = jnp.concatenate([t, z], axis=-1)
        
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = self.act(z)
        z = jnp.squeeze(z)
        return z * z_in