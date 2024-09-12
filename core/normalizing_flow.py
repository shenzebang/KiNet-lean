import flax.linen as nn
from typing import Any
import numpy as np
import jax.numpy as jnp
import math
from abc import ABC, abstractmethod

class TimeEmbedding(nn.Module):
    dim: int
    mul: int = 1
    act: str = 'celu'

    @nn.compact
    def __call__(self, inputs):
        time_dim = self.dim * self.mul

        se = SinusoidalEmbedding(self.dim)(inputs)

        x = nn.Dense(time_dim)(se)
        x = ActivationFactory.create(self.act)(x)
        x = nn.Dense(time_dim)(x)

        return x

class SinusoidalEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, inputs):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # emb = math.log(100) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        # emb = jnp.expand_dims(inputs, -1) * emb
        emb = inputs * emb
        # emb = jnp.expand_dims(10 * inputs, -1) * emb
        assert(emb.ndim == 1)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb


class TDPFBase(nn.Module, ABC):
    '''
    Base class for a jax model of time-dependent push forward (TDPF) of
    the form (x, t) -> (y).
    '''
    dim: int

    @abstractmethod
    def __call__(self, t, x0, reverse):
        '''
        Args:
          t: ()
          x0: (D,)
          reverse: bool
        Returns:
          A tuple, (xt, ldj) where
            xt: (D,), pushed samples
            ldj:
              () or None, log|detJT|(x).
              Hence density at xt is log_p(x) - ldj.
        '''
        pass

class ActivationModule(nn.Module):
    fn: Any


    def __call__(self, x):
        return self.fn(x)


class ActivationFactory:
    @staticmethod
    def create(name):
        if name == 'relu':
            fn = nn.relu
        elif name == 'tanh':
            fn = nn.tanh
        elif name == 'celu':
            fn = nn.celu
        elif name == 'gelu':
            fn = nn.gelu
        elif name == 'elu':
            fn = nn.elu
        elif name == 'silu':
            fn = nn.silu
        elif name == 'softplus':
            fn = nn.softplus
        elif name == 'prelu':
            fn = nn.activation.PReLU()
        else:
            raise Exception(f'Unknown activation name: {name}')
        return ActivationModule(fn)


class BasicMLP(nn.Module):
    out_dim: int
    act: str

    @nn.compact
    def __call__(self, X):
        out = nn.Sequential([
            nn.Dense(32),
            ActivationFactory.create(self.act),
            nn.Dense(64),
            ActivationFactory.create(self.act),
            nn.Dense(64),
            ActivationFactory.create(self.act),
            nn.Dense(self.out_dim),
        ])(X)
        return out


class CouplingLayer(nn.Module):
    mask : np.ndarray # coordinates to keep identical
    soft_init: float
    ignore_time: bool
    act: str
    time_emb: Any

    def setup(self):
        dim = self.mask.shape[0]
        self.scaling_factor = self.param('scaling_factor',
                                         nn.initializers.zeros,
                                         (dim,))
        self.scale_net = BasicMLP(out_dim=dim,
                                  act=self.act)
        self.translate_net = BasicMLP(
            out_dim=dim, act=self.act)


    def __call__(self, t, x, reverse):
        assert(x.ndim == 1)

        if not self.ignore_time:
            if self.time_emb is not None:
                xt_cat = jnp.concatenate([x * self.mask, self.time_emb(t)])
            else:
                xt_cat = jnp.append(x * self.mask, t)
        else:
            xt_cat = x * self.mask
        # Modify scale/translate so that at t=0 the resulting map is identity.
        scale = self.scale_net(xt_cat)
        translate = self.translate_net(xt_cat)
        if not self.ignore_time and self.soft_init == 0.:
            scale = t * scale
            translate = t * translate

        sf = jnp.exp(self.scaling_factor)
        scale = nn.tanh(scale / sf) * sf

        scale = scale * (1 - self.mask)
        translate = translate * (1 - self.mask)

        if reverse:
            x = (x + translate) * jnp.exp(scale)
            ldj = scale.sum()
        else:
            x = (x * jnp.exp(-scale)) - translate
            ldj = -scale.sum()

        return x, ldj


class MNF(TDPFBase):
    '''
    Masked normalizing flow by Dinh et al.
    '''
    couple_mul: int
    mask_type: str # ['loop', 'random']
    soft_init: float # if 0, then use hard parameterization
    ignore_time: bool # whether to remove time dependency
    activation_layer: str
    embed_time_dim: int # 0 if not embedding_time

    def setup(self):
        if self.embed_time_dim > 0:
            self.time_emb = TimeEmbedding(self.embed_time_dim)
        else:
            self.time_emb = None
        couple_layers = []
        num_layer = (self.couple_mul if self.mask_type == 'random'
                     else self.dim * self.couple_mul)
        if self.mask_type == 'random':
            rng_state = np.random.RandomState(seed=888)
            prev_mask = np.zeros(self.dim, dtype=int)
        for i in range(num_layer):
            if self.mask_type == 'loop':
                # Change one coordinate at a time; okay in low dimensions.
                mask = np.ones(self.dim)
                mask[i % self.dim] = 0
            else:
                while True:
                    mask = rng_state.binomial(1, p=0.5, size=[self.dim])
                    if not (mask.sum() in [0, self.dim] or (mask == prev_mask).all()):
                        prev_mask = mask
                        break
            # print(f'mask {i} = {mask}')
            couple_layers.append(CouplingLayer(
                time_emb=self.time_emb,
                mask=mask,
                act=self.activation_layer,
                soft_init=self.soft_init,
                ignore_time=self.ignore_time))
        self.couple_layers = couple_layers


    def __call__(self, t, x0, reverse=False):
        ldj_sum = 0

        couple_layers = self.couple_layers
        if reverse:
            couple_layers = reversed(couple_layers)
        x = x0
        for layer in couple_layers:
            x, ldj = layer(t, x, reverse)
            ldj_sum += ldj
        return x, ldj_sum

class RealNVP(nn.Module):
    mnf: MNF
    log_prob_0: Any

    def __call__(self, t, x0):
        x, ldj_sum = self.mnf(t, x0, reverse=True)
        return jnp.exp(self.log_prob_0(x) + ldj_sum)

