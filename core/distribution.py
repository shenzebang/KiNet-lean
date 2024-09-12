from abc import ABC

import jax.numpy as jnp
import jax.random
import jax.random as random
from utils.common_utils import v_matmul
from typing import List
import warnings
from core.rw_sampler import rw_metropolis_sampler


class Distribution(ABC):
    def sample(self, batch_size: int, key):
        raise NotImplementedError

    def score(self, x: jnp.ndarray):
        raise NotImplementedError

    def logdensity(self, x: jnp.ndarray):
        raise NotImplementedError
    
    def density(self, x: jnp.ndarray):
        raise NotImplementedError

class DistributionKineticDeterministic(Distribution):
    def __init__(self, distribution_x: Distribution, u):
        self.distribution_x = distribution_x
        self.u = u
    
    def sample(self, batch_size: int, key):
        x = self.distribution_x.sample(batch_size, key)
        v = self.u(x)
        z = jnp.concatenate([x, v], axis=-1)
        return z
    
    def density(self, z: jnp.ndarray):
        x, _ = jnp.split(z, indices_or_sections=2, axis=-1)
        return self.distribution_x.density(x)

class DistributionKinetic(Distribution):
    def __init__(self, distribution_x: Distribution, distribution_v: Distribution):
        warnings.warn("Currently DistributionKinetic only supports when x and v are independent")
        self.distribution_x = distribution_x
        self.distribution_v = distribution_v

    def sample(self, batch_size: int, key):
        key_x, key_v = jax.random.split(key, 2)
        x = self.distribution_x.sample(batch_size, key_x)
        v = self.distribution_v.sample(batch_size, key_v)
        z = jnp.concatenate([x, v], axis=-1)
        return z

    def score(self, z: jnp.ndarray):
        x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        score_x = self.distribution_x.score(x)
        score_v = self.distribution_v.score(v)
        score_z = jnp.concatenate([score_x, score_v], axis=-1)
        return score_z

    def logdensity(self, z: jnp.ndarray):
        x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        logdensity_x = self.distribution_x.logdensity(x)
        logdensity_v = self.distribution_v.logdensity(v)
        logdensity_z = logdensity_x + logdensity_v
        return logdensity_z
    
    def density(self, z: jnp.ndarray):
        return jnp.exp(self.logdensity(z))

class Gaussian(Distribution):
    def __init__(self, mu: jnp.ndarray, cov: jnp.ndarray):
        assert mu.ndim == 1 and cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] == mu.shape[0]
        warnings.warn("cov is assumed to be positive definite!")
        self.dim = mu.shape[0]
        self.mu = mu
        self.cov = cov
        U, S, _ = jnp.linalg.svd(cov)
        self.inv_cov = jnp.linalg.inv(self.cov)
        self.det = jnp.linalg.det(self.cov * 2 * jnp.pi)
        self.log_det = jnp.log(self.det)
        self.cov_half = U @ jnp.diag(jnp.sqrt(S)) @ jnp.transpose(U)

    def sample(self, batch_size: int, key):
        return v_matmul(self.cov_half, random.normal(key, (batch_size, self.dim))) + self.mu

    def score(self, x: jnp.ndarray):
        if x.ndim == 1:
            return jnp.matmul(self.inv_cov, self.mu - x)
        else:
            return v_matmul(self.inv_cov, self.mu - x)

    def logdensity(self, x: jnp.ndarray):
        if x.ndim == 1:  # no batching
            quad = jnp.dot(x - self.mu, self.inv_cov @ (x - self.mu))
        elif x.ndim == 2:  # the first dimension is batch
            offset = x - self.mu  # broadcasting
            quad = jnp.sum(offset * v_matmul(self.inv_cov, offset), axis=(-1,))
        else:
            raise NotImplementedError
        return - .5 * (self.log_det + quad)

    def density(self, x: jnp.ndarray):
        return jnp.exp(self.logdensity(x))

class Uniform_over_3d_Ball(Distribution):
    def __init__(self, r):
        self.r = r
        self.volume = 4. / 3. * jnp.pi * r ** 3

    def sample(self, batch_size: int, key):
        return jax.random.ball(key, d=3, p=2, shape=[batch_size]) * self.r

    def score(self, x: jnp.ndarray):
        return jnp.zeros_like(x)
    
    def density(self, x: jnp.ndarray):
        warnings.warn("This implementation is not complete. Should check if x is within the ball!")
        if x.ndim == 1:
            return jnp.ones([]) / self.volume
        elif x.ndim == 2:
            return jnp.ones(x.shape[0]) / self.volume
        else:
            raise ValueError("The input should be either 1D (unbatched) or 2D (batched).")


class GaussianMixtureModel(Distribution):
    def __init__(self, mus: List[jnp.ndarray], covs: List[jnp.ndarray], weights: jnp.ndarray | None = None):
        self.n_Gaussians = len(mus)
        assert self.n_Gaussians == len(covs)
        warnings.warn("covs is a deprecated input. It is always taken as identity!")

        # check weights
        if weights == None:
            weights = jnp.ones([self.n_Gaussians])/self.n_Gaussians
        
        
        if weights.ndim != 1:
            raise ValueError("weights should be a 1D array")
        
        if weights.shape[0] != self.n_Gaussians:
            raise ValueError("The number of weights does not match the number of Gaussians!")
        
        if jnp.any(weights <= 0):
            raise ValueError("The weights should be positive!")
        
        self.log_weights = jnp.log(weights)
        
        # check shape of mus and covs
        assert mus[0].ndim == 1
        self.mus = jnp.stack(mus, axis=0)
        self.dim = self.mus.shape[-1]

        assert covs[0].ndim == 2
        self.covs = jnp.stack(covs, axis=0)
        assert self.covs.shape[-1] == self.dim and self.covs.shape[-2] == self.dim
        
        def get_half_cov(cov):
            U, S, _ = jnp.linalg.svd(cov)
            return U @ jnp.diag(jnp.sqrt(S)) @ jnp.transpose(U)


        self.inv_covs = jnp.stack([jnp.linalg.inv(cov) for cov in covs], axis=0)
        self.half_covs = jnp.stack([get_half_cov(cov) for cov in covs], axis=0) 
        self.dets_2pi = jnp.stack([jnp.linalg.det(cov * 2 * jnp.pi) for cov in covs], axis=0)
        self.coefficients = -.5 * jnp.log(self.dets_2pi) + self.log_weights

    def sample(self, batch_size: int, key):
        key_cat, key_gaussian = random.split(key, 2)
        sample_index = random.categorical(key_cat, self.log_weights, shape=[batch_size])
        _, n_sample_per_center = jnp.unique(sample_index, return_counts=True)
        keys = jax.random.split(key_gaussian, self.n_Gaussians)
        # return jnp.concatenate(sample_gaussians(n_sample_per_center, keys, self.mus, self.half_covs), axis=0)
        samples = []
        # keys = jax.random.split(key_gaussian, self.n_Gaussians)
        for i, (n_sample_i, _key) in enumerate(zip(n_sample_per_center.tolist(), keys)):
            # mu, sigma = self.mus[i, :], self.sigmas[i]
            # samples.append(v_matmul(sigma, random.normal(_key, (n_sample_i, self.dim))) + mu)
            samples.append(sample_gaussians(n_sample_i, _key, self.mus[i], self.half_covs[i]))

        return jnp.concatenate(samples, axis=0)

    def logdensity(self, xs: jnp.ndarray):
        return v_logdensity_gmm(xs, self.mus, self.inv_covs, self.coefficients)

    def score(self, xs: jnp.ndarray):
        return v_score_gmm(xs, self.mus, self.inv_covs, self.coefficients)


def sample_gaussians(batch_size, key, mu, half_cov):
    return v_matmul(half_cov, random.normal(key, (batch_size, mu.shape[0])) + mu)
# sample_gaussians = jax.vmap(sample_gaussians, in_axes=[0, 0, 0, 0])     


class Uniform(Distribution):
    def __init__(self, mins: jnp.ndarray, maxs: jnp.ndarray):
        if not (mins.ndim == 1 and maxs.ndim == 1):
            raise ValueError("mins and maxs should be 1-d array")
        if mins.shape[0] != maxs.shape[0]:
            raise ValueError("mins and maxs should have the same length")
        self.dim = mins.shape[0]
        self.mins = mins
        self.maxs = maxs

    def sample(self, batch_size: int, key):
        return jax.random.uniform(key=key, shape=[batch_size, self.dim], minval=self.mins, maxval=self.maxs)

    def score(self, x: jnp.ndarray):
        pass

    def logdensity(self, x: jnp.ndarray):
        pass


class UniformMixture(Distribution):
    def __init__(self, uniforms: List[Uniform]):
        self.uniforms = uniforms
        self.n_uniforms = len(uniforms)

    def sample(self, batch_size: int, key):
        if batch_size % self.n_uniforms != 0:
            raise ValueError(f"batch_size should be a multiple of n_uniforms {self.n_uniforms}!")

        _n = batch_size // self.n_uniforms
        _samples = []
        _keys = jax.random.split(key, self.n_uniforms)
        for _key, uniform in zip(_keys, self.uniforms):
            _samples.append(uniform.sample(_n, _key))
        return jnp.concatenate(_samples)


def K(t):
    return 1. - jnp.exp(-t/8.)/2.

class BKW(Distribution):
    def __init__(self, t_0: jnp.ndarray = jnp.zeros([]), dim: int = 2, n_sample = 5000) -> None:
        super().__init__()
        self.t_0 = t_0
        self.dim = dim
        self.n_sample = n_sample
    
    def sample(self, batch_size: int, key):
        rng_keys = jax.random.split(key, batch_size)  # (nchains,)
        initial_position = jnp.zeros([batch_size, self.dim])
        run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 0), out_axes=0)

        return run_mcmc(rng_keys, self.n_sample, self.logdensity, initial_position)

    def logdensity(self, x: jnp.ndarray):
        x_norm_2 = jnp.sum(x ** 2, axis=-1)
        K_t = K(self.t_0)
        return jnp.log(1 / 2 / jnp.pi / K_t) - x_norm_2 / 2. / K_t + jnp.log(
            (2. * K_t - 1.) / K_t + (1. - K_t) / 2. / K_t ** 2 * x_norm_2)

    def density(self, x: jnp.ndarray):
        raise NotImplementedError
    def score(self, x: jnp.ndarray):

        def log_densities(x):
            return jnp.sum(self.logdensity(x))

        score_fn = jax.grad(log_densities)

        return score_fn(x)
    

def get_uniforms_over_box_boundary(mins: jnp.ndarray, maxs: jnp.ndarray):
    if not (mins.ndim == 1 and maxs.ndim == 1):
        raise ValueError("mins and maxs should be 1-d array")
    if mins.shape[0] != maxs.shape[0]:
        raise ValueError("mins and maxs should have the same length")
    dim = mins.shape[0]
    uniforms = []
    for i in range(dim):
        basis_i = [0.] * dim
        basis_i[i] = 1.
        basis_i = jnp.array(basis_i)
        min_i, max_i = mins[i], maxs[i]
        _mins_i_min = mins
        _maxs_i_min = maxs + (-max_i + min_i) * basis_i
        uniforms.append(Uniform(_mins_i_min, _maxs_i_min))
        _mins_i_max = mins + (-min_i + max_i) * basis_i
        _maxs_i_max = maxs
        uniforms.append(Uniform(_mins_i_max, _maxs_i_max))

    return uniforms


def _density_gaussian(x, mu, inv_cov, det):
    # computes the density in a single Gaussian of a single point
    a = x - mu
    dim = x.shape[0]
    if inv_cov.ndim == 1:
        return jnp.squeeze(jnp.exp(- .5 * jnp.dot(a, a) * inv_cov) / jnp.sqrt((2 * jnp.pi) ** dim * det))
    else:
        return jnp.exp(- .5 * jnp.dot(a, jnp.matmul(inv_cov, a))) / jnp.sqrt((2 * jnp.pi) ** dim * det)


v_density_gaussian = jax.vmap(_density_gaussian, in_axes=[None, 0, 0, 0])


# computes the density in several Gaussians of a single point


def get_quads(x, mu, inv_cov):
    x_mu = x - mu
    # return jnp.dot(x_mu, jnp.matmul(inv_cov, x_mu))
    # return x_mu.dot(inv_cov).dot(x_mu)
    return jnp.sum(x_mu ** 2, axis=-1)

get_quads = jax.vmap(get_quads, in_axes=[None, 0, 0])

def _logdensity_gmm(x, mus, inv_covs, coefficients):
    return jax.scipy.special.logsumexp(-.5 * get_quads(x, mus, inv_covs) + coefficients)
    # # computes log densities of gmm of multiple points
    # densities = v_density_gaussian(x, mus, inv_covs, dets)
    # # densities : (self.n_Gaussians)
    # return jnp.log(jnp.mean(densities, axis=0))


v_logdensity_gmm = jax.vmap(_logdensity_gmm, in_axes=[0, None, None, None])
# computes log densities of gmm of multiple points

_score_gmm = jax.grad(_logdensity_gmm)
# compute the gradient w.r.t. x

v_score_gmm = jax.vmap(_score_gmm, in_axes=[0, None, None, None])

