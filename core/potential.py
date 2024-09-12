import jax.numpy as jnp
import jax
from utils.common_utils import v_matmul
import warnings
from typing import List
from core.distribution import GaussianMixtureModel

class Potential(object):
    def gradient(self, x: jnp.ndarray):
        raise NotImplementedError


class QuadraticPotential(Potential):
    def __init__(self, mu: jnp.ndarray, cov: jnp.ndarray):
        assert mu.ndim == 1 and cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] == mu.shape[0]
        warnings.warn("cov is assumed to be positive definite!")
        self.dim = mu.shape[0]
        self.mu = mu
        self.cov = cov
        self.inv_cov = jnp.linalg.inv(self.cov)

    def gradient(self, x):
        if x.ndim == 1:
            return self.inv_cov @ (x - self.mu)
        else:
            return v_matmul(self.inv_cov, x - self.mu)


class VoidPotential(Potential):
    def gradient(self, x: jnp.ndarray):
        return jnp.zeros_like(x)


def gmm_V(x, mus, cov):
    # we use the broadcasting mechanism
    a = - jnp.sum((x-mus)**2, axis=-1)/(2 * cov)
    return - jax.scipy.special.logsumexp(a)

g_gmm_V = jax.grad(gmm_V) # by default, grad computes gradient w.r.t. the first input, i.e. argnums = 0

# def g_gmm_V(x, mus, sigma):
#     x = x - mus
#     a = jnp.exp(-jnp.sum(x**2, axis=1)/(2 * sigma**2))
#     normalized_a = a / jnp.sum(a)
#     return jnp.sum(x * normalized_a[:, None], axis=0)/sigma**2


vg_gmm_V = jax.vmap(g_gmm_V, in_axes=[0, None, None]) # only apply autobatching to the first input

class GMMPotential(Potential):
    # def __init__(self, mus: jnp.ndarray, covs: jnp.ndarray, weights: jnp.ndarray):
    #     # we assume that the Gaussian component has the same cov for simplicity
    #     assert mus.ndim == 2
    #     self.n_Gaussian = mus.shape[0]
    #     self.dim = mus.shape[-1]
    #     self.mus = mus

    #     if covs.ndim == 0:
    #         self.cov = covs
    #     elif covs.ndim == 1 and len(covs) == 1:
    #         self.cov = covs[0]
    #     else:
    #         raise ValueError("sigma should be a scalar!")

    # def gradient(self, x):
    #     assert x.ndim == 1 or x.ndim == 2
    #     assert x.shape[-1] == self.dim

    #     if len(x.shape) == 1:
    #         return g_gmm_V(x, self.mus, self.cov)
    #     elif len(x.shape) == 2:
    #         return vg_gmm_V(x, self.mus, self.cov)
    #     else:
    #         raise ValueError("x should be either 1D (un-batched) or 2D (batched) array.")
    
    def __init__(self, mus: List[jnp.ndarray], covs: List[jnp.ndarray], weights: jnp.ndarray):
        warnings.warn("Only Identity covariance matrix is supported!")

        self.gmm = GaussianMixtureModel(mus, covs, weights)        

    def gradient(self, x):
        # (why we need the negative sign) \nabla log rho_\infty = - \nabla V, so \nabla V = - \nabla log rho_\infty
        return - self.gmm.score(x)
   
