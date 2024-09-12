from functools import partial
import jax
import jax.numpy as np

# @partial(jax.jit, static_argnums=(1,))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob):
    """Moves the chains by one step using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    rng_key: jax.random.PRNGKey
      Key for the pseudo random number generator.
    logpdf: function
      Returns the log-probability of the model given a position.
    position: np.ndarray, shape (n_dims,)
      The starting position.
    log_prob: float
      The log probability at the starting position.

    Returns
    -------
    Tuple
        The next positions of the chains along with their log probability.
    """
    score_fn = jax.grad(logpdf)

    key1, key2 = jax.random.split(rng_key)
    step_size = 0.1
    move_proposal = jax.random.normal(key1, shape=position.shape) * step_size
    proposal = position + move_proposal * np.sqrt(2) - score_fn(position) * (step_size ** 2)
    proposal_log_prob = logpdf(proposal)

    log_uniform = np.log(jax.random.uniform(key2))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = np.where(do_accept, proposal, position)
    log_prob = np.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


# @partial(jax.jit, static_argnums=(1, 2))
def rw_metropolis_sampler(rng_key, n_samples, logpdf, initial_position):
    """Generate samples using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    rng_key: jax.random.PRNGKey
        Key for the pseudo random number generator.
    n_samples: int
        Number of samples to generate per chain.
    logpdf: function
      Returns the log-probability of the model given a position.
    inital_position: np.ndarray (n_dims, n_chains)
      The starting position.

    Returns
    -------
    (n_samples, n_dim)
    """

    def mh_update(i, state):
        key, position, log_prob = state
        _, key = jax.random.split(key)
        new_position, new_log_prob = rw_metropolis_kernel(key, logpdf, position, log_prob)
        return (key, new_position, new_log_prob)

    logp = logpdf(initial_position)
    rng_key, position, log_prob = jax.lax.fori_loop(0, n_samples, mh_update, (rng_key, initial_position, logp))
    return position

