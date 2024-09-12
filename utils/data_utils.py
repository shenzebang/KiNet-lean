from core.distribution import Distribution
from typing import Generator
import jax.random as random

import jax



def distribution_to_generator(  distribution_time: Distribution,
                                distribution_space: Distribution,
                                batch_size: int, key, label_fn=None, batch_size_time=None) -> Generator:
    key1 = key
    if batch_size_time is None:
        batch_size_time = batch_size

    def generate(key1, key2):
        if label_fn is None:
            return distribution_time.sample(batch_size_time, key2), distribution_space.sample(batch_size, key1)
        else:
            samples_t = distribution_time.sample(batch_size_time, key2)
            samples_s = distribution_space.sample(batch_size, key1)
            rng, _ = jax.random.split(key2, 2)
            return samples_t, samples_s, label_fn(samples_t, samples_s, rng)
    generate = jax.jit(generate)
    while True:
        key1, key2 = random.split(key1, 2)
        yield generate(key1, key2)