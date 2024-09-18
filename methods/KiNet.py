import methods.KiNet_instances.kinetic_fokker_planck as kinetic_fokker_planck
import methods.KiNet_instances.euler_poisson as euler_poisson
import methods.KiNet_instances.flocking as flocking
import methods.KiNet_instances.hlandau as hlandau
import methods.KiNet_instances.fokker_planck as fokker_planck
from api import Method
from functools import partial
import jax.random as random
from utils.common_utils import evolve_data_score_logprob
import jax.numpy as jnp
import jax
INSTANCES = {
    'Fokker-Planck'          : fokker_planck,
    'Kinetic-Fokker-Planck'  : kinetic_fokker_planck,
    'Euler-Poisson'          : euler_poisson,
    'Homogeneous-Landau'     : hlandau,
    'Flocking'               : flocking,
}


class KiNet(Method):
    def create_model_fn(self):
        return INSTANCES[self.pde_instance.instance_name].create_model_fn(self.pde_instance)

    def test_fn(self, forward_fn, params, time_interval, rng):
        data = None
        if self.cfg.pde_instance.test.evolve_data:
            rng, rng_data = random.split(rng, 2)
            data = self.sample_data(rng_data, forward_fn, params["previous"], time_interval["previous"],
                                    batch_size=self.cfg.test.batch_size,
                                    batch_size_ref=0,
                                    evolve_score=self.cfg.pde_instance.test.evolve_score,
                                    evolve_logprob=self.cfg.pde_instance.test.evolve_logprob)

        forward_fn = partial(forward_fn, params["current"])
        return INSTANCES[self.pde_instance.instance_name].test_fn(forward_fn=forward_fn, 
                                                                  data=data, time_interval=time_interval, 
                                                                  pde_instance=self.pde_instance, rng=rng
                                                                  )

    def value_and_grad_fn(self, forward_fn, params, time_interval, rng):
        rng_sample, rng_vg = random.split(rng, 2)
        # Sample data
        data = self.sample_data(rng_sample, forward_fn, params["previous"], time_interval["previous"],
                                batch_size=self.cfg.train.batch_size,
                                batch_size_ref=self.cfg.solver.train.batch_size_ref,
                                evolve_score=self.cfg.pde_instance.train.evolve_score,
                                evolve_logprob=self.cfg.pde_instance.train.evolve_logprob)
        # compute function value and gradient
        config_train = {
            "ODE_tolerance" : self.cfg.ODE_tolerance,
        }
        return INSTANCES[self.pde_instance.instance_name].value_and_grad_fn(forward_fn=forward_fn, params=params["current"], 
                                                                            data=data, time_interval=time_interval, rng=rng_vg, 
                                                                            config=config_train, pde_instance=self.pde_instance
                                                                            )
        # TODO: a better implementation for the time_offset!

    def sample_data(self, rng, forward_fn, params_previous, time_interval_previous, batch_size: int, batch_size_ref: int,
                    evolve_score: bool=False, evolve_logprob: bool=False):
        rng_initial, rng_ref = random.split(rng, 2)
        data = {
            "data_initial"  : self.pde_instance.distribution_0.sample(batch_size, rng_initial),
            "data_ref"      : self.pde_instance.distribution_0.sample(batch_size_ref, rng_ref),
        }

        get_score = lambda x: self.pde_instance.score_0(x)
        data["score_initial"] = get_score(data["data_initial"]) if evolve_score else None
        data["score_ref"] = get_score(data["data_ref"]) if evolve_score else None

        get_logprob = lambda x: self.pde_instance.logprob_0(x)
        data["logprob_initial"] = get_logprob(data["data_initial"]) if evolve_logprob else None
        data["logprob_ref"] = get_logprob(data["data_ref"]) if evolve_logprob else None
        
        get_weight = lambda x: self.pde_instance.density_0(x) / self.pde_instance.distribution_0.density(x)
        data["weight_initial"] = get_weight(data["data_initial"]) 
        data["weight_ref"] = get_weight(data["data_ref"])

        # @jax.jit
        def preprocess_data_and_score(data, score, logprob):
            # preprocess the data and score based on the params in params_collection
            time_offset = jnp.zeros([])
            for params, time_interval in zip(params_previous, time_interval_previous):
                dynamics_fn = self.pde_instance.forward_fn_to_dynamics(partial(forward_fn, params), time_offset)
                data, score, logprob = evolve_data_score_logprob(dynamics_fn, time_interval, data, score, logprob)
                time_offset = time_offset + time_interval[-1]
            return data, score, logprob
        
        data["data_initial"], data["score_initial"], data["logprob_initial"] = \
            preprocess_data_and_score(data["data_initial"], data["score_initial"], data["logprob_initial"])
        data["data_ref"],     data["score_ref"],     data["logprob_ref"]     = \
            preprocess_data_and_score(data["data_ref"],     data["score_ref"],     data["logprob_ref"])


        return data
    
    def metric_fn(self, forward_fn, params, time_interval, rng):
        # decide the metric based on figuration
        if self.cfg.pde_instance.metric == "trend-to-equilibrium":
            data = self.sample_data(rng, forward_fn, params["previous"], time_interval["previous"],
                            batch_size=self.cfg.test.batch_size,
                            batch_size_ref=0,
                            evolve_score=True,
                            evolve_logprob=True,)
            return INSTANCES[self.pde_instance.instance_name].distance_to_equilibrium(data, self.pde_instance, None)
        elif self.cfg.pde_instance.metric == "functional-decay":
            data = self.sample_data(rng, forward_fn, params["previous"], time_interval["previous"],
                            batch_size=self.cfg.test.batch_size,
                            batch_size_ref=0,
                            evolve_score=False,
                            evolve_logprob=False,)
            return INSTANCES[self.pde_instance.instance_name].functional_decay_fn(data, self.pde_instance, None)
        else:
            raise ValueError("unknown metric!")