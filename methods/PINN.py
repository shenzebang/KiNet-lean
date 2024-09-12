import methods.PINN_instances.kinetic_fokker_planck as kinetic_fokker_planck
import methods.PINN_instances.euler_poisson as euler_poisson
from api import Method, ProblemInstance
from functools import partial
import jax.random as random
import jax.numpy as jnp
INSTANCES = {
    '2D-Kinetic-Fokker-Planck': kinetic_fokker_planck,
    '3D-Euler-Poisson'        : euler_poisson
}


class PINN(Method):
    def create_model_fn(self):
        if "Kinetic-Fokker-Planck" in self.pde_instance.instance_name:
            return kinetic_fokker_planck.create_model_fn(self.pde_instance)
        elif "Euler-Poisson" in self.pde_instance.instance_name:
            return euler_poisson.create_model_fn(self.pde_instance)
        else:
            raise NotImplementedError


    def test_fn(self, forward_fn, params, rng):

        forward_fn = partial(forward_fn, params)
        config_test = {}
        if "Kinetic-Fokker-Planck" in self.pde_instance.instance_name:
            return kinetic_fokker_planck.test_fn(forward_fn=forward_fn, config=config_test, pde_instance=self.pde_instance,
                                                rng=rng)
        elif "Euler-Poisson" in self.pde_instance.instance_name:
            return euler_poisson.test_fn(forward_fn=forward_fn, config=config_test, pde_instance=self.pde_instance,
                                                rng=rng)
        else:
            raise NotImplementedError

    def plot_fn(self, forward_fn, params, rng):
        pass
        # forward_fn = partial(forward_fn, params)

        # config_plot = {}

        # return INSTANCES[self.args.PDE].plot_fn(forward_fn=forward_fn, config=config_plot, pde_instance=self.pde_instance,
        #                                         rng=rng)

    def value_and_grad_fn(self, forward_fn, params, rng):
        rng_sample, rng_vg = random.split(rng, 2)
        # Sample data
        data = self.sample_data(rng_sample)
        # compute function value and gradient
        weights = {
            "weight_initial": 1,
            "weight_train": 1,
            "mass_change": 1
        }
        config_train = {
            "ODE_tolerance" : self.cfg.ODE_tolerance,
            "weights": weights,
        }
        if "Kinetic-Fokker-Planck" in self.pde_instance.instance_name:
            return kinetic_fokker_planck.value_and_grad_fn(forward_fn=forward_fn, params=params, data=data, rng=rng_vg,
                                                          config=config_train, pde_instance=self.pde_instance)
        elif "Euler-Poisson" in self.pde_instance.instance_name:
            return euler_poisson.value_and_grad_fn(forward_fn=forward_fn, params=params, data=data, rng=rng_vg,
                                                          config=config_train, pde_instance=self.pde_instance)
        else:
            raise NotImplementedError

    def sample_data(self, rng):
        rng_train_t, rng_train_domain, rng_initial = random.split(rng, 3)
        data_0 = self.pde_instance.distribution_domain.sample(self.cfg.solver.train.batch_size_initial, rng_initial)
        # density_0 = jnp.exp(self.pde_instance.distribution_0.logdensity(data_0))

        if self.cfg.neural_network.name == "RealNVP":
            time_train = self.pde_instance.distribution_t.sample(self.cfg.solver.train.batch_size, rng_train_t)
        else:
            time_train = self.pde_instance.distribution_t.sample(10, rng_train_t)
            
        data_train = self.pde_instance.distribution_domain.sample(self.cfg.solver.train.batch_size, rng_train_domain)
        data = {
            "data_initial": data_0,
            "data_train": (time_train, data_train),
        }
        return data