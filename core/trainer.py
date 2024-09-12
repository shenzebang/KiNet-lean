import optax
import wandb
import jax
import jax.numpy as jnp
from utils.common_utils import compute_pytree_norm, normalize_grad, compute_pytree_difference
from api import Method
import jax.random as random
from optax import GradientTransformation
import copy
from flax.training import orbax_utils
import orbax.checkpoint
from utils.logging_utils import get_checkpoint_directory_from_cfg

class JaxTrainer:
    def __init__(self,
                 cfg,
                 method: Method,
                 rng: jnp.ndarray,
                 optimizer: GradientTransformation,
                 forward_fn,
                 params: optax.Params,
                 ):
        self.cfg = cfg
        self.forward_fn = forward_fn
        self.params_initial = params
        self.optimizer = optimizer
        self.method = method
        self.rng = rng
        self.params = {"current": copy.deepcopy(params), "previous": []}
        self.time_per_shard = self.cfg.pde_instance.total_evolving_time / self.cfg.train.number_of_time_shard
        self.time_interval = {"current": jnp.array([0, self.time_per_shard]), "previous": []}
        self.checkpoint_directory = get_checkpoint_directory_from_cfg(self.cfg)

    def fit(self, ):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(self.checkpoint_directory, orbax_checkpointer, options)



        # jit or pmap the gradient computation for efficiency
        def _value_and_grad_fn(params, time_interval, rng):
            return self.method.value_and_grad_fn(self.forward_fn, params, time_interval, rng)

        if self.cfg.backend.use_pmap_train and jax.local_device_count() > 1:
            _value_and_grad_fn = jax.pmap(_value_and_grad_fn, in_axes=(None, None, 0))

            def value_and_grad_fn(params, time_interval, rng):

                rngs = random.split(rng, jax.local_device_count())
                # compute in parallel
                v_g_etc = _value_and_grad_fn(params, time_interval, rngs)
                v_g_etc = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), v_g_etc)
                return v_g_etc
        else:
            value_and_grad_fn = jax.jit(_value_and_grad_fn)
            # value_and_grad_fn = _value_and_grad_fn

        @jax.jit
        def step_fn(params, opt_state, grad, scale=1):
            updates, opt_state = self.optimizer.update(grad, opt_state, params)   
            updates = jax.tree_util.tree_map(lambda g: scale * g, updates)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        # @jax.jit
        if self.cfg.backend.use_pmap_test and jax.local_device_count() > 1:
            def _test(params, time_interval, rng):
                return self.method.test_fn(self.forward_fn, params, time_interval, rng)
            
            _test_fn = jax.pmap(_test, in_axes=(None, None, 0))

            def test_fn(params, time_interval, rng):
                rngs = random.split(rng, jax.local_device_count())
                test_results = _test_fn(params, time_interval, rngs)
                test_results = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), test_results)
                return test_results
        else:
            def test_fn(params, time_interval, rng):
                return self.method.test_fn(self.forward_fn, params, time_interval, rng)
            
        test_fn = jax.jit(test_fn)

        # @jax.jit
        def plot_fn(params, time_interval, rng):
            return self.method.plot_fn(self.forward_fn, params, time_interval, rng)
        
        @jax.jit
        def metric_fn(params, time_interval, rng):
            return self.method.metric_fn(self.forward_fn, params, time_interval, rng)
        
        def test_metric_and_log_fn(params, time_interval, rng, shard_id):
            metrics = metric_fn(params, time_interval, rng)
            log_dict_metric = {"metric/step": (shard_id+1) * self.time_per_shard}
            for key in metrics:
                log_dict_metric[f"metric/{key}"] = metrics[key]
            wandb.log(log_dict_metric)

        minimum_loss_collection = []
        rng_metri_0, rng_0 = jax.random.split(self.rng)
        rngs_shard = jax.random.split(rng_0, self.cfg.train.number_of_time_shard)
        wandb.define_metric("metric/step")
        wandb.define_metric("metric/*", step_metric="metric/step")
        if self.cfg.pde_instance.test_metric:
            test_metric_and_log_fn(self.params, self.time_interval, rng_metri_0, shard_id=-1)
            
        opt_state = self.optimizer.init(self.params["current"])
        for shard_id, rng_shard in enumerate(rngs_shard):
            if self.cfg.train.reduce_step_after_first_shard:
                scale = 0.1 if shard_id > 0 else 1
            else:
                scale = 1
            minimum_loss = jnp.inf
            best_model_shard_id = self.params["current"]
            model_initial = self.params["current"]
            # self.time_interval["current"] = jnp.array([shard_id * self.time_per_shard, (shard_id+1) * self.time_per_shard])
            self.time_interval["current"] = jnp.array([0, self.time_per_shard])
            # initialize the opt_state
            if self.cfg.train.optimizer.reinitialize_per_shard:
                opt_state = self.optimizer.init(self.params["current"])
            rng_shard, rng_plot, rng_metric = jax.random.split(rng_shard, 3)
            rngs = jax.random.split(rng_shard, self.cfg.train.number_of_iterations)

            # logging related
            wandb.define_metric(f"shard {shard_id}/step")
            wandb.define_metric(f"shard {shard_id}/*", step_metric=f"shard {shard_id}/step")
            for epoch in range(self.cfg.train.number_of_iterations):
                # print(epoch)
                rng = rngs[epoch]
                rng_train, rng_test = random.split(rng, 2)

                v_g_etc = value_and_grad_fn(self.params, self.time_interval, rng_train)
                grad = normalize_grad(v_g_etc["grad"], v_g_etc["grad norm"]) if self.cfg.train.normalize_grad else v_g_etc["grad"]
                self.params["current"], opt_state = step_fn(self.params["current"], opt_state, grad, scale)

                # update the best model based on loss
                if v_g_etc["loss"] < minimum_loss:
                    best_model_shard_id = self.params["current"]
                    minimum_loss = v_g_etc["loss"]
                
                # log stats
                v_g_etc.pop("grad")
                v_g_etc["params_norm"] = compute_pytree_norm(self.params["current"])
                v_g_etc["distance to initial"] = compute_pytree_difference(model_initial, self.params["current"])

                log_dict_epoch ={
                    f"shard {shard_id}/step": epoch,
                }
                for key in v_g_etc:
                    log_dict_epoch[f"shard {shard_id}/{key}"] = v_g_etc[key]
                
                # perform test
                if self.cfg.pde_instance.perform_test and (epoch % self.cfg.test.frequency == 0 or epoch >= self.cfg.train.number_of_iterations - 3):
                    result_epoch = test_fn(self.params, self.time_interval, rng_test)
                    for key in result_epoch:
                        log_dict_epoch[f"shard {shard_id}/{key}"] = result_epoch[key]
                    if self.cfg.test.verbose:
                        msg = f"In epoch {epoch + 1: 5d}, "
                        for key in v_g_etc:
                            msg = msg + f"{key} is {v_g_etc[key]: .3e}, "
                        for key in result_epoch:
                            msg = msg + f"{key} is {result_epoch[key]: .3e}, "
                        print(msg)
                wandb.log(log_dict_epoch)

            # record the testing error of the saved model (the best model for the shard)            
            if self.cfg.pde_instance.perform_test:
                self.params["current"] = best_model_shard_id
                rng_test, _ = jax.random.split(rng_test)
                result_shard = test_fn(self.params, self.time_interval, rng_test)
                log_dict_best_model_test = {"best model/step": (shard_id+1) * self.time_per_shard}
                for key in result_shard:
                    log_dict_best_model_test[f"best model/{key}"] = result_shard[key]
                wandb.log(log_dict_best_model_test)

            minimum_loss_collection.append(minimum_loss)
            # store the params for a specific time shard
            self.params["previous"].append(copy.deepcopy(best_model_shard_id))
            self.time_interval["previous"].append(copy.deepcopy(self.time_interval["current"]))
            self.params["current"] = copy.deepcopy(best_model_shard_id)

            # save checkpoint
            ckpt = {"model": self.params, "time_interval": self.time_interval, "cfg": self.cfg}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(shard_id, ckpt, save_kwargs={'save_args': save_args})
            checkpoint_manager.wait_until_finished()
            
            plot_fn(self.params, self.time_interval, rng_plot) 
            # evaluate metric, e.g. trend to equilibrium, flocking, Landau damping
            if self.cfg.pde_instance.test_metric:
                test_metric_and_log_fn(self.params, self.time_interval, rng_metric, shard_id)
            


