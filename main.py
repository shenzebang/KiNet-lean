import wandb
import hydra
from omegaconf import OmegaConf
import jax.random as random
from core.trainer import JaxTrainer
from register import get_pde_instance, get_method
from utils.optimizer import get_optimizer


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    wandb.login()
    pde_instance_name = f"{cfg.pde_instance.domain_dim}D-{cfg.pde_instance.name}"
    run = wandb.init(
        # Set the project where this run will be logged
        project=f"{pde_instance_name}-{cfg.solver.name}-{cfg.pde_instance.total_evolving_time}",
        # Track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg),
        # hyperparameter tuning mode or normal mode.
        # name=cfg.mode
    )

    rng_problem, rng_method, rng_trainer = random.split(random.PRNGKey(cfg.seed), 3)

    # create problem instance
    pde_instance = get_pde_instance(cfg)(cfg=cfg, rng=rng_problem)

    # create method instance
    method = get_method(cfg)(pde_instance=pde_instance, cfg=cfg, rng=rng_method)

    # create model
    net, params = method.create_model_fn()

    # create optimizer
    optimizer = get_optimizer(cfg.train)

    # Construct the JaxTrainer
    trainer = JaxTrainer(cfg=cfg, method=method, rng=rng_trainer, forward_fn=net.apply,
                         params=params, optimizer=optimizer)

    # Fit the model
    trainer.fit()

    # Test the model

    wandb.finish()


if __name__ == '__main__':
    main()
