from omegaconf import DictConfig
import optax

def get_optimizer(train_cfg: DictConfig):
    optimizer_cfg = train_cfg.optimizer
    if optimizer_cfg.learning_rate.scheduling == "None":
            lr_schedule = optimizer_cfg.learning_rate.initial
    elif optimizer_cfg.learning_rate.scheduling == "cosine":
        lr_schedule = optax.cosine_decay_schedule(optimizer_cfg.learning_rate.initial, train_cfg.number_of_iterations, 0.1)
    elif optimizer_cfg.learning_rate.scheduling == "warmup-cosine":
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=optimizer_cfg.learning_rate.initial,peak_value=optimizer_cfg.learning_rate.initial,
                                                         warmup_steps=train_cfg.number_of_iterations // 4, decay_steps=train_cfg.number_of_iterations, 
                                                         end_value=optimizer_cfg.learning_rate.initial * .1)
    else:
        raise NotImplementedError
    if optimizer_cfg.grad_clipping.type=="adaptive":
        clip = optax.adaptive_grad_clip
    elif optimizer_cfg.grad_clipping.type=="non-adaptive":
        clip = optax.clip
    elif optimizer_cfg.grad_clipping.type=="global":
        clip = optax.clip_by_global_norm
    else:
        raise ValueError("type of clipping should be either adaptive or non-adaptive!")
    if optimizer_cfg.method == "SGD":
        optimizer = optax.chain(clip(optimizer_cfg.grad_clipping.threshold),
                                optax.add_decayed_weights(optimizer_cfg.weight_decay),
                                optax.sgd(learning_rate=lr_schedule, momentum=optimizer_cfg.momentum)
                                )
    elif optimizer_cfg.method == "ADAM":
        optimizer = optax.chain(clip(optimizer_cfg.grad_clipping.threshold),
                                optax.add_decayed_weights(optimizer_cfg.weight_decay),
                                optax.adam(learning_rate=lr_schedule, b1=optimizer_cfg.momentum)
                                )
    elif optimizer_cfg.method == "ADAGRAD":
        optimizer = optax.chain(clip(optimizer_cfg.grad_clipping.threshold),
                                optax.add_decayed_weights(optimizer_cfg.weight_decay),
                                optax.adagrad(learning_rate=lr_schedule)
                                )
    else:
        raise NotImplementedError
    return optimizer