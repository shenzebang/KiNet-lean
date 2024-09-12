from typing import Dict
import jax.numpy as jnp
import pandas as pd
import os, json
import uuid
import os

def save_to_csv(data_dict: Dict, save_file: str):
    data_arrays = jnp.stack([jnp.array(data_dict[key]) for key in data_dict], axis=1)
    result = pd.DataFrame(data_arrays, columns=list(data_dict.keys()))
    result.to_csv(save_file, index=False)


def save_config(save_directory, args):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with open(save_directory + '/config.json', 'w') as f:
        json.dump(vars(args), f)

def get_checkpoint_directory_from_cfg(cfg):
    pde_instance_name = f"{cfg.pde_instance.domain_dim}D-{cfg.pde_instance.name}"
    directory = f"{pde_instance_name}-{cfg.solver.name}-{cfg.pde_instance.total_evolving_time}"
    return f"{os.path.expanduser("~")}/checkpoint/{directory}"
