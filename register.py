from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from example_problems.flocking_example import Flocking
from example_problems.euler_poisson_with_drift import EulerPoissonWithDrift
from example_problems.hlandau_example import HomogeneousLandau
from example_problems.fokker_planck_example import FokkerPlanck
from methods.KiNet import KiNet
from methods.PINN import PINN
from omegaconf import DictConfig

def get_pde_instance(cfg: DictConfig):
    if cfg.pde_instance.name == "Kinetic-Fokker-Planck":
        return KineticFokkerPlanck
    elif cfg.pde_instance.name == "Fokker-Planck":
        return FokkerPlanck
    elif cfg.pde_instance.name == "Flocking":
        return Flocking
    elif cfg.pde_instance.name == "Euler-Poisson":
        return EulerPoissonWithDrift
    elif cfg.pde_instance.name == "Homogeneous-Landau":
        return HomogeneousLandau
    else:
        return NotImplementedError

def get_method(cfg: DictConfig):
    if cfg.solver.name == "KiNet":
        return KiNet
    elif cfg.solver.name == "PINN":
        return PINN
    else:
        raise NotImplementedError
