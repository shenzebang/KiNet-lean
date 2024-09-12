CUDA_VISIBLE_DEVICES=0,1, \
python main.py \
    neural_network=RealNVP \
    pde_instance=nd_fokker_planck \
    pde_instance.domain_dim=4 \
    solver=PINN \
    pde_instance.domain_min=-4 \
    pde_instance.domain_max=4 \
    train.number_of_iterations=800000 \
    train.optimizer.grad_clipping.type=non-adaptive \
    solver.train.batch_size=10000  \
    train.optimizer.weight_decay=0 \
    train.optimizer.learning_rate.initial=1e-3\
    backend.use_pmap_train=True \
    backend.use_pmap_test=True \
    test.batch_size=25000