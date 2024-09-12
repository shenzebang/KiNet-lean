CUDA_VISIBLE_DEVICES=0,1,2 \
python main.py \
    pde_instance=nd_fokker_planck_OU \
    pde_instance.total_evolving_time=10 \
    pde_instance.domain_dim=10 \
    train.optimizer.weight_decay=0 \
    solver.train.batch_size_ref=0 \
    neural_network.hidden_dim=64 \
    neural_network.layers=2 \
    train.batch_size=256 \
    backend.use_pmap_train=True \
    train.optimizer.learning_rate.initial=1e-3 \
    neural_network.time_embedding_dim=20 \
    train.optimizer.method=ADAM \
    train.optimizer.grad_clipping.threshold=999999 \
    train.optimizer.learning_rate.scheduling=warmup-cosine \
    neural_network.activation=gelu
    # train.optimizer.grad_clipping.type=global
