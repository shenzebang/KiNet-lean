CUDA_VISIBLE_DEVICES=0 \
python main.py \
    pde_instance=fokker_planck \
    pde_instance.domain_dim=3 \
    pde_instance.potential=OU\
    train.optimizer.weight_decay=0 \
    solver.train.batch_size_ref=0 \
    neural_network.hidden_dim=256 \
    neural_network.layers=2 \
    train.batch_size=1024 \
    backend.use_pmap_train=False \
    train.optimizer.learning_rate.initial=1e-3 \
    neural_network.time_embedding_dim=20 \
    train.optimizer.method=ADAM \
    train.optimizer.grad_clipping.threshold=999999 \
    train.optimizer.learning_rate.scheduling=warmup-cosine \
    neural_network.activation=tanh \
    plot.frequency=999999 \
    train.pretrain=True \
    pde_instance.total_evolving_time=1 \
    train.number_of_time_shard=1 \
    train.number_of_iterations=20000 \
    test.batch_size=10000\
    train.pretrain=False
    # train.optimizer.grad_clipping.type=global
