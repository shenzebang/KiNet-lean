CUDA_VISIBLE_DEVICES=0 \
    python main.py \
        pde_instance=nd_hlandau \
        train.optimizer.weight_decay=0 \
        neural_network.hidden_dim=64 \
        neural_network.layers=2 \
        plot.frequency=999999 \
        train.optimizer.learning_rate.initial=1e-3 \
        train.batch_size=200 \
        train.optimizer.method=ADAM \
        train.optimizer.grad_clipping.threshold=999999 \
        train.optimizer.learning_rate.scheduling=warmup-cosine \
        neural_network.time_embedding_dim=20 \
        neural_network.activation=gelu
        # train.batch_size=2000