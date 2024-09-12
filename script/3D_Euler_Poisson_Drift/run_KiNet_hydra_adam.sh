CUDA_VISIBLE_DEVICES=2 \
    python main.py \
        train.optimizer.weight_decay=0 \
        neural_network.hidden_dim=128 \
        neural_network.layers=3 \
        plot.frequency=999999 \
        train.optimizer.learning_rate.initial=1e-3\
        train.optimizer.method=ADAM \
        train.optimizer.grad_clipping.threshold=999999 \
        train.optimizer.learning_rate.scheduling=warmup-cosine
        # neural_network.time_embedding_dim=20 
        # train.optimizer.learning_rate.scheduling=warmup-cosine
        # train.batch_size=2000