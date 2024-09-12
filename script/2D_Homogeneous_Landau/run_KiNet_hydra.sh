CUDA_VISIBLE_DEVICES=0 \
    python main.py \
        pde_instance=nd_hlandau \
        train.optimizer.weight_decay=0 \
        neural_network.hidden_dim=64 \
        neural_network.layers=3 \
        plot.frequency=999999 \
        train.optimizer.learning_rate.initial=1e-2 \
        train.batch_size=200
        # neural_network.time_embedding_dim=20 \
        # train.batch_size=2000