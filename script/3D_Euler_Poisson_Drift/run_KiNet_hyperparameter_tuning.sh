CUDA_VISIBLE_DEVICES=3 python main.py --multirun \
        train.optimizer.weight_decay=0.0005,0.00025 \
        neural_network.hidden_dim=128,256 \
        neural_network.layers=3 \
        plot.frequency=999999