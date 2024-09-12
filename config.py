import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)

    ## task configuration
    parser.add_argument('--domain_min', type=float, default=-10)
    parser.add_argument('--domain_max', type=float, default=10)
    parser.add_argument('--domain_dim', type=int, default=1, choices=[1, 2, 3], help='only 1D/2D/3D problems are supported now')
    parser.add_argument('--batch_size_initial', type=int, default=1000)
    parser.add_argument('--batch_size_boundary', type=int, default=1000)
    parser.add_argument('--batch_size_ref', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_test', type=int, default=10000)
    parser.add_argument('--batch_size_test_ref', type=int, default=10000)



    parser.add_argument('--dim', type=int, default=2, choices=[2, 3], help='only 2/3D problems are supported now')
    parser.add_argument('--method', type=str, default='KiNet',
                        choices=['KiNet', 'PINN',])
    parser.add_argument('--PDE', type=str, default='2D-Navier-Stokes', choices=[    '1D-Burgers',
                                                                                    '2D-Navier-Stokes',
                                                                                    '2D-Fokker-Planck',
                                                                                    '2D-Kinetic-Fokker-Planck',
                                                                                    '3D-Euler-Poisson',
                                                                                    '3D-Flocking',
                                                                                    '3D-Euler-Poisson-Drift'])
    parser.add_argument('--boundary_condition', type=str, default='None', choices=['None',
                                                                                   'Periodic',
                                                                                   'Neumann'])
    parser.add_argument('--diffusion_coefficient', type=float, default=0.)
    parser.add_argument('--init_weight', type=float, default=1.)
    parser.add_argument('--total_mass_weight', type=float, default=1.)
    parser.add_argument('--mass_change_weight', type=float, default=1.)
    parser.add_argument('--total_evolving_time', type=float, default=2.)


    ## network configuration
    parser.add_argument('--embed_scale', type=float, default=1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dims', type=str, default="64-64-64")
    parser.add_argument('--s_hidden_dims', type=str, default="64-64-64-64-64")
    parser.add_argument('--t_hidden_dims', type=str, default="64-64-64")


    ## train
    parser.add_argument('--number_of_iterations', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--refer_batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--ODE_tolerance', type=float, default=1e-7)
    parser.add_argument('--grad_clip_norm', type=float, default=-1.)
    parser.add_argument('--momentum', type=float, default=-1,
                        help="momentum of the sgd optimizer, negative value means no momentum"
                        )

    ## test/plot
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--test_domain_size', type=float, default=10.)
    parser.add_argument('--test_frequency', type=int, default=100)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--plot_frequency', type=int, default=100)
    parser.add_argument('--plot_domain_size', type=float, default=2.)
    parser.add_argument('--log_frequency', type=int, default=100)

    ## save/load
    parser.add_argument('--load_model', action='store_true', help='load a pretrained model')
    parser.add_argument('--load_total_evolving_time', type=float, default=2.)
    parser.add_argument('--save_model', action='store_true', help='save the trained model')
    parser.add_argument('--save_frequency', type=int, default=1000)
    parser.add_argument('--plot_save_directory', type=str, default='./plot')
    parser.add_argument('--save_directory', type=str, default='./save')
    parser.add_argument('--model_save_directory', type=str, default='./save')

    ## backend
    parser.add_argument('--use_pmap_train', action='store_true', help='use pmap to accelerate training')
    parser.add_argument('--use_pmap_test', action='store_true', help='use pmap to accelerate testing')

    args = parser.parse_args()
    return args
