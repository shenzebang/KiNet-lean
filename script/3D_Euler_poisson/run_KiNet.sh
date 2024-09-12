args=(--PDE 3D-Euler-Poisson
      --method KiNet
      --boundary_condition None
      --domain_dim 3
      --number_of_iterations 40000
      --learning_rate 8e-4
      --total_evolving_time 2
      --batch_size_initial 1000
      --batch_size_ref 5000
      --batch_size_test_ref 60000
      --ODE_tolerance 1e-4
)

CUDA_VISIBLE_DEVICES=0 python main.py "${args[@]}"
