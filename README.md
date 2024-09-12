# KiNet: Neural Kinetic Dynamics
A Jax-based implementation. The proposed KiNet loss is minimized via the adjoint method.

## Important Classes:
1. `ProblemInstance`
    - `ground_truth`
3. `Method`
    - `value_and_grad_fn`
    - `test_fn`
    - `plot_fn`
    - `create_model_fn`
4. `Distribution`
    - `sample`
    - `score`
    - `logdensity`
## Examples
### Kinetic Fokker-Planck Equation
Let $x, v \in \mathbb{R}^{d}$ and use $z = [x, v]\in \mathbb{R}^{2d}$ to denote their concatenation.

$$\frac{\partial}{\partial t} \rho(t, z) + \nabla \cdot \left( \rho(t, z)
	\begin{bmatrix}
		v \\
		- \beta \nabla U(x) - 4 \beta v/ \Gamma - \Gamma \beta \nabla_v \log \rho(t, z)
	\end{bmatrix}
	\right) = 0, \quad  \rho(0, z) = \rho_0(z),
$$
- $d$: `domain_dim`
- $U:\mathbb{R}^d \rightarrow \mathbb{R}$: `target_potential` 
- $\Gamma \in \mathbb{R}$: `Gamma` 
- $\beta \in \mathbb{R}$: `beta`
- $\rho_0$: `distribution_0`

### Vlasov-Poisson Equation
Let $x, v \in \mathbb{R}^{d}$ and use $z = [x, v]\in \mathbb{R}^{2d}$ to denote their concatenation.

$$
\frac{\partial }{\partial t} \rho(t, z) + \mathrm{div} \left(\rho(t, z)
	\begin{bmatrix}
		v \\
		- \nabla \phi(t, x)
	\end{bmatrix}
	\right) = 0,\ \text{with}\ \Delta \phi(t, x) = - \mu(t, x),\ \text{and}\ \mu(t, x) = \int \rho(t, x, v) d v
$$

### Euler-Poisson Equation
The Euler-Poisson equation can be recovered from the Vlasov-Poisson equation with the mono-kinetic intialization ansatz $\rho_0(x, v) = \mu_0(x) \delta_{v = u(x)}$.

$$
\frac{\partial }{\partial t} \mu(t, x) + \mathrm{div} \left(\mu(t, x) u(t, x) \right) = 0,\ \frac{\partial }{\partial t} u(t, x) = -\nabla \phi(t, x) -  (u(t, x)\cdot \nabla) u(t, x).
$$
