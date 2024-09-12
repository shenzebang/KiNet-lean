import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from matplotlib.animation import FuncAnimation, PillowWriter
import wandb
import tempfile
import warnings

def plot_velocity(total_time, XV_0T: jnp.ndarray):
    if XV_0T.shape[-1] == 6:
        plot_velocity_3d(total_time, XV_0T)
    elif XV_0T.shape[-1] == 4:
        plot_velocity_2d(total_time, XV_0T)
    else:
        msg = f"Plotting {XV_0T.shape[-1]/2}D problem is not supported! Only 2D and 3D problems are supported."
        warnings.warn(msg)


def plot_velocity_2d(total_time, XV_0T: jnp.ndarray):
    X_0T, V_0T = jnp.split(XV_0T, indices_or_sections=2, axis=-1)
    C = jnp.hypot(V_0T[0, :, 0], V_0T[0, :, 1])
    T, N, D = X_0T.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    quiver = ax.quiver(X_0T[0, :, 0], X_0T[0, :, 1], V_0T[0, :, 0], V_0T[0, :, 1], C, angles='xy', scale_units='xy',
                       scale=2)

    xy_min = jnp.min(X_0T, axis=[0, 1])
    xy_max = jnp.max(X_0T, axis=[0, 1])
    scaling = 1.2
    ax.set_xlim(xy_min[0] * scaling, xy_max[0] * scaling)
    ax.set_ylim(xy_min[1] * scaling, xy_max[1] * scaling)
    title = ax.set_title("Time: 0")
    time_unit = total_time / (T-1)
    def update(t):
        C = jnp.hypot(V_0T[t, :, 0], V_0T[t, :, 1])
        quiver.set_UVC(V_0T[t, :, 0], V_0T[t, :, 1], C=C)
        quiver.set_offsets(X_0T[t])
        title.set_text(f"Time: {time_unit * t:.2f}")

    ani = FuncAnimation(fig, update, frames=range(T), interval=100, repeat_delay=2000)
    # ani.save('./velocity_field.gif', writer='pillow')
    tf = tempfile.NamedTemporaryFile(dir="./", suffix='.gif')

    writergif = PillowWriter()
    ani.save(tf.name, writer=writergif)
    wandb.log({"video": wandb.Video(tf.name, fps=10, format='gif')})
    plt.close(fig)

def plot_velocity_3d(total_time, XV_0T: jnp.ndarray):
    X_0T, V_0T = jnp.split(XV_0T, indices_or_sections=2, axis=-1)
    # C = jnp.hypot(V_0T[0, :, 0], V_0T[0, :, 1], )
    T, N, D = X_0T.shape

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    quiver = [ax.quiver(X_0T[0, :, 0], X_0T[0, :, 1], X_0T[0, :, 2],
                        V_0T[0, :, 0], V_0T[0, :, 1], V_0T[0, :, 2])]

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    title = ax.set_title("Time: 0")

    def update(t):
        quiver[0].remove()
        quiver[0] = ax.quiver(X_0T[t, :, 0], X_0T[t, :, 1], X_0T[t, :, 2],
                              V_0T[t, :, 0], V_0T[t, :, 1], V_0T[t, :, 2])
        title.set_text(f"Time: {t}")

    ani = FuncAnimation(fig, update, frames=range(T), interval=200, repeat_delay=2000)
    # ani.save('./velocity_field.gif', writer='pillow')
    tf = tempfile.NamedTemporaryFile(dir="./", suffix='.gif')

    writergif = PillowWriter()
    ani.save(tf.name, writer=writergif)
    wandb.log({"video": wandb.Video(tf.name, fps=10, format='gif')})
    plt.close(fig)


def plot_scatter_2d(X, mins=np.array([-10, -10]), maxs=np.array([10, 10])):
    T, N, D = X.shape
    # Reshape and create DataFrame
    X_reshaped = X.reshape(-1, D)
    df = pd.DataFrame(X_reshaped)

    # Add time and particle index
    df["time"] = np.repeat(np.arange(T), N)
    df["particle"] = np.tile(np.arange(N), T)

    # Plot
    for t in range(T):
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df[df["time"] == t], x=0, y=1, hue="particle", palette="viridis")
        # only the first two dimensions are used, even in the Kinetic case
        plt.ylim(mins[0], maxs[0])
        plt.xlim(mins[1], maxs[1])
        plt.title(f"Scatter plot at time {t}")
        plt.show()

def plot_density_2d(f, config=None):
    # Sample data
    if config is None:
        side = jnp.linspace(-10, 10, 256)
        X, Y = jnp.meshgrid(side, side)
    else:
        mins = config["mins"]
        maxs = config["maxs"]
        side_x = jnp.linspace(mins[0], maxs[0], 256)
        side_y = jnp.linspace(mins[1], maxs[1], 256)
        X, Y = jnp.meshgrid(side_x, side_y)

    XY = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    Z = f(XY)
    Z = Z.reshape(X.shape)

    # Plot the density map using nearest-neighbor interpolation
    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(X, Y, Z)
    fig.colorbar(pcm, ax=ax)
    plt.show()