from configparser import Interpolation

import jax
import jax.numpy as jnp
from astropy.visualization import ZScaleInterval
from jax import lax


def solve_delta(N_i, T_prev, f_eq, n_iter=3):
    """
    Determine the distortion by solving:
        f_eq(N + delta) = T_prev - delta
    """

    def newton_step(delta, _):
        x = N_i + delta
        f = f_eq(x) + delta - T_prev
        fp = jax.grad(lambda d: f_eq(N_i + d) + d)(delta)
        return delta - f / fp, None

    delta0 = jnp.zeros(())
    delta_final = jax.lax.fori_loop(
        0, n_iter, lambda i, d: newton_step(d, None)[0], delta0
    )

    return delta_final


def distort_line(true_pixvals, f_eq, n_iter=3):
    """
    Apply the pocket effect distortion to undistorted pixels
    """

    def step(T_prev, N_i):
        delta = solve_delta(N_i, T_prev, f_eq, n_iter=n_iter)
        T_new = T_prev - delta
        return T_new, delta

    T_init = jnp.zeros(())
    _, deltas = jax.lax.scan(step, T_init, true_pixvals)

    return true_pixvals + deltas


def invert_line(distorted_pixvals, f_eq):
    """
    Invert the pocket effect distortion and return undistorted pixels.
    """

    # much simpler: no iteration required
    def step(T_prev, N_dist_i):
        # since (T_new, N_dist_i) are on the equililibrium line
        # T_new is directly given by the equilibrium relation
        T_new = f_eq(N_dist_i)
        delta = T_new - T_prev
        return T_new, delta

    T_init = jnp.zeros(())
    _, deltas = jax.lax.scan(step, T_init, distorted_pixvals)

    return distorted_pixvals + deltas


def predict(true_pixvals, f_eq, n_iter=3):
    """
    Apply the distortion to a full 2D image
    """
    return jax.vmap(distort_line, in_axes=(0, None, None), out_axes=0)(
        true_pixvals, f_eq, n_iter
    )


def invert(distorted_pixvals, f_eq):
    """
    Invert the distortion on a full 2D image
    """
    return jax.vmap(invert_line, in_axes=(0, None), out_axes=0)(distorted_pixvals, f_eq)


def plot_1d(
    f_eq,
    true_pixvals=None,
    seed=0,
    nstars=3,
    line_size=200,
    skylev=150.0,
    overscan_width=30,
    psf_sigma=2.0,
    add_poisson_noise=True,
    readout_noise_sigma=1.0,
    n_iter=3,
    figsize=(16, 9),
    **kwargs,
):
    """
    Correction on 1D pixels.
    """
    import matplotlib.pyplot as plt

    import sims

    key = jax.random.PRNGKey(seed)
    if true_pixvals is None:
        if skylev is None:
            skylev = 150.0
        k_stars, k_poisson, k_readout = jax.random.split(key, 3)

        line = sims.Line(size=line_size, skylev=skylev, overscan_width=overscan_width)
        stars = line.gen_stars(k_stars, nstars)
        line.add_stars(stars, sims.GaussianPSF1D(sigma=psf_sigma))
        if add_poisson_noise:
            line.add_noise(k_poisson)
        true_pixvals = line.true_data
        orig_pixvals = line.orig_data
    else:
        k_readout = key
        true_pixvals = jnp.asarray(true_pixvals)
        orig_pixvals = true_pixvals

    print(true_pixvals)
    distorted_pixvals = distort_line(true_pixvals, f_eq, n_iter=n_iter)

    if readout_noise_sigma is not None and float(readout_noise_sigma) > 0.0:
        distorted_pixvals = distorted_pixvals + (
            jax.random.normal(k_readout, shape=distorted_pixvals.shape)
            * float(readout_noise_sigma)
        )
    recovered_pixvals = invert_line(distorted_pixvals, f_eq)

    # _, pocket_evolution = predict_pocket_evolution_line(true_pixvals, pars)
    # if skylev is None:
    #    skylev = jnp.median(true_pixvals)
    # pocket_eq_val = pocket_eq(skylev, pars, n_warmup=n_warmup)

    fig, axes = plt.subplots(figsize=figsize, nrows=5, ncols=1, sharex=True)
    if orig_pixvals is not None:
        axes[0].plot(jnp.asarray(orig_pixvals), "k.", label="orig (not noisy)")
    axes[0].plot(true_pixvals, "g+", label="true")
    axes[0].plot(distorted_pixvals, "r+", label="distorted")
    axes[0].legend(loc="upper right")

    axes[1].plot(distorted_pixvals - true_pixvals, "k.", label="distorted - true")
    axes[1].legend(loc="lower right")

    # axes[2].plot(pocket_evolution, "c.", label="pocket contents")
    # axes[2].axhline(pocket_eq_val, color="r", ls=":", label="pocket at equilibrium")
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("pocket_q")

    axes[3].plot(true_pixvals, "g+", label="true")
    axes[3].plot(distorted_pixvals, "r+", label="distorted")
    axes[3].plot(recovered_pixvals[:-overscan_width], "b.", label="recovered")
    axes[3].legend(loc="upper right")

    axes[4].plot(
        (recovered_pixvals - true_pixvals)[:-overscan_width],
        "k.",
        label="recovered - true",
    )
    if readout_noise_sigma is not None:
        s = float(readout_noise_sigma)
        axes[4].axhspan(-s, s, color="c", alpha=0.25)
    axes[4].legend(loc="lower left")
    axes[4].set_xlabel("col")
    axes[4].set_ylabel("residual")
    # axes[4].set_ylim((-5.0 * s, 5.0 * s))

    plt.subplots_adjust(hspace=0.025, wspace=0.025)
    return fig, axes, true_pixvals, distorted_pixvals, recovered_pixvals - true_pixvals


def plot_2d(
    f_eq,
    true_pixvals=None,
    nstars=100,
    shape=(200, 200),
    overscan_width=30,
    psf_sigma=1.5,
    skylev=150.0,
    add_poisson_noise=True,
    readout_noise_sigma=1.0,
    n_iter=3,
    figsize=(10, 10),
    seed=0,
):
    """
    Correction on 2D pixels.
    """
    import matplotlib.pyplot as plt

    import sims

    key = jax.random.PRNGKey(seed)
    if true_pixvals is None:
        if skylev is None:
            skylev = 150.0
        k_stars, k_poisson, k_readout = jax.random.split(key, 3)

        im = sims.Image(shape=shape, skylev=skylev, overscan_width=overscan_width)
        stars = im.gen_stars(k_stars, nstars)
        im.add_stars(stars, sims.GaussianPSF2D(sigma=psf_sigma))
        if add_poisson_noise:
            im.add_noise(k_poisson)
        true_pixvals = im.true_data
        orig_pixvals = im.orig_data
    else:
        k_readout = key
        true_pixvals = jnp.asarray(true_pixvals)
        orig_pixvals = true_pixvals

    distorted_pixvals = predict(true_pixvals, f_eq, n_iter=n_iter)

    if readout_noise_sigma is not None and float(readout_noise_sigma) > 0.0:
        distorted_pixvals = distorted_pixvals + (
            jax.random.normal(k_readout, shape=distorted_pixvals.shape)
            * float(readout_noise_sigma)
        )
    recovered_pixvals = invert(distorted_pixvals, f_eq)

    #
    fig, axes = plt.subplots(
        figsize=figsize, nrows=2, ncols=2, sharex=True, sharey=True
    )

    interval = ZScaleInterval()

    # original (distorted) frame
    axes[0, 0].imshow(
        distorted_pixvals,
        interpolation="none",
    )
    axes[0, 0].set_title("distorted")

    # distortion
    vmin, vmax = interval.get_limits(distorted_pixvals - true_pixvals)
    axes[0, 1].imshow(
        distorted_pixvals - true_pixvals,
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title("distorted - true")
    axes[1, 0].imshow(
        recovered_pixvals - distorted_pixvals,
        interpolation="none",
        vmin=-25.0,
        vmax=25.0,
    )
    axes[1, 0].set_title("recovered - distorted (correction)")
    axes[1, 1].imshow(
        recovered_pixvals - true_pixvals,
        interpolation="none",
        vmin=-25.0,
        vmax=25.0,
    )
    axes[1, 1].set_title("recovered - true")
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return (
        fig,
        axes,
        true_pixvals,
        distorted_pixvals,
        recovered_pixvals,
        recovered_pixvals - true_pixvals,
    )
