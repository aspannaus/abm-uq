import argparse
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as grid_spec
import scipy

from ridgeplot import ridgeplot

from jax.scipy import stats

import utils


font = {"family": "sans-serif", "weight": "regular", "size": 24}
mpl.rc("font", **font)
plt.rcParams["text.usetex"] = True


class ParamError(Exception):
    """
    Simple class to catch exceptions and print helpful messages.
    """

    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


def main(args):

    print("Reading args from command-line ")

    if os.path.isfile(args.data):
        data = np.load(args.data, allow_pickle=True)
        mod_args = data["params"][()]
        abm_data = data["abm_I"].flatten()
    else:
        raise ParamError("""The data file is required to plot the results.""")

    plot_args = {
        "T_max": mod_args["SMC_args"]["T_max"],
        "tau": mod_args["SMC_args"]["tau"],
        "N": mod_args["epi_args"]["N"],
        "I0": mod_args["epi_args"]["I0"],
        "beta": mod_args["epi_args"]["beta"],
        "gamma": mod_args["epi_args"]["gamma"],
    }

    start = 36
    end = 55
    I = data["X"][start:end, :, 1]
    _w = data["lw"][start:end]

    time = np.linspace(start, end, endpoint=False, num=I.shape[0]) * plot_args["tau"]

    i_min = 0.3
    i_max = 0.6
    pts = np.linspace(i_min, i_max, endpoint=True, num=500)

    times = [t for t in time]

    gs = grid_spec.GridSpec(len(times), 1)
    fig = plt.figure(figsize=(16, 9), dpi=300)

    cmap = mpl.cm.viridis
    ax_objs = []

    for i, time in enumerate(times):
        w = np.exp(_w[i] - _w[i].max())
        W = w / w.sum()
        vals = np.linspace(i_min, i_max, I[i].shape[0], endpoint=True)
        kde = stats.gaussian_kde(dataset=I[i], weights=W, bw_method="scott")
        probs = kde.pdf(vals)
        res = scipy.stats.ecdf(probs, )

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        ax_objs[i].plot(vals, probs, lw=1, alpha=0.5)
        ax_objs[i].fill_between(vals, probs, alpha=0.5)

        # setting uniform x and y lims
        ax_objs[i].set_xlim(i_min, i_max)
        ax_objs[i].set_ylim(0, max(probs) * 1.05)

        # make background transparent
        rect = ax_objs[i].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[i].set_yticklabels([])
        ax_objs[i].axes.get_yaxis().set_visible(False)
        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[i].spines[s].set_visible(False)
        
        if i == len(times) - 1:
            ax_objs[i].set_xlabel("Proportion Infected", )
        else:
            ax_objs[i].set_xticklabels([])

        if i % 2 == 0:
            ax_objs[i].text(0.29, 0, int(time), ha="right")

    gs.update(hspace=-0.2)

    fig.text(0.45, 0.95, "Infected Densities")
    fig.text(0.02, 0.5, "Time", rotation="vertical")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="",
        help="""specify the data path""",
        required=True,
    )

    main(parser.parse_args())
