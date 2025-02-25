import argparse
import os

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp


font = {"family": "sans-serif", "weight": "regular", "size": 16}
matplotlib.rc("font", **font)
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


def sir(t, y, log_beta, log_gamma):
    (S, I, R,) = y

    gamma = np.exp(log_gamma)
    beta = np.exp(log_beta)
    dS = -(beta * S * I)
    dI = (beta * S * I) - I * gamma
    dR = I * gamma
    return np.array([dS, dI, dR])


def clip(x):
    x = np.asarray(x)
    x = np.where(x > 0, x, 0.0)
    x = np.where(x < 1, x, 1.0)
    return x


def load_abm_data(in_file):
    # data = np.loadtxt(in_file, delimiter=",", skiprows=1, usecols=(1,2,3,4,6))
    data = pd.read_csv(in_file)
    idxs = np.where(data["hour"] == 0.0)
    data = data.iloc[idxs]
    data = data[["S_susceptible", "I_total_infect", "R_total_recov"]]
    return data.to_numpy() / 4000


def plot(plot_args, abm, data, soln):

    T_max = plot_args["T_max"]
    ts = np.linspace(0, T_max, endpoint=True, num=data["X"].shape[0])

    S = soln["y"][0]
    I = soln["y"][1]
    R = soln["y"][2]
    t_epi = np.linspace(0, T_max, endpoint=True, num=S.shape[0])

    moments = data["moments"]

    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=300)
    ax[0, 0].plot(ts, [m["mean"][0] for m in moments], label="S SMC")
    ax[0, 0].plot(t_epi, S, label="S ODE", color="tab:orange")
    ub = clip([m["mean"][0] + 2 * np.sqrt(m["var"][0]) for m in moments])
    lb = clip([m["mean"][0] - 2 * np.sqrt(m["var"][0]) for m in moments])
    ax[0, 0].fill_between(ts, lb, ub, alpha=0.2, color="tab:blue")
    ax[0, 0].set_ylabel("Population Proportion")
    # ax[0, 0].set_xlabel("t")
    ax[0, 0].legend()

    ax[0, 1].plot(ts, [m["mean"][1] for m in moments], label="I SMC")
    ax[0, 1].scatter(np.arange(abm.shape[0]), abm, marker="x", color="m")
    ax[0, 1].plot(t_epi, I, label="I ODE", color="tab:orange")
    ub = clip([m["mean"][1] + 2 * np.sqrt(m["var"][1]) for m in moments])
    lb = clip([m["mean"][1] - 2 * np.sqrt(m["var"][1]) for m in moments])
    ax[0, 1].fill_between(ts, lb, ub, alpha=0.2, color="tab:blue")
    # ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("Population Proportion")
    ax[0, 1].legend()

    ax[1, 0].plot(ts, [m["mean"][2] for m in moments], label="R SMC")
    ax[1, 0].plot(t_epi, R, label="R ODE", color="tab:orange")
    ub = clip([m["mean"][2] + 2 * np.sqrt(m["var"][2]) for i, m in enumerate(moments)])
    lb = clip([m["mean"][2] - 2 * np.sqrt(m["var"][2]) for i, m in enumerate(moments)])
    ax[1, 0].fill_between(ts, lb, ub, alpha=0.2, color="tab:blue")
    ax[1, 0].set_xlabel("Time (days)")
    ax[1, 0].set_ylabel("Population Proportion")
    ax[1, 0].legend()

    # beta and gamma
    ax[1, 1].plot(
        ts,
        jnp.log(plot_args["beta"]) * jnp.ones(ts.shape[0]),
        # label=r"$\log \beta$",
        alpha=1,
        c="darkorange",
    )
    ax[1, 1].plot(
        ts,
        [m["mean"][3] for m in moments],
        label=r"$\log \beta$ SMC",
        alpha=0.9,
        color="lightskyblue",
    )
    ub = [m["mean"][3] + 2 * np.sqrt(m["var"][3]) for m in moments]
    lb = [m["mean"][3] - 2 * np.sqrt(m["var"][3]) for m in moments]
    ax[1, 1].fill_between(ts, lb, ub, alpha=0.2, color="lightskyblue")

    ax[1, 1].plot(
        ts,
        jnp.log(plot_args["gamma"]) * jnp.ones(ts.shape[0]),
        # label=r"$\log \gamma$",
        alpha=0.8,
        c="orange",
    )

    ax[1, 1].plot(
        ts,
        [m["mean"][4] for m in moments],
        label=r"$\log \gamma$ SMC",
        alpha=0.8,
        c="darkturquoise",
    )

    ub = [m["mean"][4] + 2 * np.sqrt(m["var"][4]) for m in moments]
    lb = [m["mean"][4] - 2 * np.sqrt(m["var"][4]) for m in moments]

    ax[1, 1].fill_between(ts, lb, ub, alpha=0.2, color="darkturquoise")
    ax[1, 1].legend(loc="lower right")
    ax[1, 1].set_ylabel("Log gamma, log beta")
    ax[1, 1].set_xlabel("Time (days)")

    plt.tight_layout()
    plt.show()


def main(cli_args):

    print("Reading config file from command-line ")

    if os.path.isfile(cli_args.data):
        if cli_args.model == "static":
            data = np.load(cli_args.data, allow_pickle=True)
            # fancy indexing to get the dict of model params
            mod_args = data["params"][()]
            in_file = mod_args["SMC_args"]["abm_file"]
            _abm_data = load_abm_data(in_file)
            abm_data = _abm_data[:, 1]
        elif cli_args.model == "streaming":
            data = np.load(cli_args.data, allow_pickle=True)
            mod_args = data["params"][()]
            abm_data = data["abm_I"].flatten()
        else:
            raise ParamError("""The data file is required to plot the results.""")
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
    # need to compute the truth
    N = plot_args["N"]
    # Initial number of infected and recovered individuals, I0 and R0.
    I0 = plot_args["I0"]
    R0 = 0.0
    S0 = N - I0 - R0
    # Initial conditions vector
    y0 = np.asarray([S0, I0, R0]) / N

    ts = np.linspace(0, mod_args["SMC_args"]["T_max"], endpoint=True, num=250)
    # Integrate the SIR equations over the time grid, t.
    ret = solve_ivp(
        sir,
        t_span=[0, mod_args["SMC_args"]["T_max"]],
        t_eval=ts,
        y0=y0,
        method="RK45",
        args=(jnp.log(plot_args["beta"]), jnp.log(plot_args["gamma"])),
    )

    plot(plot_args, abm_data, data, ret)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="",
        help="""specify the model type, either streaming or static""",
        required=True,
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
