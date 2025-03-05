import argparse
import datetime
import math
import os
import yaml

import jax.numpy as jnp
import jax.random as jr

import ssm
import fk

import collectors


class ParamError(Exception):
    """
    Simple class to catch exception errors and print helpful debugging messages.
    """

    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


def main(cli_args):

    if len(cli_args.model_args) > 0:  # get config file from cli
        print("Reading config file from command-line ")
        if os.path.isfile(cli_args.model_args):
            with open(cli_args.model_args, "r", encoding="utf-8") as f_in:
                mod_args = yaml.safe_load(f_in)
        else:
            raise ParamError(
                " The config yaml file is needed to set the plotting params."
            )
    else:
        raise ParamError(" The config yaml file is needed to set the plotting params.")

    T_max = mod_args["SMC_args"]["T_max"]
    tau = mod_args["SMC_args"]["tau"]

    N = mod_args["epi_args"]["N"]
    I0 = mod_args["epi_args"]["I0"]

    key = jr.key(mod_args["SMC_args"]["seed"])
    model_key, fk_key, alg_key = jr.split(key, 3)

    sir = ssm.SIRSDE(
        N=float(N / N),
        I0=float(I0 / N),
        sigma2_y=mod_args["epi_args"]["sigma2_x"],
        sigma2_x=mod_args["epi_args"]["sigma2_y"],
        tau=tau,
        key=model_key,
        abm_file=mod_args["SMC_args"]["abm_file"],
    )
    fk_model = ssm.Bootstrap(ssm=sir, data=sir.y.copy())

    alg = fk.SMC(
        T_max,
        tau,
        fk=fk_model,
        N=mod_args["SMC_args"]["n_particles"],
        store_history=True,
        collect=[collectors.Moments(), collectors.ESSs()],
        Y_STEP=math.ceil(1.0 / tau),
        key=fk_key,
        verbose=False,
        # resampler="parallel"
    )
    alg.run(key=alg_key)
    print("Log weight variance: ", jnp.var(alg.wgts.lw))
    out_file = f"results/{mod_args['save_name']}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.npz"
    print(f"Writing results to: {out_file}")

    jnp.savez(
        out_file,
        X=alg.hist.X,
        moments=alg.summaries.moments,
        lw=alg.hist.wgts,
        ess=alg.summaries.ESSs,
        abm_I=sir.y[:,1],
        params=mod_args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_args",
        "-args",
        type=str,
        default="",
        help="""specify the model config""",
        required=True,
    )

    main(parser.parse_args())
