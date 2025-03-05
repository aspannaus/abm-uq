"""Coupled ABM and UQ module

"""

import copy
import datetime
import math
import os
from dataclasses import dataclass
import yaml

from mpi4py import MPI
import numpy as np

from repast4py.network import write_network, read_network

from repast4py import core, random, schedule, logging
from repast4py import context as ctx

import jax
import jax.random as jr
import jax.numpy as jnp

import collectors
import fk
import ssm
import utils



class ParamError(Exception):
    """
    Simple class to catch exception errors and print helpful debugging messages.
    """

    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


def generate_network_file(fname: str, n_ranks: int, n_agents: int, g):
    """Generates a network file using repast4py.network.write_network.

    Args:
        fname: the name of the file to write to
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    # g = nx.complete_graph(n_agents)
    try:
        import nxmetis

        write_network(g, "network", fname, n_ranks, partition_method="metis")
    except ImportError:
        write_network(g, "network", fname, n_ranks)


class NodeAgent(core.Agent):

    def __init__(
        self,
        nid: int,
        agent_type: int,
        rank: int,
        received_infect=False,
        recovered=False,
    ):
        super().__init__(nid, agent_type, rank)
        self.received_infect = received_infect
        self.recovered = recovered

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data (self.received_rumor).

        Returns:
            The agent's state
        """
        return (self.uid, self.received_infect, self.recovered)

    # def update(self, data: bool, data_recov: bool):
    #     """Updates the state of this agent when it is a ghost
    #     agent on some rank other than its local one.

    #     Args:
    #         data: the new agent state (received_rumor)
    #     """

    #     if not self.received_infect and data:
    #         # only update if the received rumor state
    #         # has changed from false to true
    #         model.infect.append(self)
    #         self.received_infect = data
    #     if not self.recovered and data_recov:
    #         model.recover.append(self)
    #         self.recovered = data_recov


def create_node_agent(nid, agent_type, rank, **kwargs):
    return NodeAgent(nid, agent_type, rank)


def restore_agent(agent_data):
    uid = agent_data[0]
    return NodeAgent(uid[0], uid[1], uid[2], agent_data[1])


@dataclass
class InfectedCounts:
    total_infect: int
    new_infect: int
    total_recov: int
    new_recov: int


@dataclass
class HistoryCounts:
    total_infect_history: list[int]
    total_recover_history: list[int]


class Model:

    def __init__(
        self,
        comm,
        params,
        susceptible=None,
        infect=None,
        recover=None,
        net=None,
        context=None,
        num_nodes=None,
        avg_deg=None,
    ):
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)
        self.rank = comm.Get_rank()
        if context is None:
            self.context = ctx.SharedContext(comm)
            fpath = params["network_file"]
            read_network(fpath, self.context, create_node_agent, restore_agent)
        else:
            self.context = context
        if net is None:
            self.net = self.context.get_projection("network")
        else:
            self.net = net

        if susceptible is None:
            self.susceptible = []
            for agent in self.context.agents(
                agent_type=None, count=None, shuffle=False
            ):
                self.susceptible.append(agent)
        else:
            self.susceptible = susceptible

        if infect is None:
            self.infect = []
            self._seed_infect(params["initial_infect_count"], comm)
        else:
            self.infect = infect

        if recover is None:
            self.recover = []
            self._initial_recover(params["initial_recover_count"], comm)
        else:
            self.recover = recover

        self.y = []

        infect_count = len(self.infect)
        recover_count = len(self.recover)

        self.counts = InfectedCounts(
            infect_count, infect_count, recover_count, recover_count
        )
        self.counts_history = HistoryCounts([infect_count], [recover_count])

        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params["counts_file"])
        self.data_set.log(0)

        self.num_nodes = params["noNodes"]
        self.avg_deg = params["avgDeg"]

        self.trans_probability = params["trans_probability"]
        self.recov_probability = params["recov_probability"]

    def _seed_infect(self, init_count: int, comm):
        world_size = comm.Get_size()
        # np array of world size, the value of i'th element of the array
        # is the number of infects to seed on rank i.
        infect_counts = np.zeros(world_size, np.int32)
        if self.rank == 0:
            for _ in range(init_count):
                idx = random.default_rng.integers(0, high=world_size)
                infect_counts[idx] += 1

        infect_count = np.empty(1, dtype=np.int32)
        comm.Scatter(infect_counts, infect_count, root=0)

        for agent in self.context.agents(count=infect_count[0], shuffle=True):
            agent.received_infect = True
            self.infect.append(agent)
            self.susceptible.remove(agent)

    def _initial_recover(self, init_recover_count: int, comm):
        world_size = comm.Get_size()

        # np array of world size, the value of i'th element of the array
        # is the number of recovers to seed on rank i.
        recover_counts = np.zeros(world_size, np.int32)
        if self.rank == 0:
            for _ in range(init_recover_count):
                idx = random.default_rng.integers(0, high=world_size)
                recover_counts[idx] += 1

        recover_count = np.empty(1, dtype=np.int32)
        comm.Scatter(recover_counts, recover_count, root=0)

        for agent in np.random.choice(
            self.susceptible, recover_count[0], replace=False
        ):
            if agent in self.infect:
                self.infect.remove(agent)

            self.recover.append(agent)

    def at_end(self):
        self.data_set.close()

    def step(self):
        new_infect = []
        new_recov = []
        rng = random.default_rng
        t = self.runner.schedule.tick / 1
        for agent in self.infect:
            for ngh in self.net.graph.neighbors(agent):
                if (
                    not ngh.received_infect
                    and ngh.local_rank == self.rank
                    and rng.uniform() <= self.trans_probability
                ):
                    ngh.received_infect = True
                    new_infect.append(ngh)
            if rng.uniform() <= self.recov_probability:
                self.recover.append(agent)
                self.infect.remove(agent)

                new_recov.append(agent)

        self.infect += new_infect
        self.counts.new_infect = len(new_infect)
        self.counts.new_recov = len(new_recov)
        self.counts.total_infect = len(self.infect)
        self.counts.total_recov += self.counts.new_recov

        self.counts_history.total_infect_history.append(self.counts.total_infect)
        self.counts_history.total_recover_history.append(self.counts.total_recov)

        if self.runner.schedule.tick % 1 == 0:
            self.data_set.log(self.runner.schedule.tick)

        self.context.synchronize(restore_agent)

    def start(self):
        self.runner.execute()


def run(model):
    # global model
    model.start()
    I = model.counts_history.total_infect_history
    R = model.counts_history.total_recover_history
    S = model.num_nodes - jnp.array(I) - jnp.asarray(R)
    model.y = jnp.asarray([S, I, R]) / model.num_nodes


def update_abm_params(abm_params, ssm_params, counts, num_nodes, avg_deg):
    abm_params["initial_infect_count"] = int(counts[1, -1] * num_nodes)
    abm_params["initial_recover_count"] = int(counts[2, -1] * num_nodes)
    abm_params["trans_probability"] = jnp.exp(ssm_params[0]) / avg_deg  # for node degree
    abm_params["recov_probability"] = jnp.exp(ssm_params[1])
    return abm_params


def main(cli_args):

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # on GPU, put each process on its own device
    if mpi_rank != 0:
        jax.default_device(jax.devices("cpu")[0])

    print(f"MPI rank {mpi_rank}, MPI size {mpi_size}, Jax devices: {jax.devices()}")

    if mpi_rank == 0:
        if os.path.isfile(cli_args.smc_args):
            with open(cli_args.smc_args, "r", encoding="utf-8") as f_in:
                smc_args = yaml.safe_load(f_in)
        else:
            raise ParamError(
                " The config yaml file is needed to set the smc model params."
            )

        if os.path.isfile(cli_args.abm_args):
            with open(cli_args.abm_args, "r", encoding="utf-8") as f_in:
                abm_params = yaml.safe_load(f_in)
        else:
            raise ParamError(
                " The config yaml file is needed to set the network model params."
            )

        T_max = smc_args["SMC_args"]["T_max"]
        tau = smc_args["SMC_args"]["tau"]
        seed = smc_args["SMC_args"]["seed"]
        N = smc_args["epi_args"]["N"]

        key = jr.key(smc_args["SMC_args"]["seed"])

        output_dict = {"beta": [], "gamma": [], "S": [], "I": [], "R": []}

        key = jr.key(seed)
        model_key, fk_key, alg_key, key = jr.split(key, 4)

        rands = jr.uniform(key, minval=-20, maxval=-1, shape=2)
        beta0 = jnp.exp(rands[0])
        gamma0 = jnp.exp(rands[1])
        # both start from same ICs
        sir = ssm.SIRSDE(
            N=float(N / N),
            I0=abm_params["initial_infect_count"] / abm_params["noNodes"],
            sigma2_y=smc_args["epi_args"]["sigma2_y"],
            sigma2_x=smc_args["epi_args"]["sigma2_x"],
            tau=tau,
            key=model_key,
            abm_file=None,
            beta0=jnp.log(beta0),
            gamma0=jnp.log(gamma0) 
        )

        abm_params["trans_probability"] = (
            beta0 / abm_params["avgDeg"]
        )  # for node degree
        abm_params["recov_probability"] = gamma0
        output_dict["beta"].append(beta0)
        output_dict["gamma"].append(gamma0)
        mpi_comm.send(abm_params, dest=1, tag=11)

    # in global space
    if mpi_rank == 1:
        abm_params = mpi_comm.recv(source=0, tag=11)
    n_weeks = 7
    model = Model(MPI.COMM_WORLD, abm_params)
    run(model)

    if mpi_rank == 0:
        ys = model.y.copy()
        print("I", model.y[1])
        output_dict["S"].append(model.y[0])
        output_dict["I"].append(model.y[1])
        output_dict["R"].append(model.y[2])

        fk_model = ssm.Bootstrap(ssm=sir, data=ys.T)

        alg = fk.SMC(
            T=int(T_max / n_weeks),
            tau=tau,
            fk=fk_model,
            N=smc_args["SMC_args"]["n_particles"],
            store_history=True,
            resampler="parallel",
            collect=[collectors.Moments(), collectors.ESSs()],
            Y_STEP=math.ceil(1.0 / tau),
            key=fk_key,
            verbose=False,
        )
        alg.run(key=alg_key)
        
        ssm_params = jnp.zeros(2)
        ssm_params = ssm_params.at[0].set(
            utils.weighted_percentile(alg.X[:, 3], alg.wgts.W, 0.5)
        )
        ssm_params = ssm_params.at[1].set(
            utils.weighted_percentile(alg.X[:, 4], alg.wgts.W, 0.5)
        )
        abm_params = update_abm_params(
            abm_params, ssm_params, model.y, abm_params["noNodes"], abm_params["avgDeg"]
        )
        output_dict["beta"].append(ssm_params[0])
        output_dict["gamma"].append(ssm_params[1])
        moments = alg.summaries.moments

    # the repast bits need to be in the global space
    abm_params = mpi_comm.bcast(abm_params, root=0)
    old_s = copy.copy(model.susceptible)
    old_i = copy.copy(model.infect)
    old_r = copy.copy(model.recover)

    for week in range(1, n_weeks):
        # run abm in global namespace
        model = Model(
            MPI.COMM_WORLD,
            abm_params,
            susceptible=old_s,
            infect=old_i,
            recover=old_r,
            net=model.net,
            context=model.context,
        )
        run(model)

        if mpi_rank == 0:
            print("week", week)
            output_dict["S"].append(model.y[0])
            output_dict["I"].append(model.y[1])
            output_dict["R"].append(model.y[2])
            new_key, key, y0_key = jr.split(key, 3)

            fk_model.data = model.y.T.copy()
            # passing conditions from the abm to initialize the next smc round
            # set smc initial conds from abm
            sir.Y0 = alg.hist.X[-1]
            tmp_i = model.y[1, 0] + (sir.sigma2_y * jr.normal(y0_key, shape=(alg.N)))
            tmp_i = tmp_i.at[:].set(jnp.where(tmp_i > 0, tmp_i, jnp.abs(tmp_i)))
            tmp_i = tmp_i.at[:].set(jnp.where(tmp_i < 1.0, tmp_i, 1.0 / tmp_i))
            sir.Y0 = sir.Y0.at[:, 1].set(tmp_i)
            alg.X = sir.Y0.copy()
            # run smc
            alg.run(key=key, abm=True)
            # get distribution est from smc
            ssm_params = jnp.zeros(2)
            ssm_params = ssm_params.at[0].set(
                utils.weighted_percentile(alg.X[:, 3], alg.wgts.lw, 0.5)
            )
            ssm_params = ssm_params.at[1].set(
                utils.weighted_percentile(alg.X[:, 4], alg.wgts.lw, 0.5)
            )
            abm_params = update_abm_params(
                abm_params, ssm_params, model.y, abm_params["noNodes"], abm_params["avgDeg"]
            )
            output_dict["beta"].append(ssm_params[0])
            output_dict["gamma"].append(ssm_params[1])
            # jax randomness
            key = new_key

        # update abm params in global space
        abm_params = mpi_comm.bcast(abm_params, root=0)
        old_s = copy.copy(model.susceptible)
        old_i = copy.copy(model.infect)
        old_r = copy.copy(model.recover)

    if mpi_rank == 0:
        out_file = f"results/{smc_args['save_name']}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.npz"
        print(f"Writing results to: {out_file}")
        jnp.savez(
            out_file,
            X=alg.hist.X,
            lw=alg.hist.wgts,
            moments=alg.summaries.moments,
            ess=alg.summaries.ESSs,
            params=smc_args,
            abm_beta=output_dict["beta"],
            abm_gamma=output_dict["gamma"],
            abm_S=output_dict["S"],
            abm_I=output_dict["I"],
            abm_R=output_dict["R"],
        )


if __name__ == "__main__":
    import argparse
    # mpi_comm = MPI.COMM_WORLD
    # mpi_rank = mpi_comm.Get_rank()
    # if mpi_rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abm_args",
        "-args",
        type=str,
        default="",
        help="""specify the network model config""",
        required=True,
    )
    parser.add_argument(
        "--smc_args",
        "-smc",
        type=str,
        default="",
        help="""specify the smc model config""",
        required=True,
    )
    parser.add_argument("--parameters", type=str, required=False)
    args = parser.parse_args()
    num_gpus_per_node = jax.device_count()

    main(args)
