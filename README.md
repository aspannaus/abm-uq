# abm-uq
Improved Uncertainty Quantification for Agent-Based Models

The python dependencies are listed in the `requirements.txt` and also requires a 
working Cuda [toolkit](https://developer.nvidia.com/cuda-toolkit) installation for the Thrust prefix sum. 

You will also need:
  - NVCC
  - GNU C and C++ compilers

## Compiling shared parallel-scan library

To create the shared library, you'll need to be on a machine with CUDA 12.4 installed. The commands are then:

```
make objs/thrust_scan.o
make libthrustscan
```
The shared library will be saved in the `parallel-sum` directory and accessible from the python
scripts.

You can then load the shared library into python, a small example is given in `scan_test.py`.

## Running simulations

To run a uq simulation with static data, run `$ python ssm_inference.py -args configs/smc_config.yaml`. 

To run a uq simulation with streaming data, run `$ mpiexec -n N python abm_coupling.py -args networks/ABM_BTER.yaml -smc configs/smc_streaming.yaml`, where N is the number of processors to use with the MPI backend for the agent based model.

### Configuration file

Following is a brief description of the required keywords in the config file.

 - save_name: "smc_static"  # output file name, a time stamp is appended
 - SMC_args:
    - T_max: 49  # number of simulation days
    - tau: 0.5   # time-step between observations
    - abm_file: "data/abm_data/counts_1.csv"  # used for static (offline) data assimilation
    - seed: 4211  #  jax random number seed
    - n_particles: 16384  # number of particles in sequential Monte Carlo

- epi_args:
  - N: 4000  # population size
  - I0: 10   # initial number of infections
  - beta: 0.5  # used for plotting ODE solution
  - gamma: 0.1  # used for plotting ODE solution
  - sigma2_x: 0.05  # variance of ensemble noise
  - sigma2_y: 0.1   # variance of observational noise

