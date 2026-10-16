# Distributed State-Vector Quantum Circuit Simulator with Lossy and All-Zero Compression

This is *QompSim*, a scalable distributed state-vector simulator that integrates error-bounded lossy compression and all-zero data compression. The implementation is based on *Qulacs* [1,2] and employs the stage-level block-wise lossy compression technique proposed in *BMQSim* [3].

The Python distribution is named `qompsim`, but its import package is `qulacs`. QompSim is a separate project from upstream Qulacs; because both use the `qulacs` import namespace, installing them in the same Python environment is not supported and may cause conflicts.

## Build Instructions

### Required Packages
* Python 3.10 or later
* CMake 3.5 or later (3.30.4 or later when using RMM)
* OpenMPI
* Boost
* CUDA 12 (only for the GPU implementation)

### Install SZ3 (used by the CPU implementation)

```Shell
mkdir -p $HOME/local/tools
cd $HOME/local/tools
git clone https://github.com/szcompressor/SZ3.git -b v3.3.2
cd SZ3
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME/local ..
make -j $(nproc)
make install
```

* The provided environment configuration sets `SZ3_ROOT=$HOME/local`, so installing SZ3 under `$HOME/local` is recommended.
* Before building SZ3, we recommend commenting out the `printf` calls on lines 36 and 127 of `include/SZ3/api/impl/SZImplOMP.hpp`.

### Install NVIDIA nvCOMP (used by the GPU implementation)

```Shell
cd $HOME/local/tools
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz
tar xf nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz
```

* No build is required; just extract the tarball.
* The provided environment configuration sets `NVCOMP_ROOT=$HOME/local/tools/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive`, so extracting nvCOMP under `$HOME/local/tools` is recommended.

### Install RMM (optional, used for GPU memory allocation)

```Shell
cd $HOME/local/tools
git clone https://github.com/rapidsai/rmm.git -b v25.12.00
cd rmm/cpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME/local ..
make -j $(nproc)
make install
```

* RMM is disabled by default. Install it only when building with `USE_RMM=Yes`.
* The pragma on line 409 of `cpp/tests/device_buffer_tests.cu` may need to be commented out.

### Build QompSim

```Shell
git clone https://github.com/FujitsuResearch/Compression-based-State-Vector-Simulator.git
cd Compression-based-State-Vector-Simulator
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install qulacsvis mpi4py
source env.sh  # Adjust the environment settings as needed.
C_COMPILER=mpicc CXX_COMPILER=mpicxx \
  USE_MPI=Yes USE_GPU_COMP=Yes pip install .
```

* This command installs the GPU-enabled implementation as a Python module.
* For a CPU-only build, set `USE_GPU_COMP=No`.
* To enable RMM for GPU memory allocation, install RMM and set `USE_RMM=Yes`.
* Remove the `build` directory before rebuilding with different build options.

## How to Run

### Quick Start

Run a 28-qubit GHZ circuit with two MPI processes, assigning one process to each socket and GPU:

```Shell
mkdir -p logdir
mpirun -n 2 --map-by ppr:1:socket --bind-to socket ./bind_gpu.sh \
  python script/run_sim.py \
    --simulator fidelity \
    --circuit_name ghz \
    --nqubits 28 \
    --outdir logdir
```

### Simulation Modes

| Mode | Description |
| --- | --- |
| `qulacs` | Run upstream Qulacs only. |
| `qompsim` | Run QompSim only. |
| `fidelity` | Run both simulators and compare their state vectors to calculate the QompSim fidelity. |

### Options

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `-s`, `--simulator` | Yes | - | Simulation mode: `qulacs`, `qompsim`, or `fidelity`. |
| `-c`, `--circuit_name` | Yes | - | Circuit to simulate. See the supported circuits below. |
| `-n`, `--nqubits` | Yes | - | Number of qubits. |
| `-o`, `--outdir` | No | Standard output | Directory in which to store generated JSON result files. |
| `-a`, `--append` | No | Disabled | Append results to an existing output file. |
| `-d`, `--draw_circuit` | No | Disabled | Render the circuit with Qulacs Visualizer. |
| `-b`, `--block_size` | No | Automatic | Base-2 logarithm of the number of amplitudes in each state-vector block. |
| `-i`, `--inner_size` | No | `4` | Base-2 logarithm of the number of blocks in each state-vector group. |

Supported circuits:

`hadamard`, `seq_hadamard`, `ghz`, `qft`, `ising`, `hea`, `grover`, `cc`, `bv`, `qsvm`, `qaoa`, `wstate`

> When `--outdir` is set, output file names are generated automatically. If a matching file already exists, the simulation is skipped unless `--append` is specified.

### Compression Layout

For `--block_size B` and `--inner_size I`:

| Quantity | Size |
| --- | ---: |
| Amplitudes per block | $2^B$ |
| Blocks per group | $2^I$ |
| Working set excluding the compressed state vector | $2^{B + I + 4}$ bytes |

Run `python script/run_sim.py --help` to view the complete command-line reference.

### Compression Environment Variables

The following environment variables control the error bounds and all-zero block
detection. Set them before starting the simulation. GPU and CPU compression use
different variables, as shown below.

| Variable | Compression backend | Description | Default |
| --- | --- | --- | --- |
| `BITCOMP_REL_EB` | GPU / Bitcomp | Relative error bound for lossy compression. | `1e-3` |
| `BITCOMP_ABS_EB` | GPU / Bitcomp | Absolute error bound for lossy compression. | `1e-7` |
| `SZ3_EB_MODE` | CPU / SZ3 | Error-bound mode. Set to `rel` for relative error; other values, including `abs`, select absolute error. | `abs` |
| `SZ3_EB_VAL` | CPU / SZ3 | Error-bound value used for either absolute or relative mode. | `1e-7` |
| `ALL_ZERO_THRESHOLD` | CPU / SZ3 | Values whose real and imaginary parts are both within this threshold are treated as zero when deciding whether to store an all-zero block. | `10 * SZ3_EB_VAL` |

For example, configure the GPU-based Bitcomp backend with separate relative and
absolute error bounds:

```Shell
export BITCOMP_REL_EB=1e-3
export BITCOMP_ABS_EB=1e-7
mpirun -n 2 python script/run_sim.py \
  --simulator qompsim --circuit_name ghz --nqubits 28
```

For the CPU-based SZ3 backend, use relative error control and explicitly set the
all-zero threshold:

```Shell
export SZ3_EB_MODE=rel
export SZ3_EB_VAL=1e-6
export ALL_ZERO_THRESHOLD=1e-5
mpirun -n 2 python script/run_sim.py \
  --simulator qompsim --circuit_name ghz --nqubits 28
```

## References

[1] Y. Suzuki, Y. Kawase, Y. Masumura, Y. Hiraga, M. Nakadai, J. Chen, K. M. Nakanishi, K. Mitarai, R. Imai, S. Tamiya, T. Yamamoto, T. Yan, T. Kawakubo, Y. O. Nakagawa, Y. Ibe, Y. Zhang, H. Yamashita, H. Yoshimura, A. Hayashi, and K. Fujii, “Qulacs: a fast and versatile quantum circuit simulator for research purpose,” Quantum, vol. 5, p. 559, Oct. 2021.

[2] A. Tabuchi, S. Imamura, M. Yamazaki, T. Honda, A. Kasagi, H. Nakao, N. Fukumoto, and K. Nakashima, “mpiQulacs: A scalable distributed quantum computer simulator for ARM-based clusters,” in Proceedings of the 2023 IEEE International Conference on Quantum Computing and Engineering (QCE), 2023, pp. 959–969.

[3] B. Zhang, B. Fang, F. Ye, L. Guo, F. Song, N. Tallent, and D. Tao, “BMQSim: Overcoming memory constraints in quantum circuit simulation with a high-fidelity compression framework,” in Proceedings of the 39th ACM International Conference on Supercomputing (ICS), 2025, pp. 689–704.

## Citation

If you use QompSim in your research or report results obtained with it, please cite the following paper:

> S. Imamura, H. Fujii, N. Aoki, A. Tabuchi, R. Kawano and A. Kasagi, "QompSim: A Distributed State-Vector Simulator with Lossy and All-Zero Compression", in Proceedings of the IEEE International Conference on Quantum Computing and Engineering (QCE), 2026.


## License

QompSim is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

This repository contains software derived from Qulacs, which is licensed under the MIT License.
See [THIRD_PARTY_LICENSES/Qulacs-LICENSE.txt](THIRD_PARTY_LICENSES/Qulacs-LICENSE.txt).

nvCOMP is proprietary NVIDIA software and is subject to NVIDIA's applicable license terms; it is not included in this repository.
