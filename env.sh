#source $HOME/spack/share/spack/setup-env.sh
#spack load boost openmpi

WORKDIR=$HOME

source venv/bin/activate
export NVCOMP_ROOT=$WORKDIR/local/tools/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive
export SZ3_ROOT=$WORKDIR/local
export CMAKE_PREFIX_PATH=$WORKDIR/local
export LD_LIBRARY_PATH=$NVCOMP_ROOT:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$WORKDIR/local/lib:$LD_LIBRARY_PATH
