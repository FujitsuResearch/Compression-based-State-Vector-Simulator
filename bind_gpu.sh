#!/bin/bash

source env.sh

LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
echo "rank${OMPI_COMM_WORLD_RANK} on $(hostname): gpu=${LOCAL_RANK}"
export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
$@
