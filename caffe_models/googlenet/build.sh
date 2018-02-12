#!/bin/bash

if [ $# -ge 1 ] ; then
    name=$1
    shift
else
    name=gnet_mean
fi

source ../evgen_util.bash

comm_opts="$base_opts --name ${name} --id 0xcaffe"

runcmd $evgen $comm_opts --generator host_fixed --outdir gen_${name} "$@"
wr_bld_vof      gen_${name}/host_fixed/build.sh
wr_bld_vof_dump gen_${name}/host_fixed/build_dump.sh
