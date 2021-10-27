# Utilities to be sourced in another script

if [ -z "$name" ] ; then
    name="test"
fi

if [[ -z "$EV_CNNSDK_HOME" || ! -d $EV_CNNSDK_HOME ]] ; then
    echo "EV_CNNSDK_HOME not correctly set in the environment"
    exit 1
fi

evgen=${EV_CNNSDK_HOME}/evgencnn/scripts/evgencnn
blobr=${EV_CNNSDK_HOME}/install/bin/blobreader

if [ ! -f $evgen ] ; then
    echo "evgencnn executable not found : $evgen"
    exit 1
fi

# Secret evgencnn options : wof & vof
export __SAVER=1
# More verbose evgencnn output
export show_classify=1

if [ -f "${name}.opts" ] ; then
    base_opts=$(<${name}.opts)
    base_opts=$(echo $base_opts)
    echo "Using ${name}.opts"
fi

function runcmd()
{
    echo "$@"
    eval "$@"
    if [ $? -ne 0 ] ; then
        echo "Failed to run command"
        exit 1
    fi
}

# Verification info loaded from bin files
function wr_bld_vof()
{
    local out=$1
    local dir=$(dirname ${out})
    if [ ! -d $dir ] ; then
        echo "Dir nto found : $dir"
        exit 2
    fi
    cat <<- EOF > $out
	#!/bin/bash -ex
	g++ -O2 -DWEIGHTS_IN_FILE -DVERIFY_IN_FILE \\
	  -I ./include \\
	  code/${name}_impl.cc \\
	  verify/${name}_verify.cc \\
	  -o ${name}.exe
	EOF
    chmod u+x $out
    echo "Created $out"
    pushd $dir
    runcmd ./$(basename ${out})
    popd
}

# Verification info loaded from bin files + dump intermediate values
function wr_bld_vof_dump()
{
    local out=$1
    cat <<- EOF > $out
	#!/bin/bash -ex
	g++ -O2 -DWEIGHTS_IN_FILE -DVERIFY_IN_FILE -DCNN_NATIVE_VERIFY \\
	  -I ./include -I \${EV_CNNSDK_HOME}/include \\
	  code/${name}_impl.cc \\
	  verify/${name}_verify.c \\
	  -o ${name}_dump.exe
	EOF
    chmod u+x $out
    echo "Created $out"
}

# Statically link all the input data (found in .c files).
# This leads to slow compilation and slow startup. Unfortunatelly it is
# the default mode of operation for the CNN tools.
function wr_bld_static()
{
    local out=$1
    cat <<- EOF > $out
	#!/bin/bash -ex
	# Argv 1 : the name of the C file containing the input image
	input_img=\${1:-no_input_specified}
	g++ -O2 -DCNN_NATIVE_VERIFY \\
	  -I ./include -I \${EV_CNNSDK_HOME}/include \\
	  code/${name}_impl.cc \\
	  weights/${name}_filters.c \\
	  verify/${name}_verify.c \\
	  \${input_img} \\
	  -o ${name}_static.exe
	EOF
    chmod u+x $out
    echo "Created $out"
}

