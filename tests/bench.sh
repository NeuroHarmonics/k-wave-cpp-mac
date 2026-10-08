#!/bin/zsh
# Time one run of a kspaceFirstOrder binary on a benchmark input from make_bench_inputs.m.
#
# Usage: ./bench.sh <binary> <N> [threads]
#   e.g. ./bench.sh ../kspaceFirstOrder-OMP/kspaceFirstOrder-OMP 256 6
#   (run from the folder holding the bench_*.h5 files)
#
# Prints the total execution time and the time-stepping (compute) time reported by the binary.
# The thread option is only passed when given (the OpenMP binary uses it, a GPU binary may not).

bin=$1; n=$2; threads=$3
args=(-i bench_$n.h5 -o out_${n}_$$.h5)
[ -n "$threads" ] && args+=(-t $threads)

out=$($bin $args 2>&1)
rm -f out_${n}_$$.h5

total=$(echo "$out" | grep 'Total execution' | grep -oE '[0-9.]+s')
# Elapsed time lines are input loading, pre-processing, the progress table header, then time stepping
compute=$(echo "$out" | grep 'Elapsed time' | sed -n '4p' | grep -oE '[0-9.]+s')
echo "$bin N=$n t=${threads:-default} total=$total compute=$compute"
