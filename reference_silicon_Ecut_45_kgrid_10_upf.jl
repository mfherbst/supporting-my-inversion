#!/bin/sh
#=
BN=$(basename "$0" .jl)
mpiexecjl --project -np 16 julia --project -t1 $BN.jl | tee $BN.log
exit $?
=#
include("reference.jl")
using LazyArtifacts

Ecut      = 45
kspacing  = 0.12u"1/Å"
system    = load_system("Si.extxyz")
system    = attach_psp(system; Si=artifact"pd_nc_sr_pbe_stringent_0.4.1_upf/Si.upf")
prefix, _ = splitext(@__FILE__)
run_reference(system, prefix; kspacing, Ecut)
