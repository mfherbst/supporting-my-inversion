#!/bin/sh
#=
BN=$(basename "$0" .jl)
mpiexecjl --project -np 16 julia --project -t1 $BN.jl | tee $BN.log
exit $?
=#
include("reference.jl")
using LazyArtifacts

Ecut      = 83
kspacing  = 0.12u"1/Å"
system    = load_system("GaAs.extxyz")
system    = attach_psp(system; Ga=artifact"pd_nc_sr_pbe_standard_0.4.1_upf/Ga.upf",
                               As=artifact"pd_nc_sr_pbe_standard_0.4.1_upf/As.upf")
prefix, _ = splitext(@__FILE__)
run_reference(system, prefix; kspacing, Ecut)
