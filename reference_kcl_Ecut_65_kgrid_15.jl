#!/bin/sh
#=
BN=$(basename "$0" .jl)
mpiexecjl --project -np 16 julia --project -t1 $BN.jl | tee $BN.log
exit $?
=#
include("reference.jl")
using LazyArtifacts

Ecut      = 65
kspacing  = 0.12u"1/Å"
system    = load_system("KCl.extxyz")
system    = attach_psp(system;  K=artifact"pd_nc_sr_pbe_standard_0.4.1_upf/K.upf",
                               Cl=artifact"pd_nc_sr_pbe_standard_0.4.1_upf/Cl.upf")
prefix, _ = splitext(@__FILE__)
run_reference(system, prefix; kspacing, Ecut)
