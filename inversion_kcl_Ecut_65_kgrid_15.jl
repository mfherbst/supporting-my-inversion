#!/bin/sh
#=
BN=$(basename "$0" .jl)
julia --project -t4 $BN.jl | tee $BN.log
exit $?
=#

include("inversion.jl")
reference = "reference_kcl_Ecut_65_kgrid_15.jld2"
prefix, _ = splitext(@__FILE__)
run_exact_inversion(prefix, reference; verbose=true,
                    δ=1e-2, method=InversionVxc(), logε=(0, -8), ρtol=5e-13)
