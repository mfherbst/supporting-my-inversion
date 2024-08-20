#!/bin/sh
#=
BN=$(basename "$0" .jl)
julia --project -t4 $BN.jl | tee $BN.log
exit $?
=#

include("inversion.jl")
reference = "reference_silicon_Ecut_45_kgrid_10_upf.jld2"
prefix, _ = splitext(@__FILE__)

unperturbed = "inversion_silicon_Ecut_45_kgrid_10_upf_vxc.jld2"
verbose_unperturbed_data = jldopen(unperturbed) do jld
    (; vs=jld["inversion_vs"], ρs=jld["inversion_ρs"], εs=jld["inversion_εs"], ρref=jld["inversion_ρref"], )
end
run_inversion(prefix, reference, TruncateBasis(30);
              verbose=true, δ=1e-2, method=InversionVxc(), logε=(0, -7), verbose_unperturbed_data)
