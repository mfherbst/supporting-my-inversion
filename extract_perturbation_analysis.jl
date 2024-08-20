#!/bin/sh
#=
julia -t1 --project -e "include(\"$0\"); main()"
exit $?
=#

using LinearAlgebra
using JLD2
using JSON3
using DFTK
include("kohn_sham_inversion.jl")

function is_inversion(filename)
    bn, _ = splitext(basename(filename))
    startswith(bn, "inversion_") || startswith(bn, "truncate")
end

function perturbfile(filename)
    @assert isfile(filename)
    bn, _ = splitext(filename)
    bn * "_perturb.json"
end

function compute_perturbation_analysis(basis, unperturbed_res, perturbed_res)
    data_length = min(length(unperturbed_res.εs), length(perturbed_res.εs))
    @assert unperturbed_res.εs[1:data_length] ≈ perturbed_res.εs[1:data_length]

    Δρ     = perturbed_res.ρref - unperturbed_res.ρref
    Δρ_hm1 = norm_hm1(basis, Δρ)

    Qεs = Float64[]
    Rεs = Float64[]
    Sεs = Float64[]
    for i in 1:data_length
        ρs_ref = unperturbed_res.ρs
        (; εs, ρs, vs ) = perturbed_res

        # Δv̅ = (ρ̅^ε - ρ̃̅^ε) / 4πε
        Δvmean =  (mean(ρs_ref[i]) - mean(ρs[i])) / (εs[i] * 4π)

        Δv = unperturbed_res.vs[i] .- vs[i] .+ Δvmean  # Δv = v^ε - v̅^ε + Δv̅
        ΔvmJ = Δv .- apply_J(basis, Δρ, εs[i])         # Δv - J(Δρ/ε)

        push!(Rεs, 4π * εs[i] * norm_h1(basis, Δv)     / Δρ_hm1)  # 4πε ⋅ ‖Δv‖_H^1 / ‖Δρ‖_H^-1
        push!(Sεs, 4π * εs[i] * norm_h1(basis, ΔvmJ)   / Δρ_hm1)  # 4πε ⋅ ‖Δv - J(Δρ/ε)‖_H^1 / ‖Δρ‖_H^-1
        push!(Qεs, norm_hm1(basis, ρs_ref[i] .- ρs[i]) / Δρ_hm1)  # ‖ρ^ε - ρ̃^ε‖_H^-1 / ‖Δρ‖_H^-1
    end

    (; Δρ_hm1, εs=unperturbed_res.εs[1:data_length], Qεs, Rεs, Sεs)
end
function compute_perturbation_analysis(unperturbed::AbstractString, perturbed::AbstractString)
    # unperturbed: File with the reference (unperturbed) inversion
    # perturbed:   File with the inversion

    @assert isfile(unperturbed)
    @assert is_inversion(unperturbed)
    ref_scfres = load_scfres(unperturbed; skip_hamiltonian=true);
    refres = jldopen(unperturbed) do jld
        (; vs=jld["inversion_vs"],
           ρs=jld["inversion_ρs"],
           εs=jld["inversion_εs"],
           ρref=jld["inversion_ρref"],)
    end

    @assert isfile(perturbed)
    @assert is_inversion(perturbed)
    perturbres = jldopen(perturbed) do jld
        (; vs=jld["inversion_vs"],
           ρs=jld["inversion_ρs"],
           εs=jld["inversion_εs"],
           ρref=jld["inversion_ρref"],)
    end
    compute_perturbation_analysis(ref_scfres.basis, refres, perturbres)
end

function extract_perturbation_analysis(unperturbed::AbstractString, perturbed::AbstractString)
    computed = compute_perturbation_analysis(unperturbed, perturbed)
    data = Dict(
        "Δρ_hm1" => computed.Δρ_hm1,
        "εs"     => computed.εs,
        "Qεs"    => computed.Qεs,
        "Rεs"    => computed.Rεs,
        "Sεs"    => computed.Sεs,
    )
    open(perturbfile(perturbed), "w") do fp
        JSON3.write(fp, data)
    end
    nothing
end

function main()
    #
    # Truncate on silicon Ecut 45 kgrid 10 Vxc
    #
    unperturbed_file = "inversion_silicon_Ecut_45_kgrid_10_upf_vxc.jld2"
    truncate_files = [
        "truncate$(trunc)_silicon_Ecut_45_kgrid_10_upf_vxc.jld2"
        for trunc in (10, 15, 20, 25, 30)
    ]
    for trunc in truncate_files
        extract_perturbation_analysis(unperturbed_file, trunc)
    end
end
