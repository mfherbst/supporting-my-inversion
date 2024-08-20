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

# Extracts from the jld files the density and potentials along reference paths

#
# ===========================
#

function is_reference(filename)
    bn, _ = splitext(basename(filename))
    startswith(bn, "reference_")
end

function is_inversion(filename)
    bn, _ = splitext(basename(filename))
    startswith(bn, "inversion_") || startswith(bn, "truncate")
end

function pathfile(filename)
    @assert isfile(filename)
    bn, _ = splitext(filename)
    bn * "_path.json"
end

function testsystem_path(basis; tol=1e-6)
    @assert basis.fft_size[1] == basis.fft_size[2] == basis.fft_size[3]
    sz = basis.fft_size[1]

    if all(in([:Si]), atomic_symbol.(basis.model.atoms))
        # Find point sitting between two bonds
        @assert basis.model.positions[1] == zeros(3)
        @assert basis.model.positions[2] == ones(3)/4
        δ, index = findmin(r -> norm(r - ones(3)/8), r_vectors(basis))
        @assert abs(δ) < tol
    else
        error("Can't figure out system")
    end

    start = index.I
    branch_001 = [mod1.(start           .+ (0, 0, i), sz) for i in 1:sz]
    branch_110 = [mod1.(branch_001[end] .+ (i, i, 0), sz) for i in 1:sz]
    branch_111 = [mod1.(branch_110[end] .- (i, i, i), sz) for i in 1:sz]

    indices = CartesianIndex.(append!([start], branch_001, branch_110, branch_111))
    path_r_vectors = r_vectors(basis)[indices]

    branch_starts = [(1, "001"), (1 + length(branch_001), "110"),
                     (1 + length(branch_001) + length(branch_110), "111")]
    atoms = [i for (i, rvec) in enumerate(path_r_vectors)
             if any(r -> norm(r - rvec) < tol, basis.model.positions)]
    atom_symbols = [only(atomic_symbol(basis.model.atoms[i_atom])
                         for (i_atom, pos) in enumerate(basis.model.positions)
                         if norm(pos - path_r_vectors[atoms[j]]) < tol)
                    for j in 1:length(atoms)]
    atom_symbols = string.(atom_symbols)

    pathlength = [0.0]
    for i in 2:length(indices)
        δr = DFTK.normalize_kpoint_coordinate(path_r_vectors[i-1] - path_r_vectors[i])
        push!(pathlength, pathlength[end] + norm(basis.model.lattice * δr))
    end
    @assert length(pathlength) == length(indices)

    (; indices, pathlength, branch_starts, atoms, atom_symbols)
end

function extract_paths_from_reference(filename)
    @assert isfile(filename)
    scfres = load_scfres(filename; skip_hamiltonian=false);
    path = testsystem_path(scfres.basis)

    data = Dict(
        "kind"          => "reference",
        "pathlength"    => path.pathlength,
        "branch_starts" => path.branch_starts,
        "atoms"         => path.atoms,
        "atom_symbols"  => path.atom_symbols,
    )

    erp = extract_reference_potential
    data["ρ"]   = scfres.ρ[path.indices]
    data["vxc"] = erp(InversionVxc(), scfres.ham, scfres.ρ)[path.indices]

    open(pathfile(filename), "w") do fp
        JSON3.write(fp, data)
    end
    nothing
end

function extract_paths_from_inversion(filename)
    @assert isfile(filename)
    scfres = load_scfres(filename; skip_hamiltonian=true);
    path   = testsystem_path(scfres.basis)
    invres = jldopen(filename) do jld
        ρref=jld["inversion_ρref"]
        (; vref=jld["inversion_vref"],
           vs=jld["inversion_vs"],
           ρs=jld["inversion_ρs"],
           εs=jld["inversion_εs"],
           ρref, ρorig=get(jld, "inversion_ρorig", ρref))
    end

    data = Dict(
        "kind"          => "inversion",
        "pathlength"    => path.pathlength,
        "branch_starts" => path.branch_starts,
        "atoms"         => path.atoms,
        "atom_symbols"  => path.atom_symbols,
        "εs"            => invres.εs,
        "vref"          => invres.vref[path.indices],
        "ρref"          => invres.ρref[path.indices],
        "ρorig"         => invres.ρorig[path.indices],
        #
        "vs"            => [v[path.indices] for v in invres.vs],
        "ρs"            => [ρ[path.indices] for ρ in invres.ρs],
    )

    open(pathfile(filename), "w") do fp
        JSON3.write(fp, data)
    end
    nothing
end

function extract_paths(filename)
    if is_reference(filename)
        extract_paths_from_reference(filename)
    elseif is_inversion(filename)
        extract_paths_from_inversion(filename)
    else
        error("Unknown file type")
    end
end

function main()
    # Go through all jld2 and extract paths

    files_to_process = filter(readdir()) do fn
        endswith(fn, ".jld2") || return false
        (is_inversion(fn) || is_reference(fn)) || return false
        isfile(pathfile(fn)) && return false
        true
    end
    !isempty(files_to_process) && println("Processing files: ")
    for fn in files_to_process
        println("   - ", fn)
        extract_paths(fn)
    end
end
