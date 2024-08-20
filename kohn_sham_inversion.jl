using DFTK: Mixing
using AtomsBase
using DFTK
using DoubleFloats
using GenericLinearAlgebra
using MPI
using Statistics
using LineSearches
include("term_dualmap.jl")

struct InversionVxc end   # Invert with -½Δ + Vext + Vₕ
function basis_inversion(::InversionVxc, system::AbstractSystem, ρref_generator, ε::Number, T;
                         kwargs_model, kwargs_basis)
    model = model_atomic(system;
                         model_name="InversionVxc",
                         extra_terms=[Hartree(), DualMap(ρref_generator, ε, T)],
                         kwargs_model...)
    PlaneWaveBasis(model; kwargs_basis...)
end
function extract_reference_potential(::InversionVxc, ham_reference::Hamiltonian, ρref)
    @assert ham_reference.basis.model.n_spin_components == 1
    i_term = only(i for (i, t) in enumerate(ham_reference.basis.model.term_types) if t isa Xc)
    pot = ham_reference[1].operators[i_term].potential
    reshape(pot, ham_reference.basis.fft_size..., 1)
end

function apply_J(basis, ρ, ε)
    # Apply dual map J without rescaling by mean
    term = TermDualMap(basis, zero(ρ), ε, Float64, zero_DC=false)
    ψ = nothing
    occupation = nothing
    (; ops) = DFTK.ene_ops(term, basis, ψ, occupation; ρ)

    @assert basis.model.n_spin_components == 1
    ops[1].potential
end


function kohn_sham_inversion(system::AbstractSystem, ρref_generator;
                             method=InversionVxc(),
                             kwargs_model, kwargs_basis,
                             δ=1e-2, εs=exp10.(0:-0.25:-3),
                             ψ=nothing, verbose=false, ρtol=0,
                             verbose_unperturbed_data=nothing,
                             maxiter=300, Vref=nothing,
                             Tinversion=Double64,
                             kwargs...)
    # TODO Notes:
    #      - Using Double64 was nice for testing, but experiments show
    #        that running in Float64 does not lead to too many additional issues
    #        (i.e. it leads to issues, but at around the same ε values as the
    #        limit of numerical accuracy in ρref is anyway reached)
    function DmConvergenceInversion(ρref, ε, δ)
        n_satisfied = 0
        n_unstable  = 0
        function callback(info)
            # Rationale is that the change is smaller than our error to the
            # ground state density so no point continuing the iterations ...
            #
            # Also we use the duality mapping to ensure we have a convergence
            # to δ in the potential.
            hm1_norm_ρdiff = norm_hm1(info.ham.basis, info.ρout - ρref)
            hm1_norm_Δρ    = norm_hm1(info.ham.basis, info.ρout - info.ρin)
            l2_norm_Δρ     = norm_l2(info.ham.basis,  info.ρout - info.ρin)
            is_satisfied   = hm1_norm_Δρ < hm1_norm_ρdiff && hm1_norm_Δρ < ε * δ
            is_unstable    = l2_norm_Δρ  < ρtol / ε

            is_satisfied  && (n_satisfied += 1)
            !is_satisfied && (n_satisfied  = 0)
            is_unstable   && (n_unstable  += 1)
            !is_unstable  && (n_unstable   = 0)

            n_satisfied = MPI.bcast(n_satisfied, 0, MPI.COMM_WORLD)
            n_unstable  = MPI.bcast(n_unstable,  0, MPI.COMM_WORLD)
            if mpi_master() && n_unstable > 1
                println("    ", "Stopping to avoid numerical instabilities.")
            end
            return n_satisfied > 1 || n_unstable > 1
        end
    end
    function KSinversionCallback(ρref, ε)
        function callback(info)
            if info.stage == :iterate
                hm1_norm_ρdiff = norm_hm1(info.ham.basis, info.ρout - ρref)
                hm1_norm_Δρ    = norm_hm1(info.ham.basis, info.ρout - info.ρin)

                if !isnothing(Vref)
                    error_V = extract_inverted_potential(info.ham) - Vref
                    error_V_h1 = norm_h1(info.ham.basis, error_V .- mean(error_V))
                end

                if mpi_master()
                    print("    ", "hm1_norm_ρdiff=$hm1_norm_ρdiff hm1_norm_Δρ=$hm1_norm_Δρ")
                    !isnothing(Vref) && print("  error_V_h1=$error_V_h1")
                    # !isnothing(verbose_unperturbed_data) && print("  Qε=$Qε")
                    if hasproperty(info, :optim_state)
                        print("  α = $(round(info.optim_state.alpha; sigdigits=3))")
                    end
                    println("")
                end
            end
            info
        end
    end

    hm1_norm_ρdiff = 0
    scfres = nothing
    ρs = []
    vs = []
    energies = []
    for ε in εs
        if mpi_master()
            println()
            println("# --- ε=$ε")
            println()
        end

        basis = basis_inversion(method, system, ρref_generator, ε, Tinversion;
                                kwargs_model, kwargs_basis)
        T = eltype(basis)
        ρref = T.(only(t for t in basis.terms if t isa TermDualMap).ρref)
        callback = (verbose ? KSinversionCallback(ρref, ε) ∘ DFTK.ScfDefaultCallback()
                            : DFTK.ScfDefaultCallback())

        # Strip extra bands if there are any
        n_bands = let
            filled_occ = DFTK.filled_occupation(basis.model)
            div(basis.model.n_electrons, basis.model.n_spin_components * filled_occ, RoundUp)
        end
        if size(ψ[1], 2) > n_bands
            ψ = [@view ψk[:, 1:n_bands] for ψk in ψ]
        end

        extra_args = (; )
        algorithm  = direct_minimization
        extra_args = (; linesearch=LineSearches.MoreThuente())
        scfres = algorithm(basis; ψ, callback, maxiter, kwargs..., extra_args...,
                           is_converged=DmConvergenceInversion(ρref, ε, δ))
        if !scfres.converged
            println("Trying again without preconditioner ...")
            scfres = direct_minimization(basis; scfres.ψ, callback, maxiter, kwargs..., extra_args...,
                                         is_converged=DmConvergenceInversion(ρref, ε, δ),
                                         prec_type=PreconditionerNone)
        end

        ψ = scfres.ψ
        push!(vs, extract_inverted_potential(scfres.ham))
        push!(ρs, scfres.ρ)
        push!(energies, DFTK.todict(scfres.energies))

        error_V_h1  = nothing
        hm1_norm_ρdiff = norm_hm1(basis, scfres.ρ - ρref)
        if !isnothing(Vref)
            error_V = vs[end] - Vref
            error_V_h1 = norm_h1(basis, error_V .- mean(error_V))
        end
        if !isnothing(verbose_unperturbed_data)
            εmatch, i_unpert = findmin(εi -> abs(εi - ε), verbose_unperturbed_data.εs)
            @assert εmatch < 1e-10
            unper_ρ = verbose_unperturbed_data.ρs[i_unpert]
            Δρ     = ρref - verbose_unperturbed_data.ρref
            Δρ_hm1 = norm_hm1(scfres.ham.basis, Δρ)
            Qε = norm_hm1(scfres.ham.basis, unper_ρ .- scfres.ρ) / Δρ_hm1
        end

        if mpi_master()
            println()
            print("#-- |ρε - ρref| = $hm1_norm_ρdiff vs. $(δ*ε) = δ*ε")
            !isnothing(Vref) && print("  error_V_h1=$error_V_h1")
            !isnothing(verbose_unperturbed_data) && print("  Qε=$Qε")
            println()
            println()
        end
    end

    (; scfres, energies, εs, vs, ρs)
end

function norm_sobolev(basis::PlaneWaveBasis, x, s::Real)
    if ndims(x) > 3
        x = total_density(x)
    end
    x_fourier = fft(basis, x)
    norm_weights = (1 .+ DFTK.norm2.(G_vectors_cart(basis))) .^ s
    sqrt(sum(abs, x_fourier .* x_fourier .* norm_weights))
end
norm_h1(basis, x)    = norm_sobolev(basis, x, 1)
norm_hm1(basis, x)   = norm_sobolev(basis, x, -1)
norm_l2(basis, x)    = norm(x) * sqrt(basis.dvol)
norm_linf(basis, x)  = maximum(abs, x)

function extract_inverted_potential(ham::Hamiltonian)
    model = ham.basis.model
    @assert model.n_spin_components == 1
    if startswith(model.model_name, "Inversion")
        i_dual = only(i for (i, t) in enumerate(model.term_types) if t isa DualMap)
        return ham[1].operators[i_dual].potential
    else
        error("No inversion hamiltonian !")
    end
end
