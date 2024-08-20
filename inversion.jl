using DFTK
using JLD2
using JSON3
include("kohn_sham_inversion.jl")

setup_threading(; n_blas=1)

struct Noop end
(::Noop)(ρ, ::Any) = ρ

struct TruncateBasis
    Ecut::Float64
end
function (t::TruncateBasis)(ρ::AbstractArray, basis::PlaneWaveBasis)
    @assert t.Ecut ≤ basis.Ecut
    # Maybe should use ensure_smallprimes=false here:
    fft_size_small = DFTK.compute_fft_size(basis.model, t.Ecut)
    mpi_master() && println("truncated grid size:" ,fft_size_small) 
    ρ_small = DFTK.interpolate_density(ρ, fft_size_small)
    DFTK.interpolate_density(ρ_small, basis.fft_size)
end

function run_exact_inversion(prefix, referencefile; kwargs...)
    run_inversion(prefix, referencefile, Noop(); kwargs...)
end

function run_inversion(prefix, referencefile, perturb_density;
                       method, logε=[0, -6], δ=1e-2, kwargs...)
    @assert logε[1] > logε[2]
    εs = exp10.(logε[1]:-0.25:logε[2])

    if mpi_master()
        println()
        DFTK.versioninfo()
        println()
    end
    DFTK.reset_timer!(DFTK.timer)

    scfres = load_scfres(referencefile; skip_hamiltonian=false)
    system = periodic_system(scfres.basis.model)

    kwargs_model = (; scfres.basis.model.temperature,
                      scfres.basis.model.smearing)
    kwargs_basis = (; scfres.basis.Ecut,
                      scfres.basis.kgrid,
                      scfres.basis.fft_size)

    ρorig = scfres.ρ
    ρref  = perturb_density(ρorig, scfres.basis)
    referror_ρ_l2  = norm_l2(scfres.basis,  ρref - ρorig)
    referror_ρ_hm1 = norm_hm1(scfres.basis, ρref - ρorig)
    if mpi_master()
        println("|ρref - ρorig|hm1 = ", referror_ρ_hm1)
        println("|ρref - ρorig|l2  = ", referror_ρ_l2)
    end

    vref = extract_reference_potential(method, scfres.ham, scfres.ρ)
    ρref_generator = basis -> ρref  # Always return ρref
    res = kohn_sham_inversion(system, ρref_generator; kwargs_model, kwargs_basis,
                              scfres.ψ, method, εs, δ, Vref=vref, kwargs...)

    #
    # Post-processing
    #

    # Remove mean from potentials to shift them to a uniform zero
    vref_m = vref .- mean(vref)
    vs_m   = [v .- mean(v) for v in res.vs]

    # Compute error norms
    norms_ρ_hm1    = [norm_hm1(scfres.basis, ρ  - ρref)    for ρ  in res.ρs]
    norms_ρ_l2     = [ norm_l2(scfres.basis, ρ  - ρref)    for ρ  in res.ρs]
    norms_v_h1     = [ norm_h1(scfres.basis, vm - vref_m)  for vm in vs_m]
    norms_v_l2     = [ norm_l2(scfres.basis, vm - vref_m)  for vm in vs_m]
    refnorm_v_l2   = norm_l2(scfres.basis,  vref_m)
    refnorm_v_h1   = norm_h1(scfres.basis,  vref_m)
    refnorm_ρ_l2   = norm_l2(scfres.basis,  ρref)
    refnorm_ρ_hm1  = norm_hm1(scfres.basis, ρref)

    if mpi_master()
        println("|v-vref|h1 = ", norms_v_h1[max(1, end-5):end])
        println("|v-vref|l2 = ", norms_v_l2[max(1, end-5):end])
    end

    inversion_type = "exact"
    if !(perturb_density isa Noop)
        inversion_type = string(perturb_density)
    end

    #
    # Storage
    #
    extra_data = Dict(
        "inversion_type"      => inversion_type,
        "inversion_reference" => referencefile,
        "inversion_method"    => string(method),
        "inversion_δ"         => δ,
        "inversion_εs"        => res.εs,
        "inversion_energies"  => res.energies,
        #
        "inversion_errors_ρ_hm1"  => norms_ρ_hm1,
        "inversion_errors_ρ_l2"   => norms_ρ_l2,
        "inversion_errors_v_h1"   => norms_v_h1,
        "inversion_errors_v_l2"   => norms_v_l2,
        "inversion_refnorm_ρ_l2"  => refnorm_ρ_l2,
        "inversion_refnorm_ρ_hm1" => refnorm_ρ_hm1,
        "inversion_refnorm_v_l2"  => refnorm_v_l2,
        "inversion_refnorm_v_h1"  => refnorm_v_h1,
        "inversion_referror_ρ_l2"  => referror_ρ_l2,
        "inversion_referror_ρ_hm1" => referror_ρ_hm1,
    )
    invres = merge(res.scfres, (; optim_res=nothing))
    save_scfres(prefix * ".json", invres; extra_data, save_ψ=false, save_ρ=false)

    extra_data["inversion_vs"]    = vs_m
    extra_data["inversion_ρs"]    = res.ρs
    extra_data["inversion_ρref"]  = ρref
    extra_data["inversion_ρorig"] = ρorig
    extra_data["inversion_vref"]  = vref_m

    # Remove things which cannot be serialised
    invres = merge(res.scfres, (; optim_res=nothing, scfres.basis))
    save_scfres(prefix * ".jld2", invres; extra_data, save_ψ=false, save_ρ=false)

    mpi_master() && println(DFTK.timer)
    nothing
end
