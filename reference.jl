using AtomsIO
using DFTK
using JLD2
using JSON3
using Unitful
using UnitfulAtomic

disable_threading()

function run_reference(system, prefix;
                       kspacing, Ecut, kwargs_model=(; ), ensure_div2=true,
                       start_with_temperature=false, kwargs...)
    if mpi_master()
        println()
        DFTK.versioninfo()
        println()
    end
    DFTK.reset_timer!(DFTK.timer)

    model = model_PBE(system; kwargs_model...)
    kgrid = kgrid_from_maximal_spacing(model, kspacing)
    if ensure_div2
        fft_size = nextprod.(Ref([2]), compute_fft_size(model, Ecut))
    else
        fft_size = nothing
    end
    basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size)
    if mpi_master()
        show(stdout, "text/plain", basis)
        println()
    end

    ψ = nothing
    ρ = guess_density(basis)
    if start_with_temperature
        model_init  = model_PBE(system;
                                temperature=1e-3, smearing=Smearing.Gaussian(),
                                kwargs_model...)
        basis_init  = PlaneWaveBasis(model_init; Ecut, kgrid, fft_size)
        scfres_init = self_consistent_field(basis_init; tol=1e-2, kwargs...)
        ψ = scfres_init.ψ
        ρ = scfres_init.ρ
    end

    scfres = self_consistent_field(basis; tol=1e-12, ψ, ρ, kwargs...)
    save_scfres(prefix * ".jld2", scfres; save_ψ=true,  save_ρ=true)
    save_scfres(prefix * ".json", scfres; save_ψ=false, save_ρ=false)

    mpi_master() && println(DFTK.timer)
    nothing
end
