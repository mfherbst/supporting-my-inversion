using DFTK
using LinearAlgebra

"""
DualMap: Implements

1/(2ε) ∫ρ(x)[ρ-ρref](y) G(x-y) dx dy

where G(x, y) = exp(-|x-y|) / |x-y| (the Yukawa kernel)
and the integral running both times over the unit cell.

Note that the floating-point type of ρref sets the precision at which
the dual map term is computed.
"""
struct DualMap
    ρref_generator  # Reference density (on the same discretisation as basis)
    ε               # Regularisation parameter
    Tpotential      # Precision for computing the potential
end
DualMap(ρref_generator, ε) = DualMap(ρref_generator, ε, nothing)
(term::DualMap)(basis) = TermDualMap(basis, term.ρref_generator(basis),
                                     term.ε, term.Tpotential)
Base.show(io::IO, term::DualMap) = print(io, "DualMap(ε=$(term.ε))")

struct TermDualMap{Tref,Tarr,Tbfft} <: DFTK.TermNonlinear
    ρref_tot_fourier::Array{Complex{Tref}, 3}
    ε::Tref
    yukawa_coeffs::Array{Tref, 3}
    ρref::Tarr
    opBFFT::Tbfft
end
function TermDualMap(basis::PlaneWaveBasis{T}, ρref::AbstractArray, ε, Tpotential; zero_DC=true) where {T}
    @assert size(ρref) == (basis.fft_size..., basis.model.n_spin_components)
    Tref = something(Tpotential, T)
    if T == Tref
        opFFT  = basis.opFFT
        opBFFT = basis.opBFFT
    else
        opFFT, opBFFT = let
            dummy = similar(basis.G_vectors, complex(Tref), basis.fft_size)
            (ipFFT, opFFT, ipBFFT, opBFFT) = DFTK.build_fft_plans!(dummy)
            opFFT, opBFFT
        end
    end
    ρref_tot_fourier = Tref(basis.fft_normalization) .* (opFFT * complex.(Tref.(total_density(ρref))))
    yukawa_coeffs = 1 ./ 4Tref(π) ./Tref(ε) ./ (1 .+ DFTK.norm2.(G_vectors_cart(basis)))
    zero_DC && (yukawa_coeffs[1] = 0.0)
    TermDualMap(ρref_tot_fourier, Tref(ε), yukawa_coeffs, ρref, opBFFT)
end

DFTK.@timing "ene_ops: dualmap" function DFTK.ene_ops(term::TermDualMap{Tref}, basis::PlaneWaveBasis{T},
                                                      ψ, occupation; ρ, kwargs...) where {T,Tref}
    cTref = complex(Tref)
    ρ_fourier = cTref.(fft(basis, total_density(ρ)))
    ρtot_fourier = ρ_fourier - term.ρref_tot_fourier
    pot_fourier  = term.yukawa_coeffs .* ρtot_fourier

    E        = T(real(dot(pot_fourier, ρtot_fourier) / 2))
    pot_real = T.(real(basis.ifft_normalization * (term.opBFFT * pot_fourier)))

    ops = [DFTK.RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (; E, ops)
end


function apply_kernel(term::TermDualMap, basis::PlaneWaveBasis{T}, δρ::AbstractArray{Tδρ};
                      kwargs...) where {T, Tδρ}
    δV = zero(δρ)
    δρtot = total_density(δρ)
    # Note broadcast here: δV is 4D, and all its spin components get the same potential.
    δV .= T.(real(basis.ifft_normalization * (term.opBFFT * (term.yukawa_coeffs .* fft(basis, δρtot)))))
    δV
end
