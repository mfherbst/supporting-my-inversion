using LinearAlgebra
using Plots
using LaTeXStrings
using JLD2
using JSON3
using DFTK
using Polynomials
include("term_dualmap.jl")

function setup_plots()
    # Setup environment for making automated plots
    ENV["GKS_ENCODING"] = "utf8"
    ENV["GKSwstype"]    = "100"
    ENV["PLOTS_TEST"]   = "true"

    gr()
    default(guidefontsize = 14,
            tickfontsize = 12,
            legendfontsize = 11,
            titlefontsize = 16,
            fontfamily = "Computer Modern")
end

function load_path(pathfile)
    @assert endswith(pathfile, "_path.json")
    open(JSON3.read, pathfile)
end

function add_path!(p, pathdata::AbstractDict)
    add_path!(p, pathdata["pathlength"], pathdata["branch_starts"],
              pathdata["atoms"], pathdata["atom_symbols"])
end
function add_path!(p, pathlength, branch_starts, atoms, atom_symbols)
    vline!(p, getindex.(Ref(pathlength), first.(branch_starts)), label="", c=:grey, ls=:dashdot)
    vline!(p, [pathlength[end]], label="", c=:grey, ls=:dashdot)
    vline!(p, getindex.(Ref(pathlength), atoms), label="", c=:grey, ls=:dash)

    # Manually setting path ticks for silicon
    ticks = getindex.(Ref(pathlength), first.(branch_starts))
    push!(ticks, pathlength[end])
    labels = [raw"$O$", raw"$(001)$", raw"$O'$", raw"$(110)$", raw"$O''$", raw"$(111)$", raw"$O$"]
    l = length(ticks)
    for i in range(1,l+1,step=2)
        insert!(ticks, i+1, (ticks[i]+ticks[i+1])/2)
    end

    j = 1
    for i in getindex.(Ref(pathlength), atoms)
        push!(ticks, i)
        push!(labels, atom_symbols[j])
        j += 1
    end
    xticks!(p, ticks, labels)

    p
end

function is_reference(filename)
    bn, _ = splitext(basename(filename))
    startswith(bn, "reference_")
end
function is_inversion(filename)
    bn, _ = splitext(basename(filename))
    startswith(bn, "inversion_") || startswith(bn, "truncate")
end

function load_convergence_data(basename; ε_last=0, relative_error=false)
    @assert is_inversion(basename)
    @assert isfile(basename * ".json")
    data = open(JSON3.read, basename * ".json", "r")
    εs = data["inversion_εs"]
    εmask = εs .≥ ε_last
    εs = εs[εmask]

    errors_ρ_hm1 = data["inversion_errors_ρ_hm1"][εmask]
    errors_ρ_l2  = data["inversion_errors_ρ_l2"][εmask]
    errors_v_h1  = data["inversion_errors_v_h1"][εmask]
    errors_v_l2  = data["inversion_errors_v_l2"][εmask]

    referror_ρ_hm1 = get(data, "inversion_referror_ρ_hm1", 0.0)
    refnorm_ρ_hm1  = data["inversion_refnorm_ρ_hm1"]
    refnorm_ρ_l2   = data["inversion_refnorm_ρ_l2"]
    refnorm_v_h1   = data["inversion_refnorm_v_h1"]
    refnorm_v_l2   = data["inversion_refnorm_v_l2"]

    norms_ρ_hm1 = errors_ρ_hm1
    norms_ρ_l2  = errors_ρ_l2
    norms_v_h1  = errors_v_h1
    norms_v_l2  = errors_v_l2
    if relative_error
        norms_ρ_hm1 /= refnorm_ρ_hm1
        norms_ρ_l2  /= refnorm_ρ_l2
        norms_v_h1  /= refnorm_v_h1
        norms_v_l2  /= refnorm_v_l2
    end

    (; εs, errors_ρ_hm1, errors_ρ_l2, errors_v_h1, errors_v_l2,
       refnorm_ρ_hm1, refnorm_ρ_l2, refnorm_v_h1, refnorm_v_l2,
       referror_ρ_hm1,
       norms_ρ_hm1, norms_ρ_l2, norms_v_h1, norms_v_l2
   )
end

#
# ===========================
#

function plot_potential(basename; ε_last=0, refkey="vxc", lims=(-Inf, Inf))
    @assert isfile(basename * "_path.json")
    path = load_path(basename * "_path.json")
    if path["kind"] == "inversion"
        εmask = path["εs"] .≥ ε_last
        data  = (; vs=path["vs"][εmask], εs=path["εs"][εmask])

        p = plot(ylabel=L"$v_{\textrm{xc}}(\mathbf{r})$")
        snapshots = [(0, 0.7), (1, 0.6), (2, 0.5), (4, 0.4)]
        length(data.εs) > 6 && push!(snapshots, (6, 0.2))
        length(data.εs) > 8 && push!(snapshots, (8, 0.1))
        for (i, α) in reverse(snapshots)
            expon  = floor(Int, log10(data.εs[end-i]))                       # Computing labels for each  
            prefac = round(10^(log10(data.εs[end-i]) - expon); digits = 1)   # 
            label =  LaTeXString(raw"$\varepsilon=" * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
            plot!(p, path["pathlength"], data.vs[end-i]; c=cgrad(:speed)[α], label)
        end
        plot!(p, path["pathlength"], path["vref"]; label=L"$v_\textrm{xc}$", c=:black, ls=:dash)
    elseif path["kind"] == "reference"
        p = plot(path["pathlength"], path[refkey]; label=L"$v_\textrm{xc}$", c=:black)
    else
        error("Unknown kind")
    end
    add_path!(p, path)
    ylims!(p, lims...)
end

function plot_potential_error(basename;
                              relative_error=false, ε_last=0, errorlims=nothing)
    if relative_error
        errorlims = something(errorlims, (-1, 1))
    else
        errorlims = something(errorlims, (-Inf, Inf))
    end

    @assert isfile(basename * "_path.json")
    path = load_path(basename * "_path.json")
    @assert path["kind"] == "inversion"
    εmask = path["εs"] .≥ ε_last
    data = (; vref=path["vref"], vs=path["vs"][εmask], εs=path["εs"][εmask])
    v_error = [abs.(v - data.vref) for v in data.vs]
    if relative_error
        v_error = [abs.(v ./ data.vref) for v in v_error]
    end

    ylabel = relative_error ? "Relative error" : "Absolute pointwise error"
    p = plot(; ylabel)
    snapshots = [(0, 0.7), (1, 0.6), (2, 0.5), (4, 0.4)]
    length(data.εs) > 6 && push!(snapshots, (6, 0.2))
    length(data.εs) > 8 && push!(snapshots, (8, 0.1))
    for (i, α) in reverse(snapshots)
        label = "ε = $(round(data.εs[end-i]; sigdigits=2))"
        plot!(p, path["pathlength"], v_error[end-i]; c=cgrad(:speed)[α], label)
    end
    add_path!(p, path)
    ylims!(p, errorlims...)
end

function plot_density(basename)
    @assert isfile(basename * "_path.json")
    data = load_path(basename * "_path.json")
    if data["kind"] == "inversion"
        p = plot(data["pathlength"], data["ρref"];  label="ρref",  lw=1.5, c=1)
        plot!(p, data["pathlength"], data["ρorig"]; label="ρorig", lw=2.0, ls=:dot, c=2)
    elseif data["kind"] == "reference"
        p = plot(data["pathlength"], data["ρ"]; label="ρ", lw=1.5, c=1)
    else
        error("Unknown kind")
    end
    add_path!(p, data)
end

function plot_convergence(basename;
                          relative_error=false, ε_last=0, errorlims=nothing)
    if relative_error
        errorlims = something(errorlims, (-1, 1))
    else
        errorlims = something(errorlims, (1e-6, 15))
    end
    data = load_convergence_data(basename; ε_last, relative_error)

    common = (; mark=:x, lw=1.5)
    ylabel = relative_error ? "relative error" : "absolute error"
    p = plot(; xaxis=:log, yaxis=:log, xflip=true, xlabel=L"ε", ylabel, legend=:bottomleft)
    plot!(p, data.εs, data.norms_ρ_hm1; common..., c=1,
          label=L"‖ρ - ρ_{\textrm{ref}}‖_{H-1}")
    plot!(p, data.εs, data.norms_ρ_l2;  common..., c=1, ls=:dot,
          label=L"‖ρ - ρ_{\textrm{ref}}‖_{L^2}")
    plot!(p, data.εs, data.norms_v_h1;  common..., c=2,
          label=L"‖V - V_{\textrm{ref}}‖_{H^1}")
    plot!(p, data.εs, data.norms_v_l2;  common..., c=2, ls=:dot,
          label=L"‖V - V_{\textrm{ref}}‖_{L^2}")

    slope_1    = data.εs        .* data.norms_ρ_hm1[floor(Int, end/2)] ./ data.εs[floor(Int, end/2)]
    slope_sqrt = sqrt.(data.εs) .* data.norms_v_h1[floor(Int, end/2)]  ./ sqrt(data.εs[floor(Int, end/2)])
    plot!(p, data.εs, slope_1;    c=3, label=L"ε", ls=:dot, lw=2)
    plot!(p, data.εs, slope_sqrt; c=4, label=L"√{ε}", ls=:dot, lw=2)
    if data.referror_ρ_hm1 > 1e-12
        hline!(p, [data.referror_ρ_hm1]; label=L"‖ρ_{\textrm{ref}} - ρ_{\textrm{orig}}‖_{H-1}", ls=:dot, c=5, lw=2)
        vline!(p, [data.referror_ρ_hm1]; label="", ls=:dot, c=5, lw=2)
    end

    ylims!(p, errorlims...)
end

function plot_convergence_comparison(basenames::AbstractVector;
                                     labels=basenames,
                                     marks=fill(:x, length(basenames)),
                                     shapes=fill(:solid, length(basenames)),
                                     colors=collect(1:length(basenames)),
                                     relative_error=false,      # Plot relative error
                                     normalise_to_start=false,  # Divide by initial error
                                     ε_last=0, quantity=:norms_v_h1)
    ε_range = (Inf, -Inf)
    datas = map(bn -> load_convergence_data(bn; ε_last, relative_error), basenames)

    ylabel = string(quantity)
    if quantity == :norms_v_h1
        ylabel = L"‖V - V_{\textrm{ref}}‖_{H^1}"
    end

    p = plot(; xaxis=:log, yaxis=:log, xflip=true, xlabel=L"ε", ylabel, legend=:bottomleft)
    for (i, data) in enumerate(datas)
        label = labels[i]
        mark  = marks[i]
        ls    = shapes[i]
        c     = colors[i]

        ys = getproperty(data, quantity)
        if normalise_to_start
            ys ./= ys[1]
        end
        plot!(p, data.εs, ys; mark, lw=1.5, ls, label, c)

        ε_range = (min(ε_range[1], minimum(data.εs)), max(ε_range[2], maximum(data.εs)))
    end

    ε_range = ceil.(Int, log10.(ε_range))
    xticks!(p, 10.0 .^ (ε_range[1]:1:ε_range[2]))

    p
end

function plot_perturbation_analysis(basenames::AbstractVector{<:AbstractString};
                                    labels=basenames,
                                    colors=collect(1:length(basenames)))
    @assert length(labels) == length(basenames)
    datas = map(basenames) do bn
        @assert isfile(bn * "_perturb.json")
        open(JSON3.read, bn * "_perturb.json", "r")
    end

    Qlabel = L"$Q_\varepsilon(\Delta \rho)$"
    Rlabel = L"$R_\varepsilon(\Delta \rho)$"
    Slabel = L"$S_\varepsilon(\Delta \rho)$"

    p_Q = plot(; yaxis=:log, xaxis=:log, xflip=true, legend=:bottomright, 
                xlabel=L"$\varepsilon$", ylabel=Qlabel)
    p_R = plot(; xaxis=:log, xflip=true, legend=:bottomleft, 
                xlabel=L"$\varepsilon$", ylabel=Rlabel)
    p_S = plot(; xaxis=:log, xflip=true, legend=:topleft, 
                xlabel= L"$\varepsilon$", ylabel=Slabel)

    for (i, data) in enumerate(datas)
        plot!(p_R, data.εs, data.Rεs; label=labels[i], color=colors[i], mark=:x)
        plot!(p_S, data.εs, data.Sεs; label=labels[i], color=colors[i], mark=:x)

        expon  = floor(Int, log10(data.Δρ_hm1))                       # Computing labels for each  
        prefac = round(10^(log10(data.Δρ_hm1) - expon); digits = 1)   # Δρ using the H^-1 norm 
        labels[i] = LaTeXString(labels[i] * raw", $‖Δρ‖ = " * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")

        plot!(p_Q, data.εs, data.Qεs; label=labels[i], color=colors[i], mark=:x)
    end

    (; p_Q, p_R, p_S)
end

#
# ===========================
#

function main()
    setup_plots()

    ε_last = 7e-8
    # Potential error in exact inversion
    p_rel = plot_potential_error("inversion_silicon_Ecut_45_kgrid_10_upf_vxc";
                                 ε_last, relative_error=true, errorlims=(2e-6, 1))
    savefig(p_rel, "inversion_silicon_Ecut_45_kgrid_10_upf_vxc_pot_relerror.pdf")

    p_abs = plot_potential_error("inversion_silicon_Ecut_45_kgrid_10_upf_vxc";
                                 ε_last, relative_error=false)
    savefig(p_abs, "inversion_silicon_Ecut_45_kgrid_10_upf_vxc_pot_abserror.pdf")

    # Potential plot in exact inversion
    p_pot = plot_potential("inversion_silicon_Ecut_45_kgrid_10_upf_vxc"; ε_last)
    savefig(p_pot, "inversion_silicon_Ecut_45_kgrid_10_upf_vxc_pot.pdf")

    # Potential comparison composite
    plot!(p_pot; xaxis=false, bottom_margin=-52*Plots.PlotMeasures.px, legend_column =4, legend=(-0.07, 1.18),
                background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(p_rel; legend=false, yaxis=:log, yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], 
                top_margin=35 * Plots.PlotMeasures.px)
    p = plot(p_pot, p_rel, layout=(2,1), top_margin=30 * Plots.PlotMeasures.px)
    savefig(p, "inversion_silicon_Ecut_45_kgrid_10_upf_vxc_pot_composite.pdf")

    # XC convergence with truncation
    p = plot_convergence_comparison([
            "inversion_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate30_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate25_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate20_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate15_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate10_silicon_Ecut_45_kgrid_10_upf_vxc"
        ]; labels=[
            "Ecut = 45",
            "Ecut = 30",
            "Ecut = 25",
            "Ecut = 20",
            "Ecut = 15",
            "Ecut = 10",
        ], ε_last=1e-8)
    ylims!(p, 0.03, 10)
    savefig(p, "inversion_silicon_Ecut_45_kgrid_10_upf_vxc_truncate.pdf")

    p_Q, p_R, p_S = plot_perturbation_analysis([
            "truncate30_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate25_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate20_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate15_silicon_Ecut_45_kgrid_10_upf_vxc",
            "truncate10_silicon_Ecut_45_kgrid_10_upf_vxc"
       ]; labels=[
            raw"$E_{\mathrm{cut}} = 30$",
            raw"$E_{\mathrm{cut}} = 25$",
            raw"$E_{\mathrm{cut}} = 20$",
            raw"$E_{\mathrm{cut}} = 15$",
            raw"$E_{\mathrm{cut}} = 10$"])
    xtick = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,1e-1,1e-0]
    plot!(p_Q, xticks=xtick, yticks=[1e-3, 1e-2, 1e-1, 1e0],
          size=(600,350), foreground_color_legend = nothing)
    savefig(p_Q, "inversion_Ecut_45_kgrid_10_upf_vxc_Qε.pdf")

    plot!(p_S; xaxis=false, xlabel="", bottom_margin=-35 * Plots.PlotMeasures.px, legend=false)
    plot!(p_R; foreground_color_legend = nothing)
    p = plot!(p_S, p_R, layout=(2,1))
    xticks!(p, xtick)
    savefig(p, "inversion_Ecut_45_kgrid_10_upf_vxc_SεRε.pdf")
end
