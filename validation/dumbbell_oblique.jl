# ==================================================================
# DUMBBELL, OBLIQUE INCIDENCE — the test ObliqueRadiation exists for
#
# In validation/dumbbell.jl the dipole travels due east and meets the
# east-facing cut at NORMAL incidence. There `ObliqueRadiation` came out
# WORSE than `NormalRadiation` (enstrophy retained 2.97 vs 1.75 relative to
# the truth), and that looks like a genuine property rather than a bug:
# Raymond & Kuo divide by |∇φ|² = (∂ₙφ)² + (∂_t̂φ)², so a large tangential
# gradient shrinks cₙ below the Orlanski value and slows the radiation. A
# dipole has enormous tangential structure — its two lobes are separated
# along the boundary — but if it arrives head-on there is no real tangential
# phase propagation to capture, so the oblique term manufactures a spurious
# cₜ and takes from cₙ for nothing.
#
# That is precisely the configuration in which oblique radiation should NOT
# help. This script runs the one in which it should: the same dipole, launched
# on a heading of θ so that it strikes the cut at θ off-normal.
#
# ⚠️ RESULT: THIS TEST CANNOT BE RUN IN THE DUMBBELL. Kept as the record of why.
#
# At θ = 25° from y₀ = y_c − 90 km the dipole's lower lobe hit the south shoulder
# at t ≈ 15 d, was shredded along the wall, and by t = 30 d only an isolated
# monopole was left — which on an f-plane does not propagate at all, so it sat at
# x ≈ 290 km spinning for the rest of the run. Nothing reached the cut, and the
# metric went null: NormalRadiation 0.961, ObliqueRadiation 0.974, and a CLOSED
# WALL 0.983. When a wall scores the same as an open boundary, nothing crossed.
#
# The cause is geometric, not a tuning failure. The dipole's realistic half-span
# is d·cosθ + ~1.6R ≈ 114 km, so it spans ≈ 230 km — WIDER than the 220 km
# throat. At θ = 0 it gets through only by being squeezed symmetrically (which is
# visibly what happens: a new, stronger dipole re-forms at the neck entrance).
# Any tilt makes that squeeze asymmetric and kills the downstream lobe. Even 15°
# leaves the flanks 14–21 km inside the land, so the parameters below are not a
# working configuration either — they are the arithmetic that proved the point.
#
# Shrinking the eddy does not rescue it: below R ≈ Rd the vortex interaction is
# screened over the deformation radius and the dipole stops propagating — the
# same trap that made the first f = 1e-4 attempt crawl at 0.067 m/s.
#
# So oblique incidence needs a geometry whose channel width is a free parameter.
# That is the neck-as-its-own-basin configuration (open at BOTH ends, exterior
# data from the full-domain run): the width is ours to choose, and the domain is
# small enough to sweep angles cheaply.
#
# Run:  julia -t 8 --project=. validation/dumbbell_oblique.jl
# ==================================================================

include(joinpath(@__DIR__, "dumbbell.jl"))

const θ_obl  = 15.0
const y₀_obl = y_c - 50kilometers

function main()
    dip = make_dipole(x₀ = x_eddy, y₀ = y₀_obl, θ = θ_obl)

    @printf("oblique dumbbell: dipole launched from (%.0f, %.0f) km on a %.0f° heading\n",
            x_eddy/1e3, y₀_obl/1e3, θ_obl)
    @printf("  at the cut (x = %.0f km) the core is at y ≈ %.0f km; throat is %.0f–%.0f km\n",
            x_c/1e3, (y₀_obl + (x_c - x_eddy)*tand(θ_obl))/1e3,
            (y_c - neck_half_width)/1e3, (y_c + neck_half_width)/1e3)
    @printf("  incidence on the east-facing cut = %.0f° off-normal\n\n", θ_obl)

    if !isfile(outpath("dumbbell_obl_truth.jld2"))
        println("--- TRUTH (oblique) ---")
        run!(dumbbell_simulation(dipole = dip, filename = "dumbbell_obl_truth"))
    else
        println("--- TRUTH (oblique): reusing dumbbell_obl_truth.jld2 ---")
    end

    ext = cut_exterior("dumbbell_obl_truth")

    cases = ("normalradiation"  => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
             "obliqueradiation" => ObliqueRadiation(inflow_timescale = 0, outflow_timescale = Inf))

    for (lab, scheme) in cases
        tag = "dumbbell_obl_west_$lab"
        isfile(outpath(tag * ".jld2")) && continue
        println("--- west / $lab (oblique) ---")
        run!(dumbbell_downscaled(side = :west, dipole = dip, scheme = scheme,
                                 exterior = ext, filename = tag))
    end
    if !isfile(outpath("dumbbell_obl_west_closed.jld2"))
        println("--- west / CLOSED (oblique control) ---")
        run!(dumbbell_downscaled(side = :west, dipole = dip, closed = true,
                                 filename = "dumbbell_obl_west_closed"))
    end

    # ---- did the eddy leave? ----
    Zt = subdomain_enstrophy("dumbbell_obl_truth"; xrange = (0.0, x_c))
    idx(ts, d) = argmin(abs.(ts .- d*86400))

    println("\n" * "="^74)
    println("OBLIQUE INCIDENCE ($(Int(θ_obl))° off-normal) — west half, the eddy must LEAVE")
    println("="^74)
    @printf("%-22s %10s %10s %10s %10s | %11s\n", "run", "t=20d", "t=35d", "t=45d", "t=55d", "end/truth")
    @printf("%-22s %10.3e %10.3e %10.3e %10.3e | %11s\n", "TRUTH (west half)",
            (Zt.Z[idx(Zt.times, d)] for d in (20, 35, 45, 55))..., "1.000")
    ratios = Dict{String,Float64}()
    for lab in ("normalradiation", "obliqueradiation", "closed")
        Z = subdomain_enstrophy("dumbbell_obl_west_$lab")
        ratios[lab] = Z.Z[end] / Zt.Z[end]
        @printf("%-22s %10.3e %10.3e %10.3e %10.3e | %11.3f\n", lab,
                (Z.Z[idx(Z.times, d)] for d in (20, 35, 45, 55))..., ratios[lab])
    end
    println("="^74)
    @printf("oblique vs normal: %+.1f%%   (negative ⇒ ObliqueRadiation lets more out — the win)\n",
            100 * (ratios["obliqueradiation"] - ratios["normalradiation"]) / ratios["normalradiation"])
    println("normal incidence, for comparison: NormalRadiation 1.746, ObliqueRadiation 2.970")

    # ---- figure ----
    fig = Figure(size = (1150, 430))
    ax = Axis(fig[1, 1], title = "Oblique incidence ($(Int(θ_obl))°) — west half",
              xlabel = "t (days)", ylabel = "∫ζ² dA  (m² s⁻²)")
    lines!(ax, Zt.times ./ 86400, Zt.Z, color = :black, linewidth = 3, label = "truth")
    for (lab, colr) in (("normalradiation", :dodgerblue), ("obliqueradiation", :crimson),
                        ("closed", :gray))
        Z = subdomain_enstrophy("dumbbell_obl_west_$lab")
        lines!(ax, Z.times ./ 86400, Z.Z, color = colr, linewidth = 2,
               linestyle = lab == "closed" ? :dash : :solid, label = lab)
    end
    axislegend(ax, position = :lb, framevisible = false, labelsize = 10)

    ζts = FieldTimeSeries(outpath("dumbbell_obl_truth.jld2"), "ζ")
    xg = collect(xnodes(ζts.grid, Face())); yg = collect(ynodes(ζts.grid, Face()))
    mk = land_mask(xg, yg)
    n = argmin(abs.(ζts.times .- 40*86400))
    ax2 = Axis(fig[1, 2], title = "truth at t = 40 d — path is $(Int(θ_obl))° off-normal",
               xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    heatmap!(ax2, xg ./ 1e3, yg ./ 1e3, Array(interior(ζts[n]))[:, :, 1] ./ f_db .* mk,
             colormap = :balance, colorrange = (-0.6, 0.6), nan_color = RGBAf(0.75, 0.72, 0.66, 1))
    vlines!(ax2, [x_c/1e3], color = :limegreen, linewidth = 2)
    lines!(ax2, [x_eddy/1e3, x_c/1e3],
           [y₀_obl/1e3, (y₀_obl + (x_c - x_eddy)*tand(θ_obl))/1e3],
           color = :black, linestyle = :dash, linewidth = 2)

    Label(fig[0, 1:2], "Dumbbell at oblique incidence — does ObliqueRadiation earn its keep?",
          fontsize = 18, tellwidth = false)
    save("dumbbell_oblique.png", fig)
    println("\nsaved dumbbell_oblique.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
