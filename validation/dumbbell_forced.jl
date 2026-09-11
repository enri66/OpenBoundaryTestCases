# ==================================================================
# DUMBBELL, CASE 2 — MOM6-style basin forcing: a baroclinic strait exchange
#
# Case 1 (validation/dumbbell.jl) sends ONE clean dipole across the boundary
# once. That is the controlled test — known initial condition, known heading,
# one coherent feature — and it is worth keeping for exactly that reason. But
# it never makes the boundary do the hard thing: handle inflow and outflow at
# the same time, indefinitely.
#
# MOM6's dumbbell forces the two reservoirs and lets them exchange through the
# neck. Of its two mechanisms, this uses the buoyancy one:
#
#   * `dumbbell_dynamic_forcing` applies an oscillating SLP see-saw between the
#     basins (DUMBBELL_SLP_AMP = 10000 Pa, and note the `deg_rad` factor makes
#     the default period 360 days, not 1). In a rotating basin that mostly
#     drives a geostrophic jet ALONG the neck: the volume actually exchanged
#     is set by how far the basin surfaces move, which here works out to
#     ~1e-4 m/s of through-flow. Not a useful driver for this test.
#
#   * `RESTOREBUOY` restores surface salinity toward different values in the
#     two reservoirs. That sets up a density contrast across the neck and hence
#     a BAROCLINIC EXCHANGE — light water one way near the surface, dense water
#     the other way beneath. This is the classic rotating-strait problem
#     (Gibraltar), it is robust in a rotating system, and it is baroclinically
#     unstable, so the neck fills with eddies rather than one tidy dipole.
#
# Why that matters here: the exchange is genuinely BAROCLINIC, so it lives in
# the vertical shear — the part of the solution the split-explicit barotropic
# corrector does NOT overwrite, and therefore the only part an oblique
# baroclinic radiation scheme can act on. It also puts a real tracer gradient
# across the boundary, which is what `TracerReservoir` was built for.
#
# Run:  julia -t 8 --project=. validation/dumbbell_forced.jl
# ==================================================================

include(joinpath(@__DIR__, "dumbbell.jl"))

# ---------------- the forcing ----------------
const Δb_basin = 2e-3            # buoyancy contrast between the basins [m s⁻²] (Δρ ≈ 0.2 kg m⁻³)
const τ_basin  = 5days           # restoring timescale
const b_front  = 60kilometers    # width of the target's transition across the neck
const mask_pad = 40kilometers    # restoring acts only outside the neck, plus this pad

# Restore only in the two basins, never inside the neck: the exchange there must
# be free to organise itself.
@inline basin_mask(x, y, z) =
    0.5 * (1 + tanh((abs(x - x_c) - (neck_half_length + mask_pad)) / (25kilometers)))

# West basin light, east basin dense.
@inline b_basin_target(x, y, z, t) =
    N_db^2 * z - (Δb_basin / 2) * tanh((x - x_c) / b_front)

const b_restoring = Relaxation(rate = 1 / τ_basin, mask = basin_mask, target = b_basin_target)

# Start from rest, uniformly stratified — everything that happens is forced.
const quiescent = (u = (x, y, z) -> 0.0,
                   v = (x, y, z) -> 0.0,
                   η = (x, y, z) -> 0.0,
                   b = (x, y, z) -> N_db^2 * z)

const T_FORCED = 80days

"""
    exchange_transport(filename)

Depth-resolved transport through the cut, split into the part moving east and the
part moving west. A working baroclinic exchange shows BOTH, simultaneously and
persistently — which is the thing case 1 never asked the boundary to do.
"""
function exchange_transport(filename)
    uts = FieldTimeSeries(outpath(filename * "_cut.jld2"), "u")
    times = uts.times
    Δz = H_db / Nz_db
    lf = i_cut - first(cut_slab) + 1
    east = Float64[]; west = Float64[]
    for n in eachindex(times)
        u = Array(interior(uts[n]))[lf, :, :]
        push!(east, sum(x -> x > 0 ? x : 0.0, u) * Δz * Δ_db)
        push!(west, sum(x -> x < 0 ? -x : 0.0, u) * Δz * Δ_db)
    end
    return (; times, east, west)
end

function main()
    @printf("MOM6-style basin forcing: Δb = %.1e m s⁻² across the neck, τ = %s\n",
            Δb_basin, prettytime(τ_basin))
    @printf("  ⇒ reduced gravity g' = %.1e, exchange speed scale √(g'H)/2 ≈ %.2f m/s\n",
            Δb_basin, sqrt(Δb_basin * H_db) / 2)
    @printf("  internal Rossby radius √(g'H)/f = %.0f km vs neck width %.0f km\n\n",
            sqrt(Δb_basin * H_db) / f_db / 1e3, 2neck_half_width / 1e3)

    if !isfile(outpath("dumbbell_frc_truth.jld2"))
        println("--- TRUTH (forced, full dumbbell) ---")
        run!(dumbbell_simulation(dipole = quiescent, forcing = (; b = b_restoring),
                                 stop_time = T_FORCED, filename = "dumbbell_frc_truth"))
    else
        println("--- TRUTH (forced): reusing dumbbell_frc_truth.jld2 ---")
    end

    ex = exchange_transport("dumbbell_frc_truth")
    for d in (10, 20, 40, 60, 80)
        n = argmin(abs.(ex.times .- d*86400))
        @printf("  t=%2d d   eastward %7.3f Sv   westward %7.3f Sv   net %+7.4f Sv\n",
                d, ex.east[n]/1e6, ex.west[n]/1e6, (ex.east[n]-ex.west[n])/1e6)
    end

    ext = cut_exterior("dumbbell_frc_truth")

    cases = ("normalradiation"  => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
             "obliqueradiation" => ObliqueRadiation(inflow_timescale = 0, outflow_timescale = Inf))

    # Both halves are forced and both start quiescent: with a sustained exchange
    # there is no "seeded" side any more, everything is driven.
    for side in (:west, :east)
        for (lab, scheme) in cases
            tag = "dumbbell_frc_$(side)_$lab"
            isfile(outpath(tag * ".jld2")) && continue
            println("--- $side / $lab (forced) ---")
            run!(dumbbell_downscaled(; side, dipole = quiescent, seed_dipole = false,
                                       forcing = (; b = b_restoring), scheme,
                                       exterior = ext, stop_time = T_FORCED, filename = tag))
        end
    end

    # ---- how well does each half reproduce the truth's exchange? ----
    println("\n" * "="^76)
    println("FORCED EXCHANGE — enstrophy in each half vs the truth's own half")
    println("="^76)
    idx(ts, d) = argmin(abs.(ts .- d*86400))
    for (side, xr) in ((:west, (0.0, x_c)), (:east, (x_c, Lx_db)))
        Zt = subdomain_enstrophy("dumbbell_frc_truth"; xrange = xr)
        @printf("\n%-24s %10s %10s %10s | %11s\n", "$side half", "t=30d", "t=55d", "t=80d", "end/truth")
        @printf("%-24s %10.3e %10.3e %10.3e | %11s\n", "TRUTH",
                (Zt.Z[idx(Zt.times, d)] for d in (30, 55, 80))..., "1.000")
        for lab in ("normalradiation", "obliqueradiation")
            Z = subdomain_enstrophy("dumbbell_frc_$(side)_$lab")
            @printf("%-24s %10.3e %10.3e %10.3e | %11.3f\n", lab,
                    (Z.Z[idx(Z.times, d)] for d in (30, 55, 80))..., Z.Z[end]/Zt.Z[end])
        end
    end
    println("\n" * "="^76)

    # ---- figures ----
    fig = Figure(size = (1200, 800))
    ax = Axis(fig[1, 1:2], title = "Exchange through the neck (truth) — a working strait shows both directions at once",
              xlabel = "t (days)", ylabel = "transport (Sv)")
    lines!(ax, ex.times ./ 86400, ex.east ./ 1e6, color = :firebrick, linewidth = 2, label = "eastward")
    lines!(ax, ex.times ./ 86400, ex.west ./ 1e6, color = :navy, linewidth = 2, label = "westward")
    lines!(ax, ex.times ./ 86400, (ex.east .- ex.west) ./ 1e6, color = :black,
           linestyle = :dash, linewidth = 2, label = "net")
    axislegend(ax, position = :rt, framevisible = false)

    ζts = FieldTimeSeries(outpath("dumbbell_frc_truth.jld2"), "ζ")
    xg = collect(xnodes(ζts.grid, Face())); yg = collect(ynodes(ζts.grid, Face()))
    mk = land_mask(xg, yg)
    for (col, d) in enumerate((30, 55, 80))
        n = argmin(abs.(ζts.times .- d*86400))
        local ax2 = Axis(fig[2, col], title = @sprintf("truth ζ/f, t = %d d", d),
                         xlabel = "x (km)", ylabel = col == 1 ? "y (km)" : "", aspect = DataAspect())
        heatmap!(ax2, xg ./ 1e3, yg ./ 1e3, Array(interior(ζts[n]))[:, :, 1] ./ f_db .* mk,
                 colormap = :balance, colorrange = (-0.4, 0.4), nan_color = RGBAf(0.75, 0.72, 0.66, 1))
        vlines!(ax2, [x_c/1e3], color = :limegreen, linewidth = 2)
    end
    Label(fig[0, 1:3], "Dumbbell case 2 — MOM6-style forced baroclinic exchange",
          fontsize = 18, tellwidth = false)
    save("dumbbell_forced.png", fig)
    println("saved dumbbell_forced.png")

    out = animate_handoff("dumbbell_frc_truth", "dumbbell_frc_west_normalradiation",
                          "dumbbell_frc_east_normalradiation";
                          label = "forced exchange, NormalRadiation",
                          outfile = "dumbbell_forced_handoff.mp4")
    println("saved ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
