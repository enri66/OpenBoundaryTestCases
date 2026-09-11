# ==================================================================
# Radiating Gaussian — the OBLIQUE-INCIDENCE open-boundary benchmark
#
# Issue CliMA/Oceananigans.jl#5229, test case 1; MOM6's `circle_obcs`.
#
# WHY THIS TEST EXISTS. The coastal Kelvin wave (validation/kelvin_wave.jl)
# meets its outflow boundary at exactly NORMAL incidence, so the tangential
# phase speed is ~0 and 1-D radiation is already optimal there — it cannot
# distinguish `NormalRadiation` from `ObliqueRadiation`. This case can: a
# free-surface bump released at the centre of a domain open on ALL FOUR
# SIDES radiates outward as concentric gravity waves, presenting every
# boundary with a full range of incidence angles and the corners with 45°.
# That is exactly where a boundary-normal-only scheme fails.
#
# SETUP. Flat bottom, single layer, **f = 0**. Rotation is deliberately
# switched off: with f ≠ 0 the bump would geostrophically adjust and leave a
# balanced vortex behind, and that residual would be indistinguishable from
# reflection in the metric below. At f = 0 *all* the energy must leave the
# domain, so whatever remains is reflection and nothing else.
#
# MOM6 initialises an elevated disk; we use a Gaussian (#5229 allows either).
# A disk's sharp edge rings, and the ringing would contaminate the metric.
#
# METRICS
#   1. Reflection coefficient  R = √(E_residual / E_initial), measured once
#      the outgoing ring has fully cleared the domain. Energy is the
#      shallow-water form E = ∫∫ ½(g η² + H(u²+v²)) dA. Single number,
#      rigorous, and the standard quantity MOM6/ROMS report for this case.
#   2. Angle-resolved residual — where the leftover energy sits, as a
#      function of azimuth. A boundary-normal-only scheme should leak most
#      near the corners (45°) and least at the edge midpoints (0°, 90°, …).
#      This is the diagnostic that should visibly improve with the oblique
#      term, and it makes a compelling figure for the PR.
#
# Run:  julia --project=. validation/radiating_gaussian.jl
# ==================================================================

include(joinpath(@__DIR__, "obc_output.jl"))

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalRadiation,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using Printf
using Statistics
using CairoMakie

# ---------------- parameters ----------------
const g = 9.80665
const H = 1000.0                  # m   flat bottom
const c = sqrt(g * H)             # ≈ 99 m s⁻¹
const A = 0.1                     # m   bump amplitude (A/H = 1e-4 — linear)
const σ = 100kilometers           # bump width
const L = 2000kilometers          # square domain
const N = 200                     # Δ = 10 km  (σ spans 10 cells)

# Corner-to-centre distance is L/√2 ≈ 1414 km, so the ring reaches the corners
# at ≈ 4 h and is fully out by ≈ 5 h. Run well past that.
const T_END = 8hours
const t_clear = 5hours            # metrics evaluated after this

@inline gaussian_bump(x, y) = A * exp(-((x - L/2)^2 + (y - L/2)^2) / (2σ^2))

"""
    radiating_gaussian_simulation(; baroclinic_scheme, closed, filename)

Bump released from rest at the centre of a square, flat-bottomed, non-rotating
domain. All four sides open (or all walls, if `closed = true` — the control
run that shows what "100% reflection" looks like).

`baroclinic_scheme` is the open-boundary scheme on the 3-D velocity — the piece
this project replaces. The barotropic conditions (Flather, Chapman) are
upstream's and are held fixed across comparisons.
"""
function radiating_gaussian_simulation(; baroclinic_scheme = NormalRadiation(inflow_timescale = 0,
                                                                             outflow_timescale = Inf),
                                         closed = false,
                                         Δt = 60.0,
                                         substeps = 30,
                                         stop_time = T_END,
                                         filename = "radiating_gaussian")

    grid = RectilinearGrid(CPU();
                           size = (N, N, 1),
                           x = (0, L), y = (0, L), z = (-H, 0),
                           halo = (7, 7, 7),
                           topology = (Bounded, Bounded, Bounded))

    boundary_conditions = if closed
        NamedTuple()                       # default impenetrable walls everywhere
    else
        # Nothing enters from outside: every exterior value is zero, so the
        # boundaries are pure radiation.
        u_bcs = FieldBoundaryConditions(
            west = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme),
            east = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme))

        v_bcs = FieldBoundaryConditions(
            south = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme),
            north = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme))

        U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
            west = GravityWaveRadiationBoundaryCondition((0, 0)),
            east = GravityWaveRadiationBoundaryCondition((0, 0)))

        V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
            south = GravityWaveRadiationBoundaryCondition((0, 0)),
            north = GravityWaveRadiationBoundaryCondition((0, 0)))

        η_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face());
            west  = SurfaceWaveRadiationBoundaryCondition(),
            east  = SurfaceWaveRadiationBoundaryCondition(),
            south = SurfaceWaveRadiationBoundaryCondition(),
            north = SurfaceWaveRadiationBoundaryCondition())

        (u = u_bcs, v = v_bcs, U = U_bcs, V = V_bcs, η = η_bcs)
    end

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps),
        coriolis = nothing,                 # f = 0 — see header
        momentum_advection = WENOVectorInvariant(),
        buoyancy = nothing,
        tracers = (),
        boundary_conditions)

    set!(model, η = (x, y, z) -> gaussian_bump(x, y))

    simulation = Simulation(model; Δt, stop_time)

    function progress(sim)
        η = sim.model.free_surface.displacement
        @printf("  %-9s  max|η| = %.5f m\n", prettytime(sim), maximum(abs, η))
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(1hour))

    η = model.free_surface.displacement
    u, v, w = model.velocities
    simulation.output_writers[:fields] = JLD2Writer(model, (; η, u, v);
        filename = outpath(filename * ".jld2"),
        schedule = TimeInterval(10minutes),
        overwrite_existing = true)

    return simulation
end

# ==================================================================
# metrics
# ==================================================================
"""
    radiating_gaussian_metrics(filename)

`(; reflection, E, times, azimuthal, ...)`.

`reflection = √(E_residual / E_initial)` with the residual taken as the mean
energy over the window after `t_clear`, by which time the outgoing ring has
left a perfectly transparent domain entirely.
"""
function radiating_gaussian_metrics(filename)
    ηts = FieldTimeSeries(outpath(filename * ".jld2"), "η")
    uts = FieldTimeSeries(outpath(filename * ".jld2"), "u")
    vts = FieldTimeSeries(outpath(filename * ".jld2"), "v")
    times = ηts.times
    grid = ηts.grid
    x = collect(xnodes(grid, Center()))
    y = collect(ynodes(grid, Center()))
    Δ = (x[2] - x[1]) * (y[2] - y[1])

    xc(a) = 0.5 .* (a[1:end-1, :] .+ a[2:end, :])
    yc(a) = 0.5 .* (a[:, 1:end-1] .+ a[:, 2:end])

    function energy(n)
        η = interior(ηts[n])[:, :, 1]
        u = let a = interior(uts[n])[:, :, 1]; size(a, 1) == length(x) + 1 ? xc(a) : a end
        v = let a = interior(vts[n])[:, :, 1]; size(a, 2) == length(y) + 1 ? yc(a) : a end
        return sum(@. 0.5 * (g * η^2 + H * (u^2 + v^2))) * Δ
    end

    E = [energy(n) for n in eachindex(times)]
    late = findall(t -> t ≥ t_clear, times)
    reflection = sqrt(mean(E[late]) / E[1])

    # --- reflection vs INCIDENCE ANGLE -------------------------------------
    # The physically meaningful curve: how much does the boundary reflect as a
    # function of the angle at which the wave strikes it?
    #
    # The ring is centred, so at any point P the wave travels along the radial
    # direction ĉ = (P − centre)/|P − centre|. For a cell in the band adjacent
    # to a given edge, the incidence angle is the angle between ĉ and that
    # edge's outward normal n̂: θ_inc = acos(ĉ·n̂). θ_inc = 0° is normal
    # incidence (edge midpoints), → 45° approaching the corners.
    #
    # Binning residual ENERGY DENSITY (per unit area) by θ_inc — the earlier
    # version binned raw summed energy by azimuth-from-centre, which is wrong
    # twice over: bins subtend different areas in a square domain (the
    # diagonal reaches √2 further than the axis, so those bins hold ~2× the
    # cells), and azimuth-from-centre is not the same thing as incidence angle
    # at a boundary.
    nband = 10                                   # cells in the near-boundary band
    nbins = 30
    ebin = zeros(nbins); cbin = zeros(Int, nbins)
    edges = ((:west,  (-1.0, 0.0), 1:nband,                  eachindex(y)),
             (:east,  ( 1.0, 0.0), (length(x)-nband+1):length(x), eachindex(y)),
             (:south, (0.0, -1.0), eachindex(x),             1:nband),
             (:north, (0.0,  1.0), eachindex(x),             (length(y)-nband+1):length(y)))
    for n in late
        η = interior(ηts[n])[:, :, 1]
        for (_, n̂, irange, jrange) in edges, i in irange, j in jrange
            dx, dy = x[i] - L/2, y[j] - L/2
            r = hypot(dx, dy)
            r < eps() && continue
            cosθ = clamp((dx * n̂[1] + dy * n̂[2]) / r, -1, 1)
            cosθ <= 0 && continue                # cell is not on this edge's outgoing side
            θ = acosd(cosθ)                      # 0° = normal incidence
            b = clamp(floor(Int, θ / 90 * nbins) + 1, 1, nbins)
            ebin[b] += 0.5 * g * η[i, j]^2
            cbin[b] += 1
        end
    end
    incidence = [c > 0 ? e / c : NaN for (e, c) in zip(ebin, cbin)]   # energy DENSITY
    θbins = range(0, 90, length = nbins + 1)[1:end-1] .+ (90 / nbins / 2)

    return (; reflection, E, times, incidence, θbins, x, y, ηts, late)
end

function plot_radiating_gaussian(m, filename, label)
    xk, yk = m.x ./ 1e3, m.y ./ 1e3
    fig = Figure(size = (1350, 900))

    # snapshots: initial, mid-transit, and residual
    picks = (1,
             argmin(abs.(m.times .- 2hours)),
             argmin(abs.(m.times .- 4hours)),
             length(m.times))
    for (col, n) in enumerate(picks)
        η = interior(m.ηts[n])[:, :, 1]
        rng = col == 1 ? A : (col == 4 ? 0.02A : 0.3A)
        ax = Axis(fig[1, col], title = @sprintf("t = %s", prettytime(m.times[n])),
                  aspect = DataAspect(), xlabel = "x (km)",
                  ylabel = col == 1 ? "y (km)" : "")
        heatmap!(ax, xk, yk, η, colormap = :balance, colorrange = (-rng, rng))
        text!(ax, 0.03, 0.94; text = @sprintf("±%.3g m", rng), space = :relative,
              fontsize = 11)
    end

    ax = Axis(fig[2, 1:2], title = "domain energy (normalised)", xlabel = "t (hours)",
              ylabel = "E / E₀", yscale = log10)
    lines!(ax, m.times ./ 3600, max.(m.E ./ m.E[1], 1e-12), linewidth = 2)
    vlines!(ax, [t_clear / 3600], color = (:gray, 0.7), linestyle = :dash,
            label = "metric window")
    axislegend(ax, position = :lb)

    ax = Axis(fig[2, 3:4],
              title = "residual energy density vs INCIDENCE ANGLE at the boundary",
              xlabel = "incidence angle θ (°)   —   0° = normal, 45° = corner",
              ylabel = "⟨½gη²⟩ near boundary")
    lines!(ax, m.θbins, m.incidence, linewidth = 2)
    vlines!(ax, [45.0], color = (:firebrick, 0.6), linestyle = :dash, label = "corners")
    axislegend(ax, position = :lt)

    Label(fig[0, 1:4], "Radiating Gaussian — $label", fontsize = 20, tellwidth = false)
    save(filename * ".png", fig)
    return fig
end

# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    @printf("c = %.1f m/s   L = %.0f km   Δ = %.0f km   σ = %.0f km (%.0f cells)\n",
            c, L/1e3, L/N/1e3, σ/1e3, σ/(L/N))
    @printf("ring reaches corners at ≈ %.1f h; metrics after %.0f h\n",
            (L/sqrt(2))/c/3600, t_clear/3600)

    # --- control: all walls. Nothing can leave, so R must be ≈ 1. ----------
    println("\n--- closed control (all walls) ---")
    sim = radiating_gaussian_simulation(closed = true, filename = "gaussian_closed")
    run!(sim)
    mc = radiating_gaussian_metrics("gaussian_closed")
    @printf("  closed-domain R = %.4f   (sanity check: should be ≈ 1)\n", mc.reflection)

    # --- baseline: existing NormalRadiation --------------------------------
    println("\n--- open, NormalRadiation (existing baseline) ---")
    sim = radiating_gaussian_simulation(filename = "gaussian_normal_radiation")
    run!(sim)
    m = radiating_gaussian_metrics("gaussian_normal_radiation")
    @printf("\n===== NormalRadiation (existing, baseline) =====\n")
    @printf("  reflection coefficient R = %.4f\n", m.reflection)
    @printf("  residual E/E₀            = %.3e\n", m.reflection^2)
    ok = filter(isfinite, m.incidence)
    @printf("  leakage at 45° / at 0°   = %.2f  (>1 ⇒ oblique incidence is worse)\n",
            m.incidence[argmin(abs.(m.θbins .- 45))] / m.incidence[1])
    @printf("  incidence peak / min     = %.2f\n", maximum(ok) / minimum(ok))
    plot_radiating_gaussian(m, "gaussian_normal_radiation", "NormalRadiation (baseline)")
    println("  saved gaussian_normal_radiation.png")
end
