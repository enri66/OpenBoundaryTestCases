# ==================================================================
# Coastal Kelvin wave — open-boundary validation with an EXACT solution
#
# Issue CliMA/Oceananigans.jl#5229, test case 3. The gold-standard
# barotropic OBC test: Flather at the inflow, radiation at the outflow,
# on a rotating domain, with an analytic solution to compare against
# everywhere (cf. Bishnu et al. 2024, JAMES).
#
# GEOMETRY. A channel on an f-plane. Solid wall ("the coast") at y = 0,
# fluid at y > 0, second wall at y = Ly (far enough offshore that the
# wave has decayed to nothing). Open boundaries at x = 0 (inflow) and
# x = Lx (outflow).
#
# EXACT SOLUTION. With f > 0 the Kelvin wave travels with the coast on
# its right, i.e. in +x, trapped within a Rossby radius of the wall:
#
#   c = √(gH)          gravity-wave speed (Kelvin waves are non-dispersive)
#   R = c / f          Rossby radius of deformation
#   ω = c k            dispersion relation
#
#   η(x,y,t) = A exp(−y/R) cos(kx − ωt)
#   u(x,y,t) = (g/c) η(x,y,t)          along-channel
#   v(x,y,t) = 0                        exactly — that is the defining property
#
# (Check: cross-shore geostrophy f u = −g ∂η/∂y holds identically, since
#  ∂η/∂y = −η/R and (g/R) = g f / c.)
#
# WHAT IS BEING TESTED. The wave is injected at the west boundary from
# the analytic solution and must leave through the east boundary without
# reflecting. Reflection contaminates the interior with a standing wave,
# which both metrics below detect.
#
# METRICS
#   1. rmse(η_model − η_exact) / A over the interior — overall fidelity.
#   2. Reflection coefficient from the along-coast amplitude envelope.
#      A perfect radiation condition leaves max|η|(x) flat along the
#      coast; a reflected wave interferes with the incident one to make
#      a standing wave, so R ≈ (A_max − A_min) / (A_max + A_min).
#
# The baroclinic scheme is a PARAMETER (`baroclinic_scheme`) so this
# harness measures `NormalRadiation` today and `ObliqueRadiation` later
# on identical numbers. The barotropic conditions (Flather = Gravity-
# WaveRadiation, Chapman = SurfaceWaveRadiation) are upstream's and are
# NOT what we are changing — they are held fixed across comparisons.
#
# Run:  julia --project=. validation/kelvin_wave.jl
# ==================================================================

include(joinpath(@__DIR__, "obc_output.jl"))

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: ynode
using Oceananigans.BoundaryConditions: NormalRadiation,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using Printf
using Statistics
using CairoMakie

# ---------------- physical parameters ----------------
const g  = 9.80665
const H  = 1000.0            # m    flat bottom
const f₀ = 2e-4              # s⁻¹  f-plane
const A  = 0.1               # m    wave amplitude — A/H = 1e-4, safely linear
const T_wave = 6hours        # forcing period

const c  = sqrt(g * H)       # ≈ 99 m s⁻¹
const R  = c / f₀            # ≈ 495 km
const ω  = 2π / T_wave
const k  = ω / c
const λ  = 2π / k            # ≈ 2140 km

# ---------------- domain ----------------
const Lx = 5000kilometers    # ≈ 2.3 wavelengths
const Ly = 2000kilometers    # ≈ 4 Rossby radii — wave has decayed at the far wall
const Nx = 200               # Δx = 25 km  (~85 points per wavelength)
const Ny = 80                # Δy = 25 km  (~20 points per Rossby radius)
const Nz = 1                 # single layer — a genuinely barotropic test

# ---------------- the exact solution ----------------
@inline η_exact(x, y, t) = A * exp(-y / R) * cos(k * x - ω * t)
@inline u_exact(x, y, t) = (g / c) * η_exact(x, y, t)
@inline U_exact(x, y, t) = H * u_exact(x, y, t)     # barotropic transport

"""
    kelvin_wave_simulation(; baroclinic_scheme, stop_time, substeps, Δt, filename)

Build the channel, wire the open boundaries, and return the `Simulation`.

`baroclinic_scheme` is the open-boundary scheme applied to the 3-D velocity
`u` — the piece this project is replacing. Everything else is held fixed.
"""
function kelvin_wave_simulation(; baroclinic_scheme = NormalRadiation(inflow_timescale = 0,
                                                                     outflow_timescale = Inf),
                                  stop_time = 3 * Lx / c,     # three domain transits
                                  substeps = 60,
                                  Δt = 120.0,
                                  filename = "kelvin_wave")

    grid = RectilinearGrid(CPU();
                           size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (-H, 0),
                           halo = (7, 7, 7),
                           topology = (Bounded, Bounded, Bounded))

    # --- baroclinic (3-D) normal velocity -----------------------------------
    # West: the analytic wave enters. East: nothing enters; the interior
    # signal must radiate out. Continuous form on an x-boundary is f(y, z, t).
    @inline u_west(y, z, t) = u_exact(0,  y, t)
    @inline u_east(y, z, t) = 0.0

    u_bcs = FieldBoundaryConditions(
        west = NormalFlowBoundaryCondition(u_west; scheme = baroclinic_scheme),
        east = NormalFlowBoundaryCondition(u_east; scheme = baroclinic_scheme))

    # --- barotropic transport: Flather (upstream, held fixed) ---------------
    # Discrete form: the callback receives (j, k, grid, clock, model_fields)
    # and must return the 2-tuple (Uᵉˣᵗ, ηᵉˣᵗ).
    @inline function U_west_ext(j, k, grid, clock, model_fields)
        y = ynode(j, grid, Center())
        t = clock.time
        return (U_exact(0, y, t), η_exact(0, y, t))
    end
    @inline U_east_ext(j, k, grid, clock, model_fields) = (0.0, 0.0)

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
        west = GravityWaveRadiationBoundaryCondition(U_west_ext; discrete_form = true),
        east = GravityWaveRadiationBoundaryCondition(U_east_ext; discrete_form = true))

    # --- free surface: Chapman (upstream, held fixed) -----------------------
    η_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face());
        west = SurfaceWaveRadiationBoundaryCondition(),
        east = SurfaceWaveRadiationBoundaryCondition())

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps),
        coriolis = FPlane(f = f₀),
        momentum_advection = WENOVectorInvariant(),
        buoyancy = nothing,
        tracers = (),
        boundary_conditions = (u = u_bcs, U = U_bcs, η = η_bcs))

    # start from rest: the wave arrives entirely through the west boundary
    simulation = Simulation(model; Δt, stop_time)

    wall = Ref(time())
    function progress(sim)
        η = sim.model.free_surface.displacement
        u, v, w = sim.model.velocities
        @printf("  %-9s  max|η| = %.4f m   max|v| = %.2e m/s   (%.1f s)\n",
                prettytime(sim), maximum(abs, η), maximum(abs, v), time() - wall[])
        wall[] = time()
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(T_wave))

    η = model.free_surface.displacement
    u, v, w = model.velocities
    simulation.output_writers[:fields] = JLD2Writer(model, (; η, u, v);
        filename = outpath(filename * ".jld2"),
        schedule = TimeInterval(T_wave / 24),      # 24 samples per wave period
        overwrite_existing = true)

    return simulation
end

# ==================================================================
# metrics
# ==================================================================
"""
    kelvin_wave_metrics(filename)

Return `(; rmse_rel, reflection, envelope, x, times)` from a completed run.

Both metrics are evaluated over the LAST FULL WAVE PERIOD, by which time the
incident wave has crossed the domain several times and any reflection has had
time to establish a standing pattern.
"""
function kelvin_wave_metrics(filename)
    ηts = FieldTimeSeries(outpath(filename * ".jld2"), "η")
    times = ηts.times
    grid = ηts.grid
    x = collect(xnodes(grid, Center()))
    y = collect(ynodes(grid, Center()))

    # last full period
    last_period = findall(t -> t ≥ times[end] - T_wave, times)

    # --- metric 1: rmse against the exact solution, interior only -----------
    # skip 4 cells at each open boundary so we measure contamination of the
    # INTERIOR, not the boundary cells themselves
    ii = 5:(length(x) - 4)
    sq = Float64[]
    for n in last_period
        ηn = interior(ηts[n])[:, :, 1]
        for i in ii, j in eachindex(y)
            push!(sq, (ηn[i, j] - η_exact(x[i], y[j], times[n]))^2)
        end
    end
    rmse_rel = sqrt(mean(sq)) / A

    # --- metric 2: reflection from the along-coast amplitude envelope -------
    # j = 1 is the coastal wall, where the Kelvin wave amplitude is largest.
    # With a perfect radiation BC the envelope is flat at A; reflection makes
    # a standing wave whose ripple gives R.
    envelope = [maximum(abs(interior(ηts[n])[i, 1, 1]) for n in last_period)
                for i in eachindex(x)]
    env_int = envelope[ii]
    reflection = (maximum(env_int) - minimum(env_int)) /
                 (maximum(env_int) + minimum(env_int))

    return (; rmse_rel, reflection, envelope, x, y, times, ηts, ii)
end

function plot_kelvin_wave(m, filename, label)
    ηts = m.ηts
    n = length(m.times)
    ηn = interior(ηts[n])[:, :, 1]
    exact = [η_exact(m.x[i], m.y[j], m.times[n]) for i in eachindex(m.x), j in eachindex(m.y)]
    xk, yk = m.x ./ 1e3, m.y ./ 1e3

    # The domain is 2.5:1, so stack the maps vertically — side-by-side panels
    # crush anything sharing their column.
    fig = Figure(size = (1150, 1250))

    for (r, (fld, ttl, rng)) in enumerate(((ηn,          "η model (m) — final",        A),
                                           (exact,       "η exact (m) — final",        A),
                                           (ηn .- exact, "error η_model − η_exact (m)", 0.1A)))
        ax = Axis(fig[r, 1], title = ttl, ylabel = "y (km)",
                  xlabel = r == 3 ? "x (km)" : "")
        hm = heatmap!(ax, xk, yk, fld, colormap = :balance, colorrange = (-rng, rng))
        Colorbar(fig[r, 2], hm)
    end

    ax = Axis(fig[4, 1], title = "along-coast amplitude envelope  (flat ⇒ no reflection)",
              xlabel = "x (km)", ylabel = "max|η| (m)")
    lines!(ax, xk, m.envelope, linewidth = 2, label = "model")
    hlines!(ax, [A], color = :black, linestyle = :dash, label = "exact (A)")
    vlines!(ax, [xk[m.ii[1]], xk[m.ii[end]]], color = (:gray, 0.6),
            label = "metric window")
    axislegend(ax, position = :rb)
    rowsize!(fig.layout, 4, Relative(0.18))

    Label(fig[0, 1:2], "Coastal Kelvin wave — $label", fontsize = 20, tellwidth = false)
    save(filename * ".png", fig)
    return fig
end

# ==================================================================
# baseline run: the EXISTING NormalRadiation
# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    @printf("c = %.1f m/s   R = %.0f km   λ = %.0f km   period = %s\n",
            c, R / 1e3, λ / 1e3, prettytime(T_wave))
    @printf("Δx = %.0f km (%.0f pts/λ)   Δy = %.0f km (%.0f pts/R)\n",
            Lx / Nx / 1e3, λ / (Lx / Nx), Ly / Ny / 1e3, R / (Ly / Ny))

    sim = kelvin_wave_simulation(filename = "kelvin_normal_radiation")
    run!(sim)

    m = kelvin_wave_metrics("kelvin_normal_radiation")
    @printf("\n===== NormalRadiation (existing, baseline) =====\n")
    @printf("  relative rmse vs exact : %.4f\n", m.rmse_rel)
    @printf("  reflection coefficient : %.4f\n", m.reflection)
    plot_kelvin_wave(m, "kelvin_normal_radiation", "NormalRadiation (baseline)")
    println("  saved kelvin_normal_radiation.png")
end
