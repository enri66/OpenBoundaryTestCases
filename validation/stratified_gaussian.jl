# ==================================================================
# Stratified radiating Gaussian — BAROCLINIC + OBLIQUE incidence
#
# The test that can actually see `ObliqueRadiation`.
#
# WHY THE EARLIER BENCHMARKS COULD NOT. `kelvin_wave.jl` and
# `radiating_gaussian.jl` both use Nz = 1. The split-explicit corrector does
#
#     u += (U − ∫u dz) / H
#
# so with a single layer ∫u dz = H·u and the correction collapses to
# u = U/H exactly — the baroclinic value is wholly replaced by the
# Flather-determined barotropic transport. In a one-layer model `u` *is* the
# barotropic mode and a baroclinic radiation scheme has no degrees of freedom.
# (Confirmed empirically: a 100× canary on the oblique term changed nothing.)
#
# With Nz > 1 that same correction is depth-UNIFORM: it replaces the depth-mean
# with Flather's value while leaving the vertical shear set by the radiation
# condition intact. That shear is what the baroclinic scheme governs, so a
# stratified run is where the scheme becomes observable.
#
# SETUP. Square domain open on all four sides, f = 0 (so any residual is
# reflection, not a balanced vortex), linear stratification N² = const, and a
# MODE-1 isopycnal displacement released from rest:
#
#     ζ(x,y,z) = ζ₀ exp(−r²/2σ²) sin(πz/H)        (vanishes at surface and floor)
#     b(x,y,z) = N² z − N² ζ(x,y,z)
#
# The vertical modes of a uniformly stratified layer are sin(nπz/H) with
# eigenspeeds cₙ = NH/(nπ), so mode 1 travels at c₁ = NH/π — here ≈ 3.2 m/s,
# about 30× slower than the barotropic c₀ = √(gH) ≈ 99 m/s. The barotropic
# mode therefore clears the domain in hours while the internal mode takes days,
# which cleanly separates the two and leaves the internal wave as the thing
# crossing the boundary during the metric window.
#
# METRICS. Residual energy fraction after the internal mode has had time to
# leave, using the stratified energy
#
#     E = ∫∫∫ ½ (u² + v² + b′²/N²) dV,     b′ = b − N² z
#
# reported two ways:
#   * R_total       — all of it.
#   * R_baroclinic  — using only the DEPTH-VARYING part of the velocity,
#                     u′ = u − ū, plus the available potential energy. This is
#                     the sharper number: it excludes the barotropic mode that
#                     Flather (not our scheme) governs.
#
# Run:  julia --project=. validation/stratified_gaussian.jl
# ==================================================================

include(joinpath(@__DIR__, "obc_output.jl"))

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalRadiation, ObliqueRadiation,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using Printf
using Statistics
using CairoMakie

# ---------------- parameters ----------------
const g  = 9.80665
const Hs = 1000.0                 # m    depth
const N² = 1e-4                   # s⁻²  N = 1e-2, a thermocline value
const Ns = sqrt(N²)
const c₀ = sqrt(g * Hs)           # ≈ 99 m/s   barotropic
const c₁ = Ns * Hs / π            # ≈ 3.2 m/s  internal mode 1
const ζ₀ = 20.0                   # m    isopycnal displacement (ζ₀/H = 0.02, linear)
const σs = 80kilometers           # bump width
const Ls = 1000kilometers         # square domain
const Ns_h = 100                  # Δ = 10 km
const Nz_s = 20

# Internal mode reaches the corners at (L/√2)/c₁ ≈ 2.6 days.
const T_END_S  = 4days
const t_clear_s = 3.2days

@inline ζ_init(x, y, z) = ζ₀ * exp(-((x - Ls/2)^2 + (y - Ls/2)^2) / (2σs^2)) * sin(π * z / Hs)
@inline b_init(x, y, z) = N² * z - N² * ζ_init(x, y, z)
@inline b_background(z) = N² * z

"""
    stratified_gaussian_simulation(; baroclinic_scheme, closed, filename)

Mode-1 internal wave radiating outward from the centre of a stratified,
non-rotating square domain open on all four sides.
"""
function stratified_gaussian_simulation(; baroclinic_scheme = NormalRadiation(inflow_timescale = 0,
                                                                              outflow_timescale = Inf),
                                          closed = false,
                                          Δt = 300.0,
                                          substeps = 30,
                                          stop_time = T_END_S,
                                          filename = "stratified_gaussian")

    grid = RectilinearGrid(CPU();
                           size = (Ns_h, Ns_h, Nz_s),
                           x = (0, Ls), y = (0, Ls), z = (-Hs, 0),
                           halo = (7, 7, 7),
                           topology = (Bounded, Bounded, Bounded))

    boundary_conditions = if closed
        NamedTuple()
    else
        u_bcs = FieldBoundaryConditions(
            west = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme),
            east = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme))

        v_bcs = FieldBoundaryConditions(
            south = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme),
            north = NormalFlowBoundaryCondition(0.0; scheme = baroclinic_scheme))

        # The tracer's exterior state is the undisturbed background stratification.
        # Continuous form on an x-boundary is f(y, z, t); on a y-boundary f(x, z, t).
        b_bcs = FieldBoundaryConditions(
            west  = ValueBoundaryCondition((y, z, t) -> b_background(z); scheme = baroclinic_scheme),
            east  = ValueBoundaryCondition((y, z, t) -> b_background(z); scheme = baroclinic_scheme),
            south = ValueBoundaryCondition((x, z, t) -> b_background(z); scheme = baroclinic_scheme),
            north = ValueBoundaryCondition((x, z, t) -> b_background(z); scheme = baroclinic_scheme))

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

        (u = u_bcs, v = v_bcs, b = b_bcs, U = U_bcs, V = V_bcs, η = η_bcs)
    end

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps),
        coriolis = nothing,
        buoyancy = BuoyancyTracer(),
        tracers = (:b,),
        closure = nothing,
        momentum_advection = WENOVectorInvariant(),
        tracer_advection = WENO(),
        boundary_conditions)

    set!(model, b = b_init)

    simulation = Simulation(model; Δt, stop_time)

    function progress(sim)
        u, v, w = sim.model.velocities
        @printf("  %-9s  max|u| = %.4e  max|w| = %.4e\n",
                prettytime(sim), maximum(abs, u), maximum(abs, w))
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(12hours))

    u, v, w = model.velocities
    b = model.tracers.b
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, b);
        filename = outpath(filename * ".jld2"),
        schedule = TimeInterval(3hours),
        overwrite_existing = true)

    return simulation
end

# ==================================================================
# metrics
# ==================================================================
function stratified_gaussian_metrics(filename)
    uts = FieldTimeSeries(outpath(filename * ".jld2"), "u")
    vts = FieldTimeSeries(outpath(filename * ".jld2"), "v")
    bts = FieldTimeSeries(outpath(filename * ".jld2"), "b")
    times = bts.times
    grid = bts.grid
    x = collect(xnodes(grid, Center()))
    y = collect(ynodes(grid, Center()))
    z = collect(znodes(grid, Center()))
    Nx, Ny, Nz = length(x), length(y), length(z)

    xc(a) = 0.5 .* (a[1:end-1, :, :] .+ a[2:end, :, :])
    yc(a) = 0.5 .* (a[:, 1:end-1, :] .+ a[:, 2:end, :])

    function energies(n)
        u = let a = Array(interior(uts[n])); size(a,1) == Nx+1 ? xc(a) : a end
        v = let a = Array(interior(vts[n])); size(a,2) == Ny+1 ? yc(a) : a end
        b = Array(interior(bts[n]))
        b′ = similar(b)
        for k in 1:Nz
            @views b′[:, :, k] .= b[:, :, k] .- b_background(z[k])
        end
        ape = 0.5 * sum(b′.^2) / N²
        ke  = 0.5 * sum(@. u^2 + v^2)
        # depth-varying (baroclinic) part of the velocity
        ū = mean(u, dims = 3); v̄ = mean(v, dims = 3)
        ke_bc = 0.5 * sum(@. (u - ū)^2 + (v - v̄)^2)
        return (total = ke + ape, baroclinic = ke_bc + ape)
    end

    E  = [energies(n) for n in eachindex(times)]
    Et = [e.total for e in E]
    Eb = [e.baroclinic for e in E]
    late = findall(t -> t ≥ t_clear_s, times)

    R_total      = sqrt(mean(Et[late]) / Et[1])
    R_baroclinic = sqrt(mean(Eb[late]) / Eb[1])

    return (; R_total, R_baroclinic, Et, Eb, times, x, y, z, uts, vts, bts, late, Nx, Ny, Nz)
end

function plot_stratified(m, filename, label)
    xk, yk = m.x ./ 1e3, m.y ./ 1e3
    kmid = max(1, m.Nz ÷ 2)
    fig = Figure(size = (1350, 880))

    picks = (1,
             argmin(abs.(m.times .- 1days)),
             argmin(abs.(m.times .- 2days)),
             length(m.times))
    for (col, n) in enumerate(picks)
        b = Array(interior(m.bts[n]))
        b′ = b[:, :, kmid] .- b_background(m.z[kmid])
        rng = col == 1 ? N²*ζ₀ : (col == 4 ? 0.05*N²*ζ₀ : 0.4*N²*ζ₀)
        ax = Axis(fig[1, col], title = @sprintf("t = %s", prettytime(m.times[n])),
                  aspect = DataAspect(), xlabel = "x (km)",
                  ylabel = col == 1 ? "y (km)" : "")
        heatmap!(ax, xk, yk, b′, colormap = :balance, colorrange = (-rng, rng))
        text!(ax, 0.03, 0.94; text = @sprintf("±%.1e", rng), space = :relative, fontsize = 11)
    end
    Label(fig[1, 0], "b′ at mid-depth", rotation = π/2, tellheight = false, fontsize = 13)

    ax = Axis(fig[2, 1:2], title = "energy (normalised)", xlabel = "t (days)",
              ylabel = "E / E₀", yscale = log10)
    lines!(ax, m.times ./ 86400, max.(m.Et ./ m.Et[1], 1e-12), linewidth = 2, label = "total")
    lines!(ax, m.times ./ 86400, max.(m.Eb ./ m.Eb[1], 1e-12), linewidth = 2, label = "baroclinic")
    vlines!(ax, [t_clear_s / 86400], color = (:gray, 0.7), linestyle = :dash, label = "metric window")
    axislegend(ax, position = :lb)

    ax = Axis(fig[2, 3:4], title = "u at mid-depth, final", xlabel = "x (km)", ylabel = "y (km)",
              aspect = DataAspect())
    u = Array(interior(m.uts[end]))
    uu = size(u,1) == m.Nx+1 ? 0.5 .* (u[1:end-1,:,:] .+ u[2:end,:,:]) : u
    hm = heatmap!(ax, xk, yk, uu[:, :, kmid], colormap = :balance)
    Colorbar(fig[2, 5], hm)

    Label(fig[0, 1:4], "Stratified radiating Gaussian — $label", fontsize = 20, tellwidth = false)
    save(filename * ".png", fig)
    return fig
end

# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    @printf("c₀ = %.1f m/s (barotropic)   c₁ = %.2f m/s (internal mode 1)   ratio %.0f\n",
            c₀, c₁, c₀/c₁)
    @printf("internal mode reaches corners at %.1f days; metrics after %.1f days\n",
            (Ls/sqrt(2))/c₁/86400, t_clear_s/86400)
    @printf("grid %d×%d×%d, Δ = %.0f km, σ = %.0f km (%.0f cells)\n",
            Ns_h, Ns_h, Nz_s, Ls/Ns_h/1e3, σs/1e3, σs/(Ls/Ns_h))

    results = Dict{String,Any}()
    for (name, scheme) in ("NormalRadiation"  => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
                           "ObliqueRadiation" => ObliqueRadiation(inflow_timescale = 0, outflow_timescale = Inf))
        println("\n--- $name ---")
        tag = "stratified_" * lowercase(name)
        sim = stratified_gaussian_simulation(baroclinic_scheme = scheme, filename = tag)
        run!(sim)
        m = stratified_gaussian_metrics(tag)
        results[name] = m
        @printf("  R_total = %.5f   R_baroclinic = %.5f\n", m.R_total, m.R_baroclinic)
        plot_stratified(m, tag, name)
    end

    println("\n" * "="^64)
    @printf("%-18s %14s %16s\n", "scheme", "R_total", "R_baroclinic")
    for name in ("NormalRadiation", "ObliqueRadiation")
        @printf("%-18s %14.5f %16.5f\n", name, results[name].R_total, results[name].R_baroclinic)
    end
    rn, ro = results["NormalRadiation"], results["ObliqueRadiation"]
    @printf("change: R_total %+.1f%%   R_baroclinic %+.1f%%\n",
            100*(ro.R_total - rn.R_total)/rn.R_total,
            100*(ro.R_baroclinic - rn.R_baroclinic)/rn.R_baroclinic)
end
