# ==================================================================
# Supercritical flow — a BAROTROPIC open-boundary test
#
# Re-scoped 2026-09-09. The earlier version of this file tried to use a
# sheared supercritical flow to discriminate `ObliqueRadiation` from
# `NormalRadiation`. That was a mistake: `supercritical` is a BAROTROPIC
# test case (confirmed with the person who implemented it in MOM6). A
# single-layer flow has no baroclinic structure for an oblique *baroclinic*
# radiation scheme to act on, and MOM6's own oblique-OBC path is gated off
# in this configuration. What the case actually exercises is the barotropic
# open boundary — in Oceananigans, the `GravityWaveRadiation` (Flather 1976)
# + `SurfaceWaveRadiation` (Chapman 1985) pair on `U`/`V`/`η`. That pair is
# upstream code this OBC project leans on but did not write, and it has not
# been independently validated here. This does that.
#
# The file reproduces BOTH reference implementations, so the result can be
# checked against each:
#
# ── PART 1 — MOM6's `supercritical`, exactly ──────────────────────────
#   From ESMG-configs/ocean_only/Supercritical (MOM_input + the
#   supercritical_initialization.F90 OBC code):
#     grid 200 × 150, flat bottom, H = 1 m, single layer, non-rotating
#     uniform 8.57 m/s zonal inflow  ⇒  Fr = 8.57/√(gH) = 2.74
#     west  = SIMPLE   (fully specified inflow)
#     east  = GRADIENT (zero-gradient radiation)
#     north / south = walls
#     run 180 s, Δt = 5 s
#   MOM6's `ocean.stats.flat`: kinetic energy stays exactly ½·8.57² = 36.72
#   m² s⁻², zero truncations, a small linear drift in mean sea level from
#   the non-conservative outflow. No shock forms — the domain is a plain
#   channel and the flow just advects through. This is the minimal check
#   that a zero-gradient / radiating outflow passes uniform supercritical
#   flow without injecting noise. We assert the same three things.
#
# ── PART 2 — the ROMS `supercritical` physics: an oblique hydraulic jump ─
#   ROMS's version (the Rutgers "supercritical" test problem) adds a wall
#   kink: the south wall deflects into the flow by θ, and a standing
#   oblique hydraulic jump springs from the kink (Ippen & Harleman 1956).
#   For Fr₁ = 2.74 and the canonical θ = 8.95°, the oblique-jump relations
#   give a shock angle β ≈ 30° and a depth ratio h₂/h₁ ≈ 1.50 — verified
#   here in the interior against the analytical solution (Part 2a).
#
#   The jump is a genuine 2-D discontinuity that crosses the open east and
#   north boundaries at an oblique angle. Part 2b checks the barotropic OBC
#   passes it: compare the open-boundary run to a WIDE-DOMAIN reference
#   (east boundary pushed far downstream), and to a CLOSED-wall control
#   that must reflect. Because the flow stays supercritical downstream of a
#   weak oblique jump (Fr₂ ≈ 2.1 here), characteristics cannot carry a
#   boundary error upstream — so the measurement is the boundary band
#   itself, where a reflecting wall visibly kinks the shock and a working
#   Flather condition does not.
#
# Run:  julia --project=. validation/supercritical.jl
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

# ---------------- MOM6 ESMG `supercritical` parameters ----------------
const g_sc  = 9.81                 # G_EARTH
const H_sc  = 1.0                  # MAXIMUM_DEPTH  [m]
const U_sc  = 8.57                 # INITIAL_U_CONST / SUPERCRITICAL_ZONAL_FLOW  [m/s]
const c_sc  = sqrt(g_sc * H_sc)    # 3.132 m/s  (barotropic gravity-wave speed)
const Fr_sc = U_sc / c_sc          # 2.7365

const Nx_sc = 200                  # NIGLOBAL
const Ny_sc = 150                  # NJGLOBAL
const Δ_sc  = 200.0                # m  (CFL 0.321 in ocean.stats.flat ⇒ Δ ≈ 182 m; 200 is close)
const Lx_sc = Nx_sc * Δ_sc         # 40 km
const Ly_sc = Ny_sc * Δ_sc         # 30 km
const Δt_sc = 5.0                  # DT  [s]

# Part 1 runs for MOM6's DAYMAX (180 s, genuinely seconds). Part 2 needs a
# steady state: one domain transit is Lx/U ≈ 4670 s, so run four of them.
const T_UNIFORM = 180.0
const T_JUMP    = 4 * Lx_sc / U_sc

# ---------------- Part 2 wall kink ----------------
const θ_wall  = 8.95               # deg  — canonical Ippen deflection
const x_kink  = 8kilometers        # where the south wall starts deflecting up
const ν_sc    = 2.0e3              # m² s⁻¹  horizontal viscosity for shock capture (grid Re ≈ 0.9)

# ==================================================================
# Ippen (1951) oblique hydraulic jump — the analytical target for Part 2a
# ==================================================================
"""
    oblique_jump(Fr₁, θ_deg)

Standing oblique hydraulic jump produced when a wall deflects a supercritical
stream (Froude number `Fr₁`) into itself by `θ_deg`. Returns the shock angle
`β` (from the incoming flow direction), the depth ratio `h₂/h₁`, and the
downstream Froude number, from the shallow-water oblique-shock relations:

    tan θ = tan β · (√(1 + 8 Fr₁² sin²β) − 3) / (2 tan²β − 1 + √(1 + 8 Fr₁² sin²β))
    h₂/h₁ = ½ (√(1 + 8 Fr₁² sin²β) − 1)

Solved for the weak (small-β) root by bisection — no external dependency.
"""
function oblique_jump(Fr₁, θ_deg)
    θ = deg2rad(θ_deg)
    resid(β) = begin
        s = sqrt(1 + 8Fr₁^2 * sin(β)^2)
        tan(β) * (s - 3) / (2tan(β)^2 - 1 + s) - tan(θ)
    end

    a, b = θ + 1e-4, deg2rad(55.0)     # brackets the weak root for any physical θ
    fa = resid(a)
    for _ in 1:200
        m = (a + b) / 2
        fm = resid(m)
        (abs(fm) < 1e-13 || b - a < 1e-13) && (a = b = m; break)
        fa * fm < 0 ? (b = m) : (a = m; fa = fm)
    end
    β = (a + b) / 2

    s = sqrt(1 + 8Fr₁^2 * sin(β)^2)
    r = (s - 1) / 2                                        # h₂/h₁
    Fr₂ = Fr₁ * sqrt(cos(β)^2 + sin(β)^2 / r^2) / sqrt(r)  # downstream (still > 1 for a weak jump)
    return (; β_deg = rad2deg(β), h₂_h₁ = r, Fr₂)
end

# ==================================================================
# the simulation
# ==================================================================
"""
    supercritical_simulation(; wedge, open_east, open_north, Lx, Nx, stop_time, filename)

Single-layer supercritical channel. West is a fully-specified uniform inflow;
`wedge` adds the Ippen wall kink on the south boundary. `open_east` /
`open_north` select the barotropic radiation triad (Flather on `U`/`V`,
Chapman on `η`) versus a solid wall — the wide-domain reference and the
closed control differ only in these two flags and in `Lx`/`Nx`.
"""
function supercritical_simulation(; wedge::Bool = false,
                                    open_east::Bool = true,
                                    open_north::Bool = false,
                                    Lx = Lx_sc, Nx = Nx_sc,
                                    Ly = Ly_sc, Ny = Ny_sc,
                                    stop_time = T_UNIFORM,
                                    Δt = Δt_sc,
                                    filename = "supercritical")

    underlying = RectilinearGrid(CPU();
                                 size = (Nx, Ny, 1),
                                 x = (0, Lx), y = (0, Ly), z = (-H_sc, 0),
                                 halo = (8, 8, 8),   # WENO(5) on an immersed-boundary grid
                                 topology = (Bounded, Bounded, Bounded))

    # The wall kink is a DRY wedge in the SE: bottom raised to the surface where
    # y < (x − x_kink) tan θ_wall. Single layer, so that column is fully masked.
    grid = if wedge
        @inline wedge_bottom(x, y) = ifelse((x > x_kink) & (y < (x - x_kink) * tand(θ_wall)),
                                            zero(x), convert(typeof(x), -H_sc))
        ImmersedBoundaryGrid(underlying, GridFittedBottom(wedge_bottom))
    else
        underlying
    end

    # The undisturbed exterior everywhere is the uniform stream: transport
    # U = U_sc·H through the zonal faces, V = 0 through the meridional faces,
    # η = 0. Flather (`GravityWaveRadiation`) then radiates only the DEVIATION
    # from that stream — feeding it (0, 0) instead, as an earlier version did,
    # tells the outflow boundary the transport should be zero while 8.57·1
    # m²/s is leaving, and the barotropic mode blows up.
    Uext = U_sc * H_sc

    # west: SIMPLE — the uniform inflow, imposed on u and carried by Flather on U
    u_bcs_pairs = Pair{Symbol,Any}[:west => NormalFlowBoundaryCondition(U_sc)]
    U_bcs_pairs = Pair{Symbol,Any}[:west => GravityWaveRadiationBoundaryCondition((Uext, 0.0))]
    v_bcs_pairs = Pair{Symbol,Any}[]
    V_bcs_pairs = Pair{Symbol,Any}[]

    # east: GRADIENT baroclinic (zero-gradient radiation) + Flather barotropic, or a wall
    if open_east
        push!(u_bcs_pairs, :east => NormalFlowBoundaryCondition(0.0;
                  scheme = NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf)))
        push!(U_bcs_pairs, :east => GravityWaveRadiationBoundaryCondition((Uext, 0.0)))
    end

    # north: the undisturbed stream (v = 0) or a wall
    if open_north
        push!(v_bcs_pairs, :north => NormalFlowBoundaryCondition(0.0;
                  scheme = NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf)))
        push!(V_bcs_pairs, :north => GravityWaveRadiationBoundaryCondition((0.0, 0.0)))
    end

    u_bcs = FieldBoundaryConditions(; u_bcs_pairs...)
    v_bcs = isempty(v_bcs_pairs) ? nothing : FieldBoundaryConditions(; v_bcs_pairs...)
    U_bcs = FieldBoundaryConditions(underlying, (Face(), Center(), nothing); U_bcs_pairs...)
    V_bcs = isempty(V_bcs_pairs) ? nothing :
            FieldBoundaryConditions(underlying, (Center(), Face(), nothing); V_bcs_pairs...)

    bcs = (; u = u_bcs, U = U_bcs)
    isnothing(v_bcs) || (bcs = merge(bcs, (; v = v_bcs)))
    isnothing(V_bcs) || (bcs = merge(bcs, (; V = V_bcs)))

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
        timestepper = :SplitRungeKutta3,
        coriolis = nothing,
        buoyancy = nothing,
        tracers = (),
        closure = HorizontalScalarDiffusivity(ν = ν_sc),
        momentum_advection = WENOVectorInvariant(),
        boundary_conditions = bcs)

    set!(model, u = U_sc)     # uniform stream everywhere at t = 0

    simulation = Simulation(model; Δt, stop_time)
    conjure_time_step_wizard!(simulation, IterationInterval(10); cfl = 0.7,
                              max_Δt = Δt_sc)

    function progress(sim)
        u, v, w = sim.model.velocities
        η = sim.model.free_surface.displacement
        @printf("  %-10s  ⟨u²⟩/2 = %6.2f   max|η| = %.3f m   max|v| = %.2f m/s\n",
                prettytime(sim), mean(interior(u) .^ 2) / 2,
                maximum(abs, interior(η)), maximum(abs, interior(v)))
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(max(stop_time / 8, 30.0)))

    u, v, w = model.velocities
    η = model.free_surface.displacement
    simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, η);
        filename = outpath(filename * ".jld2"),
        schedule = TimeInterval(stop_time / 12),
        overwrite_existing = true)

    return simulation
end

# ==================================================================
# Part 1 metric — uniformity, exactly as MOM6's ocean.stats.flat reports it
# ==================================================================
function uniform_flow_metrics(filename)
    uts = FieldTimeSeries(outpath(filename * ".jld2"), "u")
    vts = FieldTimeSeries(outpath(filename * ".jld2"), "v")
    ηts = FieldTimeSeries(outpath(filename * ".jld2"), "η")

    # MOM6's ocean.stats energy for this uniform zonal flow is ½⟨u²⟩ (v ≈ 0);
    # u and v live on different faces so they cannot be added elementwise anyway.
    energy(n) = mean(interior(uts[n]) .^ 2) / 2
    E  = [energy(n) for n in eachindex(uts.times)]
    E₀ = U_sc^2 / 2

    u_final = interior(uts[end])
    v_final = interior(vts[end])
    η_final = interior(ηts[end])

    return (; times = uts.times, E, E₀,
              energy_drift = abs(E[end] - E₀) / E₀,
              max_u_dev = maximum(abs, u_final .- U_sc) / U_sc,
              max_v = maximum(abs, v_final),
              η_drift = maximum(abs, η_final),
              nan = any(isnan, u_final) || any(isnan, η_final))
end

# ==================================================================
# Part 2a metric — shock angle and depth ratio, vs the Ippen solution
# ==================================================================
function measure_oblique_jump(filename)
    ηts = FieldTimeSeries(outpath(filename * ".jld2"), "η")
    grid = ηts.grid
    x = collect(xnodes(grid, Center()))
    y = collect(ynodes(grid, Center()))
    η = Array(interior(ηts[end]))[:, :, 1]

    # h/h₁ = 1 + η (h₁ = H_sc, η₁ ≈ 0). Shock front: where η crosses ¼ of its plateau.
    η_plateau = quantile(vec(filter(isfinite, η)), 0.97)
    thresh = 0.25 * η_plateau

    # x of the front at each y, over the y-band the shock actually crosses
    x_front = fill(NaN, length(y))
    for (j, _) in enumerate(y)
        col = η[:, j]
        i = findfirst(>(thresh), col)
        (i === nothing || i == 1) && continue
        # linear interpolation between cells i-1 and i
        x_front[j] = x[i-1] + (thresh - col[i-1]) / (col[i] - col[i-1]) * (x[i] - x[i-1])
    end

    good = findall(j -> isfinite(x_front[j]) && y[j] > 2kilometers && y[j] < 0.7Ly_sc, eachindex(y))
    length(good) < 5 && return (; β_deg = NaN, h₂_h₁ = NaN, front_r2 = NaN)

    # least-squares line x_front = x_kink + y / tan β  ⇒  fit slope dy/dx
    Y = y[good]; X = x_front[good]
    slope = (mean(X .* Y) - mean(X) * mean(Y)) / (mean(X .^ 2) - mean(X)^2)   # dY/dX
    β_deg = atand(slope)
    ŷ = mean(Y) .+ slope .* (X .- mean(X))
    r2 = 1 - sum((Y .- ŷ) .^ 2) / sum((Y .- mean(Y)) .^ 2)

    # downstream depth: average η in a box below the shock, clear of walls and the wedge
    ix = findall(xi -> 0.6Lx_sc < xi < 0.85Lx_sc, x)
    jy = findall(yi -> 0.30Ly_sc < yi < 0.45Ly_sc, y)   # above the wedge, below the shock there
    h₂_h₁ = 1 + mean(filter(isfinite, η[ix, jy]))

    return (; β_deg, h₂_h₁, front_r2 = r2, x, y, η, x_front)
end

# ==================================================================
# Part 2b metric — does the open east boundary pass the shock?
# ==================================================================
function obc_boundary_error(test_file, ref_file)
    t = Array(interior(FieldTimeSeries(outpath(test_file * ".jld2"), "η")[end]))[:, :, 1]
    r_ts = FieldTimeSeries(outpath(ref_file * ".jld2"), "η")
    r = Array(interior(r_ts[end]))[:, :, 1]

    nx = size(t, 1)
    band = (nx - 9):nx                        # last 10 cells before the open east boundary
    a = vec(t[band, :]); b = vec(r[band, 1:size(t, 2)])
    boundary = sqrt(mean((a .- b) .^ 2)) / sqrt(mean(b .^ 2))

    ai = vec(t[5:nx-10, :]); bi = vec(r[5:nx-10, 1:size(t, 2)])
    interior_err = sqrt(mean((ai .- bi) .^ 2)) / sqrt(mean(bi .^ 2))

    return (; boundary, interior = interior_err)
end

# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    ip = oblique_jump(Fr_sc, θ_wall)
    @printf("Fr₁ = %.3f   wall deflection θ = %.2f°\n", Fr_sc, θ_wall)
    @printf("Ippen oblique jump:  β = %.2f°   h₂/h₁ = %.3f   Fr₂ = %.2f  (%s downstream)\n",
            ip.β_deg, ip.h₂_h₁, ip.Fr₂, ip.Fr₂ > 1 ? "still supercritical" : "subcritical")
    @printf("grid %d × %d, Δ = %.0f m  (%.0f × %.0f km)\n\n",
            Nx_sc, Ny_sc, Δ_sc, Lx_sc/1e3, Ly_sc/1e3)

    # ---------------- PART 1 : MOM6 uniform-flow anchor ----------------
    println("="^68)
    println("PART 1 — MOM6 `supercritical`: uniform flow, GRADIENT outflow")
    println("="^68)
    run!(supercritical_simulation(wedge = false, open_east = true, open_north = false,
                                  stop_time = T_UNIFORM, filename = "supercritical_uniform"))
    m1 = uniform_flow_metrics("supercritical_uniform")
    @printf("\n  kinetic energy  ½⟨|u|²⟩ = %.4f   (MOM6: ½·8.57² = %.4f, drift %.2e)\n",
            m1.E[end], m1.E₀, m1.energy_drift)
    @printf("  max |u − U_in| / U_in     = %.2e\n", m1.max_u_dev)
    @printf("  max |v|                   = %.2e m/s\n", m1.max_v)
    @printf("  free-surface drift        = %.2e m  (MOM6 shows a small non-conservative drift too)\n", m1.η_drift)
    p1_pass = !m1.nan && m1.energy_drift < 1e-3 && m1.max_u_dev < 5e-3 && m1.max_v < 1e-2
    println("  ⇒ ", p1_pass ? "PASS — outflow passes uniform supercritical flow without noise" : "FAIL")

    # ---------------- PART 2a : oblique jump vs Ippen ----------------
    println("\n" * "="^68)
    println("PART 2a — ROMS `supercritical`: oblique hydraulic jump, interior vs Ippen")
    println("="^68)
    run!(supercritical_simulation(wedge = true, open_east = true, open_north = true,
                                  stop_time = T_JUMP, filename = "supercritical_jump"))
    m2 = measure_oblique_jump("supercritical_jump")
    @printf("\n  shock angle β : measured %.2f°   Ippen %.2f°   (Δ = %.2f°, fit R² = %.3f)\n",
            m2.β_deg, ip.β_deg, m2.β_deg - ip.β_deg, m2.front_r2)
    @printf("  depth ratio   : measured %.3f    Ippen %.3f   (Δ = %.1f%%)\n",
            m2.h₂_h₁, ip.h₂_h₁, 100 * (m2.h₂_h₁ - ip.h₂_h₁) / ip.h₂_h₁)
    p2a_pass = abs(m2.β_deg - ip.β_deg) < 3 && abs(m2.h₂_h₁ - ip.h₂_h₁) / ip.h₂_h₁ < 0.1
    println("  ⇒ ", p2a_pass ? "PASS — solver reproduces the analytical oblique jump" : "FAIL")

    # ---------------- PART 2b : does Flather pass the shock? ----------------
    println("\n" * "="^68)
    println("PART 2b — barotropic OBC: open east vs wide reference vs closed wall")
    println("="^68)
    run!(supercritical_simulation(wedge = true, open_east = false, open_north = true,
                                  Lx = 2Lx_sc, Nx = 2Nx_sc,
                                  stop_time = T_JUMP, filename = "supercritical_wide"))
    run!(supercritical_simulation(wedge = true, open_east = false, open_north = true,
                                  stop_time = T_JUMP, filename = "supercritical_closed"))

    e_open   = obc_boundary_error("supercritical_jump",   "supercritical_wide")
    e_closed = obc_boundary_error("supercritical_closed", "supercritical_wide")
    @printf("\n  %-22s  %14s  %14s\n", "east boundary", "boundary band", "interior")
    @printf("  %-22s  %14.4f  %14.4f\n", "open (Flather)",  e_open.boundary,   e_open.interior)
    @printf("  %-22s  %14.4f  %14.4f\n", "closed wall",     e_closed.boundary, e_closed.interior)
    p2b_pass = e_open.boundary < 0.5 * e_closed.boundary && e_open.boundary < 0.1
    println("  ⇒ ", p2b_pass ? "PASS — Flather passes the oblique shock; the wall reflects it" : "FAIL")

    # ---------------- figure ----------------
    fig = Figure(size = (1300, 900))

    m1p = uniform_flow_metrics("supercritical_uniform")
    ax = Axis(fig[1, 1], title = "Part 1 — kinetic energy (MOM6: flat at ½·8.57²)",
              xlabel = "t (s)", ylabel = "½⟨|u|²⟩  (m² s⁻²)")
    lines!(ax, m1p.times, m1p.E, linewidth = 2)
    hlines!(ax, [m1p.E₀], color = :firebrick, linestyle = :dash, label = "½ U_in²")
    axislegend(ax, position = :rc)

    jm = measure_oblique_jump("supercritical_jump")
    ax = Axis(fig[1, 2], title = @sprintf("Part 2a — oblique jump  (β: %.1f° vs Ippen %.1f°)", jm.β_deg, ip.β_deg),
              xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, jm.x ./ 1e3, jm.y ./ 1e3, 1 .+ jm.η, colormap = :dense, colorrange = (1, ip.h₂_h₁ * 1.05))
    lines!(ax, (x_kink .+ (0:0.1:Ly_sc) ./ tand(ip.β_deg)) ./ 1e3, (0:0.1:Ly_sc) ./ 1e3,
           color = :orange, linestyle = :dash, linewidth = 2, label = "Ippen shock")
    lines!(ax, [x_kink/1e3, Lx_sc/1e3], [0, (Lx_sc - x_kink) * tand(θ_wall) / 1e3],
           color = :black, linewidth = 3, label = "wall")
    axislegend(ax, position = :lt)

    for (col, (tag, ttl)) in enumerate(("supercritical_jump"   => "Part 2b — open east (Flather)",
                                        "supercritical_closed" => "Part 2b — closed east (wall)"))
        local ηts = FieldTimeSeries(outpath(tag * ".jld2"), "η")
        local ηf = Array(interior(ηts[end]))[:, :, 1]
        local xg = collect(xnodes(ηts.grid, Center())) ./ 1e3
        local yg = collect(ynodes(ηts.grid, Center())) ./ 1e3
        local axb = Axis(fig[2, col], title = ttl, xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
        heatmap!(axb, xg, yg, 1 .+ ηf, colormap = :dense, colorrange = (1, ip.h₂_h₁ * 1.05))
        vlines!(axb, [Lx_sc/1e3], color = (:red, 0.4), linestyle = :dash)
    end

    Label(fig[0, 1:2], "Supercritical — barotropic open-boundary test (MOM6 + ROMS/Ippen)",
          fontsize = 19, tellwidth = false)
    save("supercritical.png", fig)
    println("\nsaved supercritical.png")
end
