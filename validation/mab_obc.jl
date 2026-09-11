using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
include(joinpath(@__DIR__, "obc_output.jl"))

using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation, ObliqueRadiation,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
import SeawaterPolynomials
using NCDatasets
using Downloads
using Statistics
using Printf
using CairoMakie

# ------------------------------------------------------------------
# Script 9a: Open boundaries + sponge layers — regional downscaling
#
# Scripts 1-8 all lived behind CLOSED walls: whatever the wind or
# convection did, no water, heat or salt could enter or leave. A real
# regional model is a window cut out of a larger ocean, and the ocean
# outside flows through its edges.
#
# This script keeps script 8's Mid-Atlantic Bight configuration (same
# LatitudeLongitudeGrid, same real ETOPO bathymetry, same latitude-
# dependent Coriolis, same CATKE, same SplitExplicitFreeSurface, same NE
# wind, 8-day → 6-day spin-up) and opens three of its four sides: the
# north, south and east edges become OPEN; the coast (west) stays closed.
#
# New concepts vs. script 8:
#
#   - OPEN BOUNDARIES FOR A SPLIT-EXPLICIT MODEL need a coordinated SET
#     of boundary conditions per open edge, because the model carries a
#     slow 3-D flow AND a fast barotropic (depth-integrated) mode:
#       * `u`, `v` NORMAL component + tracers: `NormalFlowBoundaryCondition`
#         / `ValueBoundaryCondition` with `PerturbationAdvection` — the
#         perturbation from the exterior state is advected out at the
#         boundary-normal velocity; a long `inflow_timescale` (3 days) and
#         `outflow_timescale = Inf` make it a "soft" boundary that mostly
#         follows the interior.
#       * `u`, `v` TANGENTIAL component: `GradientBoundaryCondition(0)`.
#         This version has no open scheme for the tangential velocity
#         (that needs oblique radiation); left at the default free-slip it
#         grows a boundary jet, so a zero-gradient Neumann condition is the
#         next-best thing.
#       * `U`, `V`  (barotropic transport): `GravityWaveRadiationBoundary-
#         Condition` — the Flather (1976) characteristic condition
#         `Uᵇ = Uᵉˣᵗ + √(gH)(ηᵇ − ηᵉˣᵗ)`, so barotropic gravity waves
#         (~200 m/s) leave at the right speed instead of reflecting.
#       * `η`  (free surface): `SurfaceWaveRadiationBoundaryCondition` —
#         Chapman (1985), lets the boundary sea level evolve so the
#         surface pressure gradient can cross the edge.
#     Omit the `U`/`V`/`η` conditions and the barotropic mode resonates:
#     the domain slowly fills with energy and the run blows up. (Learned
#     the hard way — the first version of this script did exactly that.)
#
#   - SPONGE LAYERS. A `Relaxation` forcing whose `mask` is 1 in a
#     ~5-cell buffer along each open edge and 0 in the interior, relaxing
#     u, v, T, S back toward the prescribed large-scale state. The open
#     BCs handle the boundary-NORMAL flow; the sponge damps everything
#     tangential and every tracer anomaly before it can reach the edge.
#
# The prescribed "large-scale state" the exterior imposes here is simply
# the ocean at rest with script 8's stratification, T̄(z) = 12 + 8e-3 z,
# S = 35. So this script asks a clean question: with the same NE wind as
# script 8 (but tapered to zero through the sponge zone — a uniform wind
# blowing right up to an at-rest open edge drives an Ekman layer that
# converges into the boundary cells and spins up a spurious edge jet),
# what changes when the water can leave? The Ekman transport that piled
# up against script 8's walls (setting up a ~0.28 m sea-surface tilt) now
# flows out through the open edges instead.
#
# STATUS — a milestone, not a finished result. What the verification
# confirms works:
#   - the run stays bounded, i.e. the split-explicit barotropic open BCs
#     (Flather + Chapman) are wired correctly — the first version without
#     them resonated and blew up;
#   - domain volume is conserved: d/dt ∫∫η dA → −0.01 Sv by day 6;
#   - the INTERIOR is clean — geostrophic below the Ekman layer
#     (corr = 1.00), Ekman transport near τ/(ρf), no grid-scale noise,
#     sponge zones tracking the state at rest;
#   - the sea-surface setup (0.24 m) is smaller than script 8's closed
#     box (0.28 m) — water leaves instead of piling up.
#
# What it does NOT achieve: a ~0.15 m/s current clings to the open south
# and east edges, with a slow energy creep (domain-max speed still rising
# at day 6) and a bad corner where the closed coast meets the open south
# boundary (Cape Hatteras). This is the interior circulation exiting
# through a radiation condition that is not transparent to a sheared
# current. Things tried that did NOT remove it, in Oceananigans 0.110.19:
# swapping NormalRadiation ↔ PerturbationAdvection, a Gradient(0) BC on
# the tangential velocity, tapering the wind through the sponge, and a
# 30× viscous sponge. Fixing it needs oblique radiation and/or better
# corner handling — reworked in a newer Oceananigans; revisit after the
# version bump.
#
# (A prescribed inflowing shelfbreak jet — a geostrophic slope current
# entering through the north boundary — was also deferred, 9a-plus.)
#
# CPU, ~163k cells, 6 days. ~6.5 min wall with `julia -t 8` (~23 min
# single-threaded — the 30× viscous sponge is the expensive part).
# ------------------------------------------------------------------

const λ_bounds = (-76.0, -64.0)     # °E  (Mid-Atlantic Bight)
const φ_bounds = ( 34.0,  42.0)     # °N
const ETOPO_NC = joinpath(homedir(), "Dropbox", "Models", "JuliaOceananigans", "mab_etopo.nc")

# ---------------- ETOPO download (cached) ----------------
function fetch_etopo(path, (λ₁, λ₂), (φ₁, φ₂))
    isfile(path) && return path
    url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?" *
          "altitude%5B($φ₁):($φ₂)%5D%5B($λ₁):($λ₂)%5D"
    @info "Downloading ETOPO subset for the Mid-Atlantic Bight …"
    Downloads.download(url, path)
    return path
end

# ---------------- bathymetry conditioning (identical to script 8) ----------------
function regional_bathymetry(nc, λc, φc; smoothing_passes = 12, min_depth = 20.0, max_depth)
    ds = NCDataset(nc)
    elon = Float64.(ds["longitude"][:]); elat = Float64.(ds["latitude"][:])
    ez   = Float64.(coalesce.(ds["altitude"][:, :], 0.0))   # (Nlon, Nlat), >0 = land
    close(ds)

    Δλ = λc[2] - λc[1]; Δφ = φc[2] - φc[1]
    raw = [ (v = ez[(elon .≥ λc[i]-Δλ/2) .& (elon .< λc[i]+Δλ/2),
                    (elat .≥ φc[j]-Δφ/2) .& (elat .< φc[j]+Δφ/2)];
             isempty(v) ? 0.0 : mean(v)) for i in eachindex(λc), j in eachindex(φc) ]
    wet = raw .< 0
    Nx, Ny = size(raw)

    b = copy(raw)
    for _ in 1:smoothing_passes
        bn = copy(b)
        for i in 2:Nx-1, j in 2:Ny-1
            wet[i,j] || continue
            acc = 0.0; w = 0.0
            for (di,wi) in ((-1,1.),(0,2.),(1,1.)), (dj,wj) in ((-1,1.),(0,2.),(1,1.))
                wet[i+di,j+dj] || continue
                acc += wi*wj*b[i+di,j+dj]; w += wi*wj
            end
            w > 0 && (bn[i,j] = acc/w)
        end
        b = bn
    end

    bh = [ wet[i,j] ? clamp(b[i,j], -max_depth + 1, -min_depth) : 0.0
           for i in 1:Nx, j in 1:Ny ]

    rmax(h) = maximum(begin
        r = 0.0
        for i in 1:Nx, j in 1:Ny
            h[i,j] < 0 || continue
            for (di,dj) in ((1,0),(0,1))
                (i+di > Nx || j+dj > Ny) && continue
                h[i+di,j+dj] < 0 || continue
                r = max(r, abs(h[i+di,j+dj]-h[i,j]) / abs(h[i+di,j+dj]+h[i,j]))
            end
        end
        r
    end for _ in 1:1)

    rawh = [ wet[i,j] ? raw[i,j] : 0.0 for i in 1:Nx, j in 1:Ny ]
    @printf("bathymetry: %.0f … %.0f m,  land %d / %d cells,  rₓ %.2f → %.2f (raw → smoothed)\n",
            minimum(bh), -maximum(filter(<(0), bh)), count(==(0.0), bh), Nx*Ny,
            rmax(rawh), rmax(bh))
    return bh
end

# ---------------- grid (identical to script 8) ----------------
Nλ, Nφ, Nz = 100, 68, 24
Lz = 4000
refinement, stretching = 8, 10
s(k) = (k - 1) / Nz
z_faces(k) = Lz * ((1 + (s(k) - 1) / refinement) *
                   ((1 - exp(-stretching * s(k))) / (1 - exp(-stretching))) - 1)

underlying_grid = LatitudeLongitudeGrid(CPU();
                                         size = (Nλ, Nφ, Nz), halo = (5, 5, 5),
                                         longitude = λ_bounds, latitude = φ_bounds,
                                         z = z_faces,
                                         topology = (Bounded, Bounded, Bounded))

λc = λnodes(underlying_grid, Center())
φc = φnodes(underlying_grid, Center())
zc = znodes(underlying_grid, Center())

fetch_etopo(ETOPO_NC, λ_bounds, φ_bounds)
bottom_height = regional_bathymetry(ETOPO_NC, λc, φc; max_depth = Lz)
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height))

Ω = 7.292115e-5
@printf("Coriolis f: %.2e s⁻¹ (%.0f°N)  …  %.2e s⁻¹ (%.0f°N)\n",
        2Ω*sind(φ_bounds[1]), φ_bounds[1], 2Ω*sind(φ_bounds[2]), φ_bounds[2])

# ==================================================================
# THE PRESCRIBED LARGE-SCALE STATE:  ocean at rest, script 8's stratification
# ==================================================================
const g_earth = 9.80665
const ρₒ      = 1025.0
@inline T̄(z) = 12 + 8e-3 * z
const S̄ = 35.0

# ---- the buffer-zone mask: 1 within n_sponge cells of an open edge, 0 inside ----
# used both by the sponge (below) and to taper the wind stress to zero at the
# open boundaries.
const n_sponge = 6           # cells
@inline function sponge_mask(λ, φ, z)
    dλ  = (λ_bounds[2] - λ) / (λc[2] - λc[1])
    dφs = (φ - φ_bounds[1]) / (φc[2] - φc[1])
    dφn = (φ_bounds[2] - φ) / (φc[2] - φc[1])
    d   = min(dλ, dφs, dφn)
    return clamp((n_sponge - d) / n_sponge, 0, 1) ^ 2
end

# ==================================================================
# BOUNDARY CONDITIONS
# ==================================================================
# --- surface wind stress: script 8's Qᵘ = −τ/ρₒ, ramped in time, and now
#     ALSO TAPERED TO ZERO through the sponge zone. Without the taper the model
#     drives a full Ekman layer right at the open edge, into an exterior that is
#     at rest — the transport converges in the boundary cells with nowhere to
#     go and organises into an edge jet. Tapering makes the buffer a genuine
#     quiet transition to the at-rest exterior. ---
τˣ, τʸ = 0.04, 0.04
@inline ramp(t) = tanh(t / 1day)
@inline wind_taper(λ, φ) = 1 - sponge_mask(λ, φ, 0.0)
@inline Qᵘ_wind(λ, φ, t, p) = - p.τ / p.ρ * ramp(t) * wind_taper(λ, φ)
@inline Qᵛ_wind(λ, φ, t, p) = - p.τ / p.ρ * ramp(t) * wind_taper(λ, φ)

# --- the open-boundary triad on the north, south and east edges ---
# exterior state is at rest, so every external value is zero.
#
# NORMAL velocity + tracers: `PerturbationAdvection` with a long (3-day)
# inflow_timescale and Inf outflow_timescale — a "soft" open boundary that
# lets the interior largely set the boundary value in both directions and
# only weakly pulls it back toward the exterior state on inflow.
#
# TANGENTIAL velocity: this Oceananigans version has no open-boundary scheme
# for the tangential component (that needs oblique radiation), so it was left
# at the default free-slip and grew an edge jet. Here we give it an explicit
# `GradientBoundaryCondition(0)` (zero normal-gradient — a crude Neumann /
# "do-nothing" outflow) so the boundary value follows the interior instead of
# being clamped.
# ---- VARIANT SWITCHES (see the A/B at the bottom) -------------------------
# MAB_SCHEME    : "pa" (script 9a as committed) | "oblique"
# MAB_TANGENTIAL: "gradient" (script 9a) | "radiated"
const MAB_SCHEME     = get(ENV, "MAB_SCHEME", "pa")
const MAB_TANGENTIAL = get(ENV, "MAB_TANGENTIAL", "gradient")

oarad = MAB_SCHEME == "oblique" ?
        ObliqueRadiation(inflow_timescale = 1day, outflow_timescale = Inf) :
        PerturbationAdvection(inflow_timescale = 3days, outflow_timescale = Inf)

# The tangential velocity is Center-located in the boundary-normal direction, so a
# Value-classification scheme dispatches on it. Script 9a used Gradient(0) because
# no tangential scheme was thought to exist; the neck-basin tests measured that as
# the over-dissipative end (0.330 against a truth of 1.000, versus 0.923 radiated).
tangential_bc(val) = MAB_TANGENTIAL == "radiated" ?
    ValueBoundaryCondition(val; scheme = NormalRadiation(inflow_timescale = 1day,
                                                         outflow_timescale = Inf)) :
    GradientBoundaryCondition(0.0)

u_bcs = FieldBoundaryConditions(
    top   = FluxBoundaryCondition(Qᵘ_wind, parameters = (τ = τˣ, ρ = ρₒ)),
    east  = NormalFlowBoundaryCondition(0.0; scheme = oarad),   # normal
    north = tangential_bc(0.0),                                 # tangential
    south = tangential_bc(0.0))                                 # tangential

v_bcs = FieldBoundaryConditions(
    top   = FluxBoundaryCondition(Qᵛ_wind, parameters = (τ = τʸ, ρ = ρₒ)),
    north = NormalFlowBoundaryCondition(0.0; scheme = oarad),   # normal
    south = NormalFlowBoundaryCondition(0.0; scheme = oarad),   # normal
    east  = tangential_bc(0.0))                                 # tangential

# barotropic transport — Flather (GravityWaveRadiation), external (U, η) = (0, 0)
U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
    east = GravityWaveRadiationBoundaryCondition((0, 0)))
V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
    north = GravityWaveRadiationBoundaryCondition((0, 0)),
    south = GravityWaveRadiationBoundaryCondition((0, 0)))

# free surface — Chapman (SurfaceWaveRadiation)
η_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face());
    east  = SurfaceWaveRadiationBoundaryCondition(),
    north = SurfaceWaveRadiationBoundaryCondition(),
    south = SurfaceWaveRadiationBoundaryCondition())

# tracers — radiate out / relax to the exterior profile on inflow
T_bcs = FieldBoundaryConditions(
    north = ValueBoundaryCondition((λ, z, t) -> T̄(z); scheme = oarad),
    south = ValueBoundaryCondition((λ, z, t) -> T̄(z); scheme = oarad),
    east  = ValueBoundaryCondition((φ, z, t) -> T̄(z); scheme = oarad))
S_bcs = FieldBoundaryConditions(
    north = ValueBoundaryCondition(S̄; scheme = oarad),
    south = ValueBoundaryCondition(S̄; scheme = oarad),
    east  = ValueBoundaryCondition(S̄; scheme = oarad))

# ==================================================================
# SPONGE LAYERS  (Relaxation with an edge mask)
# ==================================================================
# Two things happen in the buffer zone: (a) the prognostic fields are RELAXED
# toward the large-scale state, and (b) the horizontal viscosity is ramped up
# by ~30× so the boundary-trapped shear layer that the radiation condition
# leaves behind is smoothed away before it can organise into an edge jet.
# (Without the viscous part the first working version grew a ~1 m/s current
# pinned to the open south and east edges.)  `sponge_mask` / `n_sponge` are
# defined above (they are also used to taper the wind stress).

const sponge_rate = 1 / 30minutes
Fu = Relaxation(rate = sponge_rate, mask = sponge_mask, target = 0.0)
Fv = Relaxation(rate = sponge_rate, mask = sponge_mask, target = 0.0)
FT = Relaxation(rate = sponge_rate, mask = sponge_mask, target = (λ, φ, z, t) -> T̄(z))
FS = Relaxation(rate = sponge_rate, mask = sponge_mask, target = S̄)

const ν_interior, ν_sponge = 300.0, 1.0e4
@inline ν_horizontal(λ, φ, z, t) = ν_interior + ν_sponge * sponge_mask(λ, φ, z)

# ==================================================================
# MODEL
# ==================================================================
model = HydrostaticFreeSurfaceModel(grid;
    momentum_advection = WENOVectorInvariant(order = 5),
    tracer_advection   = WENO(order = 5),
    coriolis     = HydrostaticSphericalCoriolis(),
    free_surface = SplitExplicitFreeSurface(grid; substeps = 60),
    buoyancy     = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState()),
    tracers      = (:T, :S),
    closure      = (CATKEVerticalDiffusivity(),
                    HorizontalScalarDiffusivity(ν = ν_horizontal, κ = 100)),
    boundary_conditions = (u = u_bcs, v = v_bcs, U = U_bcs, V = V_bcs, η = η_bcs,
                           T = T_bcs, S = S_bcs),
    forcing = (u = Fu, v = Fv, T = FT, S = FS))

set!(model, T = (λ, φ, z) -> T̄(z), S = S̄)
:e in propertynames(model.tracers) && set!(model, e = 1e-6)

simulation = Simulation(model, Δt = 30, stop_time = 6days)
conjure_time_step_wizard!(simulation, cfl = 0.3, max_Δt = 4minutes)

function progress(sim)
    u, v, w = sim.model.velocities
    η = sim.model.free_surface.displacement
    @printf("  %s  Δt=%s  |u|=%.2f  |v|=%.2f  |η|=%.3f m\n",
            prettytime(sim), prettytime(sim.Δt),
            maximum(abs, u), maximum(abs, v), maximum(abs, η))
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, TimeInterval(12hours))

filename = get(ENV, "MAB_TAG", "regional_mab_obc")
u, v, w = model.velocities
η  = model.free_surface.displacement
Ū  = Field(Average(model.velocities.u, dims = 3))
V̄  = Field(Average(model.velocities.v, dims = 3))
outputs = (; u, v, T = model.tracers.T, S = model.tracers.S, η, Ū, V̄)
simulation.output_writers[:fields] = JLD2Writer(model, outputs;
    filename = outpath(filename * ".jld2"), schedule = TimeInterval(3hours), overwrite_existing = true)

run!(simulation)
println("\n✅ Simulation complete. Verifying …")

# ==================================================================
# VERIFICATION
# ==================================================================
uts = FieldTimeSeries(outpath(filename * ".jld2"), "u")
vts = FieldTimeSeries(outpath(filename * ".jld2"), "v")
Tts = FieldTimeSeries(outpath(filename * ".jld2"), "T")
ηts = FieldTimeSeries(outpath(filename * ".jld2"), "η")
Ūts = FieldTimeSeries(outpath(filename * ".jld2"), "Ū")
V̄ts = FieldTimeSeries(outpath(filename * ".jld2"), "V̄")
times = ηts.times ; Nt = length(times)
tdays = times ./ day

z   = collect(znodes(underlying_grid, Center()))
Δz  = collect(zspacings(underlying_grid, Center()))
depth = -bottom_height
wet   = depth .> 0
maskland(A) = [wet[i, j] ? A[i, j] : NaN for i in 1:Nλ, j in 1:Nφ]
xcenter(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
ycenter(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])

f = [2Ω * sind(φ) for φ in φc]
R = 6.371e6
Δλm = deg2rad(λc[2] - λc[1]) ; Δφm = deg2rad(φc[2] - φc[1])

last_day = max(1, Nt-15):Nt      # last ~2 days — filter the near-inertial wobble
tmean(ts) = mean(interior(ts[n]) for n in last_day)
un  = xcenter(tmean(uts))
vn  = ycenter(tmean(vts))
Tn  = tmean(Tts)
ηn  = tmean(ηts)[:, :, 1]
Ūn  = xcenter(tmean(Ūts))[:, :, 1]
V̄n  = ycenter(tmean(V̄ts))[:, :, 1]

# --- 1. the run stayed bounded: domain-max speed must plateau, not grow ---
umax_t = [maximum(abs, interior(uts[n])) for n in 1:Nt]
vmax_t = [maximum(abs, interior(vts[n])) for n in 1:Nt]
late_growth = (max(umax_t[Nt], vmax_t[Nt]) - max(umax_t[Nt÷2], vmax_t[Nt÷2])) / (tdays[Nt] - tdays[Nt÷2])
@printf("stability:  max speed  day %d → %.2f m/s,  day %d → %.2f m/s   (late growth %.3f m/s/day)\n",
        Int(round(tdays[Nt÷2])), max(umax_t[Nt÷2], vmax_t[Nt÷2]),
        Int(round(tdays[Nt])),   max(umax_t[Nt], vmax_t[Nt]), late_growth)

# --- 2. domain volume is conserved: d/dt ∫∫η dA ≈ 0 (the barotropic open BCs
#        let the net flux adjust so the domain neither fills nor drains) ---
cellarea = [wet[i,j] ? R^2 * cosd(φc[j]) * Δλm * Δφm : 0.0 for i in 1:Nλ, j in 1:Nφ]
Vη(n) = sum(interior(ηts[n])[i,j,1] * cellarea[i,j] for i in 1:Nλ, j in 1:Nφ)
dVdt_late = (Vη(Nt) - Vη(last_day[1])) / (times[Nt] - times[last_day[1]])       # m³/s
dVdt_early = (Vη(Nt÷2) - Vη(1)) / (times[Nt÷2] - times[1])
@printf("volume budget:  d/dt ∫∫η dA  =  %+.3f Sv (days 0–3)  →  %+.4f Sv (days 4–6)   (→ 0 = conserved)\n",
        dVdt_early/1e6, dVdt_late/1e6)
mean_η_late = mean(Vη(n) for n in last_day) / sum(cellarea)
@printf("             mean sea level (last 2 days):  %+.4f m\n", mean_η_late)

# --- 3. sponge zones track the state at rest ---
inσ = [sponge_mask(λc[i], φc[j], -10.0) > 0.5 && wet[i,j] for i in 1:Nλ, j in 1:Nφ]
Trms(sel) = sqrt(mean((Tn[i,j,k] - T̄(z[k]))^2
                      for i in 1:Nλ, j in 1:Nφ, k in 1:Nz
                      if sel(i,j) && z[k] > bottom_height[i,j]))
sprms(sel) = sqrt(mean(un[i,j,k]^2 + vn[i,j,k]^2
                       for i in 1:Nλ, j in 1:Nφ, k in 1:Nz
                       if sel(i,j) && z[k] > bottom_height[i,j]))
@printf("T − T̄ RMS:      sponge %.3f °C    free interior %.3f °C\n",
        Trms((i,j)->inσ[i,j]), Trms((i,j)-> !inσ[i,j] && wet[i,j]))
@printf("speed RMS:      sponge %.3f m/s   free interior %.3f m/s\n",
        sprms((i,j)->inσ[i,j]), sprms((i,j)-> !inσ[i,j] && wet[i,j]))

# --- 4. no grid-scale noise at the open edges ---
function wiggle(A2)
    w = Float64[]
    for i in 2:Nλ-1, j in 2:Nφ-1
        (wet[i-1,j] && wet[i+1,j] && wet[i,j-1] && wet[i,j+1]) || continue
        push!(w, abs(A2[i+1,j] - 2A2[i,j] + A2[i-1,j]) + abs(A2[i,j+1] - 2A2[i,j] + A2[i,j-1]))
    end
    mean(w)
end
speed_surf = hypot.(un[:,:,Nz], vn[:,:,Nz])
edge = [ (i ≤ 3 || i ≥ Nλ-2 || j ≤ 3 || j ≥ Nφ-2) for i in 1:Nλ, j in 1:Nφ ]
w_edge = wiggle([edge[i,j] && wet[i,j] ? speed_surf[i,j] : 0.0 for i in 1:Nλ, j in 1:Nφ])
w_all  = wiggle([wet[i,j] ? speed_surf[i,j] : 0.0 for i in 1:Nλ, j in 1:Nφ])
@printf("grid-scale wiggle:  open-edge band %.2e   whole domain %.2e   (ratio %.1f)\n",
        w_edge, w_all, w_edge / max(w_all, eps()))

# --- 5. Ekman transport in the deep interior (same diagnostic as script 8) ---
ksurf = findall(>(-40), z)
uag = un[:, :, ksurf] .- reshape(Ūn, Nλ, Nφ, 1)
vag = vn[:, :, ksurf] .- reshape(V̄n, Nλ, Nφ, 1)
Uek = dropdims(sum(uag .* reshape(Δz[ksurf], 1, 1, :), dims = 3), dims = 3)
Vek = dropdims(sum(vag .* reshape(Δz[ksurf], 1, 1, :), dims = 3), dims = 3)
deepint = [wet[i,j] && depth[i,j] > 1000 && !inσ[i,j] for i in 1:Nλ, j in 1:Nφ]
f̄ = mean(f[j] for (i,j) in Tuple.(findall(deepint)))
@printf("Ekman transport (deep interior, sponges excluded):  model (%.2f, %.2f)   theory (%.2f, %.2f) m²/s\n",
        mean(Uek[deepint]), mean(Vek[deepint]), τʸ/(ρₒ*f̄), -τˣ/(ρₒ*f̄))

# --- 6. geostrophic interior below the Ekman layer ---
dηdx = zeros(Nλ, Nφ) ; dηdy = zeros(Nλ, Nφ)
for i in 2:Nλ-1, j in 2:Nφ-1
    dηdx[i,j] = (ηn[i+1,j] - ηn[i-1,j]) / (2Δλm * R * cosd(φc[j]))
    dηdy[i,j] = (ηn[i,j+1] - ηn[i,j-1]) / (2Δφm * R)
end
ksub  = findall(zz -> -90 < zz < -45, z)
u_geo = dropdims(mean(un[:,:,ksub], dims = 3), dims = 3)
v_geo = dropdims(mean(vn[:,:,ksub], dims = 3), dims = 3)
gmask = [wet[i,j] && depth[i,j] > 150 && !inσ[i,j] && 6 ≤ i ≤ Nλ-5 && 6 ≤ j ≤ Nφ-5
         for i in 1:Nλ, j in 1:Nφ]
idx = Tuple.(findall(gmask))
fv = [f[j]*v_geo[i,j] for (i,j) in idx] ; gx = [ g_earth*dηdx[i,j] for (i,j) in idx]
fu = [f[j]*u_geo[i,j] for (i,j) in idx] ; gy = [-g_earth*dηdy[i,j] for (i,j) in idx]
r_x = cor(fv, gx) ; r_y = cor(fu, gy)
@printf("geostrophy (below the Ekman layer, sponges excluded):  corr(f·v, g∂ₓη) = %.2f   corr(f·u, -g∂ᵧη) = %.2f\n", r_x, r_y)

# --- 7. sea-surface setup vs script 8's closed box ---
η_span = maximum(filter(!isnan, maskland(ηn))) - minimum(filter(!isnan, maskland(ηn)))
@printf("sea-surface setup (max − min η, day 6):  %.3f m   (script 8, closed box: ≈ 0.28 m)\n", η_span)
@printf("wall-clock: 6 simulated days in ~%s\n", prettytime(simulation.run_wall_time))

# ==================================================================
# FIGURE
# ==================================================================
ηmax = maximum(abs, filter(!isnan, maskland(ηn)))
fig = Figure(size = (1500, 880))

axb = Axis(fig[1,1], title = "bathymetry (m) — open: N, S, E", xlabel = "°E", ylabel = "°N", aspect = DataAspect())
hmb = heatmap!(axb, λc, φc, replace(bottom_height, 0.0 => NaN), colormap = :deep)
for (p1, p2) in (((λ_bounds[1],φ_bounds[2]),(λ_bounds[2],φ_bounds[2])),
                 ((λ_bounds[1],φ_bounds[1]),(λ_bounds[2],φ_bounds[1])),
                 ((λ_bounds[2],φ_bounds[1]),(λ_bounds[2],φ_bounds[2])))
    lines!(axb, [p1[1], p2[1]], [p1[2], p2[2]], color = :orange, linewidth = 3)
end
Colorbar(fig[1,2], hmb)

axη = Axis(fig[1,3], title = @sprintf("η (m), day %d", Int(round(tdays[Nt]))), xlabel = "°E", ylabel = "°N", aspect = DataAspect())
hmη = heatmap!(axη, λc, φc, maskland(ηn), colormap = :balance, colorrange = (-ηmax, ηmax))
Colorbar(fig[1,4], hmη)

axS = Axis(fig[1,5], title = "domain-max speed (m/s)", xlabel = "t (days)", ylabel = "m/s")
lines!(axS, tdays, umax_t, label = "max |u|") ; lines!(axS, tdays, vmax_t, label = "max |v|")
axislegend(axS, position = :lt)

axU = Axis(fig[2,1], title = "depth-avg speed (m/s) + isobaths", xlabel = "°E", ylabel = "°N", aspect = DataAspect())
hmU = heatmap!(axU, λc, φc, maskland(hypot.(Ūn, V̄n)), colormap = :speed, colorrange = (0, 0.15))
contour!(axU, λc, φc, replace(depth, 0.0 => NaN), levels = [100, 200, 1000, 2000, 3000], color = :black, linewidth = 0.4)
Colorbar(fig[2,2], hmU)

axg = Axis(fig[2,3], title = @sprintf("geostrophy  (r = %.2f, %.2f)", r_x, r_y),
           xlabel = "pressure-gradient term (m s⁻²)", ylabel = "Coriolis term (m s⁻²)")
scatter!(axg, gx, fv, markersize = 3, label = "f·v vs g ∂ₓη")
scatter!(axg, gy, fu, markersize = 3, label = "f·u vs −g ∂ᵧη")
ablines!(axg, 0, 1, color = :black) ; axislegend(axg, position = :lt)

axw = Axis(fig[2,5], title = "surface speed (m/s), final", xlabel = "°E", ylabel = "°N", aspect = DataAspect())
hmw = heatmap!(axw, λc, φc, maskland(speed_surf), colormap = :speed, colorrange = (0, 0.2))
contour!(axw, λc, φc, replace(depth, 0.0 => NaN), levels = [200, 2000], color = (:black, 0.4), linewidth = 0.4)
Colorbar(fig[2,6], hmw)

save(filename * "_verification.png", fig)
println("✅ Saved verification figure to $(filename)_verification.png")

# ==================================================================
# ANIMATION
# ==================================================================
nobs = Observable(1)
surf = @lift maskland(hypot.(xcenter(interior(uts[$nobs]))[:, :, Nz],
                             ycenter(interior(vts[$nobs]))[:, :, Nz]))
ttl  = @lift @sprintf("MAB with open boundaries — %s", prettytime(times[$nobs]))
figa = Figure(size = (780, 640))
Label(figa[0, 1:2], ttl, fontsize = 17, tellwidth = false)
axa = Axis(figa[1, 1], xlabel = "°E", ylabel = "°N", aspect = DataAspect())
hma = heatmap!(axa, λc, φc, surf, colormap = :speed, colorrange = (0, 0.2))
contour!(axa, λc, φc, replace(depth, 0.0 => NaN), levels = [100, 200, 1000, 3000],
         color = (:white, 0.6), linewidth = 0.5)
Colorbar(figa[1, 2], hma, label = "surface speed (m/s)")
record(figa, outpath(filename * ".mp4"), 1:Nt, framerate = 10) do i
    nobs[] = i
end
println("✅ Saved animation to $(filename).mp4")

# ==================================================================
# INFLOW vs OUTFLOW ATTRIBUTION
#
# The open boundaries here carry ZERO exterior data: every external value is
# (0, 0), so on inflow the boundary's only instruction is "relax toward rest".
# The true inflow at the MAB's southern and eastern edges is nothing like rest,
# so no radiation scheme can invent it. If that is the binding constraint, the
# residual should sit preferentially on faces where water is ENTERING.
#
# Measured as the boundary-adjacent speed anomaly — |u| in the outermost wet
# cells relative to the free interior — split by the sign of the normal velocity.
# A large inflow/outflow ratio says the limiting factor is the exterior data, not
# the scheme; a ratio near one says the scheme itself is leaking.
# ==================================================================
let
    uts = FieldTimeSeries(outpath(filename * ".jld2"), "u")
    vts = FieldTimeSeries(outpath(filename * ".jld2"), "v")
    nlast = length(uts.times)
    ui = Array(interior(uts[nlast])); vi = Array(interior(vts[nlast]))
    nx, ny, nz = size(ui, 1) - 1, size(vi, 2) - 1, size(ui, 3)

    ucc(i, j, k) = 0.5 * (ui[i, j, k] + ui[i+1, j, k])
    vcc(i, j, k) = 0.5 * (vi[i, j, k] + vi[i, j+1, k])
    spd(i, j, k) = sqrt(ucc(i, j, k)^2 + vcc(i, j, k)^2)

    # interior reference: away from every edge and above the bottom
    ref = Float64[]
    for i in 12:(nx-12), j in 12:(ny-12), k in (nz-6):nz
        v = spd(i, j, k); isfinite(v) && v > 0 && push!(ref, v)
    end
    s_int = isempty(ref) ? NaN : sqrt(sum(abs2, ref) / length(ref))

    # open edges: east (i = nx, normal +x), north (j = ny, +y), south (j = 1, −y)
    inflow = Float64[]; outflow = Float64[]
    push_face!(sp, un_into) = begin
        (isfinite(sp) && sp > 0) || return
        push!(un_into > 0 ? inflow : outflow, sp)
    end
    for j in 6:(ny-6), k in (nz-6):nz
        push_face!(spd(nx, j, k), -ui[nx+1, j, k])   # east: entering = −u
    end
    for i in 6:(nx-6), k in (nz-6):nz
        push_face!(spd(i, ny, k), -vi[i, ny+1, k])   # north: entering = −v
        push_face!(spd(i, 1,  k),  vi[i, 1,   k])    # south: entering = +v
    end

    rms(a) = isempty(a) ? NaN : sqrt(sum(abs2, a) / length(a))
    s_in, s_out = rms(inflow), rms(outflow)
    println("\n", "="^78)
    @printf("INFLOW/OUTFLOW ATTRIBUTION  [scheme=%s tangential=%s]\n", MAB_SCHEME, MAB_TANGENTIAL)
    @printf("  free interior speed          %.4f m/s   (%d cells)\n", s_int, length(ref))
    @printf("  boundary band, INFLOW faces  %.4f m/s   (%d cells)  → %.2f× interior\n",
            s_in, length(inflow), s_in / s_int)
    @printf("  boundary band, OUTFLOW faces %.4f m/s   (%d cells)  → %.2f× interior\n",
            s_out, length(outflow), s_out / s_int)
    @printf("  inflow / outflow             %.2f\n", s_in / s_out)
    println("  >1 ⇒ the residual is concentrated where water ENTERS, i.e. the binding")
    println("       constraint is the missing exterior data rather than the scheme.")
    println("="^78)
end
