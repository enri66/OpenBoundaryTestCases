# ==================================================================
# DUMBBELL — can an eddy leave, and enter, a downscaled region?
#
# WHY THIS TEST, AND WHY IT IS SHAPED THE WAY IT IS.
#
# Open boundary conditions are not exact solutions. We impose a hyperbolic
# radiation condition on what is really a parabolic problem, so we always
# end up overspecifying and settling for stability. That is why barotropic
# tests (shallow water genuinely IS hyperbolic) look so much better, and it
# is why chasing "error against a reference solution" — which is what the
# kelvin_wave, radiating_gaussian, stratified_gaussian and supercritical
# attempts all did — cannot cleanly separate one radiation scheme from
# another. Those metrics mostly measure how badly the rest of the setup was
# overspecified.
#
# So this test asks a different, honest question:
#
#     Can a coherent eddy LEAVE the domain through an open boundary,
#     and can one ENTER through it, without being destroyed, reflected,
#     or blowing the run up over a long integration?
#
# It will never reproduce the full-domain solution. It is not supposed to.
#
# THE GEOMETRY, from MOM6's `dumbbell_initialize_topography`: two full-width
# basins joined by a narrow neck — land where |x − x_c| ≤ 0.25·L and
# |y − y_c| ≥ 0.5·frac·W. Cutting the domain AT THE NECK is what makes this
# a downscaling test: the boundary sits in the throat that the eddy must
# transit, so the eddy is forced across it.
#
#   TRUTH   : the whole dumbbell, both basins. No open boundaries.
#   LEAVING : west basin only, open boundary in the middle of the neck.
#             The eddy must exit.
#   ENTERING: east basin only, open boundary in the middle of the neck,
#             fed from TRUTH. The eddy must come in.
#
# THE EDDY. A single geostrophic vortex on an f-plane just sits there, so
# the eddy here is a DIPOLE, which self-propagates in a straight line at a
# speed we can set. It is built in exact thermal-wind balance from a
# streamfunction ψ(x,y,z) = Ψ(x,y)·F(z):
#
#     u = −∂Ψ/∂y · F(z),   v = ∂Ψ/∂x · F(z)
#     η = (f/g)·Ψ·F(0),    b = N²z + f·Ψ·F′(z)
#
# so it starts balanced and does not spend the first week radiating the
# adjustment away. F(z) = exp(z/hₑ) makes it surface-intensified — i.e.
# genuinely BAROCLINIC, which matters: with Nz = 1 the split-explicit
# barotropic corrector sets u = U/H exactly and no baroclinic open-boundary
# scheme has any degrees of freedom left (that trap ate two earlier tests).
#
# Run:  julia -t 8 --project=. validation/dumbbell.jl
# ==================================================================

include(joinpath(@__DIR__, "obc_output.jl"))

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalRadiation, ObliqueRadiation, TracerReservoir,
                                       SurfaceWaveRadiationBoundaryCondition,
                                       PerturbationAdvection,
                                       GravityWaveRadiationBoundaryCondition
using Printf
using Statistics
using CairoMakie

# ---------------- geometry (after MOM6 dumbbell_initialize_topography) ----------------
const Lx_db = 900kilometers
const Ly_db = 450kilometers
const H_db  = 1000.0

const Δ_db  = 7.5kilometers
const Nx_db = round(Int, Lx_db / Δ_db)      # 120
const Ny_db = round(Int, Ly_db / Δ_db)      # 60

# Nz is the one free lever on cost. The number of TIMESTEPS is set by
# N_steps ~ Nx/Ro (the eddy must cross Nx cells, and each step is limited by the
# internal wave speed c₁ = NH/π, which for a dipole that is not screened —
# Bu = (Rd/R)² ~ 1 — is ≈ v_max/Ro). Nz multiplies the cost per step but not the
# number of steps, so cutting it is a straight saving. Four levels still resolve
# the surface-intensified shear the oblique scheme acts on, and MOM6's own
# dumbbell is a layered model.
const Nz_db = 4

const x_c = Lx_db / 2
const y_c = Ly_db / 2
const neck_half_length = 100kilometers      # neck spans |x − x_c| ≤ this
const neck_half_width  = 110kilometers      # water in the neck where |y − y_c| < this

"""Bottom height: 0 (dry land) in the two shoulders that form the neck, −H elsewhere."""
@inline function dumbbell_bottom(x, y)
    in_neck = abs(x - x_c) <= neck_half_length
    is_land = in_neck & (abs(y - y_c) >= neck_half_width)
    return ifelse(is_land, zero(x), convert(typeof(x), -H_db))
end

# ---------------- stratification and rotation ----------------
# f is subtropical rather than mid-latitude, which raises the deformation radius
# to ≈ 64 km. That matters: in a stratified fluid a vortex's influence on its
# partner is screened over Rd, so a dipole whose lobes sit several Rd apart barely
# propagates. A first attempt at f = 1e-4 (Rd = 32 km) with the lobes 3 Rd apart
# crawled at 0.067 m/s — e⁻³ of the unscreened point-vortex estimate — and never
# reached the neck in 20 days. It is also the right regime for the eventual
# application: Gulf Stream rings arriving at the MAB.
const f_db = 5e-5                            # f-plane, ≈ 20°N
const N_db = 1e-2                            # buoyancy frequency  ⇒  Rd = NH/(fπ) ≈ 64 km
const g_db = 9.81

# ---------------- the dipole ----------------
const R_eddy = 50kilometers                  # vortex radius (≈ 0.8 Rd — compact, coherent)
const d_eddy = 0.7 * R_eddy                  # half-separation ⇒ lobes ≈ 1.1 Rd apart
const v_max  = 0.6                           # peak swirl speed [m/s]
const A_eddy = v_max * R_eddy * sqrt(ℯ)      # Ψ amplitude: max|∂Ψ/∂r| = A/R·e^(−1/2)
const h_e    = 500.0                         # e-folding depth — surface-intensified
const x_eddy = 200kilometers                 # launch point, in the west basin

@inline F_z(z)  = exp(z / h_e)
@inline dF_dz(z) = exp(z / h_e) / h_e

"""
    make_dipole(; x₀, y₀, θ)

Balanced baroclinic dipole launched from `(x₀, y₀)` and self-propagating along the
heading `θ` (degrees anticlockwise from +x). Returns the four initial-condition
functions `(u, v, η, b)`.

The lobes sit perpendicular to the heading — positive lobe to the right of it,
negative to the left — which is what makes the pair translate along `θ`. `θ = 0`
sends it due east, straight at an east-facing boundary; a non-zero `θ` makes it
strike that boundary OBLIQUELY, which is the case `ObliqueRadiation` exists for
and the one every earlier benchmark in this project failed to produce.
"""
function make_dipole(; x₀ = x_eddy, y₀ = y_c, θ = 0.0)
    xp, yp = x₀ + d_eddy * sind(θ), y₀ - d_eddy * cosd(θ)   # positive lobe
    xm, ym = x₀ - d_eddy * sind(θ), y₀ + d_eddy * cosd(θ)   # negative lobe

    gp(x, y) = exp(-((x - xp)^2 + (y - yp)^2) / (2R_eddy^2))
    gm(x, y) = exp(-((x - xm)^2 + (y - ym)^2) / (2R_eddy^2))

    Ψ(x, y)  = A_eddy * (gp(x, y) - gm(x, y))
    Ψx(x, y) = A_eddy * (-(x - xp) / R_eddy^2 * gp(x, y) + (x - xm) / R_eddy^2 * gm(x, y))
    Ψy(x, y) = A_eddy * (-(y - yp) / R_eddy^2 * gp(x, y) + (y - ym) / R_eddy^2 * gm(x, y))

    return (u = (x, y, z) -> -Ψy(x, y) * F_z(z),
            v = (x, y, z) ->  Ψx(x, y) * F_z(z),
            η = (x, y, z) ->  f_db / g_db * Ψ(x, y) * F_z(0),
            b = (x, y, z) ->  N_db^2 * z + f_db * Ψ(x, y) * dF_dz(z))
end

# Self-propagation speed: the velocity one lobe induces at the other.
const c_dipole = A_eddy * (2d_eddy / R_eddy^2) * exp(-(2d_eddy)^2 / (2R_eddy^2))

const T_DB  = 55days   # dipole reaches the cut at ≈ 30 d and clears the neck at ≈ 42 d
const Δt_db = 300.0    # 600 s blew up at Ro = 0.24; 300 s is stable

# The downscaled runs are cut here, in the middle of the neck. The exterior data
# they need is saved as a slab of the truth run straddling this face.
const i_cut = round(Int, x_c / Δ_db) + 1        # x-FACE index at x = x_c  (= 61, x = 450 km)

# The NECK sub-domain (open at BOTH ends) is cut at these two faces. Both sit
# comfortably inside the throat (x = 350–550 km), so its north and south walls are
# straight land the whole way along.
const neck_i_west = 50                         # x = 367.5 km
const neck_i_east = 72                         # x = 532.5 km

# One slab wide enough to serve every cut: the single mid-neck cut used by the
# half-domain tests, and both ends of the neck sub-domain, plus a 3-cell margin.
const cut_slab = (neck_i_west - 3):(neck_i_east + 3)

"""
    dumbbell_simulation(; filename, stop_time)

The TRUTH run: the whole dumbbell, closed everywhere. The dipole is launched in
the west basin and has to transit the neck into the east basin.
"""
function dumbbell_simulation(; dipole = make_dipole(), forcing = NamedTuple(),
                               stop_time = T_DB, Δt = Δt_db,
                               filename = "dumbbell_truth")

    underlying = RectilinearGrid(CPU();
                                 size = (Nx_db, Ny_db, Nz_db),
                                 x = (0, Lx_db), y = (0, Ly_db), z = (-H_db, 0),
                                 halo = (7, 7, 7),
                                 topology = (Bounded, Bounded, Bounded))

    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(dumbbell_bottom))

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
        coriolis = FPlane(f = f_db),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        momentum_advection = WENOVectorInvariant(),
        tracer_advection = WENO(order = 5),
        closure = HorizontalScalarDiffusivity(ν = 50, κ = 50),
        forcing)

    set!(model, u = dipole.u, v = dipole.v, b = dipole.b)
    set!(model.free_surface.displacement, dipole.η)

    simulation = Simulation(model; Δt, stop_time)

    function progress(sim)
        u, v, w = sim.model.velocities
        η = sim.model.free_surface.displacement
        @printf("  %-9s  max|u| = %.3f m/s   max|η| = %.3f m\n",
                prettytime(sim), maximum(abs, u), maximum(abs, η))
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(2days))

    u, v, w = model.velocities
    η = model.free_surface.displacement
    ζ = ∂x(v) - ∂y(u)

    # η is a 2-D top-FACE field, so it cannot share an `indices` slice with the
    # Center-located 3-D fields — it needs its own writer (same trap as script 9a).
    simulation.output_writers[:surface] = JLD2Writer(model, (; ζ, u, v);
        filename = outpath(filename * ".jld2"),
        indices = (:, :, Nz_db),
        schedule = TimeInterval(3hours),
        overwrite_existing = true)

    simulation.output_writers[:eta] = JLD2Writer(model, (; η);
        filename = outpath(filename * "_eta.jld2"),
        schedule = TimeInterval(3hours),
        overwrite_existing = true)

    # Exterior data for the downscaled runs: a full-depth slab spanning the neck.
    # Two-hourly: the eddy moves ~0.09 m/s, i.e. ~0.09 cells per frame, and the
    # slab is wide enough to serve both ends of the neck sub-domain, so hourly
    # would be ~600 MB for a long run for no gain in the baroclinic signal.
    b = model.tracers.b
    simulation.output_writers[:cut] = JLD2Writer(model, (; u, v, b);
        filename = outpath(filename * "_cut.jld2"),
        indices = (cut_slab, :, :),
        schedule = TimeInterval(2hours),
        overwrite_existing = true)

    simulation.output_writers[:cut_eta] = JLD2Writer(model, (; η);
        filename = outpath(filename * "_cut_eta.jld2"),
        indices = (cut_slab, :, :),
        schedule = TimeInterval(2hours),
        overwrite_existing = true)

    return simulation
end

# ==================================================================
# the downscaled runs — half the dumbbell, open boundary in the neck
# ==================================================================
"""
    cut_exterior(truth)

Load the truth run's slab at the cut and return time-interpolating closures for
the exterior data an open boundary needs: the normal velocity `u`, the tangential
`v`, the tracer `b`, the depth-integrated transport `U`, and `η`.

Index bookkeeping. The slab spans truth x-indices `cut_slab = i_cut-3 : i_cut+3`,
so slab-local index `l = i − first(cut_slab) + 1`. The cut is at truth x-FACE
`i_cut`, which lies between truth CENTRES `i_cut−1` and `i_cut`. Hence:

  * normal velocity, on the face itself     → local `i_cut − first + 1`
  * the west sub-domain's east halo centre  → truth centre `i_cut`     (outside it)
  * the east sub-domain's west halo centre  → truth centre `i_cut − 1` (outside it)

`η ᵉˣᵗ` for Flather is taken at the sub-domain's LAST INTERIOR centre, so that
`Uᵇ = Uᵉˣᵗ + √(gH)(ηᵇ − ηᵉˣᵗ)` returns `Uᵉˣᵗ` exactly when the downscaled solution
matches the truth — the condition is then consistent with the truth rather than
fighting it. (Getting this wrong is what blew up supercritical Part 1.)
"""
function cut_exterior(truth = "dumbbell_truth")
    uts = FieldTimeSeries(outpath(truth * "_cut.jld2"), "u")
    vts = FieldTimeSeries(outpath(truth * "_cut.jld2"), "v")
    bts = FieldTimeSeries(outpath(truth * "_cut.jld2"), "b")
    ηts = FieldTimeSeries(outpath(truth * "_cut_eta.jld2"), "η")
    times = collect(uts.times)

    U = [Array(interior(uts[n])) for n in eachindex(times)]    # (7, Ny, Nz)
    V = [Array(interior(vts[n])) for n in eachindex(times)]
    B = [Array(interior(bts[n])) for n in eachindex(times)]
    E = [Array(interior(ηts[n])) for n in eachindex(times)]    # (7, Ny, 1)

    lo = first(cut_slab)
    l_face  = i_cut - lo + 1        # the cut face
    l_east  = i_cut - lo + 1        # truth centre i_cut     — outside a WEST sub-domain
    l_west  = i_cut - 1 - lo + 1    # truth centre i_cut − 1 — outside an EAST sub-domain

    Δz = H_db / Nz_db
    Utr = [vec(sum(u[l_face, :, :], dims = 2)) .* Δz for u in U]   # depth-integrated, per j

    @inline function frame(t)
        t <= times[1]   && return (1, 1, 0.0)
        t >= times[end] && return (length(times), length(times), 0.0)
        n = searchsortedlast(times, t)
        return (n, n + 1, (t - times[n]) / (times[n+1] - times[n]))
    end

    ny, nz = size(U[1], 2), size(U[1], 3)
    cj(j) = clamp(j, 1, ny)
    ck(k) = clamp(k, 1, nz)

    @inline lerp(A, n1, n2, w, l, j, k) = (1 - w) * A[n1][l, cj(j), ck(k)] + w * A[n2][l, cj(j), ck(k)]

    return (; times,
              u = (l, j, k, t) -> (nf = frame(t); lerp(U, nf[1], nf[2], nf[3], l, j, k)),
              v = (l, j, k, t) -> (nf = frame(t); lerp(V, nf[1], nf[2], nf[3], l, j, k)),
              b = (l, j, k, t) -> (nf = frame(t); lerp(B, nf[1], nf[2], nf[3], l, j, k)),
              η = (l, j, t)    -> (nf = frame(t); lerp(E, nf[1], nf[2], nf[3], l, j, 1)),
              U = (j, t)       -> begin
                    nf = frame(t)
                    (1 - nf[3]) * Utr[nf[1]][cj(j)] + nf[3] * Utr[nf[2]][cj(j)]
                  end,
              l_face, l_east, l_west)
end

"""
    dumbbell_downscaled(; side, scheme, exterior, closed, filename)

Half the dumbbell, cut in the middle of the neck.

  * `side = :west` — the eddy starts inside and must LEAVE through the east boundary.
  * `side = :east` — the domain starts quiescent and the eddy must ENTER through
    the west boundary.

`scheme` is the open-boundary scheme on the boundary-normal velocity — the thing
under test. `closed = true` replaces the open boundary with a wall: the control
that shows what total failure looks like.
"""
function dumbbell_downscaled(; side::Symbol,
                               dipole = make_dipole(),
                               forcing = NamedTuple(),
                               seed_dipole = nothing,
                               scheme = NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
                               tracer_scheme = NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
                               exterior = nothing,
                               closed = false,
                               stop_time = T_DB, Δt = Δt_db,
                               filename = "dumbbell_sub")

    side in (:west, :east) || throw(ArgumentError("side must be :west or :east"))
    Nx_sub = round(Int, (Lx_db / 2) / Δ_db)
    xspan  = side === :west ? (0.0, x_c) : (x_c, Lx_db)

    underlying = RectilinearGrid(CPU();
                                 size = (Nx_sub, Ny_db, Nz_db),
                                 x = xspan, y = (0, Ly_db), z = (-H_db, 0),
                                 halo = (7, 7, 7),
                                 topology = (Bounded, Bounded, Bounded))

    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(dumbbell_bottom))

    bcs = NamedTuple()
    if !closed
        ext = exterior
        lc  = side === :west ? ext.l_east : ext.l_west   # centre just OUTSIDE the sub-domain
        lη  = side === :west ? ext.l_west : ext.l_east   # centre just INSIDE  (see docstring)
        lf  = ext.l_face

        u_ext(j, k, grid, clock, f) = ext.u(lf, j, k, clock.time)
        v_ext(j, k, grid, clock, f) = ext.v(lc, j, k, clock.time)
        b_ext(j, k, grid, clock, f) = ext.b(lc, j, k, clock.time)
        U_ext(j, k, grid, clock, f) = (ext.U(j, clock.time), ext.η(lη, j, clock.time))

        openside = side === :west ? :east : :west
        u_bcs = FieldBoundaryConditions(; openside =>
                    NormalFlowBoundaryCondition(u_ext; scheme, discrete_form = true))
        v_bcs = FieldBoundaryConditions(; openside =>
                    ValueBoundaryCondition(v_ext; discrete_form = true))
        b_bcs = FieldBoundaryConditions(; openside =>
                    ValueBoundaryCondition(b_ext; scheme = tracer_scheme, discrete_form = true))
        U_bcs = FieldBoundaryConditions(underlying, (Face(), Center(), nothing);
                    openside => GravityWaveRadiationBoundaryCondition(U_ext; discrete_form = true))

        bcs = (u = u_bcs, v = v_bcs, b = b_bcs, U = U_bcs)
    end

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
        coriolis = FPlane(f = f_db),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        momentum_advection = WENOVectorInvariant(),
        tracer_advection = WENO(order = 5),
        closure = HorizontalScalarDiffusivity(ν = 50, κ = 50),
        boundary_conditions = bcs,
        forcing)

    # The west sub-domain starts with the dipole; the east one starts quiescent
    # and stratified, so anything that appears in it came through the boundary.
    seeded = isnothing(seed_dipole) ? (side === :west) : seed_dipole
    if seeded
        set!(model, u = dipole.u, v = dipole.v, b = dipole.b)
        set!(model.free_surface.displacement, dipole.η)
    else
        set!(model, b = (x, y, z) -> N_db^2 * z)
    end

    simulation = Simulation(model; Δt, stop_time)

    function progress(sim)
        u, v, w = sim.model.velocities
        @printf("  [%s] %-9s  max|u| = %.3f m/s\n", side, prettytime(sim), maximum(abs, u))
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(10days))

    u, v, w = model.velocities
    ζ = ∂x(v) - ∂y(u)
    simulation.output_writers[:surface] = JLD2Writer(model, (; ζ, u, v);
        filename = outpath(filename * ".jld2"),
        indices = (:, :, Nz_db),
        schedule = TimeInterval(3hours),
        overwrite_existing = true)

    return simulation
end

# ==================================================================
# diagnostics + animation
# ==================================================================
"""Land mask on Center points, NaN over land, for plotting."""
function land_mask(x, y)
    m = ones(length(x), length(y))
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
        dumbbell_bottom(xi, yj) >= 0 && (m[i, j] = NaN)
    end
    return m
end

"""
    open_water_mask(x, y; buffer = 3)

True on wet points at least `buffer` cells from any land. The sharp 90° corners
where the neck meets the shoulders generate grid-scale vorticity spikes that are
much stronger than the eddy itself — an earlier version of `eddy_track` reported
peak |ζ|/f rising 0.64 → 1.89 and I nearly concluded the dipole was
disintegrating, when in fact it was perfectly coherent at ≈ 0.6 f and the metric
was reading two corner cells. Everything below is measured on this mask.
"""
function open_water_mask(x, y; buffer = 3)
    wet = [dumbbell_bottom(xi, yj) < 0 for xi in x, yj in y]
    ok = copy(wet)
    for i in eachindex(x), j in eachindex(y)
        wet[i, j] || continue
        for di in -buffer:buffer, dj in -buffer:buffer
            ii, jj = i + di, j + dj
            (1 <= ii <= length(x) && 1 <= jj <= length(y)) || continue
            wet[ii, jj] || (ok[i, j] = false)
        end
    end
    return ok
end

"""
    eddy_track(filename)

Where the eddy is and how coherent it stays. Position is the ζ²-weighted centroid
(ζ² so the two lobes both count and weak filaments do not dominate); coherence is
the 99.9th percentile of |ζ|, both taken only over open water away from the neck
corners — see [`open_water_mask`](@ref).
"""
function eddy_track(filename)
    ζts = FieldTimeSeries(outpath(filename * ".jld2"), "ζ")
    grid = ζts.grid
    x = collect(xnodes(grid, Face()))
    y = collect(ynodes(grid, Face()))
    times = ζts.times
    ok = open_water_mask(x, y)

    xbar = Float64[]; ζpk = Float64[]
    for n in eachindex(times)
        ζ = Array(interior(ζts[n]))[:, :, 1]
        w = ifelse.(ok .& isfinite.(ζ), ζ .^ 2, 0.0)
        s = sum(w)
        push!(xbar, s > 0 ? sum(w .* x) / s : NaN)
        vals = abs.(ζ[ok .& isfinite.(ζ)])
        push!(ζpk, isempty(vals) ? NaN : quantile(vals, 0.999))
    end
    return (; times, xbar, ζpk, x, y, ok, ζts)
end

"""
    subdomain_enstrophy(filename; xrange = nothing)

∫ζ² dA over open water, optionally restricted to an x-range so the truth run can
be compared to a sub-domain on equal terms. This is the quantity that answers the
question: did the eddy actually leave (enstrophy falls) or actually arrive
(enstrophy rises)?
"""
function subdomain_enstrophy(filename; xrange = nothing)
    ζts = FieldTimeSeries(outpath(filename * ".jld2"), "ζ")
    x = collect(xnodes(ζts.grid, Face()))
    y = collect(ynodes(ζts.grid, Face()))
    ok = open_water_mask(x, y)
    if xrange !== nothing
        ok = ok .& [xrange[1] <= xi <= xrange[2] for xi in x]
    end
    dA = Δ_db^2
    Z = [sum(ifelse.(ok .& isfinite.(z), z .^ 2, 0.0)) * dA
         for z in (Array(interior(ζts[n]))[:, :, 1] for n in eachindex(ζts.times))]
    return (; times = ζts.times, Z)
end

"""
    animate_handoff(truth, west, east, labels; outfile)

Two stacked panels: the TRUTH on top, and beneath it the two downscaled runs drawn
in a single axis at their real x positions with the cut marked. The eddy should be
seen leaving the left box and appearing in the right one.
"""
function animate_handoff(truth, west, east; label = "", outfile = "dumbbell_handoff.mp4",
                         framerate = 14)
    T = FieldTimeSeries(outpath(truth * ".jld2"), "ζ")
    W = FieldTimeSeries(outpath(west  * ".jld2"), "ζ")
    E = FieldTimeSeries(outpath(east  * ".jld2"), "ζ")
    times = T.times

    coords(ts) = (collect(xnodes(ts.grid, Face())), collect(ynodes(ts.grid, Face())))
    xT, yT = coords(T); xW, yW = coords(W); xE, yE = coords(E)
    mT = land_mask(xT, yT); mW = land_mask(xW, yW); mE = land_mask(xE, yE)

    fig = Figure(size = (1150, 720))
    n = Observable(1)
    ttl = @lift @sprintf("Dumbbell handoff — surface ζ/f   t = %.1f days", times[$n] / 86400)
    Label(fig[0, 1], ttl, fontsize = 19, tellwidth = false)

    fld(ts, m) = @lift Array(interior(ts[$n]))[:, :, 1] ./ f_db .* m
    kw = (colormap = :balance, colorrange = (-0.6, 0.6), nan_color = RGBAf(0.75, 0.72, 0.66, 1))

    ax1 = Axis(fig[1, 1], title = "TRUTH — full dumbbell", ylabel = "y (km)", aspect = DataAspect())
    heatmap!(ax1, xT ./ 1e3, yT ./ 1e3, fld(T, mT); kw...)
    vlines!(ax1, [x_c/1e3], color = :limegreen, linewidth = 2)
    xlims!(ax1, 0, Lx_db/1e3); ylims!(ax1, 0, Ly_db/1e3); hidexdecorations!(ax1, grid = false)

    ax2 = Axis(fig[2, 1], title = "DOWNSCALED — two independent runs, open boundary at the cut" *
                                  (isempty(label) ? "" : "  ($label)"),
               xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    heatmap!(ax2, xW ./ 1e3, yW ./ 1e3, fld(W, mW); kw...)
    heatmap!(ax2, xE ./ 1e3, yE ./ 1e3, fld(E, mE); kw...)
    vlines!(ax2, [x_c/1e3], color = :limegreen, linewidth = 2)
    xlims!(ax2, 0, Lx_db/1e3); ylims!(ax2, 0, Ly_db/1e3)

    CairoMakie.record(fig, outfile, eachindex(times); framerate) do i
        n[] = i
    end
    return outfile
end

"""
    animate_dumbbell(files, labels; outfile)

Side-by-side animation of surface relative vorticity ζ/f. `files` may be one
run (the truth) or several (truth vs downscaled).
"""
function animate_dumbbell(files, labels; outfile = "dumbbell.mp4", framerate = 12)
    series = [FieldTimeSeries(outpath(f * ".jld2"), "ζ") for f in files]
    times = series[1].times
    ζscale = f_db

    fig = Figure(size = (620 * length(series), 420))
    n = Observable(1)
    title = @lift @sprintf("Dumbbell — surface ζ/f   t = %.1f days", times[$n] / 86400)
    Label(fig[0, 1:length(series)], title, fontsize = 19, tellwidth = false)

    for (col, (ts, lab)) in enumerate(zip(series, labels))
        gx = collect(xnodes(ts.grid, Face())) ./ 1e3
        gy = collect(ynodes(ts.grid, Face())) ./ 1e3
        mask = land_mask(collect(xnodes(ts.grid, Face())), collect(ynodes(ts.grid, Face())))
        ζn = @lift Array(interior(ts[$n]))[:, :, 1] ./ ζscale .* mask

        ax = Axis(fig[1, col], title = lab, xlabel = "x (km)",
                  ylabel = col == 1 ? "y (km)" : "", aspect = DataAspect())
        heatmap!(ax, gx, gy, ζn, colormap = :balance, colorrange = (-0.5, 0.5),
                 nan_color = RGBAf(0.75, 0.72, 0.66, 1))
        # neck outline
        vlines!(ax, [(x_c - neck_half_length)/1e3, (x_c + neck_half_length)/1e3],
                color = (:black, 0.25), linestyle = :dot)
        xlims!(ax, 0, Lx_db/1e3); ylims!(ax, 0, Ly_db/1e3)
    end

    CairoMakie.record(fig, outfile, eachindex(times); framerate) do i
        n[] = i
    end
    return outfile
end

# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    Rd = N_db * H_db / (f_db * π)
    @printf("dumbbell %.0f × %.0f km, Δ = %.1f km, Nz = %d\n", Lx_db/1e3, Ly_db/1e3, Δ_db/1e3, Nz_db)
    @printf("neck: |x − %.0f km| ≤ %.0f km, open where |y − %.0f km| < %.0f km;  CUT at x = %.0f km\n",
            x_c/1e3, neck_half_length/1e3, y_c/1e3, neck_half_width/1e3, x_c/1e3)
    @printf("Rd = %.0f km,  eddy R = %.0f km (Bu = %.2f),  Ro = %.2f\n\n",
            Rd/1e3, R_eddy/1e3, (Rd/R_eddy)^2, v_max/(f_db*R_eddy))

    # ---- TRUTH (reuse if already on disk: it is the slowest piece) ----
    if !isfile(outpath("dumbbell_truth.jld2"))
        println("--- TRUTH: full dumbbell, closed ---")
        run!(dumbbell_simulation(filename = "dumbbell_truth"))
    else
        println("--- TRUTH: reusing dumbbell_truth.jld2 ---")
    end
    tr = eddy_track("dumbbell_truth")
    @printf("  eddy centroid %.0f → %.0f km over %.0f days (%.3f m/s); peak |ζ|/f %.2f → %.2f\n\n",
            tr.xbar[1]/1e3, tr.xbar[end]/1e3, tr.times[end]/86400,
            (tr.xbar[end]-tr.xbar[1])/tr.times[end], tr.ζpk[1]/f_db, tr.ζpk[end]/f_db)

    ext = cut_exterior("dumbbell_truth")

    schemes = ("NormalRadiation"  => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
               "ObliqueRadiation" => ObliqueRadiation(inflow_timescale = 0, outflow_timescale = Inf))

    # ---- the six downscaled runs ----
    for side in (:west, :east)
        for (name, scheme) in schemes
            tag = "dumbbell_$(side)_$(lowercase(name))"
            println("--- $side / $name ---")
            run!(dumbbell_downscaled(; side, scheme, exterior = ext, filename = tag))
        end
        println("--- $side / CLOSED (control) ---")
        run!(dumbbell_downscaled(; side, closed = true, filename = "dumbbell_$(side)_closed"))
    end

    # ---- did the eddy leave, and did it arrive? ----
    Zt_w = subdomain_enstrophy("dumbbell_truth"; xrange = (0.0, x_c))
    Zt_e = subdomain_enstrophy("dumbbell_truth"; xrange = (x_c, Lx_db))

    println("\n" * "="^76)
    @printf("%-34s %14s %14s %10s\n", "run", "Z(25 d)", "Z(end)", "Z_end/Z_truth")
    println("="^76)
    idx(ts, d) = argmin(abs.(ts .- d*86400))
    rows = Any[]
    for (side, Zt) in ((:west, Zt_w), (:east, Zt_e))
        @printf("%-34s %14.3e %14.3e %10s\n", "TRUTH ($side half)",
                Zt.Z[idx(Zt.times,25)], Zt.Z[end], "—")
        for lab in ("normalradiation", "obliqueradiation", "closed")
            tag = "dumbbell_$(side)_$lab"
            Z = subdomain_enstrophy(tag)
            push!(rows, (side, lab, Z))
            @printf("%-34s %14.3e %14.3e %10.3f\n", "  $lab",
                    Z.Z[idx(Z.times,25)], Z.Z[end], Z.Z[end]/Zt.Z[end])
        end
    end
    println("="^76)
    println("west: enstrophy must FALL as the eddy leaves — a closed wall traps it.")
    println("east: enstrophy must RISE as the eddy arrives — a closed wall admits nothing.")

    # ---- figures ----
    fig = Figure(size = (1150, 430))
    for (col, (side, Zt, ttl)) in enumerate(((:west, Zt_w, "WEST — the eddy must LEAVE"),
                                             (:east, Zt_e, "EAST — the eddy must ENTER")))
        ax = Axis(fig[1, col], title = ttl, xlabel = "t (days)",
                  ylabel = col == 1 ? "∫ζ² dA  (m² s⁻²)" : "")
        lines!(ax, Zt.times ./ 86400, Zt.Z, color = :black, linewidth = 3, label = "truth (this half)")
        for (lab, colr) in (("normalradiation", :dodgerblue), ("obliqueradiation", :crimson),
                            ("closed", :gray))
            Z = subdomain_enstrophy("dumbbell_$(side)_$lab")
            lines!(ax, Z.times ./ 86400, Z.Z, color = colr, linewidth = 2,
                   linestyle = lab == "closed" ? :dash : :solid, label = lab)
        end
        col == 1 && axislegend(ax, position = :lb, framevisible = false, labelsize = 10)
    end
    Label(fig[0, 1:2], "Dumbbell — can an eddy leave and enter a downscaled region?",
          fontsize = 18, tellwidth = false)
    save("dumbbell_enstrophy.png", fig)
    println("\nsaved dumbbell_enstrophy.png")

    for (name, lab) in (("normalradiation", "NormalRadiation"), ("obliqueradiation", "ObliqueRadiation"))
        local out = animate_handoff("dumbbell_truth", "dumbbell_west_$name", "dumbbell_east_$name";
                              label = lab, outfile = "dumbbell_handoff_$name.mp4")
        println("saved ", out)
    end
    out = animate_handoff("dumbbell_truth", "dumbbell_west_closed", "dumbbell_east_closed";
                          label = "CLOSED control", outfile = "dumbbell_handoff_closed.mp4")
    println("saved ", out)
end
