# ==================================================================
# NECK BASIN — the neck as its own domain, open at BOTH ends
#
# Enri's suggestion, and it is the sharpest version of the test. The earlier
# runs cut the dumbbell in half and gave each half ONE open boundary, with a
# large interior the boundary could not reach. Here the domain IS the neck:
#
#   * TWO open boundaries. Water must enter through one and leave through the
#     other, continuously. That is the whole downscaling problem rather than
#     half of it.
#   * NO interior to hide in. The domain is ~165 × 220 km of water, so every
#     point is within an eddy diameter or two of a boundary and essentially all
#     of the error is boundary error.
#   * It is small — 22 × 60 × 4 — so it runs in seconds and angles, timescales
#     and schemes can be swept properly instead of in half-hour batches.
#   * It is literally the nesting configuration: a small regional domain fed on
#     its open sides by a coarser parent. Exactly the MAB setup.
#
# THE BOUNDARY DATA COMES FROM THE FULL-DUMBBELL TRUTH, and so does the INITIAL
# STATE. Starting the neck from rest would be a different and much harder
# problem (fill an empty box entirely through its boundaries); a nested model
# starts from the parent's state at t₀ and is then driven at its edges, which is
# what happens here.
#
# The parent is the FORCED dumbbell (validation/dumbbell_forced.jl): basin
# buoyancy restoring drives a baroclinic exchange through the neck which is
# baroclinically unstable, so by the time the sub-domain starts the throat is in
# a genuinely eddying state rather than carrying one tidy feature.
#
# A note on the Kelvin waves. Each basin develops a boundary-trapped current
# along its walls — these are Kelvin waves and they are perfectly physical. In
# the half-domain tests they were excluded from the metric as a basin-scale
# feature unrelated to the boundary. Here they are part of the test: they
# propagate along the neck's north and south walls and must EXIT through the
# open ends, which is a genuine and well-known difficulty for radiation
# conditions.
#
# Run:  julia -t 8 --project=. validation/neck_basin.jl
# ==================================================================

include(joinpath(@__DIR__, "dumbbell.jl"))

const x_neck_w = (neck_i_west - 1) * Δ_db      # 367.5 km
const x_neck_e = (neck_i_east - 1) * Δ_db      # 532.5 km
const Nx_neck  = neck_i_east - neck_i_west     # 22

"""
    neck_exterior(truth)

Exterior data for BOTH ends of the neck, sampled from the full-dumbbell truth,
plus the truth state over the whole sub-domain for initialisation.

Slab-local index `l` maps to truth index `first(cut_slab) + l − 1`. On each side
the boundary needs the normal velocity ON the face, the tangential velocity and
tracer at the halo centre just OUTSIDE, and — for Flather — `η` at the first
interior centre INSIDE, so that `Uᵇ = Uᵉˣᵗ ± √(gH)(ηᵇ − ηᵉˣᵗ)` returns `Uᵉˣᵗ`
exactly when the sub-domain matches the parent.
"""
function neck_exterior(truth = "dumbbell_frc_truth")
    uts = FieldTimeSeries(outpath(truth * "_cut.jld2"), "u")
    vts = FieldTimeSeries(outpath(truth * "_cut.jld2"), "v")
    bts = FieldTimeSeries(outpath(truth * "_cut.jld2"), "b")
    ηts = FieldTimeSeries(outpath(truth * "_cut_eta.jld2"), "η")
    times = collect(uts.times)

    U = [Array(interior(uts[n])) for n in eachindex(times)]
    V = [Array(interior(vts[n])) for n in eachindex(times)]
    B = [Array(interior(bts[n])) for n in eachindex(times)]
    E = [Array(interior(ηts[n])) for n in eachindex(times)]

    lo = first(cut_slab)
    idx = (face_w = neck_i_west - lo + 1,        # normal u, west face
           face_e = neck_i_east - lo + 1,        # normal u, east face
           out_w  = neck_i_west - 1 - lo + 1,    # centre just outside, west
           out_e  = neck_i_east - lo + 1,        # centre just outside, east
           in_w   = neck_i_west - lo + 1,        # first interior centre, west
           in_e   = neck_i_east - 1 - lo + 1)    # last  interior centre, east

    Δz = H_db / Nz_db
    Utr_w = [vec(sum(u[idx.face_w, :, :], dims = 2)) .* Δz for u in U]
    Utr_e = [vec(sum(u[idx.face_e, :, :], dims = 2)) .* Δz for u in U]
    # Tangential barotropic transport, at the halo centre just outside each end.
    # Flather is a statement about the NORMAL Riemann invariant, so on a C-grid it
    # says nothing about V at a west/east boundary — V needs its own condition.
    Vtr_w = [vec(sum(v[idx.out_w, :, :], dims = 2)) .* Δz for v in V]
    Vtr_e = [vec(sum(v[idx.out_e, :, :], dims = 2)) .* Δz for v in V]

    @inline function frame(t)
        t <= times[1]   && return (1, 1, 0.0)
        t >= times[end] && return (length(times), length(times), 0.0)
        n = searchsortedlast(times, t)
        return (n, n + 1, (t - times[n]) / (times[n+1] - times[n]))
    end

    ny, nz = size(U[1], 2), size(U[1], 3)
    cj(j) = clamp(j, 1, ny); ck(k) = clamp(k, 1, nz)
    @inline lerp(A, nf, l, j, k) =
        (1 - nf[3]) * A[nf[1]][l, cj(j), ck(k)] + nf[3] * A[nf[2]][l, cj(j), ck(k)]

    # State over the whole sub-domain at an arbitrary time, for initialisation.
    function state_at(t)
        nf = frame(t)
        w(A) = (1 - nf[3]) .* A[nf[1]] .+ nf[3] .* A[nf[2]]
        lu = idx.face_w:idx.face_e            # 23 faces
        lc = idx.face_w:(idx.face_e - 1)      # 22 centres
        return (u = w(U)[lu, :, :], v = w(V)[lc, :, :],
                b = w(B)[lc, :, :], η = w(E)[lc, :, :])
    end

    return (; times, idx, state_at,
              u = (l, j, k, t) -> lerp(U, frame(t), l, j, k),
              v = (l, j, k, t) -> lerp(V, frame(t), l, j, k),
              b = (l, j, k, t) -> lerp(B, frame(t), l, j, k),
              η = (l, j, t)    -> lerp(E, frame(t), l, j, 1),
              U_west = (j, t) -> (nf = frame(t);
                                  (1-nf[3])*Utr_w[nf[1]][cj(j)] + nf[3]*Utr_w[nf[2]][cj(j)]),
              U_east = (j, t) -> (nf = frame(t);
                                  (1-nf[3])*Utr_e[nf[1]][cj(j)] + nf[3]*Utr_e[nf[2]][cj(j)]),
              V_west = (j, t) -> (nf = frame(t); jj = clamp(j, 1, length(Vtr_w[1]));
                                  (1-nf[3])*Vtr_w[nf[1]][jj] + nf[3]*Vtr_w[nf[2]][jj]),
              V_east = (j, t) -> (nf = frame(t); jj = clamp(j, 1, length(Vtr_e[1]));
                                  (1-nf[3])*Vtr_e[nf[1]][jj] + nf[3]*Vtr_e[nf[2]][jj]))
end


"""
    boundary_sponge_viscosity(; ν_interior, amplification, width, ν_boundary)

A horizontal viscosity that ramps from `ν_interior` in the interior up to
`ν_boundary` (default `amplification * ν_interior`) within `width` of either open
end, as a smooth raised cosine so there is no kink where the sponge starts.

Baroclinic open boundaries are messy — a radiation condition is a hyperbolic
statement imposed on a parabolic problem — and the residual usually shows up as
GRID-SCALE noise near the boundary. Damping it locally is standard practice, and
one deformation radius is the usual rule of thumb for the width. Note that rule is
sized for a realistic domain: here Rd ≈ 64 km against a 165 km neck, so a full-Rd
sponge at each end would cover 17 of 22 cells. Width is therefore a parameter, not
a constant, and the whole thing is OPT-IN — `sponge = nothing` leaves the closure
uniform.

This is a plain spatially-varying `ν`, which Oceananigans already supports; the
function exists to make the intent legible and the width easy to sweep.
"""
function boundary_sponge_viscosity(; ν_interior = 50.0,
                                     amplification = 20.0,
                                     width = 4Δ_db,
                                     ν_boundary = amplification * ν_interior,
                                     west = x_neck_w, east = x_neck_e)
    @inline function ν_sponge(x, y, z, t)
        d = min(x - west, east - x)                      # distance to the nearest open end
        m = ifelse(d >= width, zero(d), (1 + cos(π * d / width)) / 2)
        return ν_interior + (ν_boundary - ν_interior) * m
    end
    return ν_sponge
end

"""
    neck_basin_simulation(; exterior, t₀, scheme, tracer_scheme, closed, filename)

The neck alone, open at both ends, initialised from the parent at `t₀`.
`closed = true` walls both ends — the control that shows what a total failure of
the open boundaries looks like.
"""
function neck_basin_simulation(; exterior,
                                 t₀ = 90days,
                                 stop_time = 40days,
                                 scheme = NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
                                 tracer_scheme = NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
                                 tangential = :prescribed,   # :prescribed | :gradient | a Value scheme
                                 tangential_barotropic = :default,   # :default | :gradient | a Value scheme
                                 closed = false,
                                 forcing = NamedTuple(),
                                 sponge = nothing,   # opt-in viscous sponge; see boundary_sponge_viscosity
                                 Δt = Δt_db,
                                 filename = "neck")

    underlying = RectilinearGrid(CPU();
                                 size = (Nx_neck, Ny_db, Nz_db),
                                 x = (x_neck_w, x_neck_e), y = (0, Ly_db), z = (-H_db, 0),
                                 halo = (7, 7, 7),
                                 topology = (Bounded, Bounded, Bounded))
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(dumbbell_bottom))

    bcs = NamedTuple()
    if !closed
        e = exterior; ix = e.idx
        # clock.time is absolute (the parent's), so the sub-domain clock is offset by t₀
        uw(j, k, grid, clock, f) = e.u(ix.face_w, j, k, clock.time + t₀)
        ue(j, k, grid, clock, f) = e.u(ix.face_e, j, k, clock.time + t₀)
        vw(j, k, grid, clock, f) = e.v(ix.out_w,  j, k, clock.time + t₀)
        ve(j, k, grid, clock, f) = e.v(ix.out_e,  j, k, clock.time + t₀)
        bw(j, k, grid, clock, f) = e.b(ix.out_w,  j, k, clock.time + t₀)
        be(j, k, grid, clock, f) = e.b(ix.out_e,  j, k, clock.time + t₀)
        Uw(j, k, grid, clock, f) = (e.U_west(j, clock.time + t₀), e.η(ix.in_w, j, clock.time + t₀))
        Ue(j, k, grid, clock, f) = (e.U_east(j, clock.time + t₀), e.η(ix.in_e, j, clock.time + t₀))

        u_bcs = FieldBoundaryConditions(
            west = NormalFlowBoundaryCondition(uw; scheme, discrete_form = true),
            east = NormalFlowBoundaryCondition(ue; scheme, discrete_form = true))
        # The TANGENTIAL velocity is the suspect for the vorticity strip that appears
        # right on the open boundary: ζ = ∂v/∂x − ∂u/∂y, so imposing v as a hard
        # Dirichlet from the parent while u is radiated puts any mismatch between the
        # prescribed halo value and the interior straight into ∂v/∂x. Oceananigans has
        # no dedicated tangential open scheme (issue #5229's missing piece 2), so this
        # switch covers the three things one can actually do:
        #   :prescribed — hard Dirichlet from the parent (what over-specifies it)
        #   :gradient   — zero normal gradient, the script-9a fallback
        #   <scheme>    — a Value-classification scheme (NormalRadiation/ObliqueRadiation)
        #                 applied to the tangential component, nudged to the parent
        v_bcs = if tangential === :gradient
            FieldBoundaryConditions(west = GradientBoundaryCondition(0.0),
                                    east = GradientBoundaryCondition(0.0))
        elseif tangential === :prescribed
            FieldBoundaryConditions(west = ValueBoundaryCondition(vw; discrete_form = true),
                                    east = ValueBoundaryCondition(ve; discrete_form = true))
        else
            FieldBoundaryConditions(
                west = ValueBoundaryCondition(vw; scheme = tangential, discrete_form = true),
                east = ValueBoundaryCondition(ve; scheme = tangential, discrete_form = true))
        end
        b_bcs = FieldBoundaryConditions(
            west = ValueBoundaryCondition(bw; scheme = tracer_scheme, discrete_form = true),
            east = ValueBoundaryCondition(be; scheme = tracer_scheme, discrete_form = true))
        U_bcs = FieldBoundaryConditions(underlying, (Face(), Center(), nothing);
            west = GravityWaveRadiationBoundaryCondition(Uw; discrete_form = true),
            east = GravityWaveRadiationBoundaryCondition(Ue; discrete_form = true))

        # The FREE SURFACE needs a boundary condition too. Without one, η at an
        # open end is still treated as a wall, which is inconsistent with letting
        # U through it — the split-explicit recipe is the full triad: a baroclinic
        # scheme on u/v, Flather (`GravityWaveRadiation`) on U/V, and Chapman
        # (`SurfaceWaveRadiation`) on η. Omitting η is what lets the barotropic
        # mode misbehave.
        η_bcs = FieldBoundaryConditions(underlying, (Center(), Center(), Face());
            west = SurfaceWaveRadiationBoundaryCondition(),
            east = SurfaceWaveRadiationBoundaryCondition())

        bcs = (u = u_bcs, v = v_bcs, b = b_bcs, U = U_bcs, η = η_bcs)

        # V is TANGENTIAL at a west/east boundary, and on a C-grid Flather constrains
        # only the NORMAL transport — so V needs its own treatment. Two things were
        # measured here:
        #   * V's Oceananigans default at those ends is `Flux` (no-flux), which for a
        #     tangential field is free-slip / zero normal gradient — NOT a wall. Setting
        #     GradientBoundaryCondition(0) explicitly gives byte-identical answers, so
        #     the default is already the right thing and `:default` is the sane choice.
        #   * Radiating V is NOT POSSIBLE with the current upstream code: NormalRadiation
        #     and ObliqueRadiation as `Value` schemes read `model_fields.u` for their
        #     advecting velocity, and during the split-explicit barotropic solve
        #     `model_fields` is `(U, V, η)` — there is no `u`. It throws.
        if tangential_barotropic !== :default
            Vw(j, k, grid, clock, f) = e.V_west(j, clock.time + t₀)
            Ve(j, k, grid, clock, f) = e.V_east(j, clock.time + t₀)
            V_bcs = if tangential_barotropic === :gradient
                FieldBoundaryConditions(underlying, (Center(), Face(), nothing);
                    west = GradientBoundaryCondition(0.0),
                    east = GradientBoundaryCondition(0.0))
            else
                FieldBoundaryConditions(underlying, (Center(), Face(), nothing);
                    west = ValueBoundaryCondition(Vw; scheme = tangential_barotropic,
                                                  discrete_form = true),
                    east = ValueBoundaryCondition(Ve; scheme = tangential_barotropic,
                                                  discrete_form = true))
            end
            bcs = merge(bcs, (; V = V_bcs))
        end
    end

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
        coriolis = FPlane(f = f_db),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        momentum_advection = WENOVectorInvariant(),
        tracer_advection = WENO(order = 5),
        closure = isnothing(sponge) ? HorizontalScalarDiffusivity(ν = 50, κ = 50) :
                                      HorizontalScalarDiffusivity(ν = sponge, κ = 50),
        boundary_conditions = bcs,
        forcing)

    # Start from the parent's own state — this is a nesting problem, not a
    # fill-an-empty-box problem.
    s = exterior.state_at(t₀)
    set!(model, u = s.u, v = s.v, b = s.b)
    set!(model.free_surface.displacement, s.η)

    simulation = Simulation(model; Δt, stop_time)

    function progress(sim)
        u, v, w = sim.model.velocities
        η = sim.model.free_surface.displacement
        @printf("  %-9s  max|u| = %.3f m/s   ⟨η⟩ = %+.4f m\n",
                prettytime(sim), maximum(abs, u), mean(filter(isfinite, interior(η))))
        return nothing
    end
    add_callback!(simulation, progress, TimeInterval(5days))

    u, v, w = model.velocities
    η = model.free_surface.displacement
    ζ = ∂x(v) - ∂y(u)
    simulation.output_writers[:surface] = JLD2Writer(model, (; ζ, u, v);
        filename = outpath(filename * ".jld2"),
        indices = (:, :, Nz_db),
        schedule = TimeInterval(6hours),
        overwrite_existing = true)
    simulation.output_writers[:eta] = JLD2Writer(model, (; η);
        filename = outpath(filename * "_eta.jld2"),
        schedule = TimeInterval(6hours),
        overwrite_existing = true)

    return simulation
end

"""
    neck_metrics(filename, truth, t₀)

Enstrophy in the neck against the parent's own neck region, and the domain-mean
`η` against the parent's. The latter is the direct test of whether a sub-domain's
sea level — which nothing pins, since volume is free — drifts away from its
parent, which is the mechanism suspected behind the secular growth seen in the
half-domain runs.
"""
function neck_metrics(filename, truth = "dumbbell_frc_truth", t₀ = 90days)
    ζs = FieldTimeSeries(outpath(filename * ".jld2"), "ζ")
    ηs = FieldTimeSeries(outpath(filename * "_eta.jld2"), "η")
    ζt = FieldTimeSeries(outpath(truth * ".jld2"), "ζ")
    ηt = FieldTimeSeries(outpath(truth * "_eta.jld2"), "η")

    xs = collect(xnodes(ζs.grid, Face())); ys = collect(ynodes(ζs.grid, Face()))
    ok = open_water_mask(xs, ys)
    xtζ = collect(xnodes(ζt.grid, Face()))
    selζ = [x_neck_w <= x <= x_neck_e for x in xtζ]
    xtη = collect(xnodes(ηt.grid, Center()))
    selη = [x_neck_w <= x <= x_neck_e for x in xtη]
    okt = open_water_mask(xtζ, collect(ynodes(ζt.grid, Face())))

    dA = Δ_db^2
    t = ζs.times
    Zs = Float64[]; Zt = Float64[]; Es = Float64[]; Et = Float64[]
    for n in eachindex(t)
        z = Array(interior(ζs[n]))[:, :, 1]
        push!(Zs, sum(ifelse.(ok .& isfinite.(z), z .^ 2, 0.0)) * dA)
        push!(Es, mean(filter(isfinite, Array(interior(ηs[n]))[:, :, 1])))

        m = argmin(abs.(ζt.times .- (t[n] + t₀)))
        zt = Array(interior(ζt[m]))[:, :, 1]
        push!(Zt, sum(ifelse.(okt .& isfinite.(zt) .& selζ, zt .^ 2, 0.0)) * dA)
        mη = argmin(abs.(ηt.times .- (t[n] + t₀)))
        push!(Et, mean(filter(isfinite, Array(interior(ηt[mη]))[selη, :, 1])))
    end
    return (; times = t, Z = Zs, Z_truth = Zt, η = Es, η_truth = Et)
end

"""
    composite_frame(ζ_parent, ζ_neck, x_parent, x_neck)

Paste the downscaled neck solution into the parent field, matching columns by
their x coordinate. The result is the full dumbbell with the neck region replaced
by the nested run — so the two open boundaries become visible SEAMS, and whether
the nested solution joins continuously onto its surroundings can be read straight
off the picture rather than inferred from a number.
"""
function composite_frame(ζp, ζn, xp, xn)
    out = copy(ζp)
    tol = 0.25 * Δ_db
    for (l, x) in enumerate(xn)
        i = findfirst(xx -> abs(xx - x) < tol, xp)
        (i === nothing || size(ζn, 2) != size(out, 2)) && continue
        out[i, :] .= ζn[l, :]
    end
    return out
end

"""
    plot_substituted(neck_file, truth, t₀; days, outfile)

Two rows: the parent dumbbell on top, and beneath it the same dumbbell with the
neck swapped out for the downscaled run.
"""
function plot_substituted(neck_file, truth, t₀; days = (10, 25, 40),
                          outfile = "neck_substituted.png", label = "")
    ζt = FieldTimeSeries(outpath(truth * ".jld2"), "ζ")
    ζn = FieldTimeSeries(outpath(neck_file * ".jld2"), "ζ")
    xt = collect(xnodes(ζt.grid, Face())); yt = collect(ynodes(ζt.grid, Face()))
    xn = collect(xnodes(ζn.grid, Face()))
    mt = land_mask(xt, yt)

    fig = Figure(size = (1400, 620))
    kw = (colormap = :balance, colorrange = (-0.4, 0.4), nan_color = RGBAf(0.75, 0.72, 0.66, 1))

    for (col, d) in enumerate(days)
        nt = argmin(abs.(ζt.times .- (t₀ + d*86400)))
        nn = argmin(abs.(ζn.times .- d*86400))
        P = Array(interior(ζt[nt]))[:, :, 1]
        N = Array(interior(ζn[nn]))[:, :, 1]
        C = composite_frame(P, N, xt, xn)

        a1 = Axis(fig[1, col], title = @sprintf("t₀ + %d days", d), aspect = DataAspect(),
                  ylabel = col == 1 ? "PARENT\ny (km)" : "")
        heatmap!(a1, xt ./ 1e3, yt ./ 1e3, P ./ f_db .* mt; kw...)
        vlines!(a1, [x_neck_w/1e3, x_neck_e/1e3], color = :limegreen, linewidth = 1.5)
        hidexdecorations!(a1, grid = false)

        a2 = Axis(fig[2, col], xlabel = "x (km)", aspect = DataAspect(),
                  ylabel = col == 1 ? "NECK SUBSTITUTED\ny (km)" : "")
        heatmap!(a2, xt ./ 1e3, yt ./ 1e3, C ./ f_db .* mt; kw...)
        vlines!(a2, [x_neck_w/1e3, x_neck_e/1e3], color = :limegreen, linewidth = 1.5)
    end

    Label(fig[0, 1:length(days)],
          "Downscaled neck pasted back into its parent" * (isempty(label) ? "" : " — $label") *
          "\ngreen lines are the open boundaries; look for a seam",
          fontsize = 17, tellwidth = false)
    save(outfile, fig)
    return outfile
end

"""
    animate_substituted(neck_file, truth, t₀; outfile)

The same substitution, animated: parent on top, parent-with-neck-replaced below.
"""
function animate_substituted(neck_file, truth, t₀; outfile = "neck_substituted.mp4",
                             label = "", framerate = 12)
    ζt = FieldTimeSeries(outpath(truth * ".jld2"), "ζ")
    ζn = FieldTimeSeries(outpath(neck_file * ".jld2"), "ζ")
    xt = collect(xnodes(ζt.grid, Face())); yt = collect(ynodes(ζt.grid, Face()))
    xn = collect(xnodes(ζn.grid, Face()))
    mt = land_mask(xt, yt)
    times = ζn.times

    fig = Figure(size = (900, 640))
    n = Observable(1)
    ttl = @lift @sprintf("Downscaled neck pasted into its parent%s — t₀ + %.1f days",
                         isempty(label) ? "" : " ($label)", times[$n] / 86400)
    Label(fig[0, 1], ttl, fontsize = 17, tellwidth = false)
    kw = (colormap = :balance, colorrange = (-0.4, 0.4), nan_color = RGBAf(0.75, 0.72, 0.66, 1))

    P = @lift Array(interior(ζt[argmin(abs.(ζt.times .- (t₀ + times[$n])))]))[:, :, 1]
    C = @lift composite_frame(Array(interior(ζt[argmin(abs.(ζt.times .- (t₀ + times[$n])))]))[:, :, 1],
                              Array(interior(ζn[$n]))[:, :, 1], xt, xn)

    a1 = Axis(fig[1, 1], title = "parent", ylabel = "y (km)", aspect = DataAspect())
    heatmap!(a1, xt ./ 1e3, yt ./ 1e3, (@lift $P ./ f_db .* mt); kw...)
    vlines!(a1, [x_neck_w/1e3, x_neck_e/1e3], color = :limegreen, linewidth = 1.5)
    hidexdecorations!(a1, grid = false)

    a2 = Axis(fig[2, 1], title = "neck substituted", xlabel = "x (km)", ylabel = "y (km)",
              aspect = DataAspect())
    heatmap!(a2, xt ./ 1e3, yt ./ 1e3, (@lift $C ./ f_db .* mt); kw...)
    vlines!(a2, [x_neck_w/1e3, x_neck_e/1e3], color = :limegreen, linewidth = 1.5)

    CairoMakie.record(fig, outfile, eachindex(times); framerate) do i
        n[] = i
    end
    return outfile
end

# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    const T0 = 90days
    const TRUN = 40days

    @printf("neck basin: x = %.1f–%.1f km (%d cells), y = 0–%.0f km, Nz = %d  ⇒ %d cells\n",
            x_neck_w/1e3, x_neck_e/1e3, Nx_neck, Ly_db/1e3, Nz_db, Nx_neck*Ny_db*Nz_db)
    @printf("throat x = %.0f–%.0f km, so both open ends sit inside it\n",
            (x_c-neck_half_length)/1e3, (x_c+neck_half_length)/1e3)
    @printf("parent = dumbbell_frc_truth; start t₀ = %.0f d, run %.0f d\n\n",
            T0/86400, TRUN/86400)

    ext = neck_exterior("dumbbell_frc_truth")
    @printf("exterior: %d frames, %.1f h apart, spanning %.0f–%.0f days\n\n",
            length(ext.times), (ext.times[2]-ext.times[1])/3600,
            ext.times[1]/86400, ext.times[end]/86400)

    cases = ("normalradiation"  => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
             "obliqueradiation" => ObliqueRadiation(inflow_timescale = 0, outflow_timescale = Inf))

    for (lab, scheme) in cases
        println("--- neck / $lab ---")
        run!(neck_basin_simulation(; exterior = ext, t₀ = T0, stop_time = TRUN,
                                     scheme, filename = "neck_$lab"))
    end
    println("--- neck / CLOSED (control) ---")
    run!(neck_basin_simulation(; exterior = ext, t₀ = T0, stop_time = TRUN,
                                 closed = true, filename = "neck_closed"))

    println("\n" * "="^78)
    @printf("%-20s %12s %12s | %12s %12s\n", "run", "Z(10d)", "Z(end)", "Z_end/truth", "η drift (m)")
    println("="^78)
    for lab in ("normalradiation", "obliqueradiation", "closed")
        m = neck_metrics("neck_$lab", "dumbbell_frc_truth", T0)
        i10 = argmin(abs.(m.times .- 10*86400))
        @printf("%-20s %12.3e %12.3e | %12.3f %12.5f\n", lab,
                m.Z[i10], m.Z[end], m.Z[end]/m.Z_truth[end], m.η[end] - m.η_truth[end])
    end
    m0 = neck_metrics("neck_normalradiation", "dumbbell_frc_truth", T0)
    @printf("%-20s %12.3e %12.3e | %12.3f %12.5f\n", "TRUTH (neck region)",
            m0.Z_truth[argmin(abs.(m0.times .- 10*86400))], m0.Z_truth[end], 1.0, 0.0)
    println("="^78)
    println("η drift is the sub-domain's mean sea level minus the parent's over the same")
    println("region. Nothing pins it — volume is free — so if Flather converts that drift")
    println("into spurious transport, this is where it shows.")

    # ---- figures ----
    fig = Figure(size = (1200, 760))
    ax = Axis(fig[1, 1], title = "Enstrophy in the neck", xlabel = "t since t₀ (days)",
              ylabel = "∫ζ² dA (m² s⁻²)")
    lines!(ax, m0.times ./ 86400, m0.Z_truth, color = :black, linewidth = 3, label = "truth")
    for (lab, c) in (("normalradiation", :dodgerblue), ("obliqueradiation", :crimson),
                     ("closed", :gray))
        m = neck_metrics("neck_$lab", "dumbbell_frc_truth", T0)
        lines!(ax, m.times ./ 86400, m.Z, color = c, linewidth = 2,
               linestyle = lab == "closed" ? :dash : :solid, label = lab)
    end
    axislegend(ax, position = :lt, framevisible = false, labelsize = 10)

    ax2 = Axis(fig[1, 2], title = "mean sea level: sub-domain − parent",
               xlabel = "t since t₀ (days)", ylabel = "Δ⟨η⟩ (m)")
    for (lab, c) in (("normalradiation", :dodgerblue), ("obliqueradiation", :crimson),
                     ("closed", :gray))
        m = neck_metrics("neck_$lab", "dumbbell_frc_truth", T0)
        lines!(ax2, m.times ./ 86400, m.η .- m.η_truth, color = c, linewidth = 2,
               linestyle = lab == "closed" ? :dash : :solid, label = lab)
    end
    hlines!(ax2, [0.0], color = (:black, 0.4), linestyle = :dot)
    axislegend(ax2, position = :lt, framevisible = false, labelsize = 10)

    ζt = FieldTimeSeries(outpath("dumbbell_frc_truth.jld2"), "ζ")
    ζn = FieldTimeSeries(outpath("neck_normalradiation.jld2"), "ζ")
    xt = collect(xnodes(ζt.grid, Face())); yt = collect(ynodes(ζt.grid, Face()))
    xn = collect(xnodes(ζn.grid, Face())); yn = collect(ynodes(ζn.grid, Face()))
    mt = land_mask(xt, yt); mn = land_mask(xn, yn)
    for (col, d) in enumerate((10, 25, 40))
        nt = argmin(abs.(ζt.times .- (T0 + d*86400)))
        nn = argmin(abs.(ζn.times .- d*86400))
        local a1 = Axis(fig[2, col], title = @sprintf("t₀+%d d — parent", d),
                        ylabel = col == 1 ? "y (km)" : "", aspect = DataAspect())
        heatmap!(a1, xt ./ 1e3, yt ./ 1e3, Array(interior(ζt[nt]))[:, :, 1] ./ f_db .* mt,
                 colormap = :balance, colorrange = (-0.4, 0.4), nan_color = RGBAf(0.75,0.72,0.66,1))
        vlines!(a1, [x_neck_w/1e3, x_neck_e/1e3], color = :limegreen, linewidth = 2)
        local a2 = Axis(fig[3, col], title = "neck basin", xlabel = "x (km)",
                        ylabel = col == 1 ? "y (km)" : "", aspect = DataAspect())
        heatmap!(a2, xn ./ 1e3, yn ./ 1e3, Array(interior(ζn[nn]))[:, :, 1] ./ f_db .* mn,
                 colormap = :balance, colorrange = (-0.4, 0.4), nan_color = RGBAf(0.75,0.72,0.66,1))
        xlims!(a2, x_neck_w/1e3, x_neck_e/1e3)
    end
    Label(fig[0, 1:3], "Neck basin — open at both ends, nested in the forced dumbbell",
          fontsize = 18, tellwidth = false)
    save("neck_basin.png", fig)
    println("\nsaved neck_basin.png")
end
