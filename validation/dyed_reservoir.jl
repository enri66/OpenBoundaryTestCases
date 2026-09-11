# ==================================================================
# Dyed open boundary — the TRACER-RESERVOIR benchmark
#
# After MOM6's `dyed_obcs` test case (`dyed_obcs_initialization.F90`).
#
# WHY THIS TEST EXISTS. Every velocity benchmark in this project
# (kelvin_wave, radiating_gaussian, stratified_gaussian, supercritical)
# measures a baroclinic scheme through a barotropic corrector that
# overwrites the depth mean — the scheme governs only the shear, so its
# signal is second order and hard to isolate. A TRACER has no barotropic
# mode. Its open boundary condition is the only thing setting the halo
# value, so its effect is first order and directly measurable.
#
# THE PHYSICS BEING TESTED. A prescribed, spatially uniform, oscillating
# flow u(t) = U₀ sin(ωt) carries water out through the east boundary and
# then brings the SAME WATER back in. With a memoryless boundary condition
# the returning water carries the exterior value instead: the domain
# exports its own water and imports someone else's, losing tracer every
# cycle even though the net displacement over a cycle is exactly zero.
# A reservoir remembers what left and gives it back.
#
# WHY THE ANSWER IS KNOWN EXACTLY. The velocity is PRESCRIBED and uniform,
# so the tracer equation is pure translation: c(x, t) = c₀(x − X(t)) with
# X(t) = (U₀/ω)(1 − cos ωt). After one full period X = 0 and the field must
# be exactly back where it started. A REFERENCE run on a domain extended far
# enough east that nothing reaches a boundary supplies that answer including
# the identical WENO numerical diffusion, so the comparison isolates the
# boundary condition and nothing else.
#
# TWO CONFIGURATIONS
#
#   A. WATER MASS.  c ≡ 1 inside, cᵉˣᵗ = 0 outside — "how much of this water
#      came from my domain?". The exact answer is c ≡ 1 everywhere at every
#      whole period. This is the failure mode the reservoir was invented for
#      and it is the decisive test: whatever deficit appears is boundary-
#      manufactured, full stop.
#
#   B. DYE PATCH, PARTIAL EXIT.  A Gaussian 3σ inside the boundary, with the
#      excursion tuned so its leading flank — not the whole patch — crosses.
#      Harder, and the honest one: a reservoir holds ONE number per boundary
#      point, so it can return a lumped memory but not the outgoing profile.
#      (An earlier version of this script drove the patch almost entirely out
#      of the domain. Nothing can recover from that, and the numbers said so;
#      the excursion below is set so that a recoverable fraction exits.)
#
# Run:  julia --project=. validation/dyed_reservoir.jl
# ==================================================================

include(joinpath(@__DIR__, "obc_output.jl"))

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Oceananigans.BoundaryConditions: TracerReservoir, NormalRadiation, PerturbationAdvection
using Printf
using Statistics
using CairoMakie

# ---------------- shared parameters ----------------
const Lx     = 400kilometers      # open-domain length
const Lx_ref = 520kilometers      # reference domain: east wall far beyond the excursion
const Δx     = 2kilometers
const H      = 100.0              # m (irrelevant: the velocity is prescribed)
const T      = 2days              # oscillation period
const ω      = 2π / T

# Peak displacement of a fluid parcel over the cycle is 2 U₀ / ω.
peak_displacement(U₀) = 2U₀ / ω
velocity_for(displacement) = displacement * ω / 2

const σ  = 10kilometers                    # patch width, configuration B
const x₀ = Lx - 3σ                         # patch centre, configuration B

# Configuration B excursion: the patch centre ends 1σ short of the boundary at
# maximum displacement, so its leading flank crosses and its core does not.
const U_PATCH = velocity_for(2σ)
const U_MASS  = velocity_for(4σ)           # configuration A: a broad slug crosses

@inline gaussian_patch(x) = exp(-(x - x₀)^2 / (2σ^2))
@inline uniform_water(x)  = 1.0

"""
    dyed_simulation(; scheme, U₀, initial, reference, filename)

Uniform oscillating flow advecting a tracer across an open east boundary.

`scheme` is the open-boundary scheme on the tracer; `reference = true` instead
runs the identical problem on a domain long enough that nothing reaches the east
wall — the exact answer, carrying the same numerical diffusion.
"""
function dyed_simulation(; scheme = TracerReservoir(),
                           U₀ = U_MASS,
                           initial = uniform_water,
                           reference = false,
                           Δt = 300.0,
                           stop_time = T,
                           filename = "dyed")

    L  = reference ? Lx_ref : Lx
    Nx = round(Int, L / Δx)

    # Flat in y; a few cells in z only because WENO wants a halo it can fill —
    # w = 0, so nothing happens in the vertical.
    grid = RectilinearGrid(CPU();
                           size = (Nx, 4),
                           x = (0, L), z = (-H, 0),
                           halo = (5, 4),
                           topology = (Bounded, Flat, Bounded))

    u_prescribed(x, z, t) = U₀ * sin(ω * t)

    # Reference: closed east wall, unreachable. Open: the scheme under test,
    # relaxing toward a clean exterior (cᵉˣᵗ = 0).
    c_bcs = reference ? FieldBoundaryConditions() :
            FieldBoundaryConditions(east = ValueBoundaryCondition(0.0; scheme))

    model = HydrostaticFreeSurfaceModel(grid;
        velocities = PrescribedVelocityFields(u = u_prescribed),
        momentum_advection = nothing,          # the velocities are prescribed
        tracer_advection = WENO(order = 5),
        coriolis = nothing,
        buoyancy = nothing,
        closure = nothing,
        tracers = :c,
        boundary_conditions = (; c = c_bcs))

    set!(model, c = (x, z) -> initial(x))

    simulation = Simulation(model; Δt, stop_time)
    simulation.verbose = false

    simulation.output_writers[:fields] = JLD2Writer(model, (; c = model.tracers.c);
        filename = outpath(filename * ".jld2"),
        schedule = TimeInterval(T / 48),
        overwrite_existing = true)

    return simulation
end

"""
    dyed_metrics(filename, ref_filename)

Compare the final tracer profile against the reference on the open domain's own
cells. `retention` is the domain tracer content as a fraction of the reference's;
`profile_error` is the relative L2 difference.
"""
function dyed_metrics(filename, ref_filename)
    cts = FieldTimeSeries(outpath(filename * ".jld2"), "c")
    rts = FieldTimeSeries(outpath(ref_filename * ".jld2"), "c")
    times = cts.times
    x = collect(xnodes(cts.grid, Center()))

    c     = interior(cts[length(times)])[:, 1, 1]
    c_ref = interior(rts[length(rts.times)])[1:length(x), 1, 1]  # reference is longer

    mass  = [sum(interior(cts[m])[:, 1, 1]) * Δx for m in eachindex(times)]
    m_ref = sum(c_ref) * Δx

    retention     = (sum(c) * Δx) / m_ref
    profile_error = sqrt(sum((c .- c_ref) .^ 2) / sum(c_ref .^ 2))

    return (; retention, profile_error, mass, times, x, c, c_ref)
end

# The schemes under test. `TracerReservoir()` with both length scales zero is
# MOM6's own default and is memoryless — the reservoir is opt-in.
schemes() = ("memoryless (L_in = 0)"       => TracerReservoir(),
             "reservoir L_in = 10 km"      => TracerReservoir(inflow_length_scale = 10kilometers),
             "reservoir L_in = 30 km"      => TracerReservoir(inflow_length_scale = 30kilometers),
             "reservoir L_in = 100 km"     => TracerReservoir(inflow_length_scale = 100kilometers),
             "reservoir frozen (L_in = ∞)" => TracerReservoir(inflow_length_scale = Inf),
             "NormalRadiation"             => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
             "PerturbationAdvection"       => PerturbationAdvection(inflow_timescale = 0, outflow_timescale = Inf))

function run_configuration(label, tag; U₀, initial)
    @printf("\n%s\n%s\n%s\n", "="^72, label, "="^72)
    @printf("U₀ = %.3f m/s   peak parcel displacement = %.1f km   period = %s\n",
            U₀, peak_displacement(U₀)/1e3, prettytime(T))

    run!(dyed_simulation(; U₀, initial, reference = true, filename = tag * "_reference"))

    results = []
    for (name, scheme) in schemes()
        file = tag * "_" * replace(lowercase(name), r"[^a-z0-9]+" => "_")
        run!(dyed_simulation(; scheme, U₀, initial, filename = file))
        m = dyed_metrics(file, tag * "_reference")
        push!(results, (name, m))
    end

    @printf("\n%-30s %14s %16s\n", "tracer boundary scheme", "retained", "profile error")
    println("-"^62)
    for (name, m) in results
        @printf("%-30s %14.4f %16.4f\n", name, m.retention, m.profile_error)
    end
    return results
end

# ==================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    mass  = run_configuration("A. WATER MASS — c ≡ 1 inside, cᵉˣᵗ = 0 outside. Exact answer: c ≡ 1.",
                              "dyedA"; U₀ = U_MASS, initial = uniform_water)

    @printf("\npatch: centre %.0f km, σ = %.0f km (%.0f cells); boundary at %.0f km — the leading flank crosses\n",
            x₀/1e3, σ/1e3, σ/Δx, Lx/1e3)
    patch = run_configuration("B. DYE PATCH — Gaussian 3σ inside, leading flank exits and returns.",
                              "dyedB"; U₀ = U_PATCH, initial = gaussian_patch)

    # ---------------- figure ----------------
    fig = Figure(size = (1300, 880))
    for (row, (results, title, ylab)) in enumerate(((mass,  "A. water mass (c ≡ 1, exact answer c ≡ 1)", "c"),
                                                    (patch, "B. dye patch, leading flank exits", "c")))
        ax = Axis(fig[row, 1], title = title * " — profile after one full oscillation",
                  xlabel = "x (km)", ylabel = ylab)
        m1 = results[1][2]
        lines!(ax, m1.x ./ 1e3, m1.c_ref, color = :black, linewidth = 3.5, label = "reference (exact)")
        # The memoryless reservoir, NormalRadiation and PerturbationAdvection all
        # snap the halo to cᵉˣᵗ on inflow, so their curves coincide — dash the last
        # two so the overlap reads as agreement rather than as a missing line.
        for (k, (name, m)) in enumerate(results)
            style = k > 5 ? :dash : :solid
            lines!(ax, m.x ./ 1e3, m.c, linewidth = 1.8, linestyle = style, label = name)
        end
        xlims!(ax, row == 1 ? (Lx/1e3 - 150, Lx/1e3) : (x₀/1e3 - 60, Lx/1e3))
        row == 1 && axislegend(ax, position = :lb, framevisible = false, labelsize = 9)

        ax = Axis(fig[row, 2], title = "tracer content in the domain", xlabel = "t / period",
                  ylabel = "∫c dx (m)")
        for (name, m) in results
            lines!(ax, m.times ./ T, m.mass, linewidth = 1.8, label = name)
        end
    end
    Label(fig[0, 1:2], "Dyed open boundary — tracer reservoirs (MOM6 dyed_obcs analogue)",
          fontsize = 19, tellwidth = false)
    save("dyed_reservoir.png", fig)
    println("\nsaved dyed_reservoir.png")
end
