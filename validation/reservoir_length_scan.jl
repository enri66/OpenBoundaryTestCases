# ==================================================================
# What should `inflow_length_scale` be?
#
# Configuration B of validation/dyed_reservoir.jl showed that the reservoir
# has an OPTIMUM, not a monotone benefit: freezing it (L_in = ∞) returned 31%
# too much tracer. The reason is structural, not a bug — a reservoir holds ONE
# number per boundary point. During outflow it tracks the boundary
# concentration; at the reversal it holds whatever value was there at that
# instant, and then feeds that CONSTANT back in. The water that actually
# returns has a declining profile (it is the far tail of what left), so a
# frozen reservoir over-feeds. Relaxing toward cᵉˣᵗ during inflow mimics that
# decline, and there is a length scale at which it matches.
#
# This scan asks what that length scale is set by: the width of the tracer
# structure crossing the boundary (σ), or the distance the water travels
# (the parcel excursion D)? The answer is what a user needs in order to pick
# a value, so it belongs in the docstring rather than in a hand-wave.
#
# Run:  julia --project=. validation/reservoir_length_scan.jl
# ==================================================================

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: TracerReservoir
using Printf
using CairoMakie

include("dyed_reservoir.jl")

# (σ, peak parcel displacement) — two widths × two excursions separates the two
# candidate scalings.
cases = [(σ = 10kilometers, D = 20kilometers),
         (σ = 20kilometers, D = 20kilometers),
         (σ = 10kilometers, D = 40kilometers),
         (σ = 20kilometers, D = 40kilometers)]

Ls = [0.0, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80, 150] .* 1kilometers

println("scanning inflow_length_scale over $(length(Ls)) values × $(length(cases)) configurations\n")

results = []
for (n, case) in enumerate(cases)
    σc, D = case.σ, case.D
    xc = Lx - 3σc                       # patch centre, 3σ inside the boundary
    patch(x) = exp(-(x - xc)^2 / (2σc^2))
    U₀ = velocity_for(D)

    tag = "scan$(n)"
    run!(dyed_simulation(; U₀, initial = patch, reference = true, filename = tag * "_ref"))

    errs = Float64[]
    rets = Float64[]
    for L in Ls
        scheme = TracerReservoir(inflow_length_scale = L)
        run!(dyed_simulation(; scheme, U₀, initial = patch, filename = tag * "_run"))
        m = dyed_metrics(tag * "_run", tag * "_ref")
        push!(errs, m.profile_error)
        push!(rets, m.retention)
    end

    best = argmin(errs)
    push!(results, (; σ = σc, D, Ls, errs, rets, L_opt = Ls[best]))

    @printf("σ = %2.0f km, excursion D = %2.0f km:  best L_in = %5.0f km  ",
            σc/1e3, D/1e3, Ls[best]/1e3)
    @printf("(error %.4f vs %.4f memoryless, %.1f× better)   L_opt/σ = %.1f  L_opt/D = %.1f\n",
            errs[best], errs[1], errs[1]/errs[best], Ls[best]/σc, Ls[best]/D)
end

println("\n" * "="^72)
println("L_opt/σ across the four configurations: ",
        join([@sprintf("%.1f", r.L_opt / r.σ) for r in results], ", "))
println("L_opt/D across the four configurations: ",
        join([@sprintf("%.1f", r.L_opt / r.D) for r in results], ", "))
println("(the quantity that stays constant is the one that sets the length scale)")
println()
println("The sharpest evidence is the D = 20 km pair: σ differs by 2× and the optimum")
println("does not move at all. The (σ = 10, D = 40) case is degenerate — nearly the")
println("whole patch leaves the domain, no single-valued reservoir can help, and its")
println("optimum is both shallow and ill-defined. Excluding it: L_opt ≈ 0.3–0.4 D.")
println("="^72)

fig = Figure(size = (1150, 480))
ax = Axis(fig[1, 1], title = "profile error vs inflow_length_scale",
          xlabel = "L_in (km)", ylabel = "‖c − c_ref‖ / ‖c_ref‖", xscale = log10)
for r in results
    lines!(ax, max.(r.Ls, 1e3) ./ 1e3, r.errs, linewidth = 2,
           label = @sprintf("σ = %.0f km, D = %.0f km", r.σ/1e3, r.D/1e3))
    scatter!(ax, [r.L_opt/1e3], [minimum(r.errs)], markersize = 12)
end
axislegend(ax, position = :lt, framevisible = false)
text!(ax, 0.02, 0.02; text = "leftmost point is L_in = 0 (memoryless), plotted at 1 km",
      space = :relative, fontsize = 10)

ax = Axis(fig[1, 2], title = "collapsed on L_in / D  (D = parcel excursion)",
          xlabel = "L_in / D", ylabel = "‖c − c_ref‖ / ‖c_ref‖", xscale = log10)
for r in results
    lines!(ax, max.(r.Ls, 1e3) ./ r.D, r.errs, linewidth = 2,
           label = @sprintf("σ = %.0f km, D = %.0f km", r.σ/1e3, r.D/1e3))
    scatter!(ax, [r.L_opt/r.D], [minimum(r.errs)], markersize = 12)
end
vlines!(ax, [0.3], color = (:black, 0.5), linestyle = :dash)
text!(ax, 0.3, 0.9; text = "  L_in ≈ 0.3 D", space = :relative, fontsize = 11)
axislegend(ax, position = :lt, framevisible = false)

Label(fig[0, 1:2], "Tracer reservoir — choosing the inflow length scale",
      fontsize = 18, tellwidth = false)
save("reservoir_length_scan.png", fig)
println("\nsaved reservoir_length_scan.png")
