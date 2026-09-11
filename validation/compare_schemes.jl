# ==================================================================
# NormalRadiation vs ObliqueRadiation on both benchmarks.
#
# Two questions:
#
#  1. REDUCTION. At normal incidence the oblique scheme must reduce to the
#     normal one — when ∂φ/∂t̂ = 0, Raymond–Kuo's cₙ = −(∂ₜφ)(∂ₙφ)/|∇φ|²
#     collapses to Orlanski's −∂ₜφ/∂ₙφ and cₜ = 0. The coastal Kelvin wave
#     arrives exactly normal to its outflow boundary, so the two schemes
#     should give the same numbers there. A difference would mean a bug.
#
#  2. IMPROVEMENT. On the radiating Gaussian, where the boundaries see a
#     full range of incidence angles, the oblique scheme should reduce the
#     reflection coefficient.
#
# Run:  julia --project=. validation/compare_schemes.jl
# ==================================================================

using Oceananigans
using Oceananigans.BoundaryConditions: NormalRadiation, ObliqueRadiation
using Printf

include("kelvin_wave.jl")
include("radiating_gaussian.jl")

schemes = ("NormalRadiation"  => NormalRadiation(inflow_timescale = 0, outflow_timescale = Inf),
           "ObliqueRadiation" => ObliqueRadiation(inflow_timescale = 0, outflow_timescale = Inf))

println("\n" * "="^70)
println("1. KELVIN WAVE — normal incidence.  Reduction check: these must agree.")
println("="^70)

kelvin = Dict{String,Any}()
for (name, scheme) in schemes
    tag = "kelvin_" * lowercase(name)
    sim = kelvin_wave_simulation(baroclinic_scheme = scheme, filename = tag)
    run!(sim)
    m = kelvin_wave_metrics(tag)
    kelvin[name] = m
    @printf("  %-18s  rmse/A = %.5f   reflection = %.5f\n", name, m.rmse_rel, m.reflection)
end
@printf("  Δ(rmse)       = %+.2e\n",
        kelvin["ObliqueRadiation"].rmse_rel - kelvin["NormalRadiation"].rmse_rel)
@printf("  Δ(reflection) = %+.2e\n",
        kelvin["ObliqueRadiation"].reflection - kelvin["NormalRadiation"].reflection)

println("\n" * "="^70)
println("2. RADIATING GAUSSIAN — mixed incidence.  The discriminator.")
println("="^70)

gauss = Dict{String,Any}()
for (name, scheme) in schemes
    tag = "gaussian_" * lowercase(name)
    sim = radiating_gaussian_simulation(baroclinic_scheme = scheme, filename = tag)
    run!(sim)
    m = radiating_gaussian_metrics(tag)
    gauss[name] = m
    @printf("  %-18s  R = %.5f   E_res/E₀ = %.3e\n", name, m.reflection, m.reflection^2)
end
Rn = gauss["NormalRadiation"].reflection
Ro = gauss["ObliqueRadiation"].reflection
@printf("  change in R           : %+.1f%%\n", 100 * (Ro - Rn) / Rn)
@printf("  change in residual E  : %+.1f%%\n", 100 * (Ro^2 - Rn^2) / Rn^2)

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
@printf("%-18s %14s %14s %14s\n", "scheme", "kelvin rmse/A", "kelvin R", "gaussian R")
for (name, _) in schemes
    @printf("%-18s %14.5f %14.5f %14.5f\n", name,
            kelvin[name].rmse_rel, kelvin[name].reflection, gauss[name].reflection)
end

# side-by-side residual maps on the discriminating case
let
    fig = Figure(size = (1200, 560))
    for (col, (name, _)) in enumerate(schemes)
        m = gauss[name]
        n = length(m.times)
        η = interior(m.ηts[n])[:, :, 1]
        ax = Axis(fig[1, col], title = "$name — residual at t = 8 h  (R = $(round(m.reflection, digits=4)))",
                  aspect = DataAspect(), xlabel = "x (km)", ylabel = col == 1 ? "y (km)" : "")
        heatmap!(ax, m.x ./ 1e3, m.y ./ 1e3, η, colormap = :balance, colorrange = (-0.002, 0.002))
    end
    Label(fig[0, 1:2], "Radiating Gaussian — residual after the ring has left (±0.002 m)",
          fontsize = 18, tellwidth = false)
    save("scheme_comparison.png", fig)
    println("\nsaved scheme_comparison.png")
end
