# ==================================================================
# MAB open boundaries — A/B across scheme and tangential treatment
#
# Reads the three runs produced by validation/mab_obc.jl and asks one question:
# is the residual at the open boundaries limited by the SCHEME, or by the fact
# that the exterior data is zero?
#
# Script 9a's open boundaries carry no exterior state — every external value is
# (0, 0) — so on inflow the boundary's only instruction is "relax toward rest",
# while the true inflow at the MAB's southern and eastern edges is nothing like
# rest. If that is the binding constraint, the residual should concentrate on
# faces where water is ENTERING, and no change of scheme will move it much.
# ==================================================================

include(joinpath(@__DIR__, "obc_output.jl"))
using Oceananigans, Printf, Statistics

const RUNS = ("mab_a_baseline" => "9a as committed  (PerturbationAdvection 3d, Gradient(0) tangential)",
              "mab_b_oblique"  => "+ ObliqueRadiation, τ_in = 1 day        (Gradient(0) tangential)",
              "mab_c_obl_tang" => "+ radiated tangential                    (NormalRadiation 1 day)")

function analyse(tag)
    uts = FieldTimeSeries(outpath(tag * ".jld2"), "u")
    vts = FieldTimeSeries(outpath(tag * ".jld2"), "v")
    ηts = FieldTimeSeries(outpath(tag * ".jld2"), "η")
    n = length(uts.times)

    ui = Array(interior(uts[n])); vi = Array(interior(vts[n]))
    nx, ny, nz = size(ui, 1) - 1, size(vi, 2) - 1, size(ui, 3)
    ucc(i,j,k) = 0.5*(ui[i,j,k] + ui[i+1,j,k])
    vcc(i,j,k) = 0.5*(vi[i,j,k] + vi[i,j+1,k])
    spd(i,j,k) = sqrt(ucc(i,j,k)^2 + vcc(i,j,k)^2)
    ok(x) = isfinite(x) && x > 0
    rms(a) = isempty(a) ? NaN : sqrt(sum(abs2, a)/length(a))

    ks = max(1, nz-5):nz                      # near-surface
    ref = [spd(i,j,k) for i in 14:(nx-14), j in 14:(ny-14), k in ks]
    s_int = rms(filter(ok, vec(ref)))

    # edge band: 6 cells in from each OPEN edge (east, north, south)
    band = Float64[]
    for i in (nx-5):nx, j in 7:(ny-6), k in ks; push!(band, spd(i,j,k)); end
    for i in 7:(nx-6), j in (ny-5):ny, k in ks; push!(band, spd(i,j,k)); end
    for i in 7:(nx-6), j in 1:6,       k in ks; push!(band, spd(i,j,k)); end
    s_band = rms(filter(ok, band))

    # attribution: outermost wet cell, split by the sign of the normal velocity
    inflow = Float64[]; outflow = Float64[]
    add!(sp, into) = ok(sp) && push!(into > 0 ? inflow : outflow, sp)
    for j in 7:(ny-6), k in ks; add!(spd(nx,j,k), -ui[nx+1,j,k]); end
    for i in 7:(nx-6), k in ks
        add!(spd(i,ny,k), -vi[i,ny+1,k])
        add!(spd(i,1,k),   vi[i,1,k])
    end
    s_in, s_out = rms(inflow), rms(outflow)

    ηf = Array(interior(ηts[n]))
    ηv = filter(isfinite, vec(ηf))
    η_span = isempty(ηv) ? NaN : maximum(ηv) - minimum(ηv)

    umax = [maximum(filter(isfinite, abs.(Array(interior(uts[m]))))) for m in 1:n]

    return (; s_int, s_band, s_in, s_out, η_span, umax, times = uts.times,
              n_in = length(inflow), n_out = length(outflow))
end

println("\n" * "="^92)
println("MAB OPEN BOUNDARIES — does the scheme or the missing exterior data limit us?")
println("="^92)
@printf("%-46s %8s %8s %8s %8s %7s\n", "run", "interior", "edge/int", "inflow", "outflow", "in/out")
println("-"^92)
res = Dict{String,Any}()
for (tag, label) in RUNS
    r = analyse(tag); res[tag] = r
    @printf("%-46s %8.4f %8.2f %8.4f %8.4f %7.2f\n", label,
            r.s_int, r.s_band/r.s_int, r.s_in, r.s_out, r.s_in/r.s_out)
end
println("-"^92)
println("interior = free-interior rms speed (m/s); edge/int = 6-cell open-edge band relative to it;")
println("inflow / outflow = rms speed in the outermost wet cells, split by the sign of the normal flow.")
println()
@printf("%-46s %10s %12s\n", "", "η span (m)", "max|u| d6")
for (tag, label) in RUNS
    r = res[tag]
    @printf("%-46s %10.3f %12.2f\n", label, r.η_span, r.umax[end])
end
println("\nscript 8 (closed box) SSH span ≈ 0.28 m; script 9a recorded 0.24 m and a ~0.15 m/s edge current.")
println("="^92)
