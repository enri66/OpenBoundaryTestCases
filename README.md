# OpenBoundaryTestCases

Test cases for open boundary conditions in regional ocean models built on
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) and [NumericalEarth](https://github.com/NumericalEarth).
The current set was used in developing `ObliqueRadiation`
([#5962](https://github.com/CliMA/Oceananigans.jl/pull/5962)) and `TracerReservoir`
([#5964](https://github.com/CliMA/Oceananigans.jl/pull/5964)) in Oceananigans.

| script | what it tests |
|---|---|
| `kelvin_wave.jl` | coastal Kelvin wave against its exact solution (#5229, test case 3) |
| `radiating_gaussian.jl` | radiating Gaussian at oblique incidence (#5229, test case 1; MOM6 `circle_obcs`) |
| `stratified_gaussian.jl` | the same, baroclinic |
| `supercritical.jl` | supercritical barotropic flow |
| `compare_schemes.jl` | `NormalRadiation` vs `ObliqueRadiation` on the Kelvin-wave and Gaussian cases |
| `dumbbell.jl`, `dumbbell_oblique.jl`, `dumbbell_forced.jl` | an eddy crossing the cut between a downscaled region and its parent, at normal and oblique incidence, and a MOM6-style strait exchange |
| `neck_basin.jl` | the neck of the dumbbell as its own domain, open at both ends, driven by a parent run |
| `dyed_reservoir.jl`, `reservoir_length_scan.jl` | tracer reservoirs (after MOM6 `dyed_obcs`) and the choice of `inflow_length_scale` |
| `mab_obc.jl`, `mab_compare.jl` | a Mid-Atlantic Bight regional configuration with open boundaries, and an A/B comparison across schemes |

`obc_output.jl` sets where model output goes: `OBC_OUTPUT`, default `~/dev/obc-output`.

## Running

Develop an Oceananigans checkout on the relevant PR branch into this environment — #5962's branch for
the radiation scripts, #5964's for the reservoir scripts:

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path = "/path/to/Oceananigans.jl")
Pkg.instantiate()
```

Then, for example, `julia --project validation/radiating_gaussian.jl`.
