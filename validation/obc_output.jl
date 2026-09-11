# ==================================================================
# Where model output goes.
#
# A single dumbbell campaign is several GB of .jld2, and this project lives in
# Dropbox, so by default every one of those files was being synced — pure
# overhead for data that is regenerable from the scripts. Data now goes to
# OBC_OUTPUT (default ~/dev/obc-output), alongside the Oceananigans clone, which
# lives outside Dropbox for the same class of reason.
#
# Figures (.png, .mp4) deliberately stay in the project directory: they are small
# and they are the thing anyone actually wants to look at.
#
# Override with the OBC_OUTPUT environment variable.
# ==================================================================

obc_output_dir() = get(ENV, "OBC_OUTPUT", joinpath(homedir(), "dev", "obc-output"))

"""
    outpath(name)

Absolute path for a model-output file, under [`obc_output_dir`](@ref).
"""
function outpath(name)
    d = obc_output_dir()
    isdir(d) || mkpath(d)
    return joinpath(d, name)
end
