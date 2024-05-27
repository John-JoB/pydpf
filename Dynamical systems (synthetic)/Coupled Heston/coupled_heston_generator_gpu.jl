using Pkg
Pkg.activate("DynamicalSystemsGeneration")
using DiffEqGPU, CUDA
using DifferentialEquations
using Plots; plotlyjs()
using Random
using Statistics
using DifferentialEquations.EnsembleAnalysis
using StaticArrays
import RandomNumbers: Xorshifts


Random.seed!(7734)

const μ = log(1.0f0 + 0.10f0) / 252.0f0
# θ = 6.47e-5
# κ = 0.097
# σ = 5.38e-3
# ρ = -0.5
const θ = 0.0055f0 / sqrt(252.0f0)
const κ = 5.0f0
const σ = 0.3f0 / sqrt(252.0f0)
const ρ = -0.30f0
const v0 = θ
const tspan = (0.0f0, 252.0f0)

feller = 2*κ*θ > σ^2
# @assert feller

function HestonProblemStatic(μ, κ, θ, σ, ρ, u0, tspan; seed=UInt64(0), kwargs...)
    f = function (u, p, t)
        du1 = μ * u[1]
        du2 = κ * (θ - max(0.0f0, u[2]))
        return SVector(du1, du2)
    end
    g = function (u, p, t)
        du1 = sqrt(max(0.0f0, u[2])) * u[1]
        du2 = σ * sqrt(max(0.0f0, u[2]))
        return SVector(du1, du2)
    end
    Γ = SA[1.0f0 ρ; ρ 1.0f0] # Covariance Matrix
    noise_rate_prototype = nothing

    if seed == 0
        seed = rand(UInt64)
    end
    noise = CorrelatedWienerProcess!(Γ, tspan[1], zeros(2), zeros(2),
        rng=Xorshifts.Xoroshiro128Plus(seed))

    sde_f = SDEFunction{false}(f, g)
    SDEProblem(sde_f, u0, tspan, noise=noise, seed=seed, kwargs...)
end

heston_model = HestonProblemStatic(μ, κ, θ, σ, ρ, SA[50.0f0, v0], tspan, seed=rand(UInt64))

function reseed_heston(prob, i, repeat)
    HestonProblemStatic(μ, κ, θ, σ, ρ, SA[50.0f0, v0], tspan, seed=rand(UInt64))
end

check_domain(u, p, t) = u[2] < 0.0f0

heston_model = HestonProblemStatic(μ, κ, θ, σ, ρ, SA[50.0f0, v0], tspan, seed=rand(UInt64))


sol = solve(heston_model, SRIW1(), dt=1f-2, adaptive=false, saveat=tspan[1]:0.1:tspan[2], isoutofdomain=check_domain)
ensemble_prob = EnsembleProblem(heston_model, prob_func=reseed_heston, safetycopy=false)
sim = solve(ensemble_prob, GPUEM(), dt=1f-3, adaptive=false, EnsembleGPUKernel(CUDA.CUDABackend(), 0.0), trajectories=20000, saveat=tspan[1]:1f0:tspan[2])

print("Done simulating on GPU!")

# p1 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, v), 0, 2), ylabel="Sq. Volatility")
# plot(p1, p2, layout=(2, 1), size=(600, 600))

summ = EnsembleSummary(sim)
plot(summ)
# p1_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, v), 0, 2), ylabel="Sq. Volatility")
# plot(p1_ens, p2_ens, layout=(2, 1), size=(600, 600))