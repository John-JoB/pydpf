using Pkg
Pkg.activate("DynamicalSystemsGeneration")
using DifferentialEquations
using Plots; plotlyjs()
using Random
using Statistics
using DifferentialEquations.EnsembleAnalysis
import RandomNumbers: Xorshifts


Random.seed!(1)

μ = log(1 + 0.07) / 252
# θ = 6.47e-5
# κ = 0.097
# σ = 5.38e-3
# ρ = -0.5

# θ = 0.0055 / sqrt(252)
θ = 0.005 / sqrt(252)

# κ = 5.0
κ = 0.05

σ = 0.2 / sqrt(252)

ρ = -0.30
# ρ = 0.0

v0 = θ
tspan = (0.0, 252.0)

feller = 2*κ*θ > σ^2
sim = nothing
summ = nothing
GC.gc()
# @assert feller

function HestonProblem(μ, κ, θ, σ, ρ, u0, tspan; seed=UInt64(0), kwargs...)
    f = function (du, u, p, t)
        du[1] = μ * u[1]
        du[2] = κ * (θ - max(0.0, u[2]))
    end
    g = function (du, u, p, t)
        du[1] = sqrt(max(0.0, u[2])) * u[1]
        du[2] = σ * sqrt(max(0.0, u[2]))
    end
    Γ = [1 ρ; ρ 1] # Covariance Matrix
    noise_rate_prototype = nothing

    if seed == 0
        seed = rand(UInt64)
    end
    noise = CorrelatedWienerProcess!(Γ, tspan[1], zeros(2), zeros(2),
        rng=Xorshifts.Xoroshiro128Plus(seed))

    sde_f = SDEFunction{true}(f, g)
    SDEProblem(sde_f, u0, tspan, noise=noise, seed=seed, kwargs...)
end

heston_model = HestonProblem(μ, κ, θ, σ, ρ, [50.0, v0], tspan, seed=rand(UInt64))

function reseed_heston(prob, i, repeat)
    HestonProblem(μ, κ, θ, σ, ρ, [50.0, θ], tspan, seed=rand(UInt64))
end

check_domain(u, p, t) = u[2] < 0.0

heston_model = HestonProblem(μ, κ, θ, σ, ρ, [50.0, v0], tspan, seed=rand(UInt64))


sol = solve(heston_model, SRIW1(), dt=1e-2, adaptive=false, saveat=tspan[1]:0.1:tspan[2], isoutofdomain=check_domain)
ensemble_prob = EnsembleProblem(heston_model, prob_func=reseed_heston, safetycopy=false)
sim = solve(ensemble_prob, SRIW1(), dt=1e-2, adaptive=false, EnsembleThreads(), trajectories=100, saveat=tspan[1]:1.0:tspan[2])

print("Done simulating!")

p1 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
p2 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, v), 0, 2), ylabel="Sq. Volatility")
plot(p1, p2, layout=(2, 1), size=(600, 600))

# summ = EnsembleSummary(sim)
# plot(summ)
# p1_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, v), 0, 2), ylabel="Sq. Volatility")
# plot(p1_ens, p2_ens, layout=(2, 1), size=(600, 600))