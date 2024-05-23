using Pkg
Pkg.activate("DynamicalSystemsGeneration")
using DifferentialEquations
using Plots;
plotlyjs();
using Random
using Statistics
using DifferentialEquations.EnsembleAnalysis
import RandomNumbers: Xorshifts


# Random.seed!(7734)

μ = log(1 + 0.10) / 252
θ = 6.47e-5
κ = 0.05
σ = 2.38e-3
ρ = -0.5
tspan = (0.0, 252.0)

feller = 2 * κ * θ > σ^2

function HestonProblemLog(μ, κ, Θ, σ, ρ, u0, tspan; seed=rand(UInt64), kwargs...)
    f = function (du, u, p, t)
        vt = exp(u[2])
        du[1] = μ * u[1]
        du[2] = κ * (Θ / vt - 1) - σ^2 / (2*vt)
    end 
    g = function (du, u, p, t)
        vt = exp(u[2])
        du[1] = sqrt(vt) * u[1]
        du[2] = σ / sqrt(vt)
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

function reseed_heston(prob, i, repeat)
    HestonProblemLog(μ, κ, θ, σ, ρ, [50.0, log(θ)], tspan, seed=rand(UInt64))
end

heston_model = HestonProblemLog(μ, κ, θ, σ, ρ, [50.0, log(θ)], tspan, seed=rand(UInt64))



# heston_noise = CorrelatedWienerProcess!(Γ, tspan[1], zeros(2), zeros(2))
# prob = SDEProblem(heston_f!, heston_g!, [50.0, θ], tspan, noise=heston_noise)
sol = solve(heston_model, SRIW1(), dt=1e-2, adaptive=false, saveat=tspan[1]:0.1:tspan[2])
ensemble_prob = EnsembleProblem(heston_model, prob_func=reseed_heston)
sim = solve(ensemble_prob, SRIW1(), dt=1e-2, adaptive=false, EnsembleThreads(), trajectories=10, saveat=0.0:0.1:20.0)

p1 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
p2 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, exp(v)), 0, 2), ylabel="Sq. Volatility")
plot(p1, p2, layout=(2, 1), size=(600, 600))

plot(sim, idxs=((t,s,v) -> (t,s,exp(v))))