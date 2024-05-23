using Pkg
Pkg.activate("DynamicalSystemsGeneration")
using DifferentialEquations
using Plots; plotlyjs()
using Random
using Statistics
using DifferentialEquations.EnsembleAnalysis
import RandomNumbers: Xorshifts


Random.seed!(7734)

μ = log(1 + 0.10) / 252
θ = 6.47e-5
κ = 0.05
σ = 2.38e-3
ρ = -0.5
tspan = (0.0, 252.0)

feller = 2*κ*θ > σ^2

function HestonProblem(μ, κ, Θ, σ, ρ, u0, tspan; seed=UInt64(0), kwargs...)
    f = function (du, u, p, t)
        du[1] = μ * u[1]
        du[2] = κ * (Θ - u[2])
    end
    g = function (du, u, p, t)
        du[1] = sqrt(abs(u[2])) * u[1]
        du[2] = σ * sqrt(abs(u[2]))
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

heston_model = HestonProblem(μ, κ, θ, σ, ρ, [50.0, θ], tspan, seed = rand(UInt64))




# heston_noise = CorrelatedWienerProcess!(Γ, tspan[1], zeros(2), zeros(2))
# prob = SDEProblem(heston_f!, heston_g!, [50.0, θ], tspan, noise=heston_noise)
sol = solve(heston_model, ImplicitEulerHeun(), dt=5e-3, adaptive=false, saveat=tspan[1]:0.1:tspan[2])

p1 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
p2 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, v), 0, 2), ylabel="Sq. Volatility")
plot(p1, p2, layout=(2, 1), size=(600, 600))