using Pkg
Pkg.activate("DynamicalSystemsGeneration")
using DifferentialEquations
using Plots;
plotlyjs();
using Random
using Statistics
using DifferentialEquations.EnsembleAnalysis
import RandomNumbers: Xorshifts
using JSON
using DataFrames
using CSV
using StatsPlots, KernelDensity
using Bootstrap



Random.seed!(12345)

μ = log(1.1) / 252
# θ = 6.47e-5
# κ = 0.097
# σ = 5.38e-3
# ρ = -0.5

# θ = 0.0055 / sqrt(252)
# θ = 0.0055 / sqrt(252)
θ = 2e-4
# θ = 0.10 / sqrt(252)


# κ = 5.0
κ = 1.0

σ = 0.2 / sqrt(252)
# σ = 0

ρ = -0.70
# ρ = 0.0

v0 = θ
tspan = (0.0, 252.0)

feller = 2 * κ * θ > σ^2
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
    Γ = [1 ρ; ρ 1] # Correlation matrix of wiener processes
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
    HestonProblem(μ, κ, θ, σ, ρ, [50.0, v0], tspan, seed=rand(UInt64))
end

check_domain(u, p, t) = u[2] < 0.0

heston_model = HestonProblem(μ, κ, θ, σ, ρ, [50.0, v0], tspan, seed=rand(UInt64))


sol = solve(heston_model, EM(), dt=1e-3, adaptive=false, saveat=tspan[1]:0.1:tspan[2], isoutofdomain=check_domain)
ensemble_prob = EnsembleProblem(heston_model, prob_func=reseed_heston, safetycopy=false)
sim = solve(ensemble_prob, EM(), dt=5e-3, adaptive=false, EnsembleThreads(), trajectories=500, saveat=tspan[1]:1.0:tspan[2])

println("Done simulating! (base problem)")

# p1 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, sqrt(v)), 0, 2), ylabel="Volatility")
# plot(p1, p2, layout=(2, 1), size=(600, 600))

# summ = EnsembleSummary(sim);
# plot(summ)
# p1_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, sqrt(v)), 0, 2), ylabel="Volatility")
# plot(p1_ens, p2_ens, layout=(2, 1), size=(600, 600))


function extract_state_and_observations(ensemble, t_steps)
    num_sims = length(ensemble)
    observation_array = Vector{Vector{Float64}}(undef, num_sims)
    state_array = Vector{Vector{Float64}}(undef, num_sims)
    for i in eachindex(ensemble)
        observation_array[i] = getindex.(ensemble[i](t_steps).u, Ref(1))
        state_array[i] = getindex.(ensemble[i](t_steps).u, Ref(2))
    end
    return observation_array, state_array
end

t_steps = range(0, 252, step=1.0)

observations, states = extract_state_and_observations(sim, t_steps)


println("Obs test stats")
println(mean([o[end] for o in observations]))
println(mean([o[end]/o[begin] for o in observations]))
println(std([o[end] for o in observations]))

println(mean([o[100] for o in observations]))
println(mean([o[100]/o[begin] for o in observations]))
println(std([o[100] for o in observations]))

println("\nState test stats")
println(mean([o[end] for o in states]))
println(std([o[end] for o in states]))

println(mean([o[100] for o in states]))
println(std([o[100] for o in states]))

states_matrix = mapreduce(permutedims, vcat, states)
obs_matrix = mapreduce(permutedims, vcat, observations)
ret_matrix = obs_matrix[:, begin+1:end] ./ obs_matrix[:, begin:end-1];
flat_rets = reduce(vcat, ret_matrix)

println("\nShould be near zero:")
println(var(ret_matrix) .- θ)
println(mean(var(ret_matrix, dims=2)) .- θ)
println((mean((ret_matrix)) - exp(μ)) / (exp(μ)-1))
# println(mean(log.(ret_matrix) .- μ))
# plot(kde(log.(flat_rets)), xlims = [-0.01, 0.01])
# plot!([μ], seriestype=:vline, color="red")
# plot!([mean(log.(ret_matrix))], seriestype=:vline, color="green")

# bs_mean_ci = bootstrap(mean, log.(flat_rets), BasicSampling(1000))
# confint(bs_mean_ci, PercentileConfInt(0.95))


# function expand_vector_col!(df, col)
#     col_width = length(df[begin, col])
#     target_cols = ["$(col)_$i" for i in 1:col_width]
#     transform!(df, col => ByRow(identity) => target_cols)
#     select!(df, Not(col))
# end

# function save_sim_obs_to_file(path, states, obs, t_steps)
#     mkpath(path)
#     dataframes_vector = Vector{DataFrame}(undef, length(obs))
#     Threads.@threads for idx in eachindex(obs)
#         dataframes_vector[idx] = DataFrame(series_id=idx, t=collect(t_steps), state=states[idx], observation=obs[idx])
#         expand_vector_col!(dataframes_vector[idx], "state")
#         expand_vector_col!(dataframes_vector[idx], "observation")
#     end
#     dataframe_joined = vcat(dataframes_vector...)
#     CSV.write(path * "simulated_data.csv", dataframe_joined)
#     return dataframe_joined
# end

# params = Dict("r" => μ, "k" => κ, "theta" => θ, "sigma" => σ, "rho" => ρ)
# save_path = "./Heston Model/data/synthetic_run/";

# open(save_path * "params.json", "w") do f
#     JSON.print(f, params, 4)
# end


# df_out = save_sim_obs_to_file(save_path, states, observations, t_steps);