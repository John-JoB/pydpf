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
using Bootstrap




Random.seed!(1234)

μ = log(1.1) / 252
σ = 0.2 / sqrt(252)

function BlackScholesProblem(μ, σ, u0, tspan; seed=UInt64(0), kwargs...)
    f = function (du, u, p, t)
        du[1] = μ * u[1]
    end
    g = function (du, u, p, t)
        du[1] = σ * u[1]
    end

    if seed == 0
        seed = rand(UInt64)
    end
    sde_f = SDEFunction{true}(f, g)
    SDEProblem(sde_f, u0, tspan, seed=seed, kwargs...)
end

bs_model = BlackScholesProblem(μ, σ, [50.0], tspan, seed=rand(UInt64))

function reseed_bs(prob, i, repeat)
    BlackScholesProblem(μ, σ, [50.0], tspan, seed=rand(UInt64))
end

bs_model = BlackScholesProblem(μ, σ, [50.0], tspan, seed=rand(UInt64))


sol = solve(bs_model, SRIW1(), adaptive=true, saveat=tspan[1]:0.1:tspan[2])
ensemble_prob = EnsembleProblem(bs_model, prob_func=reseed_bs, safetycopy=false)
sim = solve(ensemble_prob, SRIW1(), adaptive=true, EnsembleThreads(), trajectories=1000, saveat=tspan[1]:1.0:tspan[2])

println("Done simulating! (black scholes problem)")

p1 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2 = plot(sol, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, sqrt(v)), 0, 2), ylabel="Volatility")
# plot(p1, p2, layout=(2, 1), size=(600, 600))
plot(p1)

summ = EnsembleSummary(sim);
plot(summ)
p1_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=(0, 1), ylabel="Price")
# p2_ens = plot(sim, lw=1.5, size=(666, 200), dpi=300, label="", idxs=((t, v) -> (t, sqrt(v)), 0, 2), ylabel="Volatility")
# plot(p1_ens, p2_ens, layout=(2, 1), size=(600, 600))
plot(p1_ens)


function extract_observations(ensemble, t_steps)
    num_sims = length(ensemble)
    observation_array = Vector{Vector{Float64}}(undef, num_sims)
    for i in eachindex(ensemble)
        observation_array[i] = getindex.(ensemble[i](t_steps).u, Ref(1))
    end
    return observation_array
end

t_steps = range(0, 252, step=1.0)

observations = extract_observations(sim, t_steps)

obs_matrix = mapreduce(permutedims, vcat, observations)
ret_matrix = obs_matrix[:, begin+1:end] ./ obs_matrix[:, begin:end-1];
flat_rets = reduce(vcat, ret_matrix);



println("Obs test stats")
println(mean([o[end] for o in observations]))
println(mean([o[end]/o[begin] for o in observations]))
println(std([o[end] for o in observations]))

println(mean([o[100] for o in observations]))
println(mean([o[100]/o[begin] for o in observations]))
println(std([o[100] for o in observations]))

println("\nShould be close")
println(mean(ret_matrix))
println(exp(μ))

bs_mean_ci = bootstrap(mean, log.(flat_rets), BasicSampling(1000))
confint(bs_mean_ci, BasicConfInt(0.95))

# println("\nState test stats")
# println(mean([o[end] for o in states]))
# println(std([o[end] for o in states]))

# println(mean([o[100] for o in states]))
# println(std([o[100] for o in states]))

# states_matrix = mapreduce(permutedims, vcat, states)
# obs_matrix = mapreduce(permutedims, vcat, observations)
# ret_matrix = obs_matrix[:, begin+1:end] ./ obs_matrix[:, begin:end-1];

# println("\nShould be near zero:")
# println(var(ret_matrix) .- θ)
# println(mean(var(ret_matrix, dims=2)) * θ)

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