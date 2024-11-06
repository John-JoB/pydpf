using Pkg
Pkg.activate("DynamicalSystemsGeneration")

using Plots;
plotlyjs();
using Random
using LinearAlgebra
using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using DataFrames
using CSV
using Dates

Random.seed!(123456789)

function lorenz!(du, u, p, t)
    N = length(du)
    F = p[1]
    du[1] = (u[2] - u[N-1]) * u[N] - u[1] + F
    du[2] = (u[3] - u[N]) * u[1] - u[2] + F
    du[N] = (u[1] - u[N-2]) * u[N-1] - u[N] + F
    for n in 3:(N-1)
        du[n] = (u[n+1] - u[n-2]) * u[n-1] - u[n] + F
    end
    nothing
end

function σ_lorenz(du, u, p, t)
    du .= 5.0
end

function prob_func(prob, i, repeat)
    u0 = prob.u0
    u0[1] *= rand()
    remake(prob, u0=u0)
end


state_dimension = 25
num_series = 100
delta_t = 0.01
u0 = ones(state_dimension)
u0[1] = 0.01

tspan = (0.0, 10.0)

prob = SDEProblem{true}(lorenz!, σ_lorenz, u0, tspan, [8,]);

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
sim = solve(ensemble_prob, SOSRA(), EnsembleThreads(), trajectories=num_series, saveat=range(tspan..., step=delta_t))
# sim_single = solve(prob, SOSRA());
# plot(sim_single, idxs=(1, 2, 3))

function compute_linear_observations(ensemble, H, R, delta_t)
    U = (cholesky(R).U)'
    obs_dim = size(H)[1]
    num_sims = length(ensemble)
    observation_array = Vector{Vector{Vector{Float64}}}(undef, num_sims)
    Threads.@threads for i in eachindex(ensemble)
        n_obs = length(ensemble[i].t)
        observation_array[i] = (Ref(H) .* ensemble[i].u) .+ (Ref(U) .* randn.(fill(obs_dim, n_obs)))
    end
    return observation_array
end

sparsity_amount = 0.8
H = rand(5, state_dimension)
H .*= (rand(5, state_dimension) .> sparsity_amount)
R = rand(5, 5)
R *= R'
R *= 1.0^2

observations = compute_linear_observations(sim, H, R, delta_t);
print("Generated $num_series $state_dimension dimensional series of length $(length(sim[1].t)) and associated observations!")

function expand_vector_col!(df, col)
    col_width = length(df[begin, col])
    target_cols = ["$(col)_$i" for i in 1:col_width]
    transform!(df, col => ByRow(identity) => target_cols)
    select!(df, Not(col))
end

function save_sim_obs_to_file(path, sim, obs)
    mkpath(path)
    dataframes_vector = Vector{DataFrame}(undef, length(sim))
    Threads.@threads for idx in eachindex(sim)
        dataframes_vector[idx] = DataFrame(series_id=idx, t=sim[idx].t, state=sim[idx].u, observation=obs[idx])
        expand_vector_col!(dataframes_vector[idx], "state")
        expand_vector_col!(dataframes_vector[idx], "observation")
    end
    dataframe_joined = vcat(dataframes_vector...)
    CSV.write(path * "simulated_data.csv", dataframe_joined)
    return dataframe_joined
end

current_datetime_fs_string = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
save_path = "./Lorenz 96/data/L96_SYSTEM_$(current_datetime_fs_string)_$(state_dimension)_DIM_STATE_$(size(H)[1])_DIM_OBS_WITH_$(sparsity_amount*100)PERCENT_SPARSITY/";



df_out = save_sim_obs_to_file(save_path, sim, observations);