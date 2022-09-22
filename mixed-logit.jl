using Distributions
using DataFrames
using Random;
Random.seed!(16);
using Optim

# Simulate errors
function generate_errors(config)
    return rand(Gumbel(), config.N, config.J)
end

# Generate characteristics
function characteristics(config)
    # Preallocate
    X = zeros(config.N, config.J)

    # Draw repeated values (identical characteristics across people)
    x = randn(config.J)
    for i in 1:size(X, 1)
        X[i, :] = x
    end

    # X = randn(config.N, config.J)
    return X
end

# 
configuration = (
    N=100000,
    J=2,
    # beta = [50.0 20.0;]
    beta=[0.5 -0.1;]
)

# Calculate it y'all
epsilon = generate_errors(configuration)
X = characteristics(configuration)
V = X .* configuration.beta
# V[:, 1] = V[:, 1] .- V[:, 2]
# V[:, 2] .= 0
U = V .+ epsilon
choices = map(m -> m.I[2], findmax(U, dims=2)[2])
choice_count = Dict(i => sum(choices .== i) ./ length(choices) for i in 1:configuration.J)

# Calculate logit probabilities
exp_V = map(exp, V)
P = exp_V ./ sum(exp_V, dims=2)

display([choice_count[1] choice_count[2];])
mean(P, dims=1)

function mle_logit(configuration, X, choices)
    # Empirical choice probabilities
    choice_hat = zeros(1, configuration.J)
    for j in 1:configuration.J
        choice_hat[1,j] = sum(choices .== j) / length(choices)
    end

    # Define optimization function
    function ℓ(beta)
        # Compute observable utility
        V = X .* beta
        exp_V = map(exp, V)

        # Calculate choice probabilities
        P = exp_V ./ sum(exp_V, dims=2)
        # display(vcat(choice_hat, mean(P, dims=1)))
        log_P = log.(P)

        # Generate log-likelihood
        return -sum(log_P[i, choices[i]] for i in eachindex(choices))
    end

    # Optimize it
    initial_guess = zeros(size(configuration.beta)...)
    result = optimize(ℓ, initial_guess)

    return result
end

result = mle_logit(configuration, X, choices)
beta_hat = result.minimizer
true_beta = configuration.beta

DataFrame(
    beta_hat = vec(beta_hat),
    true_beta = true_beta[:]
)