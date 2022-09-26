# Imports
using Distributions
using DataFrames
using StatsFuns
using Random
Random.seed!(15)

# Design a struct that defines a bundle
struct Bundle{A,B}
    price::A
    length::B
end

# Error simulation
function generate_errors(config)
    return rand(Gumbel(), config.N, config.J)
end

function generate_articles(config)
    return rand.(config.λ_density, config.day)
end

function flatten_articles(config, articles)
    tups = []
    counter = 1
    for d in 1:config.day
        for a in 1:config.article
            for article_id in 1:articles[a][d]
                tup = (
                    d=d,
                    article_type=a,
                    article_typed_id = article_id,
                    id = counter,
                )
                counter += 1
                push!(tups, tup)
            end
        end
    end 
    return DataFrame(tups)
end

function generate_betas(config)
    β = rand(config.β_density, config.people_types)
    return β
end

function bundle_value(config, bundle, ϕ)
    B = map(p -> bundle_value(config, bundle, p, ϕ), 1:config.people_types)
    B += ϕ
    return B
end

function bundle_value(config, bundle, person_type, ϕ)
    B = 0.0
    for t in 1:bundle.length
        for a in 1:config.article
            inner = sum(k  * pdf(config.λ_density[a], k) for k in 1:100)
            inner *= config.μs[person_type][a] * config.beta[person_type, a]

            B += inner
        end
    end
    return B
end

function generate_phi(config)
    Φ = rand(config.Φ_density, config.people_types, config.day)
    return Φ
end

function generate_readership(config, articles)
    readers = []
    person_id = 1

    # Iterate through individuals
    for person_type in 1:config.people_types
        for person_count in 1:config.n_people_per
            for a in 1:config.article
                for d in 1:config.day
                    arts = filter(x -> x.article_type == a && x.d == d, articles)
                    for art in eachrow(arts)
                        article_arrived = rand(Bernoulli(config.μs[person_type][a]))
                        paywalled = article_arrived ? rand(Bernoulli(config.paywall_prob)) : false # Assume 
                        tup = (
                            person_type=person_type,
                            person_count=person_count,
                            a=a,
                            d=d,
                            person_id = person_id,
                            id = art.id,
                            arrived = article_arrived,
                            paywalled = paywalled
                        )
                        person_id += 1
                        push!(readers, tup)
                    end
                end
            end
        end
    end
    return DataFrame(readers)
end


# Set up article generating parameters
λs = [Poisson(rand(1:10)) for _ in 1:5]
μs = [[rand() for _ in 1:5] for _ in 1:2]

θ = rand(5)
θ ./= sum(θ)

configuration = (
    people_types=2,                        # X
    n_people_per=5,                        # number of individuals per type
    article=5,                             # a
    day=1,                                 # d
    θ_density = Categorical(θ),            # Article generative density, assume f(θ)=g(θ)
    λ_density = λs,                        # Density governing article production, iid poisson
    ξ_density = Normal(),                  # Unobserved utility shock. Assume Gaussian, i.i.d. shocks across X/a
    Φ_density = Normal(),                  # Distribution of the bundle shocks
    # β_density = Normal(),                # Utility coefficients. Assume Gaussian, i.i.d. shocks across X/a
    μs = μs,                               # Arrival probabilities, i.i.d.
    paywall_prob = 0.5,                    # Assume paywall prob is 50/50
    δ = 1.0,                               # Discount rate
    bundles = [Bundle(1.0, 5)],            # Set up a bundle with price=1, T=5
    beta = rand(LogNormal(), 2, 5),        # Set up coefficients
)

articles = generate_articles(configuration)
article_flat = flatten_articles(configuration, articles)
bundle_shocks = generate_phi(configuration)
bundle_value(configuration, Bundle(1.0, 3), bundle_shocks)
# betas = generate_betas(configuration)
# generate_readership(configuration, article_flat)

# @info ""  bundle_value(configuration, Bundle(1.0, 3), 1) bundle_value(configuration, Bundle(1.0, 3), 2)
