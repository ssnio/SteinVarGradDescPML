
# imports
using Statistics: mean, median
using Distances: pairwise, Euclidean


"""
    dropmean(X, d) = dropdims(mean(X, dims=d), dims=d)
"""
function dropmean(X, d)
    if ndims(X) == 1
        mean(test_a, dims = d)
    else
        dropdims(mean(X, dims = d), dims = d)
    end
end


"""
    Analytical gradient of log of Multivariae normal distribution
"""
function ana_dlogmvn(μ, Σ, x)
    - broadcast(-, x, μ) * Σ^-1
end


"""
    Squared Exponential Kernel and its analytical gradient

"""
function sq_exp_kernel(X)
    n_parts, n_dims = size(X)

    sq_pairwise_dists = pairwise(Euclidean(), X', X').^2
    #@assert size(sq_pairwise_dists) == (num_particles, num_particles)
    # l: lengrh scale
    l = sqrt(median(sq_pairwise_dists) / 2 / log(n_parts + 1))
    kxy = exp.(-sq_pairwise_dists / l.^2 / 2)

    dxkxy = - kxy * X
    sumkxy = sum(kxy, dims=2)
    for i in 1:n_dims
        dxkxy[:, i] = dxkxy[:, i] + X[:, i] .* sumkxy
    end
    dxkxy = dxkxy / (l .^ 2)

    return kxy, dxkxy
end


function update(X, dlogpdf; n_iter=20000, ϵ=0.01, α=0.9)

    n_parts, _ = size(X)

    # adagrad with momentum
    fudge_factor = 1e-6
    historical_ϕ = zero(X)

    for i in 1:n_iter

        dlogpdf_val = dlogpdf(X)

        # calculating the kernel matrix
        kxy, dxkxy = sq_exp_kernel(X)

        # gradient (step direction)
        ϕ = ((kxy * dlogpdf_val) .+ dxkxy) ./ n_parts

        # adagrad
        if i == 0
            historical_ϕ = ϕ .^ 2
        else
            historical_ϕ = (α .* historical_ϕ) .+ ((1 - α) .* (ϕ .^ 2))
        end
        adj_grad = ϕ ./ (fudge_factor .+ sqrt.(historical_ϕ))
        X = X + ϵ * adj_grad
    end
    return X
end
