
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

*Length scaled* Squared Exponential Kernel and its analytical gradient
"""
function sq_exp_kernel(X)
    n_parts, n_dims = size(X)

    sq_pairwise_dists = pairwise(Euclidean(), X', X').^2
    #@assert size(sq_pairwise_dists) == (num_particles, num_particles)
    # l: length scale
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


"""
    update(X, dlogpdf; n_epochs=20000, dt=0.01, α=0.9, opt="adagrad")
"""
function update(X, dlogpdf; n_epochs=20000, dt=0.01, α=0.9, opt="adagrad")

    n_parts = size(X, 1)  # number of particles
    ϕ_t = zero(X)  # historical grad (t-1)
    ϵ = 1e-6  # fudge factor for adagrad with momentum

    for i in 1:n_epochs

        # evaluating the derivative of log pdf on particles
        dlogpdf_val = dlogpdf(X)

        # calculating the kernel matrix
        kxy, dxkxy = sq_exp_kernel(X)

        # gradient (step direction)
        ϕ = ((kxy * dlogpdf_val) .+ dxkxy) ./ n_parts


        if opt == "adagrad" # as implemented in SVGD original repo
            if i == 0
                ϕ_t = ϕ .^ 2
            else
                ϕ_t = (α .* ϕ_t) .+ ((1 - α) .* (ϕ .^ 2))
            end
            ϕ ./= ϵ .+ sqrt.(ϕ_t)
        end
        X = X + dt .* ϕ
    end
    return X
end
