
# imports
using Statistics: mean, median
using Distances: pairwise, Euclidean


"""
    int(x) = trunc(Int, x)
"""
int(x) = trunc(Int, x)


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
<<<<<<< Updated upstream
    Analytical gradient of log of Multivariae normal distribution
=======
    very lazy implementation of Mesh-Grid for 2D-distributions
"""
function pdf_grid(x_, y_, dist_2D)
    pdf_ = Array{Float64}(undef, size(x_, 1), size(y_, 1))
    for i_ in 1:size(x_, 1)
        for j_ in 1:size(y_, 1)
            pdf_[i_, j_] = pdf(dist_2D, Vector([x_[i_], y_[j_]]))
        end
    end
    return pdf_
end


"""
    Analytical gradient of log of Multivariate normal distribution
>>>>>>> Stashed changes
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

    sq_pairwise_dists = pairwise(Euclidean(), X', X', dims=2).^2
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

    n_parts, n_dims = size(X) # number of particles
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


function update_rec(X, dlogpdf; n_epochs=20000, dt=0.01, α=0.9, opt="none")

    n_parts, n_dims = size(X)  # number of particles and dims
    X_records = Array{Float64}(undef, n_epochs+1, n_parts, n_dims)
    X_records[1, :, :] = X

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
        X_records[i+1, :, :] = X
    end
    return X, X_records
end


"""
    γ_t is the annealing schedule

"""
function update_anneal(X, dlogpdf, γ_t; n_epochs=20000, dt=0.01, α=0.9, opt="none")

    n_parts, n_dims = size(X)  # number of particles and dims
    X_records = Array{Float64}(undef, n_epochs+1, n_parts, n_dims)
    X_records[1, :, :] = X

    ϕ_t = zero(X)  # historical grad (t-1)
    ϵ = 1e-6  # fudge factor for adagrad with momentum

    for i in 1:n_epochs

        # evaluating the derivative of log pdf on particles
        dlogpdf_val = dlogpdf(X)

        # calculating the kernel matrix
        kxy, dxkxy = sq_exp_kernel(X)

        # gradient (step direction)
        ϕ = ((γ_t[i] .* (kxy * dlogpdf_val)) .+ dxkxy) ./ n_parts

        if opt == "adagrad" # as implemented in SVGD original repo
            if i == 0
                ϕ_t = ϕ .^ 2
            else
                ϕ_t = (α .* ϕ_t) .+ ((1 - α) .* (ϕ .^ 2))
            end
            ϕ ./= ϵ .+ sqrt.(ϕ_t)
        end
        X = X + dt .* ϕ
        X_records[i+1, :, :] = X
    end
    return X, X_records
end


"""
    annealing_schedule(n_epochs, method; annealing=0.8)

Returns an annealing schedule for SVGD updates

Ref: F. D’Angelo, V. Fortuin, *Annealed Stein Variational Gradient Descent*, 2020

## Arguments
- `n_epochs::Int`: number of epochs
- `method::String`: `linear`, `hyperbolic` and `cyclical` are supported
- `annealing::Float64`: the ratio of epochs where annealing is scheduled (default is 0.8)
"""
function annealing_schedule(n_epochs::Int, method::String; annealing::Float64=0.8)
    C = 4
    T = int(n_epochs*annealing)
    t = range(0., 1., length=T)
    γ_t = ones(n_epochs)
    if method == "linear"
        γ_t[1:T] = t
    elseif method == "hyperbolic"
        γ_t[1:T] = tanh.((1.3 .* t) .^ 7)
    elseif method == "cyclical"
        t_r = 1:T-1
        γ_t[1:T-1] = ((t_r .% (T/C)) ./ (T/C)) .^ 2
    else
        throw("Annealing method not supported!")
    end
    return γ_t
end
