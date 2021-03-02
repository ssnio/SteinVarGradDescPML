using Distributions: Gamma, Normal

"""     
    Analytical gradient of log joint for bayesian logistic regression.

    The observed data D = {X, y} consist of N binary class labels y_t in {-1,1},
    and d covariates for each datapoint x_t.

    Hidden variables θ = {α, w} where w consists of w_k regression coefficients
    in R and precision parameter α in R+. Hence the model is described by:

    p(y_t | x_t, w, α, a, b) = p(y_t | x_t , w) * PI[((w_k | α)] * p(α; a,b)

    where:
    p(y_t | x_t , w) = y_t * 1 / (1+e^(w^T * x_t) + (1- y_t) * 1 / (1+e^(-w^T * x_t)
    p(w_k | α) ~ N(w_k; 0, α^-1)
    p(α; a,b) ~ Gamma(α; a,b)

    About logistic regression:
    https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf

"""

σ(x:: Real) = 1.0 / (1.0 + exp(-x))

function ana_dlogblg(particles, X, Y, a0=1, b0=0.01)
    # number of regression coefficients
    k = size(particles,2) - 1
    # regression weights
    W = particles[:,1:k]
    # log(α) stored in last dimension of particles
    α = exp.(particles[:,end])
    # α = particles[:,end]

    """ Log-Likelihood:
    ∂LL / ∂W_j = ∂LL(θ) / ∂p * ∂p / ∂z * ∂z / ∂W_j
    = [y_j - sigmoid(W^T* X)] * x_j
    """

    sigmoids = σ.(W * X') # dim: particles x samples
    Y = (Y .+ 1) ./ 2 # rescaling Y
    d_DataLL_dW = (Y' .- sigmoids) * X

    """ Precision:
    ! Normal formulation, using the precision param α = 1 / σ^2
    ! See: https://en.wikipedia.org/wiki/Normal_distribution#Alternative_parameterizations

      ∂(Σ_{1:j:K} ln p(w_k;0,α^-1) )/∂θ
    =  ∂(Σ_{1:j:K} ln(sqrt(α)) - ln(sqrt(2π)) - 1/2*α*(w_k)^2 )/∂θ

    => =  ∂(Σ_{1:j:K} ln(sqrt(α)) - ln(sqrt(2π)) - 1/2*α*(w_k)^2 )/∂w_k
       =  (-α*w_k)

    => =  Σ_{1:j:K} ∂(ln(sqrt(α)) - ln(sqrt(2π)) - 1/2*α*w_k^2 )/∂α
       =  Σ_{1:j:K} 1/2α - 1/2(w_k)^2          | * α
       =  Σ_{1:j:K} 1/2 - (α/2(w_k)^2
       = k/2 - Σ_{1:j:K} α/2 * (w_k)^2

    """

    d_prec_dW = - α .* W # dim: particles x k

    d_prec_dα = 1/2 .- (α ./ 2 .* W.^2) # dim: particles x k
    d_prec_dα = dropdims(sum(d_prec_dα , dims=2), dims=2) # dim: particles x 1 (alpha)

    """ Precision prior:
      ∂(lnp(α;a0,b0))/∂α | where p(α;a0,b0) ~ Gamma(α; a,b)
    = ∂(ln(b0^a0) - ln(Gamma-fct(a0)) + (a0-1)*ln(α) - b0*α)/∂α
    = ∂( (a0-1)*ln(α) -b0*α )/∂α
    = ((a0-1)/α ) - b0  | * α
    = (a0-1) - b0 * α
    """

    # last term is jacobian term
    d_prec_prior_dα =  (a0 - 1) .- b0 .* α .+ 1 # dim: particles x 1 (alpha)

    d_joint_dw = d_DataLL_dW + d_prec_dW # dim: particles x k
    d_joint_dα = d_prec_dα + d_prec_prior_dα # dim: particles x 1(alpha)

    d_joint_dα = d_prec_dα + d_prec_prior_dα

    return hcat(d_joint_dw,d_joint_dα)

end


function predict(particles, X_test)

    n_samples = size(X_test, 1)
    coeffs = particles[:,1:end-1]
    n_particles = size(coeffs, 1)
    prob = zeros(n_particles, n_samples)

    for particle in 1:n_particles
        squares = broadcast(*,coeffs[particle, :]',X_test)
        summed_squares = dropdims(sum(squares, dims = 2), dims= 2) # Σ_{1:j:K}
        prob[particle, :] = σ.(summed_squares)

    end

    return prob

end


function gen_blg_particles(n_dims, n_particles = 100, a = 1, b=0.01)
    # Initialise n particles

    # two dimensions, convention: 1st = n_particles, 2nd = theta dimensionality
    W = zeros(n_particles, n_dims)
    # Generates precision params (= alpha values) based on prior
    a = 1
    b = 0.01
    α = rand(Gamma(a, b), n_particles)
    # Generate coeffs based on precision params
    for i in 1:n_particles
        # Precision α is the reciprocal of the variance σ
        d = Normal(0, sqrt(1/α[i]))
        W[i,:] =  rand(d, n_dims)
    end
    # Concat regression coefficients and precision params
    # Storing log(α) to use smoothness of log when doing transformations
    # init_parts = hcat(W,α)
    # Alternative: Store log(alpha), such that optimization takes space in log(α) space.
    init_particles = hcat(W, broadcast(log, α))

    return init_particles

end
