"""
    Analytical gradient of log joint for bayesian logistic regression.

    The observed data D = {X, y} consist of N binary class labels y_t in {-1,1},
    and d covariates for each datapoint x_t.

    Hidden variables θ = {α, w} where w consists of w_k regression coefficients
    and a precision parameter α in R+. We assume:

    p(y_t | x_t , w) = y_t * 1 / (1+e^(w^T * x_t) + (1- y_t) * 1 / (1+e^(-w^T * x_t)
    p(w_k | α) ~ N(w_k; 0, α^-1)
    p(α; a,b) ~ Gamma(α; a,b)

"""
##

function ana_dlogblg(particles, a0=1, b0=0.01)
    # number of particles
    n = size(particles,1)
    # number of regression coefficients
    k = size(particles,2) - 1
    # regression coefficients
    coeffs = particles[:,size(particles, 2)-1]
    # α stored in last dimension of theta
    alphas = particles[:,size(particles,2)]


    """ likelihood:

      ∂(Σ_{1:i:N} y_i * 1 / (1+e^(W^T * k_i) + (1- y_i) * 1 / (1+e^(-W^T * x_i))/∂W

    = ToDo

    """


    d_likelihood_dW = #ToDo dim: samples x particles x length(W) -> needs summation
    d_likelihood_dW_summed = # ToDo dim: samples x particles x length(W)

    """ prior:

      ∂(Σ_{1:j:K} ln p(w_k;0,α^-1) )/∂θ | (reformulating Normal, such that α is precision)

    =  ∂(Σ_{1:j:K} ln(sqrt(α)) - ln(sqrt(2π)) - 1/2*α*(w_k)^2 )/∂θ

    => =  ∂(Σ_{1:j:K} ln(sqrt(α)) - ln(sqrt(2π)) - 1/2*α*(w_k)^2 )/∂w_k
       =  -α*w_k

    => =  ∂(Σ_{1:j:K} ln(sqrt(α)) - ln(sqrt(2π)) - 1/2*α*(w_k)^2 )/∂α
       =  Σ_{1:j:K} 1/2α - 1/2(w_k)^2
       =  Σ_{1:j:K} 1/2(α - (w_k)^2)
       = k*1/2α - Σ_{1:j:K} (w_k)^2

    """
    d_prior_dW = #ToDo dim: particles x length(W)
    d_prior_dα = #ToDo dim: particles x 1


    """ hyperprior:

      ∂(lnp(α;a0,b0))/∂α

    = ∂( ln(b0^a0) - ln(Gamma-fct(a0)) + (a0-1)*ln(α) - b*α)/∂α

    = (a0-1)/α - ß0

    """
    d_alpha_dα =  (a0 - 1) ./ alphas .-b0 # dim: particles x 1

    d_joint_dw = d_likelihood_dW_summed .+ d_prior_dW
    d_joint_dα = d_prior_dα .+ d_alpha_dα

    # Then concat d_joint_dw + d_joint_dα

    d_joint_dθ = # ToDo
    return d_joint_dθ

end

##
using Distributions: Gamma, Normal
a0 = 1
b0 = 0.01

##
# Initialise particles, e.g. 200 two-dimensional particles.
n_dims = 20
n_particles = 100

# two dimensions, convention: 1st = n_particles, 2nd = theta dimensionality
coeffs = zeros(n_particles, n_dims-1)
# Generates 50 alpha values from the defined Gamma
alphas = rand(Gamma(1,0.1), n_particles)

## ...
# initialization
M = 100  # number of particles


theta = ran(0, 1/(alpha[i]))


## Concat
init_particles = hcat(coeffs,alphas)
