# imports
begin
	include("../SVGD.jl")
    using Plots
    using Distributions
    using Statistics
    using LinearAlgebra
    using ForwardDiff
end

loc_dir = "../SteinVarGradDescPML/statics"

"""
    Mean zero gaussian
"""
NormalZ(x::Float64, σ::Float64) = exp(-((x/σ)^2)/2) / (σ*sqrt(2pi))
LogNormalZ(x::Float64, σ::Float64) = - log(σ*sqrt(2pi)) - (((x/σ)^2)/2)


"""
    Neals Funnel

https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html

```math
p(y,x) = \mathsf{normal}(y|0,3) * \prod_{n=1}^9 \mathsf{normal}(x_n|0,\exp(y/2))
```
"""
NealFunnel(x::Float64, y::Float64) = NormalZ(y, 3.) * (NormalZ(x, exp(y / 2.)) ^ 9)
NealFunnel(X::Vector) = *(NormalZ(X[1], 3.), NormalZ.(X[2:10], exp(X[1] / 2.))...)
function LogNealFunnel(X::Vector)
    σ = exp(X[1] / 2.)
    logy = - log(3 * sqrt(2pi)) - ((X[1]/3.)^2 / 2)
    logx = - (9 * log(σ * sqrt(2pi))) - (sum((X[2:10] ./ σ).^2) / 2)
    return logy + logx
end
dLogNealFunnel = X -> ForwardDiff.gradient(LogNealFunnel, X)  # numeric gradient
scorefun(X) = hcat(dLogNealFunnel.(eachrow(X))...)'  # this will help us keep the dimensions correct.

begin # # parameters
	n_epochs = 240^2  # number of epochs
	frames_seq = (1:240) .^ 2
	n_parst = 240  # number of particles
	gr(size=(640,640))
end

begin # # dense initial particles
	init_particles = 10 .* (rand(n_parst, 10) .- 0.5)
	init_particles[:, 1] = init_particles[:, 1] .+ 10
	init_particles[:, 2:10] = init_particles[:, 2:10] .* 2.9
end

begin # # 2D PDF grid
	xx = -15.:0.01:15.
	yy = -10.:0.01:20.
	xxx = repeat(xx, 1, size(yy, 1))
	yyy = repeat(yy, 1, size(xx, 1))'
	NealsGrid = NealFunnel.(xxx, yyy)
end

begin # # plot PDF heatmap
	contourf(xx, yy, log.(NealsGrid'), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(init_particles[:, 2], init_particles[:, 1], color="red", markerstrokecolor=:red, ms=4)
	savefig(joinpath(loc_dir, "NealsFunnel_init.png"))
end


begin  # # training
	trans_parts, evol_rec = update_rec(init_particles, scorefun; n_epochs=n_epochs, dt=0.2, opt="none")
end

begin  # # plot PDF heatmap and evolved particles
	contourf(xx, yy, log.(NealsGrid'), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(trans_parts[:, 2], trans_parts[:, 1], color="red", markerstrokecolor=:red, ms=4)
	savefig(joinpath(loc_dir, "NealsFunnel_evolved.png"))
end

begin # # Animation
	gr(size=(640,640))
	p = contourf(xx, yy, log.(NealsGrid'), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(trans_parts[:, 2], trans_parts[:, 1], color="red", markerstrokecolor=:red, ms=4)
	anim = @animate for i in frames_seq
	    p[2] = evol_rec[i, :, 2], evol_rec[i, :, 1]
	end
	gif(anim, joinpath(loc_dir, "NealsFunnel_anim.gif"), fps=24)
end
