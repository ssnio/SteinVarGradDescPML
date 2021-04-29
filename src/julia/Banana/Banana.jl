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
    Banana Distribution
```math
p(x_1,x_2) = c \exp \left( -\frac{x_1^2 x_2^2 + x_1^2 + x_2^2 - 8x_1 - 8x_2}{2} \right)
```
"""
Banana(x1::Float64, x2::Float64) = exp(-(x1^2 * x2^2 + x1^2 + x2^2 - 8x1 - 8x2) / 2)
Banana(X::Vector) = exp(-(X[1]^2 * X[2]^2 + X[1]^2 + X[2]^2 - 8X[1] - 8X[2]) / 2)

# # Analytical Gradient
dlog_banana(x1::Float64, x2::Float64) = [-(x1 * x2^2 + x1 - 4)    -(x2 * x1^2 + x2 - 4)]
dlog_banana(X::Array) = [-(X[:, 1].*X[:, 2].^2 .+ X[:, 1] .- 4)    -(X[:, 2].*X[:, 1].^2 + X[:, 2] .- 4)]

begin # # parameters
	n_epochs = 240^2  # number of epochs
	n_parst = 128  # number of particles
	frames_seq = (1:240) .^ 2
	gr(size=(640,640))
end

# # dense initial particles
init_particles = randn(n_parst, 2) .+ [4. 4.]

begin # # 2D PDF grid
	xx, yy = -1.0:0.01:7.0, -1.0:0.01:7.0
	xxx, yyy = repeat(xx, 1, size(yy, 1)), repeat(yy, 1, size(xx, 1))'
	BananaGrid = Banana.(xxx, yyy)
end


begin # # plot PDF heatmap
	contourf(xx, yy, BananaGrid, aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=cgrad([:white, :green, :yellow], [0.3, 0.5, 0.9], scale = :log))
	scatter!(init_particles[:, 2], init_particles[:, 1], color="red", markerstrokecolor=:red, ms=4)
	savefig(joinpath(loc_dir, "Banana_init.png"))
end


begin  # # training
    trans_parts, evol_rec = update_rec(init_particles, dlog_banana; n_epochs=n_epochs)
end

begin # # plot PDF heatmap
	contourf(xx, yy, BananaGrid, aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=cgrad([:white, :green, :yellow], [0.3, 0.5, 0.9], scale = :log))
	scatter!(trans_parts[:, 2], trans_parts[:, 1], color="red", markerstrokecolor=:red, ms=4)
	savefig(joinpath(loc_dir, "Banana_evolved.png"))
end

begin # # Animation
	gr(size=(640,640))
	p = contourf(xx, yy, BananaGrid, aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=cgrad([:white, :green, :yellow], [0.3, 0.5, 0.9], scale = :log))
	scatter!(evol_rec[1, :, 2], evol_rec[1, :, 1], color="red", markerstrokecolor=:red, ms=4)
	anim = @animate for i in frames_seq
	    p[2] = evol_rec[i, :, 2], evol_rec[i, :, 1]
	end
	gif(anim, joinpath(loc_dir, "Banana_anim.gif"), fps=24)
end
