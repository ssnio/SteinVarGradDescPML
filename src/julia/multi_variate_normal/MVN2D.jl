 # imports
begin
	include("../SVGD.jl")
	using Plots
	using Distributions
	using Statistics
	using LinearAlgebra
	using ForwardDiff
end

loc_dir = "../SteinVarGradDescPML/statics";

begin # # parameters
	n_epochs=120^2
	n_frames = (1:120) .^ 2
	n_parts = 120
end

begin
	# # Data
	# # ground truth mean and covariance matrix
	cov_mat = [0.5 0.5; 0.5 1.0]
	mean_vec = [0. 0.]
	mvn = MvNormal(Vector(mean_vec[1, :]), Matrix(cov_mat))

	xx, yy = [-3.:.01:3.;], [-3.:.01:3.;]
	mvn_grid = pdf_grid(xx, yy, mvn)

end

# # initial particles
begin
	# init_particles = randn(n_parts, 2) ./ 2
	c = 1
	init_particles = Array{Float64}(undef, n_parts, 2)
	for i in -0.5:0.1:0.5
		for j in -0.5:0.1:0.5
			if i + j <= 1.0 && c <= n_parts && abs(i) + abs(j) > 0.
				init_particles[c, :] = [i, j]
				c += 1
			end
		end
	end
	init_particles .*= 5
end

begin # # plot PDF heatmap
	gr(size=(640,640))
	contourf(xx, yy, mvn_grid, xlims=(-3, 3), ylims=(-3, 3), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(init_particles[:, 2], init_particles[:, 1], color="red", markerstrokecolor=:red, ms=4)
	savefig(joinpath(loc_dir, "MVN2D_init.png"))
end

begin
	ana_dlogmvn_eval(x) = ana_dlogmvn(mean_vec, cov_mat, x)
	trans_parts, evol_rec = update_rec(init_particles, ana_dlogmvn_eval; dt=0.01)
end

begin # # plot PDF heatmap
	gr(size=(640,640))
	contourf(xx, yy, mvn_grid, xlims=(-3, 3), ylims=(-3, 3), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(trans_parts[:, 2], trans_parts[:, 1], color="red", markerstrokecolor=:red, ms=4)
	savefig(joinpath(loc_dir, "MVN2D_evolved.png"))
end

begin # # animation
	p = contourf(xx, yy, mvn_grid, xlims=(-3, 3), ylims=(-3, 3), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(evol_rec[1, :, 2], evol_rec[1, :, 1], color="red", markerstrokecolor=:red, ms=4)
	anim = @animate for i in n_frames
		p[2] = evol_rec[i, :, 2], evol_rec[i, :, 1]
	end
	gif(anim, joinpath(loc_dir, "MVN2D_anim.gif"), fps=24)
end
