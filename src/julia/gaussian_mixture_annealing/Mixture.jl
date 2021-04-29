# imports
begin
	include("../SVGD.jl")
	using Plots
	using Distributions
	using Statistics
	using LinearAlgebra
    using JLD
    using ForwardDiff
end

√ = sqrt

loc_dir = "../SteinVarGradDescPML/statics"

begin # # Mixture of Gaussians
    # # using the better stored values
    π_array = load(joinpath(loc_dir, "IsolOnes1.jld"), "pis")
    mean_mat = load(joinpath(loc_dir, "IsolOnes1.jld"), "means")
    cov_tensor = load(joinpath(loc_dir, "IsolOnes1.jld"), "covs")

    MixMVN = MixtureModel(MvNormal, [
            (Vector(mean_mat[1, :]), Matrix(cov_tensor[1, :, :])),
            (Vector(mean_mat[2, :]), Matrix(cov_tensor[2, :, :])),
            (Vector(mean_mat[3, :]), Matrix(cov_tensor[3, :, :])),
            (Vector(mean_mat[4, :]), Matrix(cov_tensor[4, :, :])),
            (Vector(mean_mat[5, :]), Matrix(cov_tensor[5, :, :])),
            ], π_array)

	Log_PDF(x) = log(pdf(MixMVN, x))  # although logpdf method exists in Distributions, it was unstable.
	dLogPDF = x -> ForwardDiff.gradient(Log_PDF, x)  # numeric gradient
	scorefun(x) = hcat(dLogPDF.(eachrow(x))...)'  # this will help us keep the dimensions correct.
end

begin # # parameters
	n_epochs = 64^3  # number of epochs
	n_parst = 64  # number of particles
	frames_seq = (1:64) .^ 3
	gr(size=(640,640))
end


# # initial particles
# init_particles = randn(n_pars_, 2) .+ [-1.0 -4.5];
begin
	c = 1
	ija = Array{Float64}(undef, 16, 2)
	ijb = Array{Float64}(undef, 16, 2)
	ijc = Array{Float64}(undef, 16, 2)
	ijd = Array{Float64}(undef, 16, 2)
	for i in 0.35:-0.1:0.05
		for j in 0.35:-0.1:0.05
			if i^2 + j^2 <= 0.33 && c <= 16 && i+j > 0.05
				ija[c, :] = [i, j]
				ijb[c, :] = [-i, j]
				ijc[c, :] = [-i, -j]
				ijd[c, :] = [i, -j]
				c += 1
			end
		end
	end
	init_particles = vcat(ija, ijb, ijc, ijd) * [√(1/2) -√(1/2); √(1/2) √(1/2)] * 2 .+ [-3.5 -3.5];
end


begin # # 2D PDF grid
	xx, yy = -7.5:.01:5.5, -7.5:.01:5.5
	mix_grid = pdf_grid(xx, yy, MixMVN)
end

begin # # plot
	contourf(xx, yy, mix_grid, aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(init_particles[:, 2], init_particles[:, 1], color="red", markerstrokecolor=:red, ms=4)
    savefig(joinpath(loc_dir, "MixMVN_init.png"))
end

begin  # # training
    trans_parts, evol_rec = update_rec(init_particles, scorefun; n_epochs=n_epochs, dt=0.02)
end

begin # # plot
	contourf(xx, yy, mix_grid, aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(trans_parts[:, 2], trans_parts[:, 1], color="red", markerstrokecolor=:red, ms=4)
    savefig(joinpath(loc_dir, "MixMVN_evolved.png"))
end

begin # # animation
	p = contourf(xx, yy, mix_grid, aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(evol_rec[1, :, 2], evol_rec[1, :, 1], color="red", markerstrokecolor=:red, ms=4)
	anim = @animate for i in frames_seq
		p[2] = evol_rec[i, :, 2], evol_rec[i, :, 1]
	end
	gif(anim, joinpath(loc_dir, "MixMVN_anim.gif"), fps=24)
end

begin  # # Annealing
	γ_t = annealing_schedule(n_epochs, "cyclical"; annealing=0.3)
	annealed_parts, anneal_rec = update_anneal(init_particles, scorefun, γ_t; n_epochs=n_epochs, dt=0.01)
end

begin # # animation
	p = contourf(xx, yy, mix_grid, xlims=(-7.5, 5.5), ylims=(-7.5, 5.5), aspect_ratio=1, legend=false, border=:none, background_color=:transparent, axis=nothing, c=:PuBu_8)
	scatter!(anneal_rec[1, :, 2], anneal_rec[1, :, 1], color="red", markerstrokecolor=:red, ms=4)
	anim = @animate for i in frames_seq
		p[2] = anneal_rec[i, :, 2], anneal_rec[i, :, 1]
	end
	gif(anim, joinpath(loc_dir, "MixMVN_anim_annealed.gif"), fps=24)
end

begin
	pyplot(size=(3*480,480))
	scalefontsizes(0.5)
	plot(1:1000, annealing_schedule(1000, "cyclical"; annealing=0.8), thickness_scaling = 2, label="cyclical", color="blue", axis=true, legend=:bottomright, xlabel="Epoch", ylabel="γ(t)")#, title="Annealing Schedule")
	plot!(annealing_schedule(1000, "hyperbolic"; annealing=0.8), label="hyperbolic")
	plot!(annealing_schedule(1000, "linear"; annealing=0.8), label="linear")
	savefig(joinpath(loc_dir, "AnnealingSchedule.png"))
end
