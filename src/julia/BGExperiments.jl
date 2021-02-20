using MAT
using MLDataUtils: shuffleobs, splitobs
using EvalMetrics
using Plots
using JSON
plotly()
include("SVGD.jl")
include("BayesianRegression.jl")

## UCI/Covertype data set. Source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
data = matread("src/data/covertype.mat")["covtype"]
n_samples = 50000

# shuffleobs and splitobs utils need X to be transposed
X = transpose(data[:,2:end])
# replace label 2 with label -1
Y = data[:,1]
Y[Y.==2] .= -1

# shuffle
Xs, Ys = shuffleobs((X,Y))

##

#take first n_samples
Xs = Xs[:, 1:n_samples]
Ys = Ys[1:n_samples]
# split
(X_train, Y_train), (X_test, Y_test) = splitobs((Xs, Ys), at= 0.7)

# transpose back
X_train = Array(transpose(X_train))
Y_train = Array(Y_train)
X_test = Array(transpose(X_test))
Y_test = Array(Y_test)

# grab dimensions from data
n_dims = size(X_train, 2)

#  generate initial particles
init_parts = gen_blg_particles(n_dims)
##
# Weights are distributed as zero-mean gaussians with governing precision parameter for each particle.
plot(W)
##
# Initial mean predictions for 1000 samples -> Distributed around 0.5
init_estimates = predict(init_parts, X_test)
means = dropdims(mean(init_estimates, dims=1), dims=1)
plot(means[1:1000])

##
dlogblg(particles) = ana_dlogblg(particles, X_train, Y_train)
@time dlogblg(init_parts)
##
@time trans_parts = update(init_parts, dlogblg, n_epochs=2000, dt=0.05, opt="adagrad")
##
# Mean predictions for 1000 samples after training
estimates = predict(trans_parts, X_test)
means = dropdims(mean(estimates, dims=1), dims=1)
plot(means[1:1000])

##
# Get classification_statistics
Y_test[Y_test.==-1] .= 0
eval_report = binary_eval_report(Y_test, means)
open("statics/regression/eval_n50000_e2000.json","w") do f
    JSON.print(f, eval_report, 4)
end
## Binarise predictions
means[means.>0.5] .= 1
means[means.<=0.5] .= 0
# Create and plot confusion matrix
cm = ConfusionMatrix(Y_test, means)
heatmap(reshape([cm.fp, cm.tn, cm.tp, cm.fn], 2, 2))
## Iris DataSet classification

X = Flux.Data.Iris.features()
Y = Flux.Data.Iris.labels()

indices_setosa = findall(x -> x == "Iris-setosa", Y)
indices_versicolor = findall(x -> x == "Iris-versicolor", Y)

X = hcat(X[:,indices_setosa], X[:,indices_versicolor])
Y = vcat(zeros(size(indices_setosa)), ones(size(indices_versicolor)))
Y[Y.==0] .= -1
Xs, Ys = shuffleobs((X,Y))
Xs = Array(Xs)
# split
(X_train, Y_train), (X_test, Y_test) = splitobs((Xs, Ys), at= 0.7)

# transpose back
X_train = Array(transpose(X_train))
Y_train = Array(Y_train)
X_test = Array(transpose(X_test))
Y_test = Array(Y_test)

# grab dimensions from data
n_dims = size(X_train, 2)
n_parts = 20
#  generate initial particles
init_parts = gen_blg_particles(n_dims, n_parts)

##
dlogblg(particles) = ana_dlogblg(particles, X_train, Y_train)
@time trans_parts = update(init_parts, dlogblg, n_epochs=1000, dt=0.05, opt="adagrad")
##
estimates = predict(trans_parts, X_test)
means = dropdims(mean(estimates, dims=1), dims=1)
Y_test[Y_test.==-1] .= 0
# Binarise predictions
means[means.>0.5] .= 1
means[means.<=0.5] .= 0
eval_report = binary_eval_report(Y_test, means)
open("statics/regression/eval_iris.json","w") do f
    JSON.print(f, eval_report, 4)
end
##
# Create and plot confusion matrix
cm = ConfusionMatrix(Y_test, means)
heatmap(reshape([cm.fp, cm.tn, cm.tp, cm.fn], 2, 2))
