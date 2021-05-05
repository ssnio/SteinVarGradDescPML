# Project in ML and AI (Prof. Opper) - SVGD
## Term Project on Stein Variational Gradient Descent - Description

Term Project conducted in winter semester 2020/21 in the course "Projects in Machine Learning and Artificial Intelligence" at Technical University Berlin, Institute of Software Engineering and Theoretical Computer Science, Research Group Methods in Artifical Intelligence (lead by Prof. Dr. Manfred Opper).

The method we analysed was Stein Variational Gradient Descent, a non-parametric variational inference method, that iteratively applies smooth transformations on a set of initial particles to construct a transport map wich moves the initial particles such that they closely approximate (in terms of KL divergence) an otherwise intractable target distribution.

We reimplemented the codebase and rerun the experiments the authors have documented in their original repository and further performed additional experiments that showcase characteristics and limitations of the SVGD.

---
## Demo

### Pseudo-Julia implementation

```julia
using Distributions
using ForwardDiff
using KernelFunctions

function SVGD(x, scorefun, kernel, grad_kernel, n_epochs, ϵ)
    n_particles = size(x, 1) # number of particles
    for i in 1:n_epochs
        scorefun_X = scorefun(x) # ∇ log p(x)
        kxy = kernel(x) # k(⋅, x)
		dxkxy = grad_kernel(x) # ∇k(⋅, x)
        ϕ = ((kxy * dlogpdf_val) .+ dxkxy) ./ n_particles # steepest descent
        x += dt .* ϕ # updating the particles
    end
    return x
end
```

### SVGD for Neal's Funnel

<p align=center>
<img src="https://raw.githubusercontent.com/ssnio/SteinVarGradDescPML/main/statics/NealsFunnel/SVGD_NealsFunnel.gif" alt="More Particles"  width=480/>
</p>

---
## Installation

We used different languages (Python 3.6; Julia 1.5) and following libraries throughout our experiments. Please install them to rerun our experiments:  

Python:
- PyTorch
- Numpy
- Scikit-Learn
- Matplotlib
- Scipy

Julia:
- Statistics
- Distances
- Random
- ForwardDiff
- LinearAlgebra
- Distributions
- Plots
- KernelFunctions
- EvalMetrics
- MLDataUtils

---
## Code structure

* presentation_slides *(mid-term and final presentation pdfs)*
* src *(code of all experiments)*
  * data *(experiment datasets from original repo)*
  * python
    * bayesian_neural_network *(pytorch implementation of SVGD, applied on boston housing dataset)*
    * multi_variate_normal *(Jupyter Notebook that applies numpy implementation on 2D gaussian example)*
    * SVGD.py *(numpy implementation of SVGD)*
  * julia
    * bayesian_regression *(bayesian regression experiments, applying Julia implementation on covertype dataset)*
    * gaussian_mixture_anealing *(gaussian mixture models experiemnts, applyuing Julia implementation on mixtures of 2D gaussians, partially using annealing SVGD)*
    * multi_variate_normal *(Jupyter Notebook that applies Julia implementation on 2D gaussian example)*
    * SVGD.jl *(julia implementation of SVGD)*
* statics *(training artifacts and generated graphics)*

---
### Team Members

* Saeed Salehi, [@ssnio](https://github.com/ssnio)
* Boshu Zhang, [@Bookiebookie](https://github.com/Bookiebookie)
* Clemens Dieffendahl, [@dissendahl](https://github.com/dissendahl)
---
## References

### Main references
* Qiang Liu and Dilin Wang [*Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm (Paper)*](https://arxiv.org/pdf/1608.04471.pdf), NIPS, 2016
* dilinwang820/Stein-Variational-Gradient-Descent [*Stein Variational Gradient Descent (SVGD) (Code Repository)*](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent), github.com, 2016

### Further Learning Resources
* Bhairav Mehta [*Depth First Learning - Stein Variational Gradient Descent Class*](https://www.depthfirstlearning.com/2020/SVGD), depthfirstlearning.com, 2020
* Blei et al. [*Variational Inference: Foundations and Modern Methods*](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf), NIPS, 2016
* Francesco D'Angelo and Vincent Fortuin [*Annealed Stein Variational Gradient Descent*](https://openreview.net/pdf?id=pw2v8HFJIYg), 3rd Symposium on Advances in Approximate Bayesian Inference, 2020
* Gianluca Detommaso et al. [*A Stein variational Newton method*](https://arxiv.org/pdf/1806.03085.pdf), arxiv.org, 2018
* José Miguel Hernández-Lobato et al. [*Probabalistic Backpropagation for Scalable Learning of Bayesian Neural Networks*](https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf), ICML, 2015
* Joseph Rooca [*Bayesian inference problem, MCMC and variational inference*](https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29), Towards Data Science Blog, 2019
* Qiang Liu et al. [*A Kernelized Stein Discrepancy for Goodness-of-Fit Tests](https://arxiv.org/pdf/1602.03253.pdf), arxiv.org, 2016
* Qiang Liu [*Stein Variational Gradient Descent as Gradient Flow*](https://arxiv.org/pdf/1704.07520.pdf), arxiv.org, 2017
* Samuel J. Gershman et al. [*Nonparametric Variational Inference*](https://icml.cc/2012/papers/360.pdf), ICML, 2012
* Yang Liu et al. [*Stein Variational Policy Gradient*](https://arxiv.org/pdf/1704.02399.pdf), arxiv.org, 2017
* Radford M. Neal, [*Slice Sampling*](https://www.jstor.org/stable/3448413), Annals of Statistics 31 (3): 705-67, 2003
