#!/usr/bin/env julia
# Run Lotka–Volterra dynamics for two sampled matrices and a known
# chaotic 4-species system. Plots show raw dynamics (α = 0.4), cumulative means,
# and analytical equilibria.

using DifferentialEquations
using LinearAlgebra
using Plots
using ColorSchemes
using Plots.PlotMeasures


gr()

const PAL = ColorSchemes.seaborn_bright6.colors
default(
    tickfontsize   = 5,
    linewidth      = 1,
    xguidefontsize = 5,
    yguidefontsize = 5
)

# ---------------------------------------------------------------------------
# Hard-coded interaction matrix (r = 1)
# Format of original files: [seed  A_row...  r]; we keep only A rows.
# ---------------------------------------------------------------------------
A_12423 = [
    -1.0         0.8824891   -1.686984    -3.2194436;
    -0.48663253 -1.0         -0.96056855   2.8984237;
     2.0790734  -1.8916421   -1.0         -2.5012732;
    -0.45135316 -0.50522804   2.2663164   -1.0
]

r_ones = ones(Float64, 4)

# Chaotic competitive LV (same as types_dynamics_n.jl)
A_chaos = [
    1.0   1.09  1.52  0.0;
    0.0   1.0   0.44  1.36;
    2.33  0.0   1.0   0.47;
    1.21  0.51  0.35  1.0
]
r_chaos = [1.0, 0.72, 1.53, 1.27]

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
function glv!(du, u, p, t)
    A, r = p
    for i in 1:length(u)
        du[i] = u[i] * (r[i] + dot(A[i, :], u))
    end
    return nothing
end

function lv_competitive!(du, u, p, t)
    A, r = p
    for i in 1:length(u)
        du[i] = r[i] * u[i] * (1.0 - dot(A[i, :], u))
    end
    return nothing
end

function equilibrium_glv(A, r)
    return -A \ r
end

function equilibrium_lv_competitive(A)
    n = size(A, 1)
    return A \ ones(n)
end

function cumulative_mean(xs::AbstractVector)
    return cumsum(xs) ./ (1:length(xs))
end

function cumulative_mean_matrix(traj::AbstractMatrix)
    n, m = size(traj)
    cum = zeros(n, m)
    for j in 1:m
        cum[:, j] .= cumulative_mean(view(traj, :, j))
    end
    return cum
end

function simulate_glv(A; r = r_ones, u0 = fill(0.2, size(A, 1)), tspan = (0.0, 250.0), saveat = 0.1)
    prob = ODEProblem(glv!, u0, tspan, (A, r))
    sol = solve(prob, Tsit5(); saveat = saveat)
    t = sol.t
    X = reduce(hcat, sol.u)'  # time x species
    return t, X
end

function simulate_lv_competitive(A, r; u0 = [0.1, 0.5, 0.05, 0.4], tspan = (0.0, 250.0), saveat = 0.1)
    prob = ODEProblem(lv_competitive!, u0, tspan, (A, r))
    sol = solve(prob, Tsit5(); saveat = saveat)
    t = sol.t
    X = reduce(hcat, sol.u)'
    return t, X
end


# ---------------------------------------------------------------------------
# Run and save plots
# ---------------------------------------------------------------------------
t1, X1 = simulate_glv(A_12423; r = r_ones)
eq1 = equilibrium_glv(A_12423, r_ones)
cum1 = cumulative_mean_matrix(X1)

p1 = plot(
    xlabel = "Time",
    ylabel = "Abundance",
    legend = false,
    framestyle = :box,
    grid = false,
    xticks = false,
    yticks = false
)
for j in 1:size(X1, 2)
    c = PAL[(j - 1) % length(PAL) + 1]
    plot!(p1, t1, X1[:, j]; color = c, alpha = 0.25, label = "", linewidth=0.75)
    plot!(p1, t1, cum1[:, j]; color = c, label = "")
    hline!(p1, [eq1[j]]; color = c, label = "", linestyle = :dot)
end

t3, X3 = simulate_lv_competitive(A_chaos, r_chaos)
eq3 = equilibrium_lv_competitive(A_chaos)
cum3 = cumulative_mean_matrix(X3)
p2 = plot(
    xlabel = "Time",
    ylabel = "",
    legend = false,
    framestyle = :box,
    grid = false,
    xticks = false, 
    yticks = false
)
for j in 1:size(X3, 2)
    c = PAL[(j - 1) % length(PAL) + 1]
    plot!(p2, t3, X3[:, j]; color = c, alpha = 0.25, label = "", linewidth=0.75)
    plot!(p2, t3, cum3[:, j]; color = c, label = "")
    hline!(p2, [eq3[j]]; color = c, linestyle = :dot, label = "")
end

# panels are twice the width of the original 5-panel figure (2300x430 → 920x430 for 2 panels)
plt_two = plot(p1, p2; layout = (1, 2), size = (330, 110))
savefig(plt_two, "pdf_files/dynamics_average_to_equilibrium.pdf")
