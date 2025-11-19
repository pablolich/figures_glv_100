#!/usr/bin/env julia
# Local stability and saturated equilibrium demos for a 3-species GLV system.

using DifferentialEquations
using LinearAlgebra
using Plots
using ColorSchemes
using Plots.PlotMeasures

gr()

const PAL = ColorSchemes.seaborn_bright6.colors
default(
    tickfontsize   = 4,
    titlefontsize  = 5,
    linewidth      = 1,
    xguidefontsize = 5,
    yguidefontsize = 5,
)

const Δt = 0.05

const A = [
    -5.0 -1.0 -2.0;
    -4.0 -1.0 -1.0;
    -4.0 -3.0 -2.0
]

function glv!(du, u, p, t)
    A, r = p
    du .= u .* (r .+ A * u)
    return nothing
end

function simulate_glv(r, u0; A_mat = A, tspan = (0.0, 60.0), callback = nothing)
    prob = ODEProblem(glv!, copy(u0), tspan, (A_mat, r))
    sol = solve(prob, Tsit5(); saveat = Δt, callback = callback)
    t = sol.t
    X = reduce(hcat, sol.u)'
    return t, X
end

function preset_callback(times::Vector{Float64}, affect!)
    return PresetTimeCallback(times, affect!; save_positions = (false, false))
end

function plot_dynamics(t, X; title, 
                       xlabel = "", ylabel = "", 
                       show_species3_from = nothing, 
                       ylim = nothing)
    plt = plot(
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        legend = false,
        framestyle = :box,
        grid = false,
        xticks = false,
        yticks = false,
        ylim = ylim
    )
    nsp = size(X, 2)
    for j in 1:nsp
        series = copy(X[:, j])
        if j == 3 && show_species3_from !== nothing
            idx = findfirst(t .>= show_species3_from)
            if idx !== nothing && idx > 1
                series[1:idx-1] .= NaN
            end
        end
        plot!(plt, t, series; color = PAL[j], label = "")
    end
    return plt
end

# helper to place perturbations at one quarter of total integration time
quarter_time(tspan) = first(tspan) + (last(tspan) - first(tspan)) / 4

# --- Scenario 1: stable equilibrium, perturb and recover ---
r_stable = [8.0, 6.0, 9.0]
x_star = [1.0, 1.0, 1.0]
span1 = (0.0, 60.0)
perturb_time1 = quarter_time(span1)
δ1 = [0.3, -0.25, 0.2]
affect1!(integrator) = integrator.u .= max.(integrator.u .+ δ1, 1e-6)
cb1 = preset_callback([perturb_time1], affect1!)
t1, X1 = simulate_glv(r_stable, x_star; callback = cb1, tspan = span1)
p1 = plot_dynamics(t1, X1;
    title = "Stable equilibrium",
    xlabel = "",
    ylabel = "abundance",
    ylim = (0, 2)
)
vline!(p1, [perturb_time1]; color = :black, linestyle = :dot, linewidth = .7)

# --- Scenario 2: unstable equilibrium after perturbation ---
r_unstable = [15.0, 10.0, 18.0]
x_star2 = [1.0, 2.0, 4.0]
scaling = Diagonal(x_star2)
A_unstable = A * scaling
u0_unstable = ones(length(x_star2))
δ2 = [0.2, -0.3, 0.4]
affect2!(integrator) = integrator.u .= max.(integrator.u .+ δ2, 1e-6)
span2 = (0.0, 30.0)
perturb_time2 = quarter_time(span2)
cb2 = preset_callback([perturb_time2], affect2!)
t2, X2 = simulate_glv(r_unstable, u0_unstable; A_mat = A_unstable, callback = cb2, tspan = span2)
p2 = plot_dynamics(t2, X2;
    title = "Unstable equilibrium",
    xlabel = "time",
    ylabel = "",
    ylim = (0, 2)
)
vline!(p2, [perturb_time2]; color = :black, linestyle = :dot, linewidth = 0.7)

# --- Scenario 3: saturated equilibrium, invasion attempt fails ---
r_sat = [6.0, 5.0, 5.0]
x_sat = [1.0, 1.0, 0.0]
span3 = (0.0, 30.0)
invasion_time = quarter_time(span3)
affect3!(integrator) = (integrator.u[3] = 0.5; nothing)
cb3 = preset_callback([invasion_time], affect3!)
t3, X3 = simulate_glv(r_sat, x_sat; callback = cb3, tspan = span3)
p3 = plot_dynamics(t3, X3;
    title = "Saturated equilibrium",
    xlabel = "",
    ylabel = "",
    ylim = (0, 2),
    show_species3_from = invasion_time
)
vline!(p3, [invasion_time]; color = :black, linestyle = :dot, linewidth =.7)

# arrange panels
plt = plot(p1, p2, p3; layout = (1, 3), size = (362, 110))
mkpath("pdf_files")
savefig(plt, "pdf_files/stability_and_saturation_dynamics.pdf")
println("Saved plot at pdf_files/stability_and_saturation_dynamics.pdf")
