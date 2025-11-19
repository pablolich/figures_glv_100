#!/usr/bin/env julia
#
# five_dynamics_row.jl
#
using DifferentialEquations
using LinearAlgebra
using Plots
using ColorSchemes
using Plots.PlotMeasures

gr()

const PAL = ColorSchemes.seaborn_bright6.colors

# --- Global style: bigger text + thicker lines ---
default(
    tickfontsize   = 20,  # tick labels
    titlefontsize  = 25,  # panel titles
    linewidth      = 5,   # thicker lines
    xguidefontsize = 25,
    yguidefontsize = 25
)

# common save step for nice resolution
const Δt = 0.05

# --------------------------
# 1) Logistic growth (1 spp)
# --------------------------
function logistic!(du, u, p, t)
    r, K = p
    du[1] = r * u[1] * (1 - u[1] / K)
end

r1, K1 = 1.0, 0.75
u0_log = [0.1]
tspan_log = (0.0, 20.0)

prob_log = ODEProblem(logistic!, u0_log, tspan_log, (r1, K1))
sol_log  = solve(prob_log, Tsit5(); saveat = Δt)

t_log = sol_log.t
x_log = [u[1] for u in sol_log.u]

p1 = plot(
    t_log, x_log;
    xlabel = "",
    ylabel = "Abundance",
    title  = "(i) Equilibrium",
    legend = false,
    ylim = (0, 1.0),
    color  = PAL[1],
    framestyle = :box,
    grid = false,
    yticks = nothing,
    xticks = nothing,
    top_margin = 7mm,
    bottom_margin = 15mm,
    left_margin = 15mm
)

# ------------------------------------
# 2) 2-species neutrally oscillating LV
# ------------------------------------
function lv2_neutral!(du, u, p, t)
    α, β, γ, δ = p
    x, y = u
    du[1] = α * x - β * x * y
    du[2] = δ * x * y - γ * y
end

params2 = (1.0, 2.0, 1.0, 2.0)  # α, β, γ, δ
α, β, γ, δ = params2
x_eq = γ / δ   # equilibrium x
y_eq = α / β   # equilibrium y

# Two amplitudes → 2 ICs (you currently use two)
δ_small = 0.1
δ_large = 0.2

u0_small_above = [x_eq - δ_small, y_eq + δ_small]
u0_small_below = [x_eq - δ_large, y_eq + δ_large]

tspan2 = (0.0, 10.0)

prob_sa = ODEProblem(lv2_neutral!, u0_small_above, tspan2, params2)
prob_sb = ODEProblem(lv2_neutral!, u0_small_below, tspan2, params2)

sol_sa = solve(prob_sa, Tsit5(); saveat = Δt)
sol_sb = solve(prob_sb, Tsit5(); saveat = Δt)

t2   = sol_sa.t
x_sa = [u[1] for u in sol_sa.u]
y_sa = [u[2] for u in sol_sa.u]
x_sb = [u[1] for u in sol_sb.u]
y_sb = [u[2] for u in sol_sb.u]

# fixed colors per species
colors_neutral = [:blue, :red]  # y, x

p2 = plot(
    t2, y_sa;
    xlabel = "",
    title  = "(ii) Neutral cycles",
    color  = PAL[1],
    linestyle = :solid,
    label  = "",
    legend = false,
    ylim = (0, 1.0),
    yticks = nothing,
    xticks = nothing,
    framestyle = :box,
    grid = false
)
plot!(p2, t2, y_sb;
    color = PAL[1],
    linestyle = :dot,
    label = "",
)
plot!(p2, t2, x_sa;
    color = PAL[2],
    linestyle = :solid,
    label = "",
)
plot!(p2, t2, x_sb;
    color = PAL[2],
    linestyle = :dot,
    label = "",
)

# ----------------------------------------
# 3) May–Leonard model: limit cycle regime
# ----------------------------------------
function may_leonard!(du, u, p, t)
    α, β = p
    x, y, z = u
    du[1] = x * (1 - x - α * y - β * z)
    du[2] = y * (1 - β * x - y - α * z)
    du[3] = z * (1 - α * x - β * y - z)
end

α_lim, β_lim = 0.8, 1.2           # α + β = 2
u0_3    = [1.0, 0.8, 0.2]         # IC 1
u0_3_2  = [0.01, 0.04, 0.05]      # IC 2
tspan3  = (0.0, 100.0)

prob3  = ODEProblem(may_leonard!, u0_3,  tspan3, (α_lim, β_lim))
prob3b = ODEProblem(may_leonard!, u0_3_2, tspan3, (α_lim, β_lim))

sol3  = solve(prob3,  Tsit5(); saveat = Δt)
sol3b = solve(prob3b, Tsit5(); saveat = Δt)

t3    = sol3.t
x1_3  = [u[1] for u in sol3.u]
x2_3  = [u[2] for u in sol3.u]
x3_3  = [u[3] for u in sol3.u]

x1_3b = [u[1] for u in sol3b.u]
x2_3b = [u[2] for u in sol3b.u]
x3_3b = [u[3] for u in sol3b.u]

p3 = plot(
    t3, x1_3;
    xlabel = "Time",
    title  = "(iii) Limit cycle",
    color  = PAL[1],
    linestyle = :solid,
    label  = "",
    legend = false,
    ylim = (0, 1.0),
    yticks = nothing,
    xticks = nothing,
    framestyle = :box,
    grid = false
)
plot!(p3, t3, x2_3;
    color = PAL[2],
    linestyle = :solid,
    label = "",
)
plot!(p3, t3, x3_3;
    color = PAL[3],
    linestyle = :solid,
    label = "",
)
plot!(p3, t3, x1_3b;
    color = PAL[1],
    linestyle = :dot,
    label = "",
)
plot!(p3, t3, x2_3b;
    color = PAL[2],
    linestyle = :dot,
    label = "",
)
plot!(p3, t3, x3_3b;
    color = PAL[3],
    linestyle = :dot,
    label = "",
)

# ---------------------------------------------------
# 4) May–Leonard model: heteroclinic cycle regime
# ---------------------------------------------------
α_het, β_het = 0.8, 1.3
u0_4   = [0.6, 0.6, 0.1]
tspan4 = (0.0, 500.0)

prob4 = ODEProblem(may_leonard!, u0_4, tspan4, (α_het, β_het))
sol4  = solve(prob4, Tsit5(); saveat = Δt)

t4   = sol4.t
x1_4 = [u[1] for u in sol4.u]
x2_4 = [u[2] for u in sol4.u]
x3_4 = [u[3] for u in sol4.u]

p4 = plot(
    t4, x1_4;
    xlabel = "",
    title  = "(iv) Heteroclinic cycle",
    color  = PAL[1],
    linestyle = :solid,
    label  = "",
    legend = false,
    ylim = (0, 1.0),
    yticks = nothing,
    xticks = nothing,
    framestyle = :box,
    grid = false
)
plot!(p4, t4, x2_4;
    color = PAL[2],
    linestyle = :solid,
    label = "",
)
plot!(p4, t4, x3_4;
    color = PAL[3],
    linestyle = :solid,
    label = "",
)

# ---------------------------------------------------------
# 5) Chaotic Lotka–Volterra (Vano et al. 2006)
# ---------------------------------------------------------
function lv_competitive!(du, u, p, t)
    A, r = p
    for i in 1:length(u)
        du[i] = r[i] * u[i] * (1.0 - dot(A[i, :], u))
    end
end

A_chaos = [
    1.0   1.09  1.52  0.0;
    0.0   1.0   0.44  1.36;
    2.33  0.0   1.0   0.47;
    1.21  0.51  0.35  1.0
]

r_chaos = [1.0, 0.72, 1.53, 1.27]

u0_5a  = [0.1, 0.5, 0.05, 0.4]   # IC₁
u0_5b  = [0.3, 0.2, 0.6, 0.1]    # IC₂
tspan5 = (0.0, 250.0)

prob5a = ODEProblem(lv_competitive!, u0_5a, tspan5, (A_chaos, r_chaos))
prob5b = ODEProblem(lv_competitive!, u0_5b, tspan5, (A_chaos, r_chaos))

sol5a = solve(prob5a, Tsit5(); saveat = Δt)
sol5b = solve(prob5b, Tsit5(); saveat = Δt)

t5    = sol5a.t
x1_5a = [u[1] for u in sol5a.u]
x2_5a = [u[2] for u in sol5a.u]
x3_5a = [u[3] for u in sol5a.u]
x4_5a = [u[4] for u in sol5a.u]

x1_5b = [u[1] for u in sol5b.u]
x2_5b = [u[2] for u in sol5b.u]
x3_5b = [u[3] for u in sol5b.u]
x4_5b = [u[4] for u in sol5b.u]

p5 = plot(
    t5, x1_5a;
    xlabel = "",
    title  = "(v) Chaos",
    color  = PAL[1],
    linestyle = :solid,
    label  = "",
    legend = false,
    ylim = (0, 1.0),
    yticks = nothing,
    xticks = nothing,
    framestyle = :box,
    grid = false,
    rightmargin = 3mm
)
plot!(p5, t5, x2_5a; color = PAL[2], linestyle = :solid, label = "")
plot!(p5, t5, x3_5a; color = PAL[3], linestyle = :solid, label = "")
plot!(p5, t5, x4_5a; color = PAL[4], linestyle = :solid, label = "")

plot!(p5, t5, x1_5b; color = PAL[1], linestyle = :dot, label = "")
plot!(p5, t5, x2_5b; color = PAL[2], linestyle = :dot, label = "")
plot!(p5, t5, x3_5b; color = PAL[3], linestyle = :dot, label = "")
plot!(p5, t5, x4_5b; color = PAL[4], linestyle = :dot, label = "")

# --------------------------
# Combine all 5 in one row
# --------------------------
plt = plot(
    p1, p2, p3, p4, p5;
    layout = (1, 5),
    size   = (2300, 430),
)

savefig(plt, "pdf_files/five_dynamics_row.pdf")
