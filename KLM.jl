# ==============================================================================
# klm.jl
# ==============================================================================
using LinearAlgebra
using Convex
import SCS
import MathOptInterface as MOI

struct KLMMethod <: AbstractMethod
    M::Float64   # Lipschitz constant bound
    R::Float64   # Distance to optimum bound ||x0 - x⋆|| <= R
    N::Int       # Total iteration budget
    k::Int       # Max memory size (active window)
end

mutable struct KLMState
    x::Vector{Float64}
    x0::Vector{Float64}
    n::Int
    f_half::Float64 # Tracks the global minimum seen over all time
    history_x::Vector{Vector{Float64}}
    history_f::Vector{Float64}
    history_g::Vector{Vector{Float64}}
end

function init_state(method::KLMMethod, x0::Vector{Float64})
    return KLMState(copy(x0), copy(x0), 1, Inf, 
                    Vector{Vector{Float64}}(), Float64[], Vector{Vector{Float64}}())
end

get_query(method::KLMMethod, state::KLMState) = state.x
get_iterate(method::KLMMethod, state::KLMState) = state.x

function update!(method::KLMMethod, state::KLMState, res)
    if method.N == 0
        return state
    end

    # Keep a running tracker for the best f seen, as history will be truncated
    state.f_half = min(state.f_half, res.f_x)

    push!(state.history_x, copy(state.x))
    push!(state.history_f, res.f_x)
    push!(state.history_g, copy(res.g))

    # Enforce limited memory 'k'
    if length(state.history_x) > method.k
        popfirst!(state.history_x)
        popfirst!(state.history_f)
        popfirst!(state.history_g)
    end

    if state.n > method.N
        return state
    end

    n = state.n
    f_half = state.f_half

    # Restrict constraints and degrees of freedom to the active memory window
    Gmat = hcat(state.history_g...)
    α = Variable(size(Gmat, 2))
    y_expr = state.x0 + Gmat * α
    zeta = Variable()
    t = Variable()

    constraints = Constraint[
        zeta >= 0,
        t >= f_half - method.M * zeta,
        norm(vcat(y_expr - state.x0, sqrt(method.N - n + 1) * zeta)) <= method.R
    ]

    # Only enforce cutting planes for the retained window
    for i in 1:length(state.history_f)
        push!(constraints, t >= state.history_f[i] + dot(state.history_g[i], y_expr - state.history_x[i]))
    end

    prob = minimize(t, constraints)

    scs_opt = MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        "eps_abs" => 1e-9,
        "eps_rel" => 1e-9, 
        "max_iters" => 50000)
		
    solve!(prob, scs_opt; silent_solver=true)

    if prob.status == MOI.OPTIMAL || prob.status == MOI.ALMOST_OPTIMAL
        y_val = evaluate(y_expr)
        state.x = y_val === nothing ? copy(state.x0) : vec(y_val)
    else
        error("KLM SOCP failed at iteration $n with status $(prob.status)")
    end

    state.n += 1
    return state
end