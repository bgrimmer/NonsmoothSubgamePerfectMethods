# ==============================================================================
# easy_klm.jl
# ==============================================================================
using LinearAlgebra
using Convex
import SCS
import MathOptInterface as MOI

struct EasyKLMMethod <: AbstractMethod
    M::Float64
    R::Float64
    N::Int
    k::Int # Number of iterations to solve the full SOCP
end

mutable struct EasyKLMState
    x::Vector{Float64}
    x0::Vector{Float64}
    n::Int
    history_x::Vector{Vector{Float64}}
    history_f::Vector{Float64}
    history_g::Vector{Vector{Float64}}
    zeta_star::Float64
end

function init_state(method::EasyKLMMethod, x0::Vector{Float64})
    # Analytical initial fallback for zeta (in case k=0)
    zeta_init = method.R / sqrt(method.N + 1)
    return EasyKLMState(copy(x0), copy(x0), 1, 
                        Vector{Vector{Float64}}(), Float64[], Vector{Vector{Float64}}(), 
                        zeta_init)
end

get_query(method::EasyKLMMethod, state::EasyKLMState) = state.x
get_iterate(method::EasyKLMMethod, state::EasyKLMState) = state.x

function update!(method::EasyKLMMethod, state::EasyKLMState, res::SubgradientResult)
    push!(state.history_x, copy(state.x))
    push!(state.history_f, res.f_x)
    push!(state.history_g, copy(res.g))
    
    if state.n > method.N
        return state
    end
    
    n = state.n
    d = length(state.x)
    
    if n <= method.k
        # --- PHASE 1: Full SOCP step ---
        f_half = minimum(state.history_f)
        
        y = Variable(d)
        zeta = Variable()
        t = Variable()
        
        constraints = Constraint[
            zeta >= 0,
            t >= f_half - method.M * zeta,
            norm(vcat(y - state.x0, sqrt(method.N - n + 1) * zeta)) <= method.R
        ]
        
        for i in 1:n
            push!(constraints, t >= state.history_f[i] + dot(state.history_g[i], y - state.history_x[i]))
        end
        
        prob = minimize(t, constraints)
        scs_opt = MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-6, "eps_rel" => 1e-6)
        solve!(prob, scs_opt; silent_solver=true)
        
        if prob.status == MOI.OPTIMAL || prob.status == MOI.ALMOST_OPTIMAL
            state.x = vec(evaluate(y))
            
            # Cache the optimal zeta for future easy steps
            opt_zeta = evaluate(zeta)
            val_zeta = opt_zeta isa AbstractArray ? opt_zeta[1] : opt_zeta
            state.zeta_star = max(0.0, Float64(val_zeta))
        else
            @warn "EasyKLM solver failed at iteration $n. Taking fallback step."
            state.x = state.x - (state.zeta_star / method.M) * res.g
        end
    else
        # --- PHASE 2: Easy Step ---
        # Takes a carefully calculated subgradient step bypassing the SOCP solver.
        # This acts as the limited-memory continuation of the method.
        state.x = state.x - (state.zeta_star / method.M) * res.g
    end
    
    state.n += 1
    return state
end