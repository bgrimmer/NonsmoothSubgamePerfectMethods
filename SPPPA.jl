# ==============================================================================
# spppa.jl
# ==============================================================================
using LinearAlgebra
using Convex
import SCS
import MathOptInterface as MOI

struct SPPPAMethod <: AbstractMethod
    L::Float64
    k::Int # Max memory size
end

struct SPPPAHistory
    y::Vector{Float64}
    f::Float64
    g::Vector{Float64}
    z_next::Vector{Float64}
    tau::Float64
end

mutable struct SPPPAState
    x::Vector{Float64}
    x0::Vector{Float64}
    n::Int
    
    tau_prime::Float64
    z_prime::Vector{Float64}
    tau_n::Float64
    
    history::Vector{SPPPAHistory}
    
    y_last::Vector{Float64}
    y_star::Vector{Float64}
    frozen::Bool
end

function init_state(method::SPPPAMethod, x0::Vector{Float64})
    return SPPPAState(
        copy(x0), copy(x0), 0,
        0.0, copy(x0), 0.0,
        SPPPAHistory[],
        copy(x0), copy(x0), false
    )
end

get_query(method::SPPPAMethod, state::SPPPAState) = (state.x, method.L)
get_iterate(method::SPPPAMethod, state::SPPPAState) = state.frozen ? state.y_star : state.y_last

function update!(method::SPPPAMethod, state::SPPPAState, res)
    if state.frozen
        return state
    end
    
    L = method.L
    
    # 1. Compute z_{n+1}
    if state.n == 0
        tau_prev = 2.0 / L
        z_next = state.x0 - tau_prev * res.g
    else
        z_next = state.z_prime - (state.tau_n - state.tau_prime) * res.g
        tau_prev = state.tau_n
    end
    
    push!(state.history, SPPPAHistory(copy(res.y), res.f_y, copy(res.g), copy(z_next), tau_prev))
    
    # Enforce limited memory 'k'
    if length(state.history) > method.k
        popfirst!(state.history)
    end
    
    W = length(state.history)
    f_vec = [h.f for h in state.history]
    
    # Define m locally within the bounded active memory window
    m_idx = argmin(f_vec)
    f_m = f_vec[m_idx]
    y_m = state.history[m_idx].y
    
    Z = hcat([h.z_next - state.x0 for h in state.history]...)
    G = hcat([h.g for h in state.history]...)
    tau_vec = [h.tau for h in state.history]
    
    a = zeros(W)
    b = zeros(W)
    for i in 1:W
        h = state.history[i]
        q_i = h.f - dot(h.g, h.y - state.x0)
        a[i] = 0.5 * norm(h.z_next - state.x0)^2 + h.tau * (h.f - f_m)
        
        # Matches exact analytical derivations in paper (no inexactness shift)
        b[i] = q_i - f_m 
    end
    
    # --- Analytical Trivial Solution Baseline (OPPA Step) ---
    tau_fallback = tau_vec[end]
    z_fallback = state.x0 + Z[:, end]
    
    tau_prime = tau_fallback
    z_prime = z_fallback
    
    # --- Optimization Subproblem ---
    mu = Variable(W)
    lam = Variable(W)
    
    obj = dot(tau_vec, mu) + sum(lam)
    
    constraints = Constraint[
        mu >= 0,
        lam >= 0,
        sumsquares(Z * mu - G * lam) <= 2.0 * (dot(a, mu) + dot(b, lam))
    ]
    
    prob = maximize(obj, constraints)
    
    scs_opt = MOI.OptimizerWithAttributes(SCS.Optimizer, 
        "eps_abs" => 1e-9, 
        "eps_rel" => 1e-9, 
        "max_iters" => 50000)
    
    solve!(prob, scs_opt; silent_solver=true)
    
    if prob.status == MOI.DUAL_INFEASIBLE || prob.status == MOI.INFEASIBLE_OR_UNBOUNDED || prob.status == MOI.OBJECTIVE_LIMIT
        # Numerically safe fallback:
        # In exact arithmetic, unboundedness can certify that a remembered point is
        # an exact minimizer. In floating point, SCS can report these statuses near
        # tiny gaps. Do not freeze; keep the OPPA fallback initialized above.
        nothing
        
    elseif prob.status == MOI.OPTIMAL || prob.status == MOI.ALMOST_OPTIMAL
        mu_ev = evaluate(mu)
        lam_ev = evaluate(lam)
        
        # Safely extract values
        mu_val = max.(0.0, mu_ev === nothing ? zeros(W) : (mu_ev isa Number ? [mu_ev] : vec(mu_ev)))
        lam_val = max.(0.0, lam_ev === nothing ? zeros(W) : (lam_ev isa Number ? [lam_ev] : vec(lam_ev)))
        
        # --- Mathematical Feasibility Rounding ---
        lhs = 0.5 * norm(Z * mu_val - G * lam_val)^2
        rhs = dot(a, mu_val) + dot(b, lam_val)
        
        if lhs > rhs
            if rhs > 1e-14
                alpha = rhs / lhs
                mu_val .*= alpha
                lam_val .*= alpha
            else
                # If rhs <= 0, scale forces variables to 0 (which triggers OPPA fallback)
                mu_val .= 0.0
                lam_val .= 0.0
            end
        end
        
        tau_opt = dot(tau_vec, mu_val) + sum(lam_val)
        z_candidate = state.x0 + Z * mu_val - G * lam_val
        cert_lhs = 0.5 * norm(Z * mu_val - G * lam_val)^2
        cert_rhs = dot(a, mu_val) + dot(b, lam_val)
        
        # Only adopt the reoptimized step when the certificate is numerically
        # reliable and it provides a meaningful improvement over the OPPA fallback.
        accept_step =
            isfinite(tau_opt) &&
            all(isfinite, z_candidate) &&
            cert_rhs >= -1e-10 &&
            cert_lhs <= cert_rhs + 1e-8 * max(1.0, abs(cert_rhs)) &&
            tau_opt > tau_fallback * (1.0 + 1e-4) + 1e-8
        
        if accept_step
            tau_prime = tau_opt
            z_prime = z_candidate
        end
    end
    
    # Extrapolate next steps based on the refined state
    tau_next = tau_prime + (1.0 + sqrt(1.0 + 2.0 * L * tau_prime)) / L
    x_next = (tau_prime / tau_next) * y_m + ((tau_next - tau_prime) / tau_next) * z_prime
    
    state.x = x_next
    state.z_prime = z_prime
    state.tau_prime = tau_prime
    state.tau_n = tau_next
    state.y_last = copy(res.y)
    state.n += 1
    
    return state
end