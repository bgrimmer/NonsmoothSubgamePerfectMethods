# ==============================================================================
# oppa.jl
# ==============================================================================
using LinearAlgebra

struct OPPAMethod <: AbstractMethod
    L::Float64
end

mutable struct OPPAState
    x::Vector{Float64}
    x0::Vector{Float64}
    n::Int
    z::Vector{Float64}
    tau_prev::Float64
    tau_curr::Float64
    y_last::Vector{Float64}
end

function init_state(method::OPPAMethod, x0::Vector{Float64})
    return OPPAState(copy(x0), copy(x0), 0, zeros(length(x0)), 0.0, 0.0, copy(x0))
end

get_query(method::OPPAMethod, state::OPPAState) = (state.x, method.L)
get_iterate(method::OPPAMethod, state::OPPAState) = state.y_last

function update!(method::OPPAMethod, state::OPPAState, res::ProximalResult)
    if state.n == 0
        tau_prev = 2.0 / method.L
        z_next = state.x0 - tau_prev * res.g
    else
        z_next = state.z - (state.tau_curr - state.tau_prev) * res.g
        tau_prev = state.tau_curr
    end

    # OPPA Induction
    tau_next = tau_prev + (1.0 + sqrt(1.0 + 2.0 * method.L * tau_prev)) / method.L
    x_next = (tau_prev / tau_next) * res.y + ((tau_next - tau_prev) / tau_next) * z_next

    state.x = x_next
    state.z = z_next
    state.tau_prev = tau_prev
    state.tau_curr = tau_next
    state.y_last = copy(res.y)
    state.n += 1
    
    return state
end