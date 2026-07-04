# ==============================================================================
# KLM_active_offset.jl
# ==============================================================================
# Kelley-like method with an active-set bundle.
#
# The bundle contains the newly observed cut plus all cuts that were binding in
# the previous KLM planning subproblem. After each planning solve, the next bundle
# is reset to the cuts satisfying
#
#     f_i + <g_i, y - x_i> ≈ t.
#
# Important numerical variant: the planning point is parameterized as
#
#     y ∈ x_current + span{active gradients},
#
# rather than y ∈ x0 + span{active gradients}. Since x_current is itself already
# in the full first-order span, this avoids losing previously made progress when
# an old gradient leaves the active bundle.

using LinearAlgebra
using Convex
import SCS
import MathOptInterface as MOI

struct KLMActiveMethod <: AbstractMethod
    M::Float64          # Lipschitz constant bound
    R::Float64          # Distance to optimum bound ||x0 - x⋆|| <= R
    N::Int              # Total iteration budget
    active_tol::Float64 # Relative tolerance for deciding binding cuts
end

KLMActiveMethod(M::Float64, R::Float64, N::Int; active_tol::Float64=1e-5) =
    KLMActiveMethod(M, R, N, active_tol)

# Convenience alias if you want the constructor name to match the file name.
const KLM_active = KLMActiveMethod

mutable struct KLMActiveState
    x::Vector{Float64}
    x0::Vector{Float64}
    n::Int
    f_half::Float64
    history_x::Vector{Vector{Float64}}
    history_f::Vector{Float64}
    history_g::Vector{Vector{Float64}}
end

function init_state(method::KLMActiveMethod, x0::Vector{Float64})
    return KLMActiveState(
        copy(x0), copy(x0), 1, Inf,
        Vector{Vector{Float64}}(), Float64[], Vector{Vector{Float64}}()
    )
end

get_query(method::KLMActiveMethod, state::KLMActiveState) = state.x
get_iterate(method::KLMActiveMethod, state::KLMActiveState) = state.x

function klm_active_indices(cut_values::Vector{Float64}, t_val::Float64, tol::Float64)
    active = Int[]
    for i in eachindex(cut_values)
        scale = max(1.0, abs(t_val), abs(cut_values[i]))
        if cut_values[i] >= t_val - tol * scale
            push!(active, i)
        end
    end

    # Guard against a fully empty bundle caused by solver tolerances.
    if isempty(active) && !isempty(cut_values)
        push!(active, argmax(cut_values))
    end

    return active
end

function update!(method::KLMActiveMethod, state::KLMActiveState, res::SubgradientResult)
    if method.N == 0
        return state
    end

    state.f_half = min(state.f_half, res.f_x)

    # Candidate bundle = previously active cuts + newly observed cut.
    push!(state.history_x, copy(state.x))
    push!(state.history_f, res.f_x)
    push!(state.history_g, copy(res.g))

    if state.n > method.N
        return state
    end

    n = state.n
    f_half = state.f_half
    W = length(state.history_f)

    Gmat = hcat(state.history_g...)
    α = Variable(size(Gmat, 2))

    # Use the current iterate as the affine offset. This keeps previous progress
    # even when directions that generated it are no longer in the active bundle.
    # The trust-region constraint is still imposed relative to x0.
    y_expr = state.x + Gmat * α
    zeta = Variable()
    t = Variable()

    constraints = Constraint[
        zeta >= 0,
        t >= f_half - method.M * zeta,
        norm(vcat(y_expr - state.x0, sqrt(method.N - n + 1) * zeta)) <= method.R
    ]

    for i in 1:W
        push!(constraints, t >= state.history_f[i] + dot(state.history_g[i], y_expr - state.history_x[i]))
    end

	println(size(constraints))

    prob = minimize(t, constraints)

    scs_opt = MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        "eps_abs" => 1e-6,
        "eps_rel" => 1e-6,
        "max_iters" => 50000,
    )
    solve!(prob, scs_opt; silent_solver=true)

    if prob.status == MOI.OPTIMAL || prob.status == MOI.ALMOST_OPTIMAL
        y_val = evaluate(y_expr)
        t_val_raw = evaluate(t)

        y_next = y_val === nothing ? copy(state.x) : vec(y_val)
        t_val = t_val_raw === nothing ? maximum([
            state.history_f[i] + dot(state.history_g[i], y_next - state.history_x[i])
            for i in 1:W
        ]) : (t_val_raw isa Number ? Float64(t_val_raw) : Float64(vec(t_val_raw)[1]))

        cut_values = [
            state.history_f[i] + dot(state.history_g[i], y_next - state.history_x[i])
            for i in 1:W
        ]
        active = klm_active_indices(cut_values, t_val, method.active_tol)

        state.x = y_next
        state.history_x = [state.history_x[i] for i in active]
        state.history_f = [state.history_f[i] for i in active]
        state.history_g = [state.history_g[i] for i in active]
    else
        # If the SOCP fails, keep the newly augmented bundle and take no step.
        # This mirrors a safe fallback while preserving the latest information.
        @warn "KLMActive SOCP failed; keeping current iterate and augmented bundle" iteration=n status=prob.status
    end

    state.n += 1
    return state
end
