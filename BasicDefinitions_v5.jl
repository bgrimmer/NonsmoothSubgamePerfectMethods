using LinearAlgebra
using Random

export AbstractProblem, AbstractMethod
export SubgradientResult, ProximalResult
export evaluate_oracle, evaluate_objective
export init_state, get_query, update!, get_iterate
export optimize
export MaxAffineProblem, generate_max_affine
export L1RegressionProblem, generate_l1_regression
export IllConditionedLSProblem, generate_ill_conditioned_ls
export DummySubgradientMethod, DummyProximalMethod

# ==============================================================================
# 1. Abstract Types & Oracle Results
# ==============================================================================
abstract type AbstractProblem end
abstract type AbstractMethod end

"""
Result from a subgradient oracle.
Contains the function value `f_x` and a subgradient `g`.
"""
struct SubgradientResult
    f_x::Float64
    g::Vector{Float64}
end

"""
Result from a proximal oracle.
Contains the proximal point `y`, function value `f_y` = f(y), 
prox-gradient `g` = L(x - y), and the inexactness shift `eps` 
such that g ∈ ∂_eps f(y).
"""
struct ProximalResult
    y::Vector{Float64}
    f_y::Float64
    g::Vector{Float64}
    eps::Float64
end

# ==============================================================================
# 2. Method Interface (To be implemented by your algorithms)
# ==============================================================================

"""
    init_state(method, x0)
Initialize and return the method's state.
"""
init_state(method::AbstractMethod, x0::Vector{Float64}) = error("Not implemented")

"""
    get_query(method, state)
Return the next query. 
- For subgradient methods, return `x::Vector{Float64}`
- For proximal methods, return `(x::Vector{Float64}, L::Float64)`
"""
get_query(method::AbstractMethod, state) = error("Not implemented")

"""
    update!(method, state, result)
Update the method's state using the oracle result (`SubgradientResult` or `ProximalResult`).
"""
update!(method::AbstractMethod, state, result) = error("Not implemented")

"""
    get_iterate(method, state)
Return the current iterate for objective tracking.
"""
get_iterate(method::AbstractMethod, state) = error("Not implemented")


# ==============================================================================
# 3. Problem 1: Finite Maximum of Affine Functions
# ==============================================================================
struct MaxAffineProblem <: AbstractProblem
    A::Matrix{Float64}
    b::Vector{Float64}
end

function generate_max_affine(d::Int, m::Int; seed::Int=123)
    rng = MersenneTwister(seed)
    
    # Base problem: gently skewed affine functions (Local M ≈ 1)
    s = range(1.0, stop=3.0, length=d)
    A = randn(rng, m, d) * Diagonal(s) ./ sqrt(d)
    b = randn(rng, m)

    # THE TRAP: One overwhelmingly steep "Decoy" hyperplane.
    # Forces global Lipschitz M to be massive (M ≈ 200).
    # But b is so negative (-10000) it is NEVER queried or active near the optimum.
    A_wall = randn(rng, 1, d)
    A_wall .*= 200.0 / norm(A_wall)
    b_wall = [-10000.0]

    A = vcat(A, A_wall)
    b = vcat(b, b_wall)

    return MaxAffineProblem(A, b)
end

function evaluate_oracle(prob::MaxAffineProblem, x::Vector{Float64})
    vals = prob.A * x + prob.b
    f_x, idx = findmax(vals)
    g = prob.A[idx, :]
    return SubgradientResult(f_x, g)
end

evaluate_objective(prob::MaxAffineProblem, x::Vector{Float64}) = maximum(prob.A * x + prob.b)


# ==============================================================================
# 4. Problem 2: L1-Regularized Regression (Inexact Proximal)
# ==============================================================================
struct L1RegressionProblem <: AbstractProblem
    A::Matrix{Float64}
    b::Vector{Float64}
    lambda::Float64
    AtA::Matrix{Float64}
    Atb::Vector{Float64}
    opnorm_A::Float64
end

function generate_l1_regression(d::Int, n::Int; lambda::Float64=0.1, sparsity::Float64=0.2, seed::Int=123)
    rng = MersenneTwister(seed)
    U = Matrix(qr(randn(rng, n, d)).Q)
    
    # Spectrum: One huge curvature direction (L_global = 10000), rest are flat (L=1)
    s = ones(d)
    s[1] = 100.0 
    
    # Notice V is implicitly the Identity matrix. 
    # This aligns the features perfectly with the singular values, ensuring 
    # soft-thresholding doesn't mix the steep and flat directions.
    A = U * Diagonal(s)
    
    # Plant the true signal ONLY in the flat features (indices 2 to d)
    x_true = zeros(d)
    for i in 2:d
        if rand(rng) < sparsity
            x_true[i] = randn(rng)
        end
    end
	
	# Noise must also be perfectly restricted to the flat subspace
    noise = 0.01 * randn(rng, n)
    noise .-= dot(noise, U[:, 1]) .* U[:, 1]
    
    b = A * x_true + noise
    
    return L1RegressionProblem(A, b, lambda, A' * A, A' * b, 100.0) # opnorm(A) = 100.0
end


function evaluate_oracle(prob::L1RegressionProblem, x::Vector{Float64}, L::Float64; max_iters::Int=5000, tol::Float64=1e-12)
    L_smooth = prob.opnorm_A^2 + L
    mu = L
    step_size = 1.0 / L_smooth
    q = mu / L_smooth
    beta = (1.0 - sqrt(q)) / (1.0 + sqrt(q)) 
    
    y = copy(x); y_old = copy(x); v = copy(x)
    
    for _ in 1:max_iters
        grad = prob.AtA * v - prob.Atb + L * (v - x)
        v_step = v - step_size * grad
        y .= sign.(v_step) .* max.(abs.(v_step) .- step_size * prob.lambda, 0.0)
        
        if norm(y - y_old) < tol
            break
        end
        v .= y + beta * (y - y_old)
        y_old .= y
    end
    
    nu = prob.A * y - prob.b
    mu_dual = L * (y - x)
    
    r = prob.AtA * y - prob.Atb + mu_dual
    r_inf = norm(r, Inf)
    alpha = min(1.0, prob.lambda / (r_inf + 1e-14))
    
    nu_feas = alpha * nu
    mu_feas = alpha * mu_dual
    
    primal = 0.5 * norm(nu)^2 + 0.5 * L * norm(y - x)^2 + prob.lambda * norm(y, 1)
    dual = -0.5 * norm(nu_feas)^2 - dot(prob.b, nu_feas) - (0.5 / L) * norm(mu_feas)^2 - dot(x, mu_feas)
    
    eps = max(0.0, primal - dual)
    f_y = 0.5 * norm(nu)^2 + prob.lambda * norm(y, 1)
    g = L * (x - y)
    
    return ProximalResult(y, f_y, g, eps)
end

evaluate_objective(prob::L1RegressionProblem, x::Vector{Float64}) = 0.5 * norm(prob.A * x - prob.b)^2 + prob.lambda * norm(x, 1)


# ==============================================================================
# 5. Problem 3: Poorly Conditioned Least Squares (Exact Proximal)
# ==============================================================================
struct IllConditionedLSProblem <: AbstractProblem
    A::Matrix{Float64}
    b::Vector{Float64}
    AtA::Matrix{Float64}
    Atb::Vector{Float64}
end

function generate_ill_conditioned_ls(d::Int, n::Int; cond_num::Float64=1e4, seed::Int=123)
    @assert n >= d 
    rng = MersenneTwister(seed)
    
    U = Matrix(qr(randn(rng, n, d)).Q)
    V = Matrix(qr(randn(rng, d, d)).Q)
    
    # Spectrum: One outlier singular value to force global L = cond_num (10,000)
    s = ones(d)
    s[1] = sqrt(cond_num)
    A = U * Diagonal(s) * V'
    
    # Plant the true solution ONLY along the flat dimensions
    x_true = randn(rng, d)
    x_true .-= dot(x_true, V[:, 1]) .* V[:, 1]
    
    # Ensure noise also lies in the benign subspace
    noise = 0.01 * randn(rng, n)
    noise .-= dot(noise, U[:, 1]) .* U[:, 1]
    
    b = A * x_true + noise
    return IllConditionedLSProblem(A, b, A'*A, A'*b)
end

function evaluate_oracle(prob::IllConditionedLSProblem, x::Vector{Float64}, L::Float64)
    # Proximal step corresponds to solving: (A'A + L*I) y = A'b + L*x
    # The L*I term acts as Tikhonov regularization, making the system strictly well-conditioned
    H = prob.AtA + L * I
    rhs = prob.Atb + L * x
    y = cholesky(Symmetric(H)) \ rhs
    
    f_y = 0.5 * norm(prob.A * y - prob.b)^2
    g = L * (x - y)
    
    return ProximalResult(y, f_y, g, 0.0) # Exact solve yields eps = 0.0
end

evaluate_objective(prob::IllConditionedLSProblem, x::Vector{Float64}) = 0.5 * norm(prob.A * x - prob.b)^2


# ==============================================================================
# 6. Top-Level Optimization Loop
# ==============================================================================

"""
    optimize(prob, method, x0; max_iters)

Applies a generic iterative method one step at a time.
Tracks the objective value and wall-clock time at each iteration.
"""
function optimize(prob::AbstractProblem, method::AbstractMethod, x0::Vector{Float64}; max_iters::Int=100)
    state = init_state(method, x0)
    
    obj_history = Float64[]
    time_history = Float64[]
    
    # Record initialization metrics
    push!(obj_history, evaluate_objective(prob, x0))
    push!(time_history, 0.0)
    
    start_time = time_ns()
    
    for _ in 1:max_iters
        query = get_query(method, state)
        
        # Determine whether it's a subgradient or proximal query based on return type
        if query isa Tuple
            res = evaluate_oracle(prob, query[1], query[2])
        else
            res = evaluate_oracle(prob, query)
        end
        
        state = update!(method, state, res)
        
        # Track metrics
        x_curr = get_iterate(method, state)
        push!(obj_history, evaluate_objective(prob, x_curr))
        push!(time_history, (time_ns() - start_time) / 1e9)
    end
    
    return state, obj_history, time_history
end


# ==============================================================================
# 7. Baseline Implementations (For testing the interface out of the box)
# ==============================================================================

# --- Dummy Subgradient Method ---
struct DummySubgradientMethod <: AbstractMethod
    step_size::Float64
end

struct DummySubgradientState
    x::Vector{Float64}
    k::Int
end

init_state(m::DummySubgradientMethod, x0::Vector{Float64}) = DummySubgradientState(copy(x0), 1)
get_query(m::DummySubgradientMethod, state::DummySubgradientState) = state.x
get_iterate(m::DummySubgradientMethod, state::DummySubgradientState) = state.x

function update!(m::DummySubgradientMethod, state::DummySubgradientState, res::SubgradientResult)
    alpha = m.step_size
    x_next = state.x - alpha * res.g
    return DummySubgradientState(x_next, state.k + 1)
end

# --- Dummy Proximal Method ---
struct DummyProximalMethod <: AbstractMethod
    L::Float64
end

struct DummyProximalState
    x::Vector{Float64}
end

init_state(m::DummyProximalMethod, x0::Vector{Float64}) = DummyProximalState(copy(x0))
get_query(m::DummyProximalMethod, state::DummyProximalState) = (state.x, m.L)
get_iterate(m::DummyProximalMethod, state::DummyProximalState) = state.x

function update!(m::DummyProximalMethod, state::DummyProximalState, res::ProximalResult)
    return DummyProximalState(res.y)
end

# ==============================================================================
# Additional paper experiments: option-2 staircase construction and proximal tests
# ==============================================================================

export generate_staircase_max_affine, generate_revealing_max_affine
export generate_lowrank_ls
export RotatedStratifiedLassoProblem, generate_rotated_stratified_lasso
export estimate_epsilon_subgradient_gap

# --- Utilities ----------------------------------------------------------------

function _logspace(a::Float64, b::Float64, n::Int)
    if n == 1
        return [sqrt(a * b)]
    end
    return exp.(range(log(a), log(b), length=n))
end

soft_threshold(x::AbstractVector, λ::Real) = sign.(x) .* max.(abs.(x) .- λ, 0.0)

function orthonormal_columns(rng::AbstractRNG, n::Int, r::Int)
    @assert 1 <= r <= n
    return Matrix(qr(randn(rng, n, r)).Q)[:, 1:r]
end

# --- Experiment 1: staircase max-affine with embedded inactive wall ------------

"""
    generate_staircase_max_affine(d; r=10, high=12.0, low=3.0, wall_M=20.0, seed=1)

Constructs the option-2 embedded staircase instance

    f(x) = max{ max_{1<=j<=r} |x_j - s_j|,
                wall_M * <u,x> - wall_M * (||s|| + 5) },

where `u` is a random unit vector orthogonal to span(e_1,...,e_r).  Thus the
mathematically relevant staircase lives in the first `r` coordinates, while the
large-norm wall is inactive near the generated solution and only increases the
global Lipschitz constant.  The remaining ambient coordinates only provide an
embedding space for `u`.
"""
function generate_staircase_max_affine(d::Int;
                                       r::Int=10,
                                       high::Float64=12.0,
                                       low::Float64=3.0,
                                       wall_M::Float64=20.0,
                                       seed::Int=1)
    @assert d >= r + 1 "Need d >= r+1 so there is a wall direction orthogonal to the staircase coordinates."

    rng = MersenneTwister(seed)

    s = collect(range(high, stop=low, length=r))
    x_star = zeros(d)
    x_star[1:r] .= s

    rows = Vector{Vector{Float64}}()
    bs = Float64[]

    # Box/staircase facets: |x_j - s_j| for j=1,...,r.
    for j in 1:r
        a_pos = zeros(d)
        a_pos[j] = 1.0
        push!(rows, a_pos)
        push!(bs, -s[j])

        a_neg = zeros(d)
        a_neg[j] = -1.0
        push!(rows, a_neg)
        push!(bs, s[j])
    end

    # Option-2 inactive wall: choose a unit direction in the orthogonal complement
    # of span(e_1,...,e_r), instead of naming a special coordinate such as x_d.
    u = zeros(d)
    u[(r + 1):d] .= randn(rng, d - r)
    u ./= norm(u)

    a_wall = wall_M .* u
    b_wall = -wall_M * (norm(s) + 5.0)
    push!(rows, a_wall)
    push!(bs, b_wall)

    A = reduce(vcat, [row' for row in rows])
    b = collect(bs)

    f_star = 0.0
    return MaxAffineProblem(A, b), x_star, f_star
end

generate_revealing_max_affine(args...; kwargs...) = generate_staircase_max_affine(args...; kwargs...)

# --- Experiment 2: low-rank least squares with exact proximal oracle -----------

"""
    generate_lowrank_ls(d, n; r=8, eig_min=1.0, eig_max=1e4, seed=3)

Constructs a rank-`r` least-squares objective

    f(x) = 1/2 ||A*x - b||^2,

with nonzero Hessian eigenvalues logarithmically spaced between `eig_min` and
`eig_max`.  The data are noiseless: `b = A*x_star`, so the returned `x_star`
is an exact minimizer and `f_star = 0`.
"""
function generate_lowrank_ls(d::Int, n::Int;
                             r::Int=8,
                             eig_min::Float64=1.0,
                             eig_max::Float64=1e4,
                             seed::Int=3)
    @assert 1 <= r <= min(d, n)
    rng = MersenneTwister(seed)

    U = orthonormal_columns(rng, n, r)
    V = orthonormal_columns(rng, d, r)
    eigs = _logspace(eig_min, eig_max, r)

    A = U * Diagonal(sqrt.(eigs)) * V'

    coeff = randn(rng, r)
    x_star = V * coeff

    # Noiseless data: x_star is an exact minimizer and the optimal value is zero.
    b = A * x_star

    prob = IllConditionedLSProblem(A, b, A' * A, A' * b)
    f_star = 0.0

    return prob, x_star, f_star, eigs
end

# --- Experiment 3: structured rotated Lasso with inexact proximal oracle -------

struct RotatedStratifiedLassoProblem <: AbstractProblem
    H::Matrix{Float64}
    c::Vector{Float64}
    lambda::Float64
    eigs::Vector{Float64}
    xstar::Vector{Float64}
    fstar::Float64
    inner_iters::Int
end

function evaluate_objective(prob::RotatedStratifiedLassoProblem, x::Vector{Float64})
    dx = x - prob.c
    return 0.5 * dot(dx, prob.H * dx) + prob.lambda * norm(x, 1)
end

function _prox_residual_bound(prob::RotatedStratifiedLassoProblem,
                              x::Vector{Float64},
                              y::Vector{Float64},
                              L::Float64)
    r = prob.H * (y - prob.c) + L * (y - x)
    s = similar(r)

    for i in eachindex(y)
        if y[i] > 0.0
            s[i] = r[i] + prob.lambda
        elseif y[i] < 0.0
            s[i] = r[i] - prob.lambda
        else
            excess = abs(r[i]) - prob.lambda
            s[i] = excess > 0.0 ? sign(r[i]) * excess : 0.0
        end
    end

    μ = minimum(prob.eigs) + L
    return norm(s)^2 / (2.0 * μ)
end

function evaluate_oracle(prob::RotatedStratifiedLassoProblem,
                         x::Vector{Float64},
                         L::Float64)
    ℓ = maximum(prob.eigs) + L
    μ = minimum(prob.eigs) + L
    step = 1.0 / ℓ
    q = μ / ℓ
    β = (1.0 - sqrt(q)) / (1.0 + sqrt(q))

    y = copy(x)
    y_old = copy(x)
    v = copy(x)

    for _ in 1:prob.inner_iters
        grad = prob.H * (v - prob.c) + L * (v - x)
        y_new = soft_threshold(v - step * grad, prob.lambda * step)
        v = y_new + β * (y_new - y_old)
        y_old = y_new
        y = y_new
    end

    f_y = evaluate_objective(prob, y)
    g = L * (x - y)
    eps = _prox_residual_bound(prob, x, y, L)

    return ProximalResult(y, f_y, g, eps)
end

function generate_rotated_stratified_lasso(d::Int;
                                           active_frac::Float64=0.10,
                                           h_min::Float64=1e-2,
                                           h_max::Float64=1e3,
                                           amp_min::Float64=1e-3,
                                           amp_max::Float64=50.0,
                                           lambda::Float64=0.02,
                                           low_signal_rank::Int=8,
                                           inactive_subgrad_fraction::Float64=0.0,
                                           inner_iters::Int=80,
                                           seed::Int=4)
    @assert 0.0 < active_frac <= 1.0
    @assert 1 <= low_signal_rank <= d
    @assert h_min > 0.0 && h_max >= h_min

    rng = MersenneTwister(seed)

    V = Matrix(qr(randn(rng, d, d)).Q)
    eigs = _logspace(h_min, h_max, d)
    H = Matrix(Symmetric(V * Diagonal(eigs) * V'))

    raw = V[:, 1:low_signal_rank] * randn(rng, low_signal_rank)
    s = max(1, round(Int, active_frac * d))
    active = sortperm(abs.(raw), rev=true)[1:s]

    xstar = zeros(d)
    amps = _logspace(amp_min, amp_max, s)
    for (kk, idx) in enumerate(active)
        xstar[idx] = sign(raw[idx] == 0.0 ? randn(rng) : raw[idx]) * amps[kk]
    end

    q = zeros(d)
    for idx in active
        q[idx] = lambda * sign(xstar[idx])
    end

    if inactive_subgrad_fraction != 0.0
        inactive = setdiff(collect(1:d), active)
        for idx in inactive
            q[idx] = inactive_subgrad_fraction * lambda * (rand(rng) < 0.5 ? -1.0 : 1.0)
        end
    end

    # KKT construction: H*(xstar-c) + q = 0, where q ∈ lambda*∂||xstar||_1.
    c = xstar + V * ((V' * q) ./ eigs)

    fstar = 0.5 * dot(xstar - c, H * (xstar - c)) + lambda * norm(xstar, 1)

    prob = RotatedStratifiedLassoProblem(H, c, lambda, eigs, xstar, fstar, inner_iters)
    return prob, xstar, fstar
end

"""
    estimate_epsilon_subgradient_gap(prob, y, g; cert_iters=1000, x_init=y)

Estimates

    epsilon = f(y) - <g,y> - min_x { f(x) - <g,x> }

for the rotated Lasso problem by running a high-accuracy accelerated proximal
solve on the strongly convex dense Lasso objective `f(x)-<g,x>`.
"""
function estimate_epsilon_subgradient_gap(prob::RotatedStratifiedLassoProblem,
                                          y::Vector{Float64},
                                          g::Vector{Float64};
                                          cert_iters::Int=1000,
                                          x_init::Vector{Float64}=copy(y))
    ℓ = maximum(prob.eigs)
    μ = minimum(prob.eigs)
    step = 1.0 / ℓ
    qratio = μ / ℓ
    β = (1.0 - sqrt(qratio)) / (1.0 + sqrt(qratio))

    x = copy(x_init)
    x_old = copy(x_init)
    v = copy(x_init)

    for _ in 1:cert_iters
        grad = prob.H * (v - prob.c) - g
        x_new = soft_threshold(v - step * grad, prob.lambda * step)
        v = x_new + β * (x_new - x_old)
        x_old = x_new
        x = x_new
    end

    val_y = evaluate_objective(prob, y) - dot(g, y)
    val_min_est = evaluate_objective(prob, x) - dot(g, x)
    return max(0.0, val_y - val_min_est)
end
