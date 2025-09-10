############# Nash Game → MCP (no masks, no θ) ################
# - Dynamics: double-integrator   (Eq. (8))
# - Bounds:   pos/vel/acc boxes   (Eqs. (9a–c))
# - Cost:     private + shared    (Eq. (10))
# - AD-ready via ForwardDiff.gradient on the Lagrangian
################################################################

using LinearAlgebra
using ForwardDiff
using MixedComplementarityProblems

# --------------------------
# Problem parameter container
# --------------------------
struct GameParams
    N::Int                    # agents
    T::Int                    # horizon (states at k=0..T; controls at k=0..T-1)
    Δ::Float64

    A::Matrix{Float64}
    BΔ::Matrix{Float64}

    # Box bounds (9a–c)
    px_min::Float64; px_max::Float64
    py_min::Float64; py_max::Float64
    vx_min::Float64; vx_max::Float64
    vy_min::Float64; vy_max::Float64
    ax_min::Float64; ax_max::Float64
    ay_min::Float64; ay_max::Float64

    W::Matrix{Float64}        # N×4 weights [w1 w2 w3 w4] per agent (10)
    goals::Matrix{Float64}    # 2×N goal positions (columns)
    x0::Matrix{Float64}       # 4×N initial [px,py,vx,vy] (columns)

    ε_dist::Float64           # small for shared cost denom
end

function make_double_integrator_params(N, T; Δ=0.1,
        px_min=-Inf, px_max=Inf, py_min=-Inf, py_max=Inf,
        vx_min=-2.0, vx_max= 2.0, vy_min=-2.0, vy_max= 2.0,
        ax_min=-1.0, ax_max= 1.0, ay_min=-1.0, ay_max= 1.0,
        W = ones(N,4), goals=zeros(2,N), x0=zeros(4,N), ε_dist=1e-3)

    I2 = Matrix(I, 2, 2)
    A  = [I2  I2*Δ;
          zeros(2,2)  I2]
    BΔ = [zeros(2,2);
          I2] * Δ

    return GameParams(N, T, Δ, A, BΔ,
                      px_min, px_max, py_min, py_max,
                      vx_min, vx_max, vy_min, vy_max,
                      ax_min, ax_max, ay_min, ay_max,
                      W, goals, x0, ε_dist)
end

# --------------------------
# Dimensions & indexing
# --------------------------
const NX_PER = 4  # [px,py,vx,vy]
const NU_PER = 2  # [ax,ay]

function dimsizes(p::GameParams)
    N, T = p.N, p.T
    nx_vars = N * T * NX_PER            # x at k=1..T (k=0 is fixed)
    nu_vars = N * T * NU_PER            # u at k=0..T-1 (allocated as T slots)
    neqs    = N * T * NX_PER            # dynamics constraints for k=0..T-1
    n_pos   = N * T * 4                 # 2 dims × (lower+upper)
    n_vel   = N * T * 4
    n_acc   = N * T * 4                 # for k=0..T-1; allocate as T slots
    nineq   = n_pos + n_vel + n_acc
    return nx_vars, nu_vars, neqs, nineq
end

struct ZLayout
    rx::UnitRange{Int}
    ru::UnitRange{Int}
    rλ::UnitRange{Int}
end
function make_layout(p::GameParams)
    nx, nu, neqs, _ = dimsizes(p)
    rx = 1:nx
    ru = (last(rx)+1):(last(rx)+nu)
    rλ = (last(ru)+1):(last(ru)+neqs)
    return ZLayout(rx, ru, rλ)
end

# xvars holds x(i,k) for k=1..T
@inline function idx_state(i::Int, k::Int, p::GameParams)
    @assert 1 ≤ k ≤ p.T
    base = (k-1)*p.N*NX_PER + (i-1)*NX_PER
    return (base+1):(base+NX_PER)
end

# uvars holds u(i,k) for k=0..T-1
@inline function idx_ctrl(i::Int, k::Int, p::GameParams)
    @assert 0 ≤ k ≤ p.T-1
    base = k*p.N*NU_PER + (i-1)*NU_PER
    return (base+1):(base+NU_PER)
end

# λ for dynamics (i,k), 4 entries
@inline function idx_eq_lambda(i::Int, k::Int, p::GameParams)
    @assert 0 ≤ k ≤ p.T-1
    base = k*p.N*NX_PER + (i-1)*NX_PER
    return (base+1):(base+NX_PER)
end

@inline unpack_z(z::AbstractVector, lay::ZLayout) = (view(z, lay.rx), view(z, lay.ru), view(z, lay.rλ))
@inline get_state(xvars, i, k, p::GameParams) = @view xvars[idx_state(i,k,p)]
@inline get_ctrl(uvars, i, k, p::GameParams)  = @view uvars[idx_ctrl(i,k,p)]

# --------------------------
# Dynamics residuals g(x,u)=0 (Eq. (8))
# --------------------------
function dynamics_residuals(xvars, uvars, p::GameParams)
    N, T   = p.N, p.T
    A, BΔ  = p.A, p.BΔ
    r      = similar(xvars, N*T*NX_PER)
    idx    = 1
    for k in 0:(T-1)
        for i in 1:N
            xik = (k == 0) ? p.x0[:, i] : get_state(xvars, i, k, p)
            xkp1 = get_state(xvars, i, k+1, p)
            uik = get_ctrl(uvars, i, k, p)
            # res = x_{k+1} - (A x_k + BΔ u_k)
            r[idx  ] = xkp1[1] - (A[1,1]*xik[1] + A[1,2]*xik[2] + A[1,3]*xik[3] + A[1,4]*xik[4] + BΔ[1,1]*uik[1] + BΔ[1,2]*uik[2])
            r[idx+1] = xkp1[2] - (A[2,1]*xik[1] + A[2,2]*xik[2] + A[2,3]*xik[3] + A[2,4]*xik[4] + BΔ[2,1]*uik[1] + BΔ[2,2]*uik[2])
            r[idx+2] = xkp1[3] - (A[3,1]*xik[1] + A[3,2]*xik[2] + A[3,3]*xik[3] + A[3,4]*xik[4] + BΔ[3,1]*uik[1] + BΔ[3,2]*uik[2])
            r[idx+3] = xkp1[4] - (A[4,1]*xik[1] + A[4,2]*xik[2] + A[4,3]*xik[3] + A[4,4]*xik[4] + BΔ[4,1]*uik[1] + BΔ[4,2]*uik[2])
            idx += 4
        end
    end
    return r
end

# --------------------------
# Box inequalities h(x,u) ≥ 0 (Eqs. (9a–c))
# --------------------------
function box_inequalities(xvars, uvars, p::GameParams)
    N, T = p.N, p.T
    total = N*T*4 + N*T*4 + N*T*4
    h = zeros(eltype(xvars), total)
    idx = 1

    # position & velocity for k=1..T
    for k in 1:T
        for i in 1:N
            xik = get_state(xvars, i, k, p)
            px, py, vx, vy = xik

            # pos (9a)
            h[idx]   = px - p.px_min; idx += 1
            h[idx]   = p.px_max - px; idx += 1
            h[idx]   = py - p.py_min; idx += 1
            h[idx]   = p.py_max - py; idx += 1

            # vel (9b)
            h[idx]   = vx - p.vx_min; idx += 1
            h[idx]   = p.vx_max - vx; idx += 1
            h[idx]   = vy - p.vy_min; idx += 1
            h[idx]   = p.vy_max - vy; idx += 1
        end
    end

    # accel for k=0..T-1  (allocated as T slots)
    for k in 0:(T-1)
        for i in 1:N
            uik = get_ctrl(uvars, i, k, p)
            ax, ay = uik

            # acc (9c)
            h[idx]   = ax - p.ax_min; idx += 1
            h[idx]   = p.ax_max - ax; idx += 1
            h[idx]   = ay - p.ay_min; idx += 1
            h[idx]   = p.ay_max - ay; idx += 1
        end
    end

    return h
end

# --------------------------
# Costs (Eq. (10))
# --------------------------
@inline function private_cost(i, k, xik, uik, p::GameParams)
    px, py, vx, vy = xik
    ax, ay         = uik
    # grab weights safely
    wi_row = @view p.W[i, :]
    w1, w2, w3 = wi_row[1], wi_row[2], wi_row[3]
    goal = @view p.goals[:, i]          # [gx, gy]

    dx = px - goal[1]
    dy = py - goal[2]

    return w1 * (dx*dx + dy*dy) +
           w2 * (vx^2 + vy^2) +
           w3 * (ax^2 + ay^2)
end

@inline function shared_cost(i, k, xvars, p::GameParams)
    # c_i,s^k = w4 * Σ_{j≠i} 1 / (‖p_i - p_j‖^2 + ε)
    wi_row = @view p.W[i, :]
    w4 = wi_row[4]
    xik = get_state(xvars, i, k, p)
    pix, piy = xik[1], xik[2]
    s = zero(eltype(xvars))
    @inbounds for j in 1:p.N
        j == i && continue
        xjk = get_state(xvars, j, k, p)
        dx = pix - xjk[1]
        dy = piy - xjk[2]
        s += 1.0 / (dx*dx + dy*dy + p.ε_dist)
    end
    return w4 * s
end

function total_cost(xvars, uvars, p::GameParams)
    N, T = p.N, p.T
    J = zero(eltype(xvars))
    for k in 1:T
        for i in 1:N
            xik = get_state(xvars, i, k, p)
            # align control slot: k uses u at k-1 (clip at 0..T-1)
            uik = get_ctrl(uvars, i, min(k-1, p.T-1), p)
            J += private_cost(i, k, xik, uik, p)
            J += shared_cost(i, k, xvars, p)
        end
    end
    return J
end

# --------------------------
# Lagrangian & MCP maps G, H
# --------------------------
function Lagrangian(z, y, p::GameParams)
    lay = make_layout(p)
    xvars, uvars, λ = unpack_z(z, lay)
    g = dynamics_residuals(xvars, uvars, p)   # equalities
    h = box_inequalities(xvars, uvars, p)     # inequalities
    J = total_cost(xvars, uvars, p)
    return J + dot(λ, g) + dot(y, h)
end

# G(z,y): ∂L/∂z  (unconstrained variables are [xvars; uvars; λ])
G(z, y; p::GameParams) = ForwardDiff.gradient(w -> Lagrangian(w, y, p), z)

# H(z,y): h(x,u) ≥ 0 
function H(z, y; p::GameParams)
    lay = make_layout(p)
    xvars, uvars, _ = unpack_z(z, lay)
    box_inequalities(xvars, uvars, p)
end

# --------------------------
# Build & solve MCP
# --------------------------
function build_and_solve(p::GameParams)
    nx, nu, neqs, nineq = dimsizes(p)
    n_uncon = nx + nu + neqs
    n_con   = nineq

    # Create functions that accept θ parameter (even though we ignore it)
    function Gwrap(z, y; θ=nothing)
        return G(z, y; p=p)
    end
    
    function Hwrap(z, y; θ=nothing) 
        return H(z, y; p=p)
    end

    mcp = MixedComplementarityProblems.PrimalDualMCP(
        Gwrap, Hwrap;
        unconstrained_dimension = n_uncon,
        constrained_dimension   = n_con,
        parameter_dimension     = 0
    )

    z0 = zeros(n_uncon)
    y0 = fill(1.0, n_con)  # strictly positive for interior-point

    sol = MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(), mcp, Float64[];
        x₀ = z0, y₀ = y0
    )
    return sol
end

# --------------------------
# Example
# --------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    N, T = 4, 50
    params = make_double_integrator_params(N, T;
        Δ=0.1,
        px_min=0.0, px_max=7.0, py_min=0.0, py_max=7.0,
        vx_min=-2, vx_max=2, vy_min=-2, vy_max=2,
        ax_min=-1, ax_max=1, ay_min=-1, ay_max=1,
        W = [1.0 1.0 0.1 2.0;
             1.0 1.0 0.1 2.0;
             1.0 1.0 0.1 2.0;
             1.0 1.0 0.1 2.0],
        goals = [6.0 1.0 5.0 2.0;
                 6.5 6.5 1.0 5.5],
        x0 = [1.0 1.0 6.0 5.5;
              1.0 5.5 1.0 6.0;
              0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0],
        ε_dist = 1e-3
    )

    sol = build_and_solve(params)
    @show length(sol.x), length(sol.y)   # basic sanity check
end
