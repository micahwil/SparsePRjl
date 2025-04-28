module SparsePRjl

using LinearAlgebra
using IterativeSolvers
using Random
using Printf
export solver_newton_projection, generate_true_signal, measure_signal

function solver_newton_projection(y_abs, A, s, opts=Dict())
    # Sparse phase retrieval using sparse spectral initialization method in the first stage
    # and the Newton projection algorithm in the second stage.
    # Converted from solver_newton_projection.m from https://github.com/jxying/SparsePR/tree/main
    maxiter = get(opts, "maxiter", 100)
    tol = get(opts, "tol", 1e-5)
    tolpcg = get(opts, "tolpcg", 1e-4)
    
    flag_true = haskey(opts, "true")
    if flag_true
        z = opts["true"]
        toltrue = get(opts, "toltrue", 1e-3)
    end
    
    flag_trunction = get(opts, "trunction", 0)
    option_svd = get(opts, "svd", "power")
    eta = get(opts, "eta", 0.95)
    tau = get(opts, "tau", 1)
    display_flag = get(opts, "display", 1)

    ## Initialize parameters
    converge = 0
    succ = 0
    m, n = size(A)
    xo = zeros(n)
    erroriter = zeros(maxiter)
    timeiter = zeros(maxiter)
    MShat = zeros(s, s)         
    y_abs2 = y_abs.^2 
    phi_sq = sum(y_abs2)/m
    phi = sqrt(phi_sq)     

    #First Stage
    Marg = vec(sum((A.^2) .* y_abs2, dims=1) / m)
    sorted_indices = sortperm(Marg, rev=true)
    MgS = sorted_indices
    Shat = sort(MgS[1:s]) 
    AShat = A[:, Shat]             

    if flag_trunction == 1
        card_Marg = ceil(Int, m/6)
        M_eval = zeros(m)

        for i in 1:m
            M_eval[i] = y_abs[i]/norm(AShat[i, :])
        end

        _, MmS = sort(M_eval, rev=true)
        Io = MmS[1:card_Marg]   
    else 
        card_Marg = m
        Io = 1:card_Marg
    end

    for i in 1:card_Marg
        ii = Io[i]
        MShat += (y_abs2[ii]) * (reshape(AShat[ii, :], :, 1) * reshape(AShat[ii, :], 1, :))
    end

    if option_svd == "svd"
        u, sigma, v = svd(MShat)
        v1 = u[:, 1]
    elseif option_svd == "power"
        v1 = svd_power(MShat)
    end

    v = zeros(n)
    v[Shat] .= v1
    x_init = phi * v
    x = copy(x_init)

    ## Second Stage
    t0 = time()
    Tx = findall(x .!= 0)
    Ax = A[:, Tx] * x[Tx]
    Ax3 = Ax.^3
    Axb3 = Ax3 - y_abs2 .* Ax
    g = A' * Axb3 / (m * m)
    T0 = Tx

    numiters = 0
    errout = 0
    for t in 1:maxiter
        x0 = copy(x)
        p = sign.(Ax)
        y_sign = y_abs .* p
        grad_i = A' * (Ax - y_sign) / m

        xtg = x0 - eta * grad_i
        _, T_indices = findmax(abs.(xtg), dims=1)
        T = [i[1] for i in sort(collect(pairs(abs.(xtg))), by=x->x[2], rev=true)[1:s]]
        
        TTc = setdiff(T0, T)
        T0 = T
        gT = g[T]
        
        Ax3b = 3 * Ax.^2 - y_abs2
        AT = A[:, T]
        H = AT' * (AT .* Ax3b) / (m * m)

        D = AT' * (A[:, TTc] .* Ax3b) / (m * m)
        
        # PCG solver for Julia
        dT = pcg_solve(H, D * x0[TTc] - gT, tolpcg)
        
        x = copy(xo)
        x[T] = x0[T] + tau * dT

        Tx = findall(x .!= 0)
        Ax = A[:, Tx] * x[Tx]
        Ax3 = Ax.^3
        Axb3 = Ax3 - y_abs2 .* Ax
        g = A' * Axb3 / (m * m)

        timepoint = time() - t0
        timeiter[t] = timepoint

        if flag_true
            err = compute_error(x, z)
            erroriter[t] = err
            if display_flag == 1
                @printf("iter: %d, relative error: %.4e, cpu time: %.3f\n", t, err, timepoint)
            end
            if err < toltrue
                converge = 1
                break
            end
            if err > 0.5
                succ += 1
            else
                succ = 0
            end

            if succ >= 10
                println("signal recovery failed: please increase measurements or regenerate sensing matrix")
                break
            end
        else
            err = compute_error(x, x0)
            if display_flag == 1
                @printf("iter: %d, difference between successive iterations: %.4e, cpu time: %.3f\n", 
                       t, err, timepoint)
            end
            if err < tol
                converge = 1
                break
            end
        end
        numiters = t
        errout = err
    end

    out = Dict{String, Any}()
    out["x"] = x
    out["time"] = timeiter[end]
    out["timeiter"] = timeiter[1:numiters+1]
    out["iterate"] = numiters
    out["converge"] = converge
    if flag_true
        out["relerror"] = errout
        out["relerror_iter"] = erroriter[1:numiters+1]
    end

    print("done solver_newton_projection")
    return out
end

function compute_error(x, z)
    err1 = norm(x - z) / norm(z)
    err2 = norm(x + z) / norm(z)
    return min(err1, err2)
end

function generate_true_signal(n, s)
    supp = randperm(n)[1:s]
    z = zeros(n)
    z[supp] = randn(s)
    z = z / norm(z)
    return z
end

function measure_signal(m, z)
    n = length(z)

    A = randn(m, n) 
    y = A * z
    y_abs = abs.(y)

    return y_abs, A
end

function svd_power(M)
    m, n = size(M)
    x = randn(n)
    for it in 1:20
        y = M * x
        y = y / norm(y)
        if norm(x - y) / norm(x) < 1e-6
            break
        end
        x = y
    end
    return x
end

# preconditioned conjugate gradient
function pcg_solve(A, b, tol)
    x = zeros(length(b))
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)
    
    max_iter = 1000
    
    for i in 1:max_iter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = dot(r, r)
        
        if sqrt(rsnew) < tol
            break
        end
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    end
    
    return x
end


function randperm(n, k=n)
# Rand permutation of n numbers, take first k
    return shuffle(1:n)[1:k]
end

end # module

