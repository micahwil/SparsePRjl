include("solver_newton_projection.jl")
using .SparsePRjl

using Random, Printf, Plots

Random.seed!(100)

# Problem setting
n = 10000     # signal dimension
m = 5000      # number of measurements
s = 100       # sparsity of the signal

# Generate signal
z = generate_true_signal(n, s)  # generate ground truth signal
y_abs, A = measure_signal(m, z)  # generate sensing matrix and phaseless measurements

# Signal reconstruction
opts = Dict(
    "maxiter" => 100,
    "true" => z,
    "toltrue" => 1e-6,
    "display" => 1
)

# Run the solver
println("Starting solver...")
@time out = solver_newton_projection(y_abs, A, s, opts)
erroriter = out["relerror_iter"]
plot((erroriter), xlabel="Iteration", ylabel="Relative Error", label="Error")
savefig("err.png")
# Display results
println("\nResults:")
println("Number of iterations: $(out["iterate"])")
println("Final relative error: $(out["relerror"])")
println("Total computation time: $(out["time"]) seconds")
println("Algorithm converged: $(out["converge"] == 1 ? "Yes" : "No")")