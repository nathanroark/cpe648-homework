# --- Mixed-integer least-squares with CVXPY ---

# Import CVXPY and NumPy.
import cvxpy as cp  # CVXPY builds the optimization model
import numpy as np  # NumPy is used to generate random test data

# For reproducibility of the random data below.
np.random.seed(0)

# Problem dimensions:
m = 40  # number of rows (observations)
n = 25  # number of columns (features)

# Generate a random m-by-n matrix A with entries in [0, 1).
# This is the "design" or "measurement" matrix in least squares.
#
# The "design" matrix A is often denoted by Φ (Phi) in literature.
# In statistics, A is often called the "model matrix" or "regression matrix".
# In signal processing, A is often called the "sensing matrix".
A = np.random.rand(m, n)


# Generate a random length-m vector b from a standard normal distribution.
# This is the observed data we aim to fit.
b = np.random.randn(m)

# Create the optimization variable:
# x is an n-dimensional vector, and `integer=True` enforces x ∈ ℤ^n (all entries integers).
x = cp.Variable(n, integer=True)

# Define the objective function:
# minimize ||A x - b||_2^2, i.e., sum of squared residuals.
# `cp.sum_squares(y)` is just ||y||_2^2.
objective = cp.Minimize(cp.sum_squares(A @ x - b))

# Wrap the objective (and any constraints—none here besides integrality) into a Problem.
prob = cp.Problem(objective)

# Solve the problem.
# Note: you must have a mixed-integer capable solver installed.
# CVXPY recommends SCIP (install via `pip install pyscipopt` or conda).
# Many installations will auto-detect a suitable solver; if not, you can specify one, e.g.:
# prob.solve(solver=cp.SCIP)
_ = prob.solve()

# After solving, report the results.
print("Status: ", prob.status)  # e.g., "optimal", "infeasible", "unbounded", etc.
print(
    "The optimal value is", prob.value
)  # Optimal objective value (sum of squared residuals).
print("A solution x is")
print(
    x.value
)  # The optimizer's integer-valued solution vector (as floats but integer-valued).
