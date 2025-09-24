# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false

"""
Mixed-Integer Quadratic Program (MIQP) — CVXPY Implementation

This script formulates and solves problems of the form:

    minimize    (1/2) x^T P x + q^T x
    subject to  G x <= h
                A x  = b
                lb <= x <= ub
                x[i] ∈ ℤ for i in integer_indices

Where:
- P ∈ ℝ^{n×n} is PSD (quadratic term), q ∈ ℝ^{n} (linear term).
- G ∈ ℝ^{m×n}, h ∈ ℝ^{m} encode inequality constraints.
- A ∈ ℝ^{p×n}, b ∈ ℝ^{p} encode equality constraints.
- Bounds (lb, ub) are per-variable lower/upper limits.
- A subset of indices in `integer_indices` are enforced integer.

CVXPY builds the model and delegates the MIQP search (branch-and-bound over a
continuous QP relaxation) to a backend solver (e.g., SCIP, CBC, ECOS_BB).
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Final, Literal, TypedDict, cast

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from cvxpy.expressions.expression import Expression
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Enumerate the solver names you want to support
SolverName = Literal[
    "SCIP", "CBC", "ECOS_BB", "OSQP", "GUROBI", "CPLEX", "GLPK_MI", "SCS", "ECOS"
]

# Map names -> cvxpy solver constants (cvxpy exposes these as ints)
SOLVER_MAP: Final[dict[SolverName, int | str]] = {
    "SCIP": cp.SCIP,
    "CBC": cp.CBC,
    "ECOS_BB": cp.ECOS_BB,
    "OSQP": cp.OSQP,
    "GUROBI": getattr(cp, "GUROBI", cp.OSQP),  # fallback if not built
    "CPLEX": getattr(cp, "CPLEX", cp.OSQP),
    "GLPK_MI": getattr(cp, "GLPK_MI", cp.ECOS_BB),
    "SCS": cp.SCS,
    "ECOS": cp.ECOS,
}


NumericLike = Number


class SolveResults(TypedDict, total=False):
    """
    Container for solve results.

    Keys:
      - status (str): Solver termination status (e.g., "optimal", "infeasible").
      - optimal_value (float | None): Objective value at the reported solution, if any.
      - solution (np.ndarray | None): Primal solution x (shape: (n,)).
      - solve_time (float | None): Wall-clock time reported by the solver (seconds).
    """

    status: str
    optimal_value: object | None
    solution: npt.NDArray[np.float64] | None
    solve_time: float | None


class MIQPSolver:
    """
    Helper class that owns CVXPY variables/constraints and wraps solve logic.

    Attributes set during lifespan:
      - n_vars (int): Number of decision variables n.
      - integer_indices (list[int]): Indices of x that must be integer.
      - verbose (bool): CVXPY solver verbosity flag.
      - x (cp.Variable | None): Decision variable vector in CVXPY.
      - problem (cp.Problem | None): The compiled CVXPY problem object.
    """

    def __init__(
        self,
        n_vars: int,
        integer_indices: list[int] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the solver wrapper.

        Args:
          n_vars: Total number of decision variables (dimension of x).
          integer_indices: Zero-based indices of x constrained to be integer.
          verbose: If True, pass CVXPY's verbose flag to the underlying solver.

        Notes:
          - Only indices in `integer_indices` are integer; all others are continuous.
          - The actual cp.Variable is created in `create_variables()`.
        """
        self.n_vars: int = n_vars
        self.integer_indices: list[int] = integer_indices or []
        self.verbose: bool = bool(verbose)  # small nicety

        self.x: cp.Variable | None = None
        self.z: cp.Variable | None = None
        self.problem: cp.Problem | None = None
        self.results: SolveResults = {}

    def create_variables(self) -> tuple[cp.Variable, cp.Variable | None]:
        """
        Allocate the CVXPY decision variable x with mixed-integrality.

        Behavior:
          - Creates a length-n variable x.
          - For indices in `integer_indices`, sets the variable to be Integer; others stay Continuous.

        Side effects:
          - Sets `self.x`.

        Raises:
          - ValueError if n_vars is not positive.
        """
        x = cp.Variable(self.n_vars)  # continuous by default

        z: cp.Variable | None = None
        if self.integer_indices:
            # Full-length integer vector (simplifies indexing constraints)
            # Only bound to x on the integer indices; other components unused.
            z = cp.Variable(len(self.integer_indices), integer=True)

        self.x = x
        self.z = z
        return x, z

    def formulate_problem(
        self,
        p: npt.NDArray[np.float64],
        q: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64] | None = None,
        h: npt.NDArray[np.float64] | None = None,
        a: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        bounds: tuple[float | None, float | None] | None = None,
    ) -> cp.Problem:
        """
        Build the CVXPY objective and constraints.

        Args:
          p: Quadratic matrix P (shape n×n). Should be PSD/symmetric for convexity.
          q: Linear vector q (shape n,).
          g: Inequality matrix G (m×n) or None if no inequalities.
          h: Inequality RHS h (m,) or None if no inequalities.
          a: Equality matrix A (p×n) or None if no equalities.
          b: Equality RHS b (p,) or None if no equalities.
          bounds: (lb, ub) where lb/ub are vectors of length n (or None) for elementwise bounds.

        Constructs:
          objective = (1/2) * quad_form(x, P) + q^T x
          constraints = []
            - If G, h given: Gx <= h
            - If A, b given: Ax == b
            - If lb/ub given: lb <= x <= ub
            - Integrality comes from variable declaration in `create_variables`.

        Side effects:
          - Populates `self.problem` with a cp.Problem(Minimize(objective), constraints).

        Raises:
          - ValueError on shape mismatches.
        """
        # Ensure arrays have the right shape/dtype
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64).reshape(-1)

        if self.x is None:
            _ = self.create_variables()

        assert self.x is not None, "Variable creation failed"

        # Symmetrize P to be safe for quad_form
        P_sym = 0.5 * (p + p.T)

        # Objective: (1/2) x^T P x + q^T x
        quadratic_term = cp.quad_form(self.x, P_sym)
        linear_term: Expression = q @ self.x
        objective = cp.Minimize(0.5 * quadratic_term + linear_term)

        constraints: list[cp.Constraint] = []

        # Gx <= h
        if g is not None and h is not None:
            g = np.asarray(g, dtype=np.float64)
            h = np.asarray(h, dtype=np.float64).reshape(-1)
            constraints.append(g @ self.x <= h)

        # Ax = b
        if a is not None and b is not None:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            constraints.append(a @ self.x == b)

        # Integer link constraints for specified indices: x[i] == z[i]
        if self.integer_indices:
            assert self.z is not None
            for k, idx in enumerate(self.integer_indices):
                constraints.append(self.x[idx] == self.z[k])

        # Global bounds (vectorized)
        if bounds is not None:
            lower_bound, upper_bound = bounds
            if lower_bound is not None:
                constraints.append(self.x >= lower_bound)
            if upper_bound is not None:
                constraints.append(self.x <= upper_bound)

        self.problem = cp.Problem(objective, constraints)

        if self.verbose:
            print("Problem Formulation Complete:")
            print(f"  - Number of variables: {self.n_vars}")
            print(f"  - Integer variables: {len(self.integer_indices)}")
            print(
                f"  - Continuous variables: {self.n_vars - len(self.integer_indices)}"
            )
            print(f"  - Number of constraints: {len(constraints)}")

        return self.problem

    def solve(
        self, solver: SolverName | None = None, **solver_options: object
    ) -> SolveResults:
        """
        Solve the MIQP using the selected backend (or try a small fallback list).

        Args:
          solver: Explicit backend selection; if None, try a reasonable sequence.

        Returns:
          A `SolveResults` dict populated with status, optimal_value, solution, and solve_time.

        Notes:
          - Mixed-integer search is handled by the backend via branch-and-bound.
          - If you have licensed/commercial solvers (e.g., GUROBI/CPLEX/MOSEK) available
            in your environment, you could extend SOLVER_MAP and the fallback list.
          - `status` may be "optimal_inaccurate" or "feasible" depending on backend.

        Raises:
          - RuntimeError if the solver fails to return a successful status.
          - ValueError if `formulate_problem()` has not been called.
        """
        if self.problem is None:
            raise ValueError("Problem not formulated. Call formulate_problem() first.")
        assert self.x is not None

        status_ok = {"optimal", "optimal_inaccurate", "feasible"}

        if self.integer_indices:
            if solver is not None:
                _ = cast(
                    float | None,
                    self.problem.solve(
                        solver=SOLVER_MAP[solver],
                        verbose=self.verbose,
                        **solver_options,
                    ),
                )
                if (self.problem.status or "").lower() not in status_ok:
                    raise RuntimeError(
                        f"MIQP solve did not succeed: {self.problem.status}"
                    )
            else:
                tried: list[SolverName] = []
                for s in cast(tuple[SolverName, ...], ("SCIP", "CBC", "ECOS_BB")):
                    tried.append(s)
                    try:
                        _ = cast(
                            float | None,
                            self.problem.solve(
                                solver=SOLVER_MAP[s],
                                verbose=self.verbose,
                                **solver_options,
                            ),
                        )
                        if (self.problem.status or "").lower() in status_ok:
                            break
                        if self.verbose:
                            print(
                                f"Solver {s} finished with status {self.problem.status}"
                            )
                    except Exception as e:
                        if self.verbose:
                            print(f"Solver {s} failed with: {e}")
                else:
                    raise RuntimeError(
                        f"No MIQP solver succeeded. Tried: {', '.join(tried)}"
                    )
        else:
            use: SolverName = solver if solver is not None else "OSQP"
            _ = cast(
                float | None,
                self.problem.solve(
                    solver=SOLVER_MAP[use],
                    verbose=self.verbose,
                    **solver_options,
                ),
            )
            if (self.problem.status or "").lower() not in status_ok:
                raise RuntimeError(f"QP solve did not succeed: {self.problem.status}")

        # Store results
        solution_value: npt.NDArray[np.float64] | None = None
        if self.x.value is not None:
            solution_value = np.array(self.x.value, dtype=np.float64).reshape(-1)

        self.results = {
            "status": self.problem.status,
            "optimal_value": self.problem.value,
            "solution": solution_value,
            "solve_time": self.problem.solver_stats.solve_time,
        }

        if self.verbose:
            self.print_results()

        return self.results

    def print_results(self) -> None:
        """
        Pretty-print the current solution stored in `self.problem`.

        Prints:
          - Status string
          - Objective value
          - First few entries of x (or full x depending on vector size)

        Safe to call after a successful `solve()`. Will no-op or guard if not solved.
        """
        print("\n" + "=" * 50)
        print("SOLUTION RESULTS")
        print("=" * 50)
        print(f"Status: {self.results.get('status', 'unknown')}")

        # Narrow optimal_value so format spec works
        ov = self.results.get("optimal_value")
        if isinstance(ov, Number):
            print(f"Optimal objective value: {ov:.6f}")
        elif ov is not None:
            print(f"Optimal objective value: {ov}")
        else:
            print("Optimal objective value: Not found")

        sol_obj = self.results.get("solution")
        if isinstance(sol_obj, np.ndarray):
            x: npt.NDArray[np.float64] = sol_obj

            # Convert to real Python floats so pyright knows the type
            xs: list[float] = [float(v) for v in x.flat]

            print("\nOptimal solution:")
            for i, v in enumerate(xs):
                var_type = "INTEGER" if i in self.integer_indices else "CONTINUOUS"
                print(f"  x[{i}] = {v:.6f} ({var_type})")
        else:
            print("\nNo solution found")

        st = self.results.get("solve_time")
        if isinstance(st, Number):
            print(f"\nSolve time: {float(st):.4f} seconds")
        print("=" * 50)


def example_miqp_problem() -> None:
    """
    Solve an example Mixed-Integer Quadratic Program.

    This demonstrates:
    - Quadratic objective (risk minimization)
    - Linear constraints (return, budget, nonnegativity)
    - Mixed integer constraints (some assets must be whole units)


    Build a small, reproducible MIQP instance for demonstration/testing.

    Returns:
      Tuple of (P, q, G, h, A, b, lb, ub, returns, prices, integer_indices)

    Where:
      - P, q, G, h, A, b: MIQP data in standard form.
      - lb, ub: Elementwise variable bounds (vectors of length n).
      - returns, prices: Toy vectors to visualize portfolio-style metrics.
      - integer_indices: Indices of x required to be integer in this example.

    Notes:
      - Values are synthetic; adjust to test different regimes (dense/sparse, tight/loose).
      - P is constructed to be PSD to keep the QP relaxation convex.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: PORTFOLIO-STYLE MIQP WITH INTEGER CONSTRAINTS")
    print("=" * 60)

    # Problem parameters
    n_assets = 5
    rng = np.random.default_rng(42)

    # Covariance (PSD)
    A_rand = rng.standard_normal((n_assets, n_assets))
    Sigma = A_rand.T @ A_rand

    # Returns (5% to 15%)
    mu = rng.uniform(0.05, 0.15, size=n_assets)

    # Prices ($50 to $150)
    prices = rng.uniform(50.0, 150.0, size=n_assets)

    budget = 1000.0
    min_return = 0.08

    # Integer variables for assets 0, 2, 4
    integer_indices = [0, 2, 4]

    solver = MIQPSolver(n_vars=n_assets, integer_indices=integer_indices, verbose=True)

    # Objective: minimize x^T Sigma x (we’ll put 1/2 outside, so use P = 2*Sigma)
    p = 2.0 * Sigma
    q = np.zeros(n_assets, dtype=np.float64)

    # Inequalities:
    #   -mu^T x <= -min_return   (i.e., mu^T x >= min_return)
    #    prices^T x <= budget
    #   -I x <= 0  (x >= 0)
    g = np.vstack(
        [
            -mu.reshape(1, -1),
            prices.reshape(1, -1),
        ]
    )
    h = np.hstack(
        [
            -min_return,
            budget,
        ]
    )

    solver.formulate_problem(p=p, q=q, g=g, h=h, bounds=(0.0, None))
    results = solver.solve()

    if results.get("solution") is not None:
        sol = results.get("solution")
        if not isinstance(sol, np.ndarray):
            print("\nNo solution found - cannot perform portfolio analysis")
            return  # or just `else: ...` and skip
        # Narrow and pin the type for pyright
        x_opt: npt.NDArray[np.float64] = sol

        # Scalars: go through numpy, then cast the scalar so pyright knows it's numeric
        ret_val = cast(np.float64, np.dot(mu, x_opt))  # shape () numpy scalar
        cost_val = cast(np.float64, np.dot(prices, x_opt))

        v: npt.NDArray[np.float64] = cast(np.ndarray, np.matmul(Sigma, x_opt))
        risk_sq = cast(np.float64, np.dot(x_opt, v))  # x' * Sigma * x

        portfolio_return: float = float(ret_val)
        portfolio_cost: float = float(cost_val)
        portfolio_risk: float = math.sqrt(float(risk_sq))

        print("\nPORTFOLIO ANALYSIS:")
        print(f"Expected Return: {portfolio_return:.2%}")
        print(f"Portfolio Risk (Std Dev): {portfolio_risk:.4f}")
        print(f"Total Cost: ${portfolio_cost:.2f}")
        print(f"Budget Utilization: {portfolio_cost / budget:.1%}")

        visualize_portfolio(x_opt, mu, prices, integer_indices)
    else:
        print("\nNo solution found - cannot perform portfolio analysis")


def visualize_portfolio(
    x: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    integer_indices: list[int],
) -> None:
    """
    Quick diagnostic plots for a portfolio-style interpretation of x.

    Plots (2×2 grid):
      1) Allocation x (bar): integers colored differently from continuous vars.
      2) Dollar exposure x ⊙ prices.
      3) Asset returns (%) as a reference bar chart.
      4) Contribution to return (%) via x ⊙ returns.

    Args:
      x: Solution vector (n,).
      returns: Per-asset returns (n,).
      prices: Per-asset prices/weights (n,).
      integer_indices: Indices of x that are integer in the model.

    Notes:
      - Intended for sanity checks; remove if your application is non-portfolio.
      - Requires matplotlib; skip calling this in non-GUI or headless CI.
    """
    n_assets: int = len(x)

    # Build figure/axes without subplots() so nothing is typed as Any
    fig: Figure = plt.figure(figsize=(12, 10))
    ax1: Axes = fig.add_subplot(2, 2, 1)
    ax2: Axes = fig.add_subplot(2, 2, 2)
    ax3: Axes = fig.add_subplot(2, 2, 3)
    ax4: Axes = fig.add_subplot(2, 2, 4)

    # Convert to plain lists of floats to avoid Any-propagation in matplotlib
    xs: list[float] = cast(list[float], x.astype(float).tolist())
    investments: list[float] = cast(list[float], (x * prices).astype(float).tolist())
    returns_pct: list[float] = cast(
        list[float], (returns * 100.0).astype(float).tolist()
    )
    contrib_pct: list[float] = cast(
        list[float], (x * returns * 100.0).astype(float).tolist()
    )

    colors: list[str] = [
        "red" if i in integer_indices else "blue" for i in range(n_assets)
    ]
    indices: list[int] = list(range(n_assets))

    # Asset allocation
    bars: BarContainer = ax1.bar(indices, xs, color=colors)
    ax1.set_xlabel("Asset Index")
    ax1.set_ylabel("Units Allocated")
    ax1.set_title("Portfolio Allocation (Red = Integer Constrained)")
    ax1.grid(True, alpha=0.3)

    for rect, val in zip(bars.patches, xs):
        r: Rectangle = rect
        ax1.text(
            r.get_x() + r.get_width() / 2.0,
            r.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )

    # Investment by asset
    ax2.bar(indices, investments, color=colors)
    ax2.set_xlabel("Asset Index")
    ax2.set_ylabel("Investment ($)")
    ax2.set_title("Investment Distribution")
    ax2.grid(True, alpha=0.3)

    # Expected returns
    ax3.bar(indices, returns_pct, color="green", alpha=0.7)
    ax3.set_xlabel("Asset Index")
    ax3.set_ylabel("Expected Return (%)")
    ax3.set_title("Expected Returns by Asset")
    ax3.grid(True, alpha=0.3)

    # Contribution to portfolio return
    ax4.bar(indices, contrib_pct, color="purple", alpha=0.7)
    ax4.set_xlabel("Asset Index")
    ax4.set_ylabel("Return Contribution (%)")
    ax4.set_title("Contribution to Portfolio Return")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Portfolio Optimization Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.show()


def main() -> None:
    """
    Entry point:
      - Builds a toy MIQP with `example_miqp_problem()`.
      - Creates variables and formulates the problem.
      - Calls `solve()` with a chosen backend (defaults to a fallback order).
      - Prints results and (optionally) visualizes.

    How to run:
      $ python miqp_solve.py
    """
    example_miqp_problem()

    print("\n" + "=" * 60)
    print("EXPLANATION OF THE ALGORITHM")
    print("=" * 60)
    print(
        """
Branch-and-bound solves the mixed-integer part by exploring subproblems where
integer variables are fixed, using continuous QP relaxations for bounds.
CVXPY delegates this to the chosen MIQP solver (SCIP/CBC/ECOS_BB).
"""
    )


if __name__ == "__main__":
    main()
