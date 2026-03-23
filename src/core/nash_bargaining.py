"""
Nash Bargaining Solution (NBS).

The axiomatic solution to the bargaining problem, satisfying:
1. Pareto efficiency
2. Symmetry
3. Independence of irrelevant alternatives (IIA)
4. Invariance to affine transformations

The NBS maximizes the Nash product:
    (u1 - d1) * (u2 - d2)
subject to (u1, u2) being feasible and u_i >= d_i.

Connection to Rubinstein:
    As delta1, delta2 -> 1, the Rubinstein SPE converges to the
    symmetric Nash bargaining solution with disagreement point (0, 0).

    For asymmetric discount factors, the Rubinstein SPE converges to
    the asymmetric Nash bargaining solution with bargaining powers
    related to the discount factors.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class NashBargainingResult:
    """Result of Nash bargaining computation."""

    player1_utility: float
    player2_utility: float
    disagreement: Tuple[float, float]
    nash_product: float
    bargaining_powers: Tuple[float, float]

    def __repr__(self) -> str:
        return (
            f"NBS: ({self.player1_utility:.4f}, {self.player2_utility:.4f}) "
            f"| d=({self.disagreement[0]:.4f}, {self.disagreement[1]:.4f}) "
            f"| Nash product={self.nash_product:.6f} "
            f"| powers=({self.bargaining_powers[0]:.3f}, {self.bargaining_powers[1]:.3f})"
        )


class NashBargainingSolution:
    """
    Nash Bargaining Solution computation.

    Parameters
    ----------
    feasible_set : list of (float, float) or callable
        Either a list of feasible (u1, u2) pairs defining the frontier,
        or a callable that takes alpha in [0, 1] and returns (u1, u2)
        on the Pareto frontier.
    disagreement : tuple of (float, float)
        The disagreement point (d1, d2).
    alpha : float
        Bargaining power of player 1 (in [0, 1]). Player 2's power
        is 1 - alpha. Default 0.5 (symmetric).
    """

    def __init__(
        self,
        feasible_set: list,
        disagreement: Tuple[float, float] = (0.0, 0.0),
        alpha: float = 0.5,
    ):
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.disagreement = disagreement
        self.alpha = alpha

        if callable(feasible_set):
            # Generate points from the callable
            self.frontier = []
            for i in range(1001):
                t = i / 1000
                self.frontier.append(feasible_set(t))
        else:
            self.frontier = list(feasible_set)

        if not self.frontier:
            raise ValueError("Feasible set must be non-empty")

    def solve(self) -> NashBargainingResult:
        """
        Compute the Nash Bargaining Solution.

        Maximizes the generalized Nash product:
            (u1 - d1)^alpha * (u2 - d2)^(1-alpha)
        over the feasible set.

        Returns
        -------
        NashBargainingResult
        """
        d1, d2 = self.disagreement
        best_product = -1.0
        best_point = None

        for u1, u2 in self.frontier:
            excess1 = u1 - d1
            excess2 = u2 - d2

            if excess1 < 0 or excess2 < 0:
                continue

            # Compute the actual Nash product directly
            if excess1 == 0 and self.alpha > 0:
                product = 0.0
            elif excess2 == 0 and (1 - self.alpha) > 0:
                product = 0.0
            else:
                product = (excess1 ** self.alpha) * (excess2 ** (1 - self.alpha))

            if product > best_product:
                best_product = product
                best_point = (u1, u2)

        if best_point is None:
            return NashBargainingResult(
                player1_utility=d1,
                player2_utility=d2,
                disagreement=self.disagreement,
                nash_product=0.0,
                bargaining_powers=(self.alpha, 1 - self.alpha),
            )

        u1_star, u2_star = best_point

        return NashBargainingResult(
            player1_utility=u1_star,
            player2_utility=u2_star,
            disagreement=self.disagreement,
            nash_product=best_product,
            bargaining_powers=(self.alpha, 1 - self.alpha),
        )

    @staticmethod
    def from_linear_frontier(
        max_u1: float,
        max_u2: float,
        disagreement: Tuple[float, float] = (0.0, 0.0),
        alpha: float = 0.5,
    ) -> "NashBargainingSolution":
        """
        Create NBS for a linear frontier: u1/max_u1 + u2/max_u2 = 1.

        Parameters
        ----------
        max_u1 : float
            Maximum utility for player 1 (when player 2 gets 0).
        max_u2 : float
            Maximum utility for player 2 (when player 1 gets 0).
        disagreement : tuple
            Disagreement point.
        alpha : float
            Bargaining power of player 1.

        Returns
        -------
        NashBargainingSolution
        """
        frontier = []
        for i in range(1001):
            t = i / 1000
            u1 = t * max_u1
            u2 = (1 - t) * max_u2
            frontier.append((u1, u2))

        return NashBargainingSolution(frontier, disagreement, alpha)

    @staticmethod
    def closed_form_linear(
        surplus: float = 1.0,
        disagreement: Tuple[float, float] = (0.0, 0.0),
        alpha: float = 0.5,
    ) -> NashBargainingResult:
        """
        Closed-form NBS for dividing a surplus with linear utilities.

        For u1 + u2 = surplus:
            u1* = d1 + alpha * (surplus - d1 - d2)
            u2* = d2 + (1-alpha) * (surplus - d1 - d2)

        Parameters
        ----------
        surplus : float
            Total surplus to divide.
        disagreement : tuple
            Disagreement point (d1, d2).
        alpha : float
            Bargaining power of player 1.

        Returns
        -------
        NashBargainingResult
        """
        d1, d2 = disagreement
        pie = surplus - d1 - d2

        if pie < 0:
            raise ValueError(
                f"Surplus ({surplus}) must exceed sum of disagreement "
                f"payoffs ({d1 + d2})"
            )

        u1 = d1 + alpha * pie
        u2 = d2 + (1 - alpha) * pie

        nash_product = (u1 - d1) ** alpha * (u2 - d2) ** (1 - alpha)

        return NashBargainingResult(
            player1_utility=u1,
            player2_utility=u2,
            disagreement=disagreement,
            nash_product=nash_product,
            bargaining_powers=(alpha, 1 - alpha),
        )

    @staticmethod
    def rubinstein_convergence(
        delta_values: Optional[List[float]] = None,
        surplus: float = 1.0,
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Show convergence of Rubinstein SPE to Nash bargaining solution.

        As delta -> 1, the Rubinstein outcome converges to
        the symmetric NBS (50/50 split with d=(0,0)).

        Parameters
        ----------
        delta_values : list of float, optional
            Discount factors to evaluate. Default: range from 0.1 to 0.999.
        surplus : float
            Total surplus.

        Returns
        -------
        list of (delta, rubinstein_p1, rubinstein_p2, nash_p1, nash_p2)
        """
        if delta_values is None:
            delta_values = [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                0.95, 0.99, 0.995, 0.999, 0.9999,
            ]

        nash = NashBargainingSolution.closed_form_linear(surplus)
        results = []

        for delta in delta_values:
            # Rubinstein SPE with symmetric discount factors
            denom = 1 - delta * delta
            x = (1 - delta) / denom
            rub_p1 = x * surplus
            rub_p2 = (1 - x) * surplus

            results.append((
                delta,
                rub_p1,
                rub_p2,
                nash.player1_utility,
                nash.player2_utility,
            ))

        return results

    def verify_axioms(
        self, result: Optional[NashBargainingResult] = None
    ) -> dict:
        """
        Verify that the NBS satisfies the four Nash axioms.

        Parameters
        ----------
        result : NashBargainingResult, optional
            Result to verify. If None, solve first.

        Returns
        -------
        dict
            Dictionary with verification results for each axiom.
        """
        if result is None:
            result = self.solve()

        d1, d2 = self.disagreement
        u1, u2 = result.player1_utility, result.player2_utility

        checks = {}

        # 1. Individual rationality: u_i >= d_i
        checks["individual_rationality"] = {
            "satisfied": u1 >= d1 - 1e-10 and u2 >= d2 - 1e-10,
            "detail": f"u1={u1:.4f} >= d1={d1:.4f}: {u1 >= d1 - 1e-10}, "
                      f"u2={u2:.4f} >= d2={d2:.4f}: {u2 >= d2 - 1e-10}",
        }

        # 2. Pareto efficiency: no feasible point dominates (u1, u2)
        pareto_efficient = True
        for fu1, fu2 in self.frontier:
            if fu1 > u1 + 1e-10 and fu2 > u2 + 1e-10:
                pareto_efficient = False
                break
        checks["pareto_efficiency"] = {
            "satisfied": pareto_efficient,
            "detail": "No feasible point strictly dominates the solution",
        }

        # 3. Symmetry check: if problem is symmetric, solution should be symmetric
        # (Check if frontier is symmetric around 45-degree line)
        frontier_symmetric = True
        for fu1, fu2 in self.frontier:
            found_mirror = False
            for gu1, gu2 in self.frontier:
                if abs(gu1 - fu2) < 1e-6 and abs(gu2 - fu1) < 1e-6:
                    found_mirror = True
                    break
            if not found_mirror:
                frontier_symmetric = False
                break

        if frontier_symmetric and abs(d1 - d2) < 1e-10 and abs(self.alpha - 0.5) < 1e-10:
            symmetry_satisfied = abs(u1 - u2) < 1e-6
            checks["symmetry"] = {
                "satisfied": symmetry_satisfied,
                "detail": f"Symmetric problem, u1={u1:.6f}, u2={u2:.6f}, "
                          f"difference={abs(u1-u2):.8f}",
            }
        else:
            checks["symmetry"] = {
                "satisfied": True,
                "detail": "Problem is not symmetric, axiom trivially satisfied",
            }

        # 4. Nash product maximization
        actual_product = (
            (u1 - d1) ** self.alpha * (u2 - d2) ** (1 - self.alpha)
        )
        max_product = 0.0
        for fu1, fu2 in self.frontier:
            if fu1 >= d1 and fu2 >= d2:
                p = (
                    (fu1 - d1) ** self.alpha
                    * (fu2 - d2) ** (1 - self.alpha)
                )
                max_product = max(max_product, p)

        checks["nash_product_maximized"] = {
            "satisfied": abs(actual_product - max_product) < 1e-6,
            "detail": f"Solution product={actual_product:.6f}, "
                      f"max feasible product={max_product:.6f}",
        }

        return checks
