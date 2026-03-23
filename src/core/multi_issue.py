"""
Multi-issue bargaining.

Players negotiate over k divisible issues simultaneously.
Each player has a valuation function over each issue.
The Pareto frontier is computed as the set of allocations
where no player can be made better off without making the
other worse off.

Key insight: When players value issues differently, there are
gains from trade. The efficient allocation gives each issue
primarily to the player who values it more.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class IssueAllocation:
    """Allocation of issues between two players."""

    shares: np.ndarray  # shape (k,) - player 1's share of each issue
    player1_utility: float
    player2_utility: float

    def __repr__(self) -> str:
        shares_str = ", ".join(f"{s:.3f}" for s in self.shares)
        return (
            f"Allocation [{shares_str}] -> "
            f"U1={self.player1_utility:.4f}, U2={self.player2_utility:.4f}"
        )


class MultiIssueBargaining:
    """
    Multi-issue bargaining model.

    Parameters
    ----------
    valuations1 : array-like
        Player 1's valuations for each issue. Shape (k,).
    valuations2 : array-like
        Player 2's valuations for each issue. Shape (k,).
    delta1 : float
        Discount factor for player 1.
    delta2 : float
        Discount factor for player 2.

    Notes
    -----
    Each issue i has a total "size" of 1, split as (x_i, 1-x_i)
    where x_i is player 1's share. Player 1's utility is
    sum(v1_i * x_i) and player 2's is sum(v2_i * (1 - x_i)).
    """

    def __init__(
        self,
        valuations1: list,
        valuations2: list,
        delta1: float = 0.9,
        delta2: float = 0.9,
    ):
        self.valuations1 = np.array(valuations1, dtype=float)
        self.valuations2 = np.array(valuations2, dtype=float)

        if len(self.valuations1) != len(self.valuations2):
            raise ValueError(
                "Valuation vectors must have the same length. "
                f"Got {len(self.valuations1)} and {len(self.valuations2)}."
            )
        if np.any(self.valuations1 < 0) or np.any(self.valuations2 < 0):
            raise ValueError("Valuations must be non-negative.")
        if not (0 < delta1 < 1):
            raise ValueError(f"delta1 must be in (0, 1), got {delta1}")
        if not (0 < delta2 < 1):
            raise ValueError(f"delta2 must be in (0, 1), got {delta2}")

        self.k = len(self.valuations1)
        self.delta1 = delta1
        self.delta2 = delta2

    def utility(
        self, shares: np.ndarray, player: int
    ) -> float:
        """
        Compute a player's utility from an allocation.

        Parameters
        ----------
        shares : ndarray
            Player 1's share of each issue, in [0, 1].
        player : int
            1 or 2.

        Returns
        -------
        float
            Total utility for the specified player.
        """
        if player == 1:
            return float(np.dot(self.valuations1, shares))
        else:
            return float(np.dot(self.valuations2, 1 - shares))

    def pareto_frontier(
        self, num_points: int = 200
    ) -> List[Tuple[float, float]]:
        """
        Compute the Pareto frontier of feasible utility pairs.

        Uses the weighted-sum scalarization method: for each weight
        alpha in [0, 1], maximize alpha * U1 + (1-alpha) * U2 over
        feasible allocations.

        For linear utilities, the optimal allocation for each issue
        is bang-bang: give the entire issue to the player whose
        weighted valuation is higher.

        Parameters
        ----------
        num_points : int
            Number of points on the frontier.

        Returns
        -------
        list of (U1, U2) tuples along the Pareto frontier
        """
        frontier = []

        for i in range(num_points + 1):
            alpha = i / num_points
            shares = self._optimal_allocation(alpha)
            u1 = self.utility(shares, 1)
            u2 = self.utility(shares, 2)
            frontier.append((u1, u2))

        # Remove dominated points and sort
        frontier = self._remove_dominated(frontier)
        return frontier

    def _optimal_allocation(self, alpha: float) -> np.ndarray:
        """
        Find the allocation maximizing alpha * U1 + (1-alpha) * U2.

        For linear utilities, for each issue i:
            Give to P1 if alpha * v1_i > (1-alpha) * v2_i
            Give to P2 otherwise
        """
        shares = np.zeros(self.k)
        for i in range(self.k):
            if alpha * self.valuations1[i] >= (1 - alpha) * self.valuations2[i]:
                shares[i] = 1.0
            else:
                shares[i] = 0.0
        return shares

    def _remove_dominated(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Remove dominated points from the frontier."""
        if not points:
            return points

        # Sort by U1
        sorted_pts = sorted(set(points), key=lambda p: p[0])

        # Keep only Pareto-optimal points
        frontier = [sorted_pts[0]]
        max_u2 = sorted_pts[0][1]

        for pt in sorted_pts[1:]:
            if pt[1] >= max_u2 - 1e-10:
                # Check if we should update or this is on the frontier
                frontier.append(pt)
                max_u2 = max(max_u2, pt[1])

        # Final cleanup: ensure monotonicity (U2 decreasing as U1 increases)
        result = []
        min_u2_so_far = float("inf")
        for pt in reversed(frontier):
            if pt[1] <= min_u2_so_far + 1e-10:
                result.append(pt)
                min_u2_so_far = min(min_u2_so_far, pt[1])
        result.reverse()

        return result

    def efficient_allocation(self) -> IssueAllocation:
        """
        Find the efficient allocation that maximizes total surplus.

        Each issue goes to the player who values it more.

        Returns
        -------
        IssueAllocation
        """
        shares = np.zeros(self.k)
        for i in range(self.k):
            if self.valuations1[i] >= self.valuations2[i]:
                shares[i] = 1.0
            else:
                shares[i] = 0.0

        return IssueAllocation(
            shares=shares,
            player1_utility=self.utility(shares, 1),
            player2_utility=self.utility(shares, 2),
        )

    def rubinstein_multi_issue_spe(self) -> IssueAllocation:
        """
        Compute the SPE of the multi-issue Rubinstein bargaining game.

        In the multi-issue setting with linear utilities, the SPE
        allocation maximizes the Nash product:
            (U1 - d1) * (U2 - d2)
        where d1, d2 are the disagreement payoffs (0, 0), weighted
        by the discount factors.

        For the Rubinstein model, the solution approaches the
        Nash bargaining solution as discount factors approach 1,
        with disagreement point (0, 0).

        We solve by finding the allocation on the Pareto frontier
        that corresponds to the Rubinstein split of the total surplus.
        """
        # Compute Rubinstein shares of total surplus
        d1 = self.delta1
        d2 = self.delta2
        denom = 1 - d1 * d2
        p1_power = (1 - d2) / denom  # Player 1's bargaining power

        # Get Pareto frontier
        frontier = self.pareto_frontier(num_points=500)

        if not frontier:
            return IssueAllocation(
                shares=np.zeros(self.k),
                player1_utility=0.0,
                player2_utility=0.0,
            )

        # Find the point on the frontier where P1 gets p1_power fraction
        # of the maximum joint surplus
        max_u1 = max(p[0] for p in frontier)
        max_u2 = max(p[1] for p in frontier)

        # Target: P1's utility should reflect their bargaining power
        target_u1 = p1_power * max_u1

        # Find closest point on frontier
        best_alloc = None
        best_dist = float("inf")

        for num_pts in range(501):
            alpha = num_pts / 500
            shares = self._optimal_allocation(alpha)
            u1 = self.utility(shares, 1)
            dist = abs(u1 - target_u1)
            if dist < best_dist:
                best_dist = dist
                best_alloc = shares.copy()

        u1 = self.utility(best_alloc, 1)
        u2 = self.utility(best_alloc, 2)

        return IssueAllocation(
            shares=best_alloc,
            player1_utility=u1,
            player2_utility=u2,
        )

    def gains_from_trade(self) -> float:
        """
        Compute the gains from trade: the difference between the
        maximum joint surplus and the minimum joint surplus.

        Returns
        -------
        float
            The gains from trade.
        """
        # Max surplus: each issue to whoever values it more
        eff = self.efficient_allocation()
        max_surplus = eff.player1_utility + eff.player2_utility

        # Equal split surplus
        equal_shares = np.full(self.k, 0.5)
        equal_surplus = (
            self.utility(equal_shares, 1) + self.utility(equal_shares, 2)
        )

        return max_surplus - equal_surplus

    def comparative_advantage(self) -> List[Tuple[int, int]]:
        """
        Compute comparative advantages: which player has a
        comparative advantage in each issue.

        Returns a list of (issue_index, advantaged_player) tuples,
        sorted by the strength of the comparative advantage.
        """
        ratios = []
        for i in range(self.k):
            if self.valuations2[i] > 0:
                ratio = self.valuations1[i] / self.valuations2[i]
            else:
                ratio = float("inf")
            ratios.append((i, ratio))

        # Sort by ratio descending - P1 has comparative advantage in
        # issues with high v1/v2 ratio
        ratios.sort(key=lambda x: -x[1])

        result = []
        for i, ratio in ratios:
            if ratio > 1:
                result.append((i, 1))
            elif ratio < 1:
                result.append((i, 2))
            else:
                result.append((i, 0))  # 0 means no advantage

        return result
