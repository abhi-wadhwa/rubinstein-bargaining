"""
Infinite-horizon Rubinstein alternating-offers bargaining model.

Two players split a surplus of size 1. Player 1 proposes first.
Each player i has discount factor delta_i in (0, 1).

Subgame Perfect Equilibrium (SPE):
    Player 1 offers x* = (1 - delta_2) / (1 - delta_1 * delta_2)
    Player 2 offers y* = (1 - delta_1) / (1 - delta_1 * delta_2)

    Player 1's equilibrium payoff: x*
    Player 2's equilibrium payoff: 1 - x*

With risk of breakdown (probability p each round):
    Player 1 offers x* = (1 - delta_2*(1-p)) / (1 - delta_1*delta_2*(1-p)^2)
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BargainingResult:
    """Result of a bargaining computation."""

    player1_share: float
    player2_share: float
    proposer: int  # 1 or 2
    round_number: int
    accepted: bool

    def __repr__(self) -> str:
        status = "accepted" if self.accepted else "rejected"
        return (
            f"Round {self.round_number}: Player {self.proposer} proposes "
            f"({self.player1_share:.4f}, {self.player2_share:.4f}) [{status}]"
        )


class InfiniteHorizonBargaining:
    """
    Rubinstein infinite-horizon alternating-offers bargaining model.

    Parameters
    ----------
    delta1 : float
        Discount factor for player 1, in (0, 1).
    delta2 : float
        Discount factor for player 2, in (0, 1).
    breakdown_prob : float
        Probability of exogenous breakdown each round, in [0, 1).
    surplus : float
        Total surplus to be divided (default 1.0).
    """

    def __init__(
        self,
        delta1: float,
        delta2: float,
        breakdown_prob: float = 0.0,
        surplus: float = 1.0,
    ):
        if not (0 < delta1 < 1):
            raise ValueError(f"delta1 must be in (0, 1), got {delta1}")
        if not (0 < delta2 < 1):
            raise ValueError(f"delta2 must be in (0, 1), got {delta2}")
        if not (0 <= breakdown_prob < 1):
            raise ValueError(
                f"breakdown_prob must be in [0, 1), got {breakdown_prob}"
            )
        if surplus <= 0:
            raise ValueError(f"surplus must be positive, got {surplus}")

        self.delta1 = delta1
        self.delta2 = delta2
        self.breakdown_prob = breakdown_prob
        self.surplus = surplus

    @property
    def effective_delta1(self) -> float:
        """Effective discount factor for player 1 incorporating breakdown risk."""
        return self.delta1 * (1 - self.breakdown_prob)

    @property
    def effective_delta2(self) -> float:
        """Effective discount factor for player 2 incorporating breakdown risk."""
        return self.delta2 * (1 - self.breakdown_prob)

    def spe_shares(self) -> Tuple[float, float]:
        """
        Compute the Subgame Perfect Equilibrium shares.

        Returns
        -------
        tuple of (float, float)
            (player1_share, player2_share) when player 1 proposes first.

        Notes
        -----
        SPE: x* = (1 - effective_delta2) / (1 - effective_delta1 * effective_delta2)
        """
        d1 = self.effective_delta1
        d2 = self.effective_delta2
        denominator = 1 - d1 * d2
        player1_share = (1 - d2) / denominator
        player2_share = 1 - player1_share
        return (
            player1_share * self.surplus,
            player2_share * self.surplus,
        )

    def spe_offers(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute both players' SPE offers.

        Returns
        -------
        tuple of two tuples
            ((p1_share_when_p1_proposes, p2_share_when_p1_proposes),
             (p1_share_when_p2_proposes, p2_share_when_p2_proposes))
        """
        d1 = self.effective_delta1
        d2 = self.effective_delta2
        denominator = 1 - d1 * d2

        # When player 1 proposes
        x1 = (1 - d2) / denominator
        offer_p1 = (x1 * self.surplus, (1 - x1) * self.surplus)

        # When player 2 proposes
        y2 = (1 - d1) / denominator
        offer_p2 = ((1 - y2) * self.surplus, y2 * self.surplus)

        return offer_p1, offer_p2

    def first_mover_advantage(self) -> float:
        """
        Compute the first-mover advantage.

        Returns the ratio of player 1's share to player 2's share
        in the SPE when player 1 proposes first.
        """
        s1, s2 = self.spe_shares()
        if s2 == 0:
            return float("inf")
        return s1 / s2

    def simulate_rounds(self, max_rounds: int = 20) -> List[BargainingResult]:
        """
        Simulate the alternating-offers process showing convergence.

        In the SPE, agreement is immediate. This method shows what each
        player would offer in each round, illustrating the logic of
        backward induction in the infinite-horizon setting.

        Parameters
        ----------
        max_rounds : int
            Number of rounds to simulate.

        Returns
        -------
        list of BargainingResult
        """
        offer_p1, offer_p2 = self.spe_offers()
        results = []

        for t in range(1, max_rounds + 1):
            if t % 2 == 1:
                # Player 1 proposes
                results.append(
                    BargainingResult(
                        player1_share=offer_p1[0],
                        player2_share=offer_p1[1],
                        proposer=1,
                        round_number=t,
                        accepted=(t == 1),  # Accepted in round 1
                    )
                )
            else:
                # Player 2 proposes
                results.append(
                    BargainingResult(
                        player1_share=offer_p2[0],
                        player2_share=offer_p2[1],
                        proposer=2,
                        round_number=t,
                        accepted=False,
                    )
                )

        return results

    def patience_analysis(
        self, delta_range: Tuple[float, float] = (0.01, 0.99), steps: int = 50
    ) -> List[Tuple[float, float, float]]:
        """
        Analyze how player 1's share changes as delta2 varies.

        Parameters
        ----------
        delta_range : tuple
            (min_delta, max_delta) range for player 2's discount factor.
        steps : int
            Number of steps in the analysis.

        Returns
        -------
        list of (delta2, player1_share, player2_share) tuples
        """
        results = []
        lo, hi = delta_range
        for i in range(steps + 1):
            d2 = lo + (hi - lo) * i / steps
            model = InfiniteHorizonBargaining(
                self.delta1, d2, self.breakdown_prob, self.surplus
            )
            s1, s2 = model.spe_shares()
            results.append((d2, s1, s2))
        return results
