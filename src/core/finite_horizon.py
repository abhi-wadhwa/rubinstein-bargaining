"""
Finite-horizon Rubinstein bargaining model.

T rounds of alternating offers. If no agreement by round T, both get 0.
Solved by backward induction: at each round, the proposer offers just
enough for the responder to accept (indifference condition).

As T -> infinity, the finite-horizon solution converges to the
infinite-horizon SPE.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FiniteRoundResult:
    """Backward-induction result for a single round."""

    round_number: int
    proposer: int  # 1 or 2
    player1_share: float
    player2_share: float

    def __repr__(self) -> str:
        return (
            f"Round {self.round_number}: Player {self.proposer} proposes "
            f"({self.player1_share:.6f}, {self.player2_share:.6f})"
        )


class FiniteHorizonBargaining:
    """
    Finite-horizon alternating-offers bargaining model.

    Parameters
    ----------
    delta1 : float
        Discount factor for player 1, in (0, 1).
    delta2 : float
        Discount factor for player 2, in (0, 1).
    total_rounds : int
        Total number of bargaining rounds (T >= 1).
    surplus : float
        Total surplus to be divided (default 1.0).
    first_proposer : int
        Player who proposes in round 1 (1 or 2, default 1).
    """

    def __init__(
        self,
        delta1: float,
        delta2: float,
        total_rounds: int,
        surplus: float = 1.0,
        first_proposer: int = 1,
    ):
        if not (0 < delta1 < 1):
            raise ValueError(f"delta1 must be in (0, 1), got {delta1}")
        if not (0 < delta2 < 1):
            raise ValueError(f"delta2 must be in (0, 1), got {delta2}")
        if total_rounds < 1:
            raise ValueError(
                f"total_rounds must be >= 1, got {total_rounds}"
            )
        if first_proposer not in (1, 2):
            raise ValueError(
                f"first_proposer must be 1 or 2, got {first_proposer}"
            )
        if surplus <= 0:
            raise ValueError(f"surplus must be positive, got {surplus}")

        self.delta1 = delta1
        self.delta2 = delta2
        self.total_rounds = total_rounds
        self.surplus = surplus
        self.first_proposer = first_proposer

    def _proposer_at_round(self, t: int) -> int:
        """Return the proposer at round t (1-indexed)."""
        if (t - 1) % 2 == 0:
            return self.first_proposer
        return 3 - self.first_proposer  # The other player

    def backward_induction(self) -> List[FiniteRoundResult]:
        """
        Solve the finite-horizon game by backward induction.

        At the last round T, the proposer takes everything (responder
        gets 0 in present value, so accepts anything >= 0).

        At round t < T, the proposer must offer the responder at least
        delta_responder * (responder's continuation value at t+1),
        discounted to period t's terms.

        Returns
        -------
        list of FiniteRoundResult
            Results for each round from T down to 1, reversed to be
            in chronological order.
        """
        T = self.total_rounds
        results = []

        # continuation_value[i] = player i's continuation value at
        # the start of the next round (in that round's terms)
        # At round T+1 (if it existed), both get 0.
        continuation = {1: 0.0, 2: 0.0}

        for t in range(T, 0, -1):
            proposer = self._proposer_at_round(t)
            responder = 3 - proposer

            # Responder's discount factor
            delta_r = self.delta1 if responder == 1 else self.delta2

            # Responder requires at least delta_r * continuation[responder]
            # (their discounted continuation value)
            responder_min = delta_r * continuation[responder]

            # Proposer takes the rest
            proposer_share = 1.0 - responder_min

            if proposer == 1:
                p1_share = proposer_share
                p2_share = responder_min
            else:
                p1_share = responder_min
                p2_share = proposer_share

            results.append(
                FiniteRoundResult(
                    round_number=t,
                    proposer=proposer,
                    player1_share=p1_share * self.surplus,
                    player2_share=p2_share * self.surplus,
                )
            )

            # Update continuation values for the round before this one.
            # At round t, if the offer is accepted:
            #   proposer gets proposer_share, responder gets responder_min
            continuation[proposer] = proposer_share
            continuation[responder] = responder_min

        # Reverse to chronological order
        results.reverse()
        return results

    def spe_outcome(self) -> Tuple[float, float]:
        """
        Compute the SPE outcome (agreement in round 1).

        Returns
        -------
        tuple of (float, float)
            (player1_share, player2_share) in the equilibrium.
        """
        results = self.backward_induction()
        return (results[0].player1_share, results[0].player2_share)

    def convergence_to_infinite(
        self, max_T: int = 100
    ) -> List[Tuple[int, float, float]]:
        """
        Show convergence of finite-horizon SPE to infinite-horizon as T grows.

        Parameters
        ----------
        max_T : int
            Maximum number of rounds to compute.

        Returns
        -------
        list of (T, player1_share, player2_share) tuples
        """
        results = []
        for T in range(1, max_T + 1):
            model = FiniteHorizonBargaining(
                self.delta1,
                self.delta2,
                T,
                self.surplus,
                self.first_proposer,
            )
            s1, s2 = model.spe_outcome()
            results.append((T, s1, s2))
        return results

    def game_tree_data(self) -> dict:
        """
        Generate game tree data for visualization.

        Returns
        -------
        dict
            Nested dictionary representing the game tree with
            offers and accept/reject branches.
        """
        bi_results = self.backward_induction()

        def _build_node(round_idx: int) -> dict:
            if round_idx >= len(bi_results):
                return {"type": "terminal", "payoff": (0.0, 0.0)}

            r = bi_results[round_idx]
            node = {
                "type": "proposal",
                "round": r.round_number,
                "proposer": r.proposer,
                "offer": (r.player1_share, r.player2_share),
                "accept": {
                    "type": "terminal",
                    "payoff": (r.player1_share, r.player2_share),
                },
                "reject": _build_node(round_idx + 1),
            }
            return node

        return _build_node(0)
