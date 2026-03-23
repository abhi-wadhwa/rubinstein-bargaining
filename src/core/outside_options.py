"""
Rubinstein bargaining with outside options.

Each player has a fallback payoff (outside option) they can take
instead of continuing to bargain. The outside option changes the
equilibrium when it is sufficiently attractive.

Three variants:
1. Always-available outside option: player can opt out at any time
2. After-rejection outside option: player can opt out only after
   rejecting an offer
3. Mutual outside options: both players have outside options
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OutsideOptionResult:
    """Result of bargaining with outside options."""

    player1_share: float
    player2_share: float
    outside_option_binding: dict  # {1: bool, 2: bool}
    agreement_type: str  # "negotiated" or "outside_option_player_X"

    def __repr__(self) -> str:
        binding = ", ".join(
            f"Player {k}: {'binding' if v else 'non-binding'}"
            for k, v in self.outside_option_binding.items()
        )
        return (
            f"Outcome: ({self.player1_share:.4f}, {self.player2_share:.4f}) "
            f"[{self.agreement_type}] ({binding})"
        )


class OutsideOptionBargaining:
    """
    Rubinstein bargaining with outside options.

    Parameters
    ----------
    delta1 : float
        Discount factor for player 1, in (0, 1).
    delta2 : float
        Discount factor for player 2, in (0, 1).
    outside1 : float
        Player 1's outside option payoff (default 0.0).
    outside2 : float
        Player 2's outside option payoff (default 0.0).
    surplus : float
        Total surplus to be divided (default 1.0).
    breakdown_prob : float
        Probability of exogenous breakdown each round.

    Notes
    -----
    The outside option is "binding" for player i if the standard
    Rubinstein outcome gives player i less than their outside option.
    When binding, the equilibrium adjusts to give that player exactly
    their outside option (if feasible).
    """

    def __init__(
        self,
        delta1: float,
        delta2: float,
        outside1: float = 0.0,
        outside2: float = 0.0,
        surplus: float = 1.0,
        breakdown_prob: float = 0.0,
    ):
        if not (0 < delta1 < 1):
            raise ValueError(f"delta1 must be in (0, 1), got {delta1}")
        if not (0 < delta2 < 1):
            raise ValueError(f"delta2 must be in (0, 1), got {delta2}")
        if outside1 < 0:
            raise ValueError(f"outside1 must be >= 0, got {outside1}")
        if outside2 < 0:
            raise ValueError(f"outside2 must be >= 0, got {outside2}")
        if outside1 + outside2 > surplus:
            raise ValueError(
                f"Sum of outside options ({outside1 + outside2}) "
                f"exceeds surplus ({surplus})"
            )

        self.delta1 = delta1
        self.delta2 = delta2
        self.outside1 = outside1
        self.outside2 = outside2
        self.surplus = surplus
        self.breakdown_prob = breakdown_prob

    def _standard_rubinstein_shares(self) -> Tuple[float, float]:
        """Compute standard Rubinstein SPE shares without outside options."""
        d1 = self.delta1 * (1 - self.breakdown_prob)
        d2 = self.delta2 * (1 - self.breakdown_prob)
        denom = 1 - d1 * d2
        x = (1 - d2) / denom
        return x * self.surplus, (1 - x) * self.surplus

    def solve_after_rejection(self) -> OutsideOptionResult:
        """
        Solve with after-rejection outside options (Shaked-Sutton model).

        A player can exercise the outside option only after rejecting
        an offer. The outside option is binding only if it exceeds
        the player's continuation value.

        In the standard Rubinstein model, player 2's continuation
        value when rejecting in round 1 is delta2 * y* where y* is
        player 2's share when player 2 proposes.

        The outside option is binding for player 2 if:
            outside2 > delta2 * y*  (where y* is P2's share when P2 proposes)

        When player 2's outside option is binding:
            Player 1 offers player 2 exactly outside2
            Player 1 keeps (surplus - outside2)

        Similarly for player 1.
        """
        s1_std, s2_std = self._standard_rubinstein_shares()

        d1 = self.delta1 * (1 - self.breakdown_prob)
        d2 = self.delta2 * (1 - self.breakdown_prob)

        # Player 2's continuation value after rejecting P1's offer
        # In standard Rubinstein: P2 gets (1-d1)/(1-d1*d2) when P2 proposes
        denom = 1 - d1 * d2
        p2_continuation = d2 * (1 - d1) / denom * self.surplus

        # Player 1's continuation value after rejecting P2's offer
        p1_continuation = d1 * (1 - d2) / denom * self.surplus

        binding = {1: False, 2: False}
        p1_share = s1_std
        p2_share = s2_std
        agreement_type = "negotiated"

        # Check if outside options are binding
        if self.outside2 > p2_continuation:
            # P2's outside option is binding
            binding[2] = True
            p2_share = self.outside2
            p1_share = self.surplus - self.outside2
            agreement_type = "negotiated_with_binding_outside_p2"

        if self.outside1 > p1_continuation:
            # P1's outside option is binding
            binding[1] = True
            p1_share = self.outside1
            p2_share = self.surplus - self.outside1
            agreement_type = "negotiated_with_binding_outside_p1"

        # Both binding: check feasibility
        if binding[1] and binding[2]:
            if self.outside1 + self.outside2 <= self.surplus:
                # Both get their outside options, surplus remainder split
                p1_share = self.outside1
                p2_share = self.outside2
                remainder = self.surplus - self.outside1 - self.outside2
                # Split remainder according to standard Rubinstein proportions
                total_std = s1_std + s2_std
                if total_std > 0:
                    p1_share += remainder * s1_std / total_std
                    p2_share += remainder * s2_std / total_std
                agreement_type = "negotiated_with_both_binding"
            else:
                agreement_type = "no_agreement_feasible"

        return OutsideOptionResult(
            player1_share=p1_share,
            player2_share=p2_share,
            outside_option_binding=binding,
            agreement_type=agreement_type,
        )

    def solve_always_available(self) -> OutsideOptionResult:
        """
        Solve with always-available outside options.

        When outside options are always available, a player can opt out
        at any point (even when it's their turn to propose). This gives
        each player a minimum payoff guarantee.

        In this case, we solve the modified Rubinstein equations:
            x1 = max(1 - delta2 * x2, outside1)  when P1 proposes
            x2 = max(1 - delta1 * x1, outside2)  when P2 proposes
        where x1 is P1's share when P1 proposes and x2 is P2's share
        when P2 proposes.
        """
        d1 = self.delta1 * (1 - self.breakdown_prob)
        d2 = self.delta2 * (1 - self.breakdown_prob)

        # Standard shares (proportions of surplus)
        denom = 1 - d1 * d2
        x1_std = (1 - d2) / denom  # P1's share when P1 proposes
        x2_std = (1 - d1) / denom  # P2's share when P2 proposes

        o1 = self.outside1 / self.surplus if self.surplus > 0 else 0
        o2 = self.outside2 / self.surplus if self.surplus > 0 else 0

        binding = {1: False, 2: False}

        # Iterative solution for the coupled equations
        x1 = x1_std
        x2 = x2_std

        for _ in range(1000):
            x1_new = max(1 - d2 * x2, o1)
            x2_new = max(1 - d1 * x1_new, o2)

            if abs(x1_new - x1) < 1e-12 and abs(x2_new - x2) < 1e-12:
                break
            x1 = x1_new
            x2 = x2_new

        binding[1] = x1 > x1_std + 1e-10
        binding[2] = x2 > x2_std + 1e-10

        p1_share = x1 * self.surplus
        p2_share = (1 - x1) * self.surplus

        agreement_type = "negotiated"
        if binding[1] and binding[2]:
            agreement_type = "negotiated_with_both_binding"
        elif binding[1]:
            agreement_type = "negotiated_with_binding_outside_p1"
        elif binding[2]:
            agreement_type = "negotiated_with_binding_outside_p2"

        return OutsideOptionResult(
            player1_share=p1_share,
            player2_share=p2_share,
            outside_option_binding=binding,
            agreement_type=agreement_type,
        )

    def solve(
        self, variant: str = "after_rejection"
    ) -> OutsideOptionResult:
        """
        Solve the bargaining problem with outside options.

        Parameters
        ----------
        variant : str
            "after_rejection" or "always_available".

        Returns
        -------
        OutsideOptionResult
        """
        if variant == "after_rejection":
            return self.solve_after_rejection()
        elif variant == "always_available":
            return self.solve_always_available()
        else:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                "Use 'after_rejection' or 'always_available'."
            )

    def sensitivity_analysis(
        self,
        player: int = 2,
        outside_range: Tuple[float, float] = (0.0, 0.49),
        steps: int = 50,
        variant: str = "after_rejection",
    ) -> list:
        """
        Analyze how the outcome changes as a player's outside option varies.

        Parameters
        ----------
        player : int
            Which player's outside option to vary (1 or 2).
        outside_range : tuple
            (min, max) range for the outside option.
        steps : int
            Number of steps.
        variant : str
            Which variant to use.

        Returns
        -------
        list of (outside_option_value, player1_share, player2_share, binding)
        """
        results = []
        lo, hi = outside_range
        for i in range(steps + 1):
            o = lo + (hi - lo) * i / steps
            if player == 2:
                model = OutsideOptionBargaining(
                    self.delta1, self.delta2, self.outside1, o,
                    self.surplus, self.breakdown_prob,
                )
            else:
                model = OutsideOptionBargaining(
                    self.delta1, self.delta2, o, self.outside2,
                    self.surplus, self.breakdown_prob,
                )
            result = model.solve(variant)
            results.append((
                o,
                result.player1_share,
                result.player2_share,
                result.outside_option_binding[player],
            ))
        return results
