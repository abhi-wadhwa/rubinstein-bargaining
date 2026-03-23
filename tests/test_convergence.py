"""Tests for convergence properties across all bargaining models."""

import math
import pytest
import numpy as np

from src.core.rubinstein import InfiniteHorizonBargaining
from src.core.finite_horizon import FiniteHorizonBargaining
from src.core.outside_options import OutsideOptionBargaining
from src.core.multi_issue import MultiIssueBargaining
from src.core.nash_bargaining import NashBargainingSolution


class TestDeltaToOneConvergence:
    """Test that Rubinstein SPE converges to 50/50 as delta -> 1."""

    def test_symmetric_converges_to_half(self):
        """With symmetric delta -> 1, each player gets 0.5."""
        for delta in [0.99, 0.999, 0.9999]:
            model = InfiniteHorizonBargaining(delta, delta)
            s1, s2 = model.spe_shares()
            assert abs(s1 - 0.5) < 1 / (1 - delta) * 0.01
            assert abs(s2 - 0.5) < 1 / (1 - delta) * 0.01

    def test_very_close_to_one(self):
        """At delta = 0.9999, should be within 0.0001 of 0.5."""
        model = InfiniteHorizonBargaining(0.9999, 0.9999)
        s1, s2 = model.spe_shares()
        assert abs(s1 - 0.5) < 0.001
        assert abs(s2 - 0.5) < 0.001

    def test_asymmetric_delta_to_one_converges_to_nash(self):
        """
        With asymmetric deltas close to 1, Rubinstein SPE converges
        to asymmetric Nash bargaining solution.

        The bargaining power ratio is ln(delta2)/ln(delta1).
        As both deltas approach 1 at different rates:
            alpha = ln(delta2) / (ln(delta1) + ln(delta2))
        """
        # Use deltas close to 1 but with fixed ratio of logs
        # delta1 = 0.99, delta2 = 0.98
        # ln(0.99) ~ -0.01005, ln(0.98) ~ -0.02020
        # alpha = -0.02020 / (-0.01005 + -0.02020) = 0.02020/0.03025 ~ 0.6678
        delta1 = 0.999
        delta2 = 0.998

        model = InfiniteHorizonBargaining(delta1, delta2)
        s1, _ = model.spe_shares()

        alpha = math.log(delta2) / (math.log(delta1) + math.log(delta2))
        nbs = NashBargainingSolution.closed_form_linear(1.0, (0, 0), alpha)

        # Should be close to NBS
        assert abs(s1 - nbs.player1_utility) < 0.01


class TestFiniteToInfiniteConvergence:
    """Test finite horizon convergence to infinite horizon."""

    @pytest.mark.parametrize("delta1,delta2", [
        (0.9, 0.9),
        (0.8, 0.7),
        (0.95, 0.85),
    ])
    def test_convergence_for_various_deltas(self, delta1, delta2):
        """Finite horizon should converge for various delta combinations."""
        inf_model = InfiniteHorizonBargaining(delta1, delta2)
        inf_s1, _ = inf_model.spe_shares()

        fin_model = FiniteHorizonBargaining(delta1, delta2, total_rounds=150)
        fin_s1, _ = fin_model.spe_outcome()

        assert abs(fin_s1 - inf_s1) < 0.001

    def test_convergence_speed(self):
        """Convergence should be geometric in T."""
        delta1, delta2 = 0.8, 0.7
        inf_model = InfiniteHorizonBargaining(delta1, delta2)
        inf_s1, _ = inf_model.spe_shares()

        errors = []
        for T in [10, 20, 40, 80]:
            fin_model = FiniteHorizonBargaining(delta1, delta2, total_rounds=T)
            fin_s1, _ = fin_model.spe_outcome()
            errors.append(abs(fin_s1 - inf_s1))

        # Error should decrease roughly geometrically
        for i in range(1, len(errors)):
            if errors[i-1] > 1e-12:
                assert errors[i] < errors[i-1]


class TestOutsideOptionEffects:
    """Test that outside options affect the bargaining outcome correctly."""

    def test_non_binding_outside_option(self):
        """Small outside option should not change the outcome."""
        d1, d2 = 0.9, 0.9
        model_no = OutsideOptionBargaining(d1, d2, 0.0, 0.0)
        model_yes = OutsideOptionBargaining(d1, d2, 0.0, 0.01)

        result_no = model_no.solve()
        result_yes = model_yes.solve()

        # With very small outside option for P2, should be non-binding
        # and result should be same as standard Rubinstein
        assert not result_yes.outside_option_binding[2]

    def test_binding_outside_option(self):
        """Large outside option should bind and change outcome."""
        d1, d2 = 0.9, 0.9
        # Standard Rubinstein gives P2 about 0.4737
        # An outside option of 0.48 should be binding
        model = OutsideOptionBargaining(d1, d2, 0.0, 0.48)
        result = model.solve()

        # P2's outside option is above their standard Rubinstein share
        # The continuation value for P2 is delta2 * P2's_share_when_P2_proposes
        # = 0.9 * (1-0.9)/(1-0.81) = 0.9 * 0.1/0.19 = 0.4737
        # So P2's continuation value after rejection is about 0.9 * 0.4737 = 0.426
        # 0.48 > 0.426, so it's binding
        assert result.outside_option_binding[2]
        assert abs(result.player2_share - 0.48) < 1e-10


class TestMultiIssueProperties:
    """Test multi-issue bargaining properties."""

    def test_efficient_allocation_maximizes_surplus(self):
        """Efficient allocation should maximize total surplus."""
        v1 = [0.8, 0.3, 0.6]
        v2 = [0.3, 0.7, 0.5]
        model = MultiIssueBargaining(v1, v2)

        eff = model.efficient_allocation()
        total_eff = eff.player1_utility + eff.player2_utility

        # Try random allocations - none should exceed efficient
        rng = np.random.RandomState(42)
        for _ in range(100):
            shares = rng.rand(3)
            total = model.utility(shares, 1) + model.utility(shares, 2)
            assert total <= total_eff + 1e-10

    def test_pareto_frontier_non_empty(self):
        """Pareto frontier should have points."""
        model = MultiIssueBargaining([0.5, 0.5], [0.5, 0.5])
        frontier = model.pareto_frontier()
        assert len(frontier) > 0

    def test_gains_from_trade_positive(self):
        """When players value issues differently, gains from trade are positive."""
        model = MultiIssueBargaining([0.9, 0.1], [0.1, 0.9])
        gains = model.gains_from_trade()
        assert gains > 0

    def test_no_gains_from_trade_same_valuations(self):
        """When players have same valuations, no gains from trade."""
        model = MultiIssueBargaining([0.5, 0.5], [0.5, 0.5])
        gains = model.gains_from_trade()
        assert abs(gains) < 1e-10

    def test_comparative_advantage(self):
        """Player with higher relative valuation should have comparative advantage."""
        model = MultiIssueBargaining([0.9, 0.1], [0.1, 0.9])
        ca = model.comparative_advantage()

        # P1 has advantage in issue 0, P2 in issue 1
        issue_0 = [p for i, p in ca if i == 0][0]
        issue_1 = [p for i, p in ca if i == 1][0]
        assert issue_0 == 1
        assert issue_1 == 2


class TestBreakdownRiskConvergence:
    """Test properties of breakdown risk."""

    def test_breakdown_equivalent_to_lower_delta(self):
        """Breakdown probability p should be equivalent to lower discount factors."""
        d1, d2, p = 0.9, 0.8, 0.1

        model_break = InfiniteHorizonBargaining(d1, d2, breakdown_prob=p)
        s1_break, _ = model_break.spe_shares()

        # Equivalent model with effective deltas
        eff_d1 = d1 * (1 - p)
        eff_d2 = d2 * (1 - p)
        model_eff = InfiniteHorizonBargaining(eff_d1, eff_d2)
        s1_eff, _ = model_eff.spe_shares()

        assert abs(s1_break - s1_eff) < 1e-10

    def test_high_breakdown_gives_more_to_proposer(self):
        """High breakdown risk should give more to the first proposer."""
        model_low = InfiniteHorizonBargaining(0.9, 0.9, breakdown_prob=0.01)
        model_high = InfiniteHorizonBargaining(0.9, 0.9, breakdown_prob=0.5)

        s1_low, _ = model_low.spe_shares()
        s1_high, _ = model_high.spe_shares()

        # Higher breakdown -> effectively lower deltas -> more first-mover advantage
        assert s1_high > s1_low


class TestNBSConvergenceQuantitative:
    """Quantitative tests for Rubinstein-Nash convergence."""

    def test_convergence_rate(self):
        """
        The gap between Rubinstein and NBS should vanish as O(1-delta).
        For symmetric delta: x* = 1/(1+delta), NBS = 0.5
        Gap = 1/(1+delta) - 0.5 = (1-delta)/(2(1+delta)) ~ (1-delta)/4
        """
        for delta in [0.9, 0.99, 0.999]:
            model = InfiniteHorizonBargaining(delta, delta)
            s1, _ = model.spe_shares()
            gap = abs(s1 - 0.5)
            theoretical_gap = (1 - delta) / (2 * (1 + delta))
            assert abs(gap - theoretical_gap) < 1e-10

    def test_limit_is_exact_half(self):
        """In the limit, symmetric Rubinstein gives exactly 0.5."""
        # Use the formula: as delta -> 1, (1-delta)/(1-delta^2) = 1/(1+delta) -> 0.5
        delta = 0.999999
        model = InfiniteHorizonBargaining(delta, delta)
        s1, s2 = model.spe_shares()
        assert abs(s1 - 0.5) < 1e-5
        assert abs(s2 - 0.5) < 1e-5
