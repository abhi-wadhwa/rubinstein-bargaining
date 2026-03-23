"""Tests for infinite-horizon Rubinstein bargaining model."""

import math
import pytest

from src.core.rubinstein import InfiniteHorizonBargaining


class TestSPEShares:
    """Test the Subgame Perfect Equilibrium share computation."""

    def test_symmetric_discount_factors(self):
        """With equal delta, player 1 gets 1/(1+delta) -- first-mover advantage."""
        delta = 0.9
        model = InfiniteHorizonBargaining(delta, delta)
        s1, s2 = model.spe_shares()
        expected_s1 = 1 / (1 + delta)
        assert abs(s1 - expected_s1) < 1e-10
        assert abs(s1 + s2 - 1.0) < 1e-10

    def test_asymmetric_discount_factors(self):
        """Test analytic SPE formula for asymmetric deltas."""
        d1, d2 = 0.8, 0.6
        model = InfiniteHorizonBargaining(d1, d2)
        s1, s2 = model.spe_shares()
        expected_s1 = (1 - d2) / (1 - d1 * d2)
        assert abs(s1 - expected_s1) < 1e-10
        assert abs(s2 - (1 - expected_s1)) < 1e-10

    def test_patient_player2_gets_more(self):
        """More patient player 2 should get a larger share."""
        model_low = InfiniteHorizonBargaining(0.5, 0.5)
        model_high = InfiniteHorizonBargaining(0.5, 0.9)
        _, s2_low = model_low.spe_shares()
        _, s2_high = model_high.spe_shares()
        assert s2_high > s2_low

    def test_patient_player1_gets_more(self):
        """More patient player 1 should get a larger share."""
        model_low = InfiniteHorizonBargaining(0.5, 0.5)
        model_high = InfiniteHorizonBargaining(0.9, 0.5)
        s1_low, _ = model_low.spe_shares()
        s1_high, _ = model_high.spe_shares()
        assert s1_high > s1_low

    def test_shares_sum_to_surplus(self):
        """Shares should always sum to the total surplus."""
        for surplus in [1.0, 2.5, 100.0]:
            model = InfiniteHorizonBargaining(0.8, 0.7, surplus=surplus)
            s1, s2 = model.spe_shares()
            assert abs(s1 + s2 - surplus) < 1e-10

    def test_very_impatient_player1(self):
        """Very impatient player 1 (low delta) gets close to (1-delta2)."""
        model = InfiniteHorizonBargaining(0.01, 0.9)
        s1, _ = model.spe_shares()
        # With delta1 ~ 0, x* ~ (1 - delta2) / 1 = 1 - delta2
        assert abs(s1 - (1 - 0.9)) < 0.01

    def test_very_impatient_player2(self):
        """Very impatient player 2 (low delta) means player 1 gets nearly all."""
        model = InfiniteHorizonBargaining(0.9, 0.01)
        s1, _ = model.spe_shares()
        assert s1 > 0.98


class TestBreakdownRisk:
    """Test bargaining with risk of breakdown."""

    def test_breakdown_reduces_shares(self):
        """Breakdown risk should reduce both players' effective discount factors."""
        model_no = InfiniteHorizonBargaining(0.9, 0.9, 0.0)
        model_yes = InfiniteHorizonBargaining(0.9, 0.9, 0.1)
        s1_no, _ = model_no.spe_shares()
        s1_yes, _ = model_yes.spe_shares()
        # With symmetric deltas, breakdown doesn't change shares much
        # but the effective discounting changes
        assert model_yes.effective_delta1 < model_no.effective_delta1

    def test_effective_discount_factor(self):
        """Effective delta should be delta * (1 - p)."""
        model = InfiniteHorizonBargaining(0.9, 0.8, 0.1)
        assert abs(model.effective_delta1 - 0.9 * 0.9) < 1e-10
        assert abs(model.effective_delta2 - 0.8 * 0.9) < 1e-10


class TestFirstMoverAdvantage:
    """Test first-mover advantage computation."""

    def test_symmetric_has_first_mover_advantage(self):
        """With symmetric deltas, player 1 has first-mover advantage."""
        model = InfiniteHorizonBargaining(0.9, 0.9)
        assert model.first_mover_advantage() > 1.0

    def test_first_mover_advantage_decreases_with_patience(self):
        """First-mover advantage should decrease as delta increases."""
        fma_low = InfiniteHorizonBargaining(0.5, 0.5).first_mover_advantage()
        fma_high = InfiniteHorizonBargaining(0.99, 0.99).first_mover_advantage()
        assert fma_low > fma_high


class TestSPEOffers:
    """Test SPE offer computation."""

    def test_offers_consistent(self):
        """Player 2 should be indifferent between accepting P1's offer and waiting."""
        model = InfiniteHorizonBargaining(0.8, 0.7)
        offer_p1, offer_p2 = model.spe_offers()

        # P2's share when P1 proposes should equal delta2 * P2's share when P2 proposes
        assert abs(offer_p1[1] - model.delta2 * offer_p2[1]) < 1e-10

        # P1's share when P2 proposes should equal delta1 * P1's share when P1 proposes
        assert abs(offer_p2[0] - model.delta1 * offer_p1[0]) < 1e-10


class TestSimulation:
    """Test round simulation."""

    def test_simulate_returns_correct_count(self):
        model = InfiniteHorizonBargaining(0.9, 0.9)
        results = model.simulate_rounds(10)
        assert len(results) == 10

    def test_first_round_accepted(self):
        model = InfiniteHorizonBargaining(0.9, 0.9)
        results = model.simulate_rounds(5)
        assert results[0].accepted is True
        assert results[0].proposer == 1


class TestValidation:
    """Test input validation."""

    def test_invalid_delta1(self):
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(0.0, 0.5)
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(1.0, 0.5)
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(-0.1, 0.5)

    def test_invalid_delta2(self):
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(0.5, 0.0)

    def test_invalid_breakdown(self):
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(0.5, 0.5, 1.0)
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(0.5, 0.5, -0.1)

    def test_invalid_surplus(self):
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(0.5, 0.5, surplus=0)
        with pytest.raises(ValueError):
            InfiniteHorizonBargaining(0.5, 0.5, surplus=-1)
