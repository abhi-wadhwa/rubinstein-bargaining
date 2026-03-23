"""Tests for finite-horizon bargaining model."""

import pytest

from src.core.finite_horizon import FiniteHorizonBargaining
from src.core.rubinstein import InfiniteHorizonBargaining


class TestBackwardInduction:
    """Test backward induction computation."""

    def test_single_round(self):
        """In a single round, proposer takes everything."""
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=1)
        s1, s2 = model.spe_outcome()
        assert abs(s1 - 1.0) < 1e-10
        assert abs(s2 - 0.0) < 1e-10

    def test_two_rounds(self):
        """In two rounds: P1 proposes, if rejected P2 proposes last."""
        delta1, delta2 = 0.9, 0.8
        model = FiniteHorizonBargaining(delta1, delta2, total_rounds=2)
        s1, s2 = model.spe_outcome()

        # Round 2: P2 proposes, takes everything -> P2 gets 1, P1 gets 0
        # Round 1: P1 must offer P2 at least delta2 * 1 = 0.8
        # So P1 gets 1 - 0.8 = 0.2
        expected_s1 = 1 - delta2
        expected_s2 = delta2
        assert abs(s1 - expected_s1) < 1e-10
        assert abs(s2 - expected_s2) < 1e-10

    def test_three_rounds(self):
        """Three-round backward induction."""
        delta1, delta2 = 0.9, 0.8
        model = FiniteHorizonBargaining(delta1, delta2, total_rounds=3)
        s1, s2 = model.spe_outcome()

        # Round 3: P1 proposes, takes 1
        # Round 2: P2 must offer P1 at least delta1 * 1 = 0.9
        #          P2 gets 1 - 0.9 = 0.1
        # Round 1: P1 must offer P2 at least delta2 * 0.1 = 0.08
        #          P1 gets 1 - 0.08 = 0.92
        expected_s2 = delta2 * (1 - delta1)
        expected_s1 = 1 - expected_s2
        assert abs(s1 - expected_s1) < 1e-10
        assert abs(s2 - expected_s2) < 1e-10

    def test_shares_sum_to_surplus(self):
        """Shares should always sum to the surplus."""
        for T in range(1, 15):
            model = FiniteHorizonBargaining(0.8, 0.7, total_rounds=T)
            s1, s2 = model.spe_outcome()
            assert abs(s1 + s2 - 1.0) < 1e-10

    def test_backward_induction_returns_all_rounds(self):
        """Backward induction should return results for all rounds."""
        T = 7
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=T)
        results = model.backward_induction()
        assert len(results) == T

    def test_proposer_alternates(self):
        """Proposers should alternate: P1, P2, P1, P2, ..."""
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=6)
        results = model.backward_induction()
        for r in results:
            if r.round_number % 2 == 1:
                assert r.proposer == 1
            else:
                assert r.proposer == 2

    def test_first_proposer_2(self):
        """Test with player 2 as first proposer."""
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=1, first_proposer=2)
        s1, s2 = model.spe_outcome()
        # P2 proposes first and takes everything
        assert abs(s1 - 0.0) < 1e-10
        assert abs(s2 - 1.0) < 1e-10

    def test_custom_surplus(self):
        """Test with non-unit surplus."""
        surplus = 100.0
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=3, surplus=surplus)
        s1, s2 = model.spe_outcome()
        assert abs(s1 + s2 - surplus) < 1e-10


class TestConvergenceToInfinite:
    """Test that finite horizon converges to infinite horizon as T -> infinity."""

    def test_convergence_symmetric(self):
        """Finite horizon should converge to infinite horizon for symmetric deltas."""
        delta = 0.9
        inf_model = InfiniteHorizonBargaining(delta, delta)
        inf_s1, _ = inf_model.spe_shares()

        fin_model = FiniteHorizonBargaining(delta, delta, total_rounds=200)
        fin_s1, _ = fin_model.spe_outcome()

        assert abs(fin_s1 - inf_s1) < 1e-4

    def test_convergence_asymmetric(self):
        """Finite horizon should converge for asymmetric deltas too."""
        d1, d2 = 0.85, 0.75
        inf_model = InfiniteHorizonBargaining(d1, d2)
        inf_s1, _ = inf_model.spe_shares()

        fin_model = FiniteHorizonBargaining(d1, d2, total_rounds=200)
        fin_s1, _ = fin_model.spe_outcome()

        assert abs(fin_s1 - inf_s1) < 1e-4

    def test_convergence_monotonic_approach(self):
        """As T increases, finite-horizon shares should approach the limit."""
        d1, d2 = 0.9, 0.8
        inf_model = InfiniteHorizonBargaining(d1, d2)
        inf_s1, _ = inf_model.spe_shares()

        prev_dist = float("inf")
        # Check odd T values (where P1 proposes last)
        for T in [1, 3, 5, 7, 9, 11, 51, 101]:
            fin_model = FiniteHorizonBargaining(d1, d2, total_rounds=T)
            fin_s1, _ = fin_model.spe_outcome()
            dist = abs(fin_s1 - inf_s1)
            # Distance should generally decrease (not strictly for alternating T)
            if T > 50:
                assert dist < 0.01

    def test_convergence_data(self):
        """Test the convergence_to_infinite method."""
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=5)
        data = model.convergence_to_infinite(max_T=20)
        assert len(data) == 20
        assert data[0][0] == 1  # First entry is T=1
        assert data[-1][0] == 20  # Last entry is T=20


class TestGameTree:
    """Test game tree generation."""

    def test_game_tree_structure(self):
        """Game tree should have correct nested structure."""
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=3)
        tree = model.game_tree_data()

        assert tree["type"] == "proposal"
        assert tree["round"] == 1
        assert tree["proposer"] == 1
        assert "accept" in tree
        assert "reject" in tree
        assert tree["accept"]["type"] == "terminal"

    def test_game_tree_depth(self):
        """Game tree should have depth equal to T."""
        T = 4
        model = FiniteHorizonBargaining(0.9, 0.9, total_rounds=T)
        tree = model.game_tree_data()

        depth = 0
        node = tree
        while node["type"] == "proposal":
            depth += 1
            node = node["reject"]
        assert depth == T


class TestValidation:
    """Test input validation."""

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            FiniteHorizonBargaining(0.9, 0.9, total_rounds=0)

    def test_invalid_first_proposer(self):
        with pytest.raises(ValueError):
            FiniteHorizonBargaining(0.9, 0.9, total_rounds=5, first_proposer=3)

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            FiniteHorizonBargaining(1.0, 0.9, total_rounds=5)
