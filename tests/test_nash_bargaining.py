"""Tests for Nash Bargaining Solution."""

import pytest
import numpy as np

from src.core.nash_bargaining import NashBargainingSolution


class TestClosedFormLinear:
    """Test closed-form NBS for linear utility frontier."""

    def test_symmetric_equal_split(self):
        """Symmetric NBS with d=(0,0) should give 50/50."""
        result = NashBargainingSolution.closed_form_linear(1.0, (0, 0), 0.5)
        assert abs(result.player1_utility - 0.5) < 1e-10
        assert abs(result.player2_utility - 0.5) < 1e-10

    def test_asymmetric_power(self):
        """Asymmetric NBS: P1 should get alpha * surplus."""
        alpha = 0.7
        result = NashBargainingSolution.closed_form_linear(1.0, (0, 0), alpha)
        assert abs(result.player1_utility - alpha) < 1e-10
        assert abs(result.player2_utility - (1 - alpha)) < 1e-10

    def test_nonzero_disagreement(self):
        """NBS with disagreement point should give d_i + alpha_i * (S - d1 - d2)."""
        d1, d2 = 0.1, 0.2
        surplus = 1.0
        alpha = 0.5
        result = NashBargainingSolution.closed_form_linear(surplus, (d1, d2), alpha)

        pie = surplus - d1 - d2
        expected_u1 = d1 + alpha * pie
        expected_u2 = d2 + (1 - alpha) * pie

        assert abs(result.player1_utility - expected_u1) < 1e-10
        assert abs(result.player2_utility - expected_u2) < 1e-10

    def test_shares_sum_to_surplus(self):
        """NBS utilities should sum to the surplus."""
        result = NashBargainingSolution.closed_form_linear(2.0, (0.1, 0.3), 0.6)
        assert abs(result.player1_utility + result.player2_utility - 2.0) < 1e-10

    def test_individual_rationality(self):
        """Each player should get at least their disagreement payoff."""
        d1, d2 = 0.15, 0.25
        result = NashBargainingSolution.closed_form_linear(1.0, (d1, d2), 0.4)
        assert result.player1_utility >= d1 - 1e-10
        assert result.player2_utility >= d2 - 1e-10


class TestNumericalSolver:
    """Test the numerical NBS solver."""

    def test_linear_frontier_matches_closed_form(self):
        """Numerical solver should match closed-form for linear frontier."""
        nbs = NashBargainingSolution.from_linear_frontier(1.0, 1.0, (0, 0), 0.5)
        result = nbs.solve()

        cf = NashBargainingSolution.closed_form_linear(1.0, (0, 0), 0.5)

        assert abs(result.player1_utility - cf.player1_utility) < 0.01
        assert abs(result.player2_utility - cf.player2_utility) < 0.01

    def test_asymmetric_frontier(self):
        """Test with asymmetric frontier (max_u1 != max_u2)."""
        nbs = NashBargainingSolution.from_linear_frontier(2.0, 1.0, (0, 0), 0.5)
        result = nbs.solve()

        # On the frontier u1/2 + u2/1 = 1
        # Maximize u1 * u2 => u1 = 1, u2 = 0.5 (for symmetric alpha)
        assert result.player1_utility > 0
        assert result.player2_utility > 0

    def test_custom_frontier(self):
        """Test with custom feasible set."""
        # Quarter circle frontier: u1^2 + u2^2 = 1
        frontier = []
        for i in range(1001):
            t = i / 1000 * np.pi / 2
            frontier.append((np.cos(t), np.sin(t)))

        nbs = NashBargainingSolution(frontier, (0, 0), 0.5)
        result = nbs.solve()

        # For symmetric NBS on quarter circle, solution is at 45 degrees
        assert abs(result.player1_utility - result.player2_utility) < 0.05

    def test_nash_product_positive(self):
        """Nash product should be positive for interior solutions."""
        result = NashBargainingSolution.closed_form_linear(1.0, (0, 0), 0.5)
        assert result.nash_product > 0


class TestAxiomVerification:
    """Test that NBS satisfies the Nash axioms."""

    def test_symmetric_problem_symmetric_solution(self):
        """Symmetric problem should have symmetric solution."""
        nbs = NashBargainingSolution.from_linear_frontier(1.0, 1.0, (0, 0), 0.5)
        result = nbs.solve()
        axioms = nbs.verify_axioms(result)

        assert axioms["symmetry"]["satisfied"]

    def test_pareto_efficiency(self):
        """Solution should be Pareto efficient."""
        nbs = NashBargainingSolution.from_linear_frontier(1.0, 1.0, (0, 0), 0.5)
        result = nbs.solve()
        axioms = nbs.verify_axioms(result)

        assert axioms["pareto_efficiency"]["satisfied"]

    def test_individual_rationality_axiom(self):
        """Solution should satisfy individual rationality."""
        nbs = NashBargainingSolution.from_linear_frontier(
            1.0, 1.0, (0.1, 0.2), 0.5
        )
        result = nbs.solve()
        axioms = nbs.verify_axioms(result)

        assert axioms["individual_rationality"]["satisfied"]

    def test_nash_product_maximized(self):
        """Solution should maximize the Nash product."""
        nbs = NashBargainingSolution.from_linear_frontier(1.0, 1.0, (0, 0), 0.5)
        result = nbs.solve()
        axioms = nbs.verify_axioms(result)

        assert axioms["nash_product_maximized"]["satisfied"]

    def test_all_axioms_satisfied(self):
        """All four axioms should be satisfied simultaneously."""
        for alpha in [0.3, 0.5, 0.7]:
            for d1, d2 in [(0, 0), (0.1, 0.1), (0.05, 0.15)]:
                nbs = NashBargainingSolution.from_linear_frontier(
                    1.0, 1.0, (d1, d2), alpha
                )
                result = nbs.solve()
                axioms = nbs.verify_axioms(result)

                for axiom_name, check in axioms.items():
                    assert check["satisfied"], (
                        f"Axiom {axiom_name} failed for alpha={alpha}, d=({d1},{d2})"
                    )


class TestRubinsteinConvergence:
    """Test convergence of Rubinstein to Nash."""

    def test_convergence_to_half(self):
        """Symmetric Rubinstein should converge to 0.5 as delta -> 1."""
        convergence = NashBargainingSolution.rubinstein_convergence()

        # Last entry should be very close to 0.5
        last = convergence[-1]
        assert abs(last[1] - 0.5) < 0.001

    def test_convergence_monotonic(self):
        """Player 1's share should monotonically decrease toward 0.5."""
        convergence = NashBargainingSolution.rubinstein_convergence()

        for i in range(1, len(convergence)):
            # As delta increases, P1's share decreases toward 0.5
            assert convergence[i][1] <= convergence[i-1][1] + 1e-10

    def test_convergence_data_length(self):
        """Should return data for all specified delta values."""
        convergence = NashBargainingSolution.rubinstein_convergence()
        assert len(convergence) > 10


class TestValidation:
    """Test input validation."""

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            NashBargainingSolution([(0, 1), (1, 0)], alpha=-0.1)
        with pytest.raises(ValueError):
            NashBargainingSolution([(0, 1), (1, 0)], alpha=1.1)

    def test_empty_feasible_set(self):
        with pytest.raises(ValueError):
            NashBargainingSolution([], (0, 0))

    def test_invalid_surplus_disagreement(self):
        with pytest.raises(ValueError):
            NashBargainingSolution.closed_form_linear(0.5, (0.3, 0.4))
