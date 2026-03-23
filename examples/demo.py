"""
Demonstration of the Rubinstein bargaining model and its extensions.

This script showcases the key features of the package:
1. Infinite-horizon SPE computation
2. Finite-horizon backward induction
3. Outside options
4. Multi-issue bargaining
5. Nash bargaining solution
6. Convergence analysis
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.rubinstein import InfiniteHorizonBargaining
from src.core.finite_horizon import FiniteHorizonBargaining
from src.core.outside_options import OutsideOptionBargaining
from src.core.multi_issue import MultiIssueBargaining
from src.core.nash_bargaining import NashBargainingSolution


def demo_infinite_horizon():
    print("=" * 70)
    print("1. INFINITE-HORIZON RUBINSTEIN BARGAINING")
    print("=" * 70)

    # Symmetric discount factors
    model = InfiniteHorizonBargaining(0.9, 0.9)
    s1, s2 = model.spe_shares()
    print(f"\nSymmetric (delta=0.9):")
    print(f"  Player 1: {s1:.4f}, Player 2: {s2:.4f}")
    print(f"  First-mover advantage: {model.first_mover_advantage():.4f}")

    # Asymmetric discount factors
    model = InfiniteHorizonBargaining(0.9, 0.5)
    s1, s2 = model.spe_shares()
    print(f"\nAsymmetric (delta1=0.9, delta2=0.5):")
    print(f"  Player 1: {s1:.4f}, Player 2: {s2:.4f}")
    print(f"  More patient player gets more!")

    # With breakdown risk
    model = InfiniteHorizonBargaining(0.9, 0.9, breakdown_prob=0.1)
    s1, s2 = model.spe_shares()
    print(f"\nWith breakdown risk p=0.1:")
    print(f"  Player 1: {s1:.4f}, Player 2: {s2:.4f}")

    # Offer timeline
    model = InfiniteHorizonBargaining(0.8, 0.7)
    print(f"\nOffer timeline (delta1=0.8, delta2=0.7):")
    for r in model.simulate_rounds(6):
        print(f"  {r}")


def demo_finite_horizon():
    print("\n" + "=" * 70)
    print("2. FINITE-HORIZON BARGAINING")
    print("=" * 70)

    delta1, delta2 = 0.9, 0.8

    print(f"\nBackward induction (delta1={delta1}, delta2={delta2}):")
    for T in [1, 2, 3, 5, 10]:
        model = FiniteHorizonBargaining(delta1, delta2, T)
        s1, s2 = model.spe_outcome()
        print(f"  T={T:2d}: P1={s1:.4f}, P2={s2:.4f}")

    # Show convergence
    inf_model = InfiniteHorizonBargaining(delta1, delta2)
    inf_s1, inf_s2 = inf_model.spe_shares()
    print(f"\n  Infinite horizon limit: P1={inf_s1:.4f}, P2={inf_s2:.4f}")

    # Detailed backward induction for T=5
    print(f"\nDetailed backward induction (T=5):")
    model = FiniteHorizonBargaining(delta1, delta2, 5)
    for r in model.backward_induction():
        print(f"  {r}")


def demo_outside_options():
    print("\n" + "=" * 70)
    print("3. OUTSIDE OPTIONS")
    print("=" * 70)

    delta1, delta2 = 0.9, 0.9

    # Standard Rubinstein for comparison
    std = InfiniteHorizonBargaining(delta1, delta2)
    s1_std, s2_std = std.spe_shares()
    print(f"\nStandard Rubinstein: P1={s1_std:.4f}, P2={s2_std:.4f}")

    # Non-binding outside option
    model = OutsideOptionBargaining(delta1, delta2, 0.0, 0.1)
    result = model.solve()
    print(f"\nP2 outside option = 0.10 (non-binding):")
    print(f"  {result}")

    # Binding outside option
    model = OutsideOptionBargaining(delta1, delta2, 0.0, 0.48)
    result = model.solve()
    print(f"\nP2 outside option = 0.48 (binding):")
    print(f"  {result}")


def demo_multi_issue():
    print("\n" + "=" * 70)
    print("4. MULTI-ISSUE BARGAINING")
    print("=" * 70)

    v1 = [0.9, 0.2, 0.5]
    v2 = [0.2, 0.8, 0.6]
    model = MultiIssueBargaining(v1, v2, 0.9, 0.9)

    print(f"\nValuations:")
    print(f"  Player 1: {v1}")
    print(f"  Player 2: {v2}")

    print(f"\nEfficient allocation:")
    eff = model.efficient_allocation()
    print(f"  {eff}")

    print(f"\nGains from trade: {model.gains_from_trade():.4f}")

    print(f"\nComparative advantages:")
    for issue, player in model.comparative_advantage():
        if player > 0:
            print(f"  Issue {issue+1}: Player {player} (v1={v1[issue]:.1f}, v2={v2[issue]:.1f})")

    print(f"\nRubinstein multi-issue SPE:")
    spe = model.rubinstein_multi_issue_spe()
    print(f"  {spe}")

    print(f"\nPareto frontier (sample points):")
    frontier = model.pareto_frontier(10)
    for u1, u2 in frontier:
        print(f"  ({u1:.4f}, {u2:.4f})")


def demo_nash_bargaining():
    print("\n" + "=" * 70)
    print("5. NASH BARGAINING SOLUTION")
    print("=" * 70)

    # Symmetric NBS
    result = NashBargainingSolution.closed_form_linear(1.0, (0, 0), 0.5)
    print(f"\nSymmetric NBS (surplus=1, d=(0,0), alpha=0.5):")
    print(f"  {result}")

    # Asymmetric NBS
    result = NashBargainingSolution.closed_form_linear(1.0, (0, 0), 0.7)
    print(f"\nAsymmetric NBS (alpha=0.7):")
    print(f"  {result}")

    # With disagreement point
    result = NashBargainingSolution.closed_form_linear(1.0, (0.1, 0.2), 0.5)
    print(f"\nNBS with d=(0.1, 0.2):")
    print(f"  {result}")

    # Axiom verification
    nbs = NashBargainingSolution.from_linear_frontier(1.0, 1.0, (0, 0), 0.5)
    nbs_result = nbs.solve()
    axioms = nbs.verify_axioms(nbs_result)
    print(f"\nAxiom verification:")
    for name, check in axioms.items():
        status = "PASS" if check["satisfied"] else "FAIL"
        print(f"  [{status}] {name}")


def demo_convergence():
    print("\n" + "=" * 70)
    print("6. RUBINSTEIN -> NASH CONVERGENCE")
    print("=" * 70)

    print(f"\n{'delta':>8} | {'Rubinstein P1':>14} | {'Nash BS P1':>12} | {'Gap':>10}")
    print(f"{'-'*8}-+-{'-'*14}-+-{'-'*12}-+-{'-'*10}")

    convergence = NashBargainingSolution.rubinstein_convergence()
    for delta, rub1, rub2, nash1, nash2 in convergence:
        gap = abs(rub1 - nash1)
        print(f"{delta:8.4f} | {rub1:14.8f} | {nash1:12.8f} | {gap:10.8f}")

    print(f"\nAs delta -> 1, the Rubinstein SPE converges to the")
    print(f"Nash Bargaining Solution (50/50 split).")


if __name__ == "__main__":
    demo_infinite_horizon()
    demo_finite_horizon()
    demo_outside_options()
    demo_multi_issue()
    demo_nash_bargaining()
    demo_convergence()

    print("\n" + "=" * 70)
    print("For interactive exploration, run:")
    print("  streamlit run src/viz/app.py")
    print("=" * 70)
