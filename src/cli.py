"""
Command-line interface for Rubinstein bargaining models.

Usage:
    python -m src.cli infinite --delta1 0.9 --delta2 0.8
    python -m src.cli finite --delta1 0.9 --delta2 0.8 --rounds 10
    python -m src.cli nash --surplus 1.0 --alpha 0.5
    python -m src.cli convergence
"""

import argparse
import sys

from src.core.rubinstein import InfiniteHorizonBargaining
from src.core.finite_horizon import FiniteHorizonBargaining
from src.core.outside_options import OutsideOptionBargaining
from src.core.multi_issue import MultiIssueBargaining
from src.core.nash_bargaining import NashBargainingSolution


def cmd_infinite(args):
    """Run infinite-horizon bargaining."""
    model = InfiniteHorizonBargaining(
        args.delta1, args.delta2, args.breakdown, args.surplus
    )
    s1, s2 = model.spe_shares()

    print("=" * 60)
    print("INFINITE-HORIZON RUBINSTEIN BARGAINING")
    print("=" * 60)
    print(f"  delta_1 = {args.delta1}")
    print(f"  delta_2 = {args.delta2}")
    print(f"  Breakdown prob = {args.breakdown}")
    print(f"  Surplus = {args.surplus}")
    print()
    print(f"  SPE Outcome:")
    print(f"    Player 1: {s1:.6f}")
    print(f"    Player 2: {s2:.6f}")
    print(f"    First-mover advantage: {model.first_mover_advantage():.4f}")
    print()

    offer_p1, offer_p2 = model.spe_offers()
    print(f"  When Player 1 proposes: ({offer_p1[0]:.6f}, {offer_p1[1]:.6f})")
    print(f"  When Player 2 proposes: ({offer_p2[0]:.6f}, {offer_p2[1]:.6f})")

    if args.verbose:
        print()
        print("  Offer Timeline:")
        for r in model.simulate_rounds(10):
            print(f"    {r}")


def cmd_finite(args):
    """Run finite-horizon bargaining."""
    model = FiniteHorizonBargaining(
        args.delta1, args.delta2, args.rounds, args.surplus
    )
    s1, s2 = model.spe_outcome()

    print("=" * 60)
    print("FINITE-HORIZON RUBINSTEIN BARGAINING")
    print("=" * 60)
    print(f"  delta_1 = {args.delta1}")
    print(f"  delta_2 = {args.delta2}")
    print(f"  Rounds = {args.rounds}")
    print(f"  Surplus = {args.surplus}")
    print()
    print(f"  SPE Outcome (Round 1):")
    print(f"    Player 1: {s1:.6f}")
    print(f"    Player 2: {s2:.6f}")
    print()

    if args.verbose:
        print("  Backward Induction:")
        for r in model.backward_induction():
            print(f"    {r}")

    # Compare with infinite
    inf_model = InfiniteHorizonBargaining(args.delta1, args.delta2, surplus=args.surplus)
    inf_s1, inf_s2 = inf_model.spe_shares()
    print()
    print(f"  Infinite-horizon comparison:")
    print(f"    Player 1: {inf_s1:.6f} (diff: {abs(s1-inf_s1):.6f})")
    print(f"    Player 2: {inf_s2:.6f} (diff: {abs(s2-inf_s2):.6f})")


def cmd_outside(args):
    """Run bargaining with outside options."""
    model = OutsideOptionBargaining(
        args.delta1, args.delta2, args.outside1, args.outside2, args.surplus
    )
    result = model.solve(args.variant)

    print("=" * 60)
    print("BARGAINING WITH OUTSIDE OPTIONS")
    print("=" * 60)
    print(f"  delta_1 = {args.delta1}, delta_2 = {args.delta2}")
    print(f"  Outside options: P1={args.outside1}, P2={args.outside2}")
    print(f"  Variant: {args.variant}")
    print()
    print(f"  {result}")


def cmd_nash(args):
    """Run Nash bargaining solution."""
    result = NashBargainingSolution.closed_form_linear(
        args.surplus, (args.d1, args.d2), args.alpha
    )

    print("=" * 60)
    print("NASH BARGAINING SOLUTION")
    print("=" * 60)
    print(f"  Surplus = {args.surplus}")
    print(f"  Disagreement = ({args.d1}, {args.d2})")
    print(f"  Alpha (P1 power) = {args.alpha}")
    print()
    print(f"  {result}")


def cmd_convergence(args):
    """Show Rubinstein -> Nash convergence."""
    print("=" * 60)
    print("RUBINSTEIN -> NASH BARGAINING CONVERGENCE")
    print("=" * 60)
    print()
    print(f"  {'delta':>8} | {'Rubinstein P1':>14} | {'Nash BS P1':>12} | {'Difference':>12}")
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")

    convergence = NashBargainingSolution.rubinstein_convergence()
    for delta, rub1, rub2, nash1, nash2 in convergence:
        diff = abs(rub1 - nash1)
        print(f"  {delta:8.4f} | {rub1:14.8f} | {nash1:12.8f} | {diff:12.8f}")

    print()
    print("  As delta -> 1, Rubinstein SPE converges to NBS (0.5, 0.5)")


def cmd_multi(args):
    """Run multi-issue bargaining."""
    v1 = list(map(float, args.v1.split(",")))
    v2 = list(map(float, args.v2.split(",")))

    model = MultiIssueBargaining(v1, v2, args.delta1, args.delta2)

    print("=" * 60)
    print("MULTI-ISSUE BARGAINING")
    print("=" * 60)
    print(f"  Player 1 valuations: {v1}")
    print(f"  Player 2 valuations: {v2}")
    print(f"  delta_1 = {args.delta1}, delta_2 = {args.delta2}")
    print()

    eff = model.efficient_allocation()
    print(f"  Efficient allocation: {eff}")
    print(f"  Gains from trade: {model.gains_from_trade():.4f}")
    print()

    spe = model.rubinstein_multi_issue_spe()
    print(f"  Rubinstein SPE: {spe}")
    print()

    print("  Comparative advantages:")
    for issue_idx, player in model.comparative_advantage():
        if player == 0:
            print(f"    Issue {issue_idx+1}: No advantage")
        else:
            print(f"    Issue {issue_idx+1}: Player {player}")


def main():
    parser = argparse.ArgumentParser(
        description="Rubinstein Bargaining Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Model to run")

    # Infinite horizon
    p_inf = subparsers.add_parser("infinite", help="Infinite-horizon Rubinstein")
    p_inf.add_argument("--delta1", type=float, default=0.9)
    p_inf.add_argument("--delta2", type=float, default=0.9)
    p_inf.add_argument("--breakdown", type=float, default=0.0)
    p_inf.add_argument("--surplus", type=float, default=1.0)
    p_inf.add_argument("--verbose", "-v", action="store_true")

    # Finite horizon
    p_fin = subparsers.add_parser("finite", help="Finite-horizon bargaining")
    p_fin.add_argument("--delta1", type=float, default=0.9)
    p_fin.add_argument("--delta2", type=float, default=0.9)
    p_fin.add_argument("--rounds", type=int, default=10)
    p_fin.add_argument("--surplus", type=float, default=1.0)
    p_fin.add_argument("--verbose", "-v", action="store_true")

    # Outside options
    p_out = subparsers.add_parser("outside", help="Outside options bargaining")
    p_out.add_argument("--delta1", type=float, default=0.9)
    p_out.add_argument("--delta2", type=float, default=0.9)
    p_out.add_argument("--outside1", type=float, default=0.0)
    p_out.add_argument("--outside2", type=float, default=0.2)
    p_out.add_argument("--surplus", type=float, default=1.0)
    p_out.add_argument("--variant", default="after_rejection",
                       choices=["after_rejection", "always_available"])

    # Nash bargaining
    p_nash = subparsers.add_parser("nash", help="Nash bargaining solution")
    p_nash.add_argument("--surplus", type=float, default=1.0)
    p_nash.add_argument("--d1", type=float, default=0.0)
    p_nash.add_argument("--d2", type=float, default=0.0)
    p_nash.add_argument("--alpha", type=float, default=0.5)

    # Convergence
    subparsers.add_parser("convergence", help="Rubinstein -> Nash convergence")

    # Multi-issue
    p_multi = subparsers.add_parser("multi", help="Multi-issue bargaining")
    p_multi.add_argument("--v1", default="0.8,0.3,0.6", help="P1 valuations (comma-separated)")
    p_multi.add_argument("--v2", default="0.3,0.7,0.5", help="P2 valuations (comma-separated)")
    p_multi.add_argument("--delta1", type=float, default=0.9)
    p_multi.add_argument("--delta2", type=float, default=0.9)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "infinite": cmd_infinite,
        "finite": cmd_finite,
        "outside": cmd_outside,
        "nash": cmd_nash,
        "convergence": cmd_convergence,
        "multi": cmd_multi,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
