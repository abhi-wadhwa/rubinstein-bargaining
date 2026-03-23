"""
Streamlit interactive visualization for Rubinstein bargaining models.

Run with: streamlit run src/viz/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.rubinstein import InfiniteHorizonBargaining
from src.core.finite_horizon import FiniteHorizonBargaining
from src.core.outside_options import OutsideOptionBargaining
from src.core.multi_issue import MultiIssueBargaining
from src.core.nash_bargaining import NashBargainingSolution


st.set_page_config(
    page_title="Rubinstein Bargaining Model",
    page_icon="handshake",
    layout="wide",
)

st.title("Rubinstein Alternating-Offers Bargaining Model")
st.markdown(
    """
    Interactive exploration of the Rubinstein bargaining model and its extensions.
    Use the sidebar to select a model and adjust parameters.
    """
)

# --- Sidebar ---
model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Infinite Horizon",
        "Finite Horizon",
        "Outside Options",
        "Multi-Issue Bargaining",
        "Nash Bargaining Solution",
        "Convergence Analysis",
    ],
)


def plot_offer_timeline(results, title="Offer Timeline"):
    """Plot alternating offers on a number line showing convergence."""
    fig, ax = plt.subplots(figsize=(10, 4))

    rounds = [r.round_number for r in results]
    p1_shares = [r.player1_share for r in results]
    p2_shares = [r.player2_share for r in results]

    colors = ["#2196F3" if r.proposer == 1 else "#FF5722" for r in results]
    markers = ["o" if r.proposer == 1 else "s" for r in results]

    for i, r in enumerate(results):
        ax.scatter(
            r.round_number,
            r.player1_share,
            c=colors[i],
            marker=markers[i],
            s=80,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.plot(rounds, p1_shares, "--", alpha=0.4, color="gray")

    # Equilibrium line
    if results:
        eq_share = results[0].player1_share
        ax.axhline(y=eq_share, color="green", linestyle=":", alpha=0.6,
                    label=f"SPE: P1={eq_share:.4f}")

    ax.set_xlabel("Round")
    ax.set_ylabel("Player 1's Share")
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add player legend
    ax.scatter([], [], c="#2196F3", marker="o", label="Player 1 proposes")
    ax.scatter([], [], c="#FF5722", marker="s", label="Player 2 proposes")
    ax.legend()

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return fig


def plot_game_tree(tree_data, max_depth=5):
    """Plot the game tree for finite-horizon bargaining."""
    fig, ax = plt.subplots(figsize=(12, 6))

    def _draw_node(node, x, y, dx, depth=0):
        if depth > max_depth or node["type"] == "terminal":
            if node["type"] == "terminal":
                payoff = node["payoff"]
                ax.text(
                    x, y, f"({payoff[0]:.3f},\n{payoff[1]:.3f})",
                    ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9"),
                )
            return

        color = "#BBDEFB" if node["proposer"] == 1 else "#FFCCBC"
        label = f"R{node['round']}\nP{node['proposer']}"
        ax.text(
            x, y, label, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color),
        )

        # Accept branch (left)
        accept_x = x - dx
        accept_y = y - 1
        ax.annotate(
            "", xy=(accept_x, accept_y + 0.3), xytext=(x, y - 0.3),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
        )
        ax.text(
            (x + accept_x) / 2 - 0.3, (y + accept_y) / 2, "Accept",
            fontsize=6, color="green",
        )
        _draw_node(node["accept"], accept_x, accept_y, dx * 0.6, depth + 1)

        # Reject branch (right)
        reject_x = x + dx
        reject_y = y - 1
        ax.annotate(
            "", xy=(reject_x, reject_y + 0.3), xytext=(x, y - 0.3),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )
        ax.text(
            (x + reject_x) / 2 + 0.3, (y + reject_y) / 2, "Reject",
            fontsize=6, color="red",
        )
        _draw_node(node["reject"], reject_x, reject_y, dx * 0.6, depth + 1)

    _draw_node(tree_data, 0, 0, 3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-max_depth - 1, 1)
    ax.axis("off")
    ax.set_title("Game Tree (Backward Induction)")
    return fig


# --- Infinite Horizon ---
if model_choice == "Infinite Horizon":
    st.header("Infinite-Horizon Rubinstein Bargaining")
    st.latex(
        r"x^* = \frac{1 - \delta_2}{1 - \delta_1 \delta_2}"
    )

    col1, col2 = st.columns(2)
    with col1:
        delta1 = st.slider("Player 1 discount factor (delta_1)", 0.01, 0.99, 0.9, 0.01)
    with col2:
        delta2 = st.slider("Player 2 discount factor (delta_2)", 0.01, 0.99, 0.9, 0.01)

    breakdown = st.slider("Breakdown probability", 0.0, 0.5, 0.0, 0.01)

    model = InfiniteHorizonBargaining(delta1, delta2, breakdown)
    s1, s2 = model.spe_shares()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Player 1's Share", f"{s1:.4f}")
    with col2:
        st.metric("Player 2's Share", f"{s2:.4f}")
    with col3:
        st.metric("First-Mover Advantage", f"{model.first_mover_advantage():.4f}")

    # Offer timeline
    results = model.simulate_rounds(20)
    fig = plot_offer_timeline(results, "SPE Offer Timeline")
    st.pyplot(fig)
    plt.close()

    # Patience analysis
    st.subheader("Effect of Player 2's Patience")
    analysis = model.patience_analysis()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    deltas = [a[0] for a in analysis]
    shares1 = [a[1] for a in analysis]
    shares2 = [a[2] for a in analysis]
    ax2.plot(deltas, shares1, label="Player 1", color="#2196F3", linewidth=2)
    ax2.plot(deltas, shares2, label="Player 2", color="#FF5722", linewidth=2)
    ax2.axvline(x=delta2, color="gray", linestyle="--", alpha=0.5, label=f"Current delta_2={delta2}")
    ax2.set_xlabel("Player 2's Discount Factor")
    ax2.set_ylabel("Equilibrium Share")
    ax2.set_title("Effect of Player 2's Patience on Equilibrium Shares")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()


# --- Finite Horizon ---
elif model_choice == "Finite Horizon":
    st.header("Finite-Horizon Bargaining")

    col1, col2, col3 = st.columns(3)
    with col1:
        delta1 = st.slider("delta_1", 0.01, 0.99, 0.9, 0.01)
    with col2:
        delta2 = st.slider("delta_2", 0.01, 0.99, 0.9, 0.01)
    with col3:
        T = st.slider("Total Rounds (T)", 1, 20, 5, 1)

    model = FiniteHorizonBargaining(delta1, delta2, T)
    results = model.backward_induction()
    s1, s2 = model.spe_outcome()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Player 1's Share (Round 1)", f"{s1:.4f}")
    with col2:
        st.metric("Player 2's Share (Round 1)", f"{s2:.4f}")

    # Show round-by-round results
    st.subheader("Backward Induction Results")
    for r in results:
        st.text(str(r))

    # Game tree
    st.subheader("Game Tree")
    tree = model.game_tree_data()
    fig = plot_game_tree(tree, max_depth=min(T, 5))
    st.pyplot(fig)
    plt.close()

    # Convergence plot
    st.subheader("Convergence to Infinite Horizon")
    conv = model.convergence_to_infinite(max_T=50)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    Ts = [c[0] for c in conv]
    p1s = [c[1] for c in conv]

    inf_model = InfiniteHorizonBargaining(delta1, delta2)
    inf_s1, _ = inf_model.spe_shares()

    ax3.plot(Ts, p1s, label="Finite Horizon P1 Share", color="#2196F3")
    ax3.axhline(y=inf_s1, color="red", linestyle="--", label=f"Infinite Horizon: {inf_s1:.4f}")
    ax3.set_xlabel("Number of Rounds (T)")
    ax3.set_ylabel("Player 1's Share")
    ax3.set_title("Finite Horizon Converges to Infinite Horizon")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close()


# --- Outside Options ---
elif model_choice == "Outside Options":
    st.header("Bargaining with Outside Options")

    col1, col2 = st.columns(2)
    with col1:
        delta1 = st.slider("delta_1", 0.01, 0.99, 0.9, 0.01)
        outside1 = st.slider("Player 1's Outside Option", 0.0, 0.49, 0.0, 0.01)
    with col2:
        delta2 = st.slider("delta_2", 0.01, 0.99, 0.9, 0.01)
        outside2 = st.slider("Player 2's Outside Option", 0.0, 0.49, 0.0, 0.01)

    variant = st.selectbox("Variant", ["after_rejection", "always_available"])

    model = OutsideOptionBargaining(delta1, delta2, outside1, outside2)
    result = model.solve(variant)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Player 1's Share", f"{result.player1_share:.4f}")
        st.text(f"Outside option binding: {result.outside_option_binding[1]}")
    with col2:
        st.metric("Player 2's Share", f"{result.player2_share:.4f}")
        st.text(f"Outside option binding: {result.outside_option_binding[2]}")

    st.text(f"Agreement type: {result.agreement_type}")

    # Sensitivity analysis
    st.subheader("Sensitivity to Player 2's Outside Option")
    sensitivity = model.sensitivity_analysis(player=2, variant=variant)
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    oos = [s[0] for s in sensitivity]
    s1s = [s[1] for s in sensitivity]
    s2s = [s[2] for s in sensitivity]
    bindings = [s[3] for s in sensitivity]

    ax4.plot(oos, s1s, label="Player 1", color="#2196F3", linewidth=2)
    ax4.plot(oos, s2s, label="Player 2", color="#FF5722", linewidth=2)

    # Shade binding region
    binding_start = None
    for i, b in enumerate(bindings):
        if b and binding_start is None:
            binding_start = oos[i]
        elif not b and binding_start is not None:
            ax4.axvspan(binding_start, oos[i-1], alpha=0.15, color="yellow",
                       label="Binding region" if binding_start == oos[i] else "")
            binding_start = None
    if binding_start is not None:
        ax4.axvspan(binding_start, oos[-1], alpha=0.15, color="yellow")

    ax4.set_xlabel("Player 2's Outside Option")
    ax4.set_ylabel("Equilibrium Share")
    ax4.set_title("Effect of Outside Options on Equilibrium")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
    plt.close()


# --- Multi-Issue ---
elif model_choice == "Multi-Issue Bargaining":
    st.header("Multi-Issue Bargaining")
    st.markdown(
        "Players negotiate over multiple issues simultaneously. "
        "Different valuations create gains from trade."
    )

    n_issues = st.slider("Number of Issues", 2, 6, 3)

    col1, col2 = st.columns(2)
    v1 = []
    v2 = []
    with col1:
        st.subheader("Player 1 Valuations")
        for i in range(n_issues):
            v = st.slider(f"Issue {i+1} (P1)", 0.0, 1.0, float(np.random.RandomState(42+i).rand()), 0.01, key=f"v1_{i}")
            v1.append(v)
    with col2:
        st.subheader("Player 2 Valuations")
        for i in range(n_issues):
            v = st.slider(f"Issue {i+1} (P2)", 0.0, 1.0, float(np.random.RandomState(99+i).rand()), 0.01, key=f"v2_{i}")
            v2.append(v)

    col1, col2 = st.columns(2)
    with col1:
        delta1 = st.slider("delta_1", 0.01, 0.99, 0.9, 0.01, key="mi_d1")
    with col2:
        delta2 = st.slider("delta_2", 0.01, 0.99, 0.9, 0.01, key="mi_d2")

    model = MultiIssueBargaining(v1, v2, delta1, delta2)

    # Pareto frontier
    frontier = model.pareto_frontier(300)
    efficient = model.efficient_allocation()
    spe = model.rubinstein_multi_issue_spe()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gains from Trade", f"{model.gains_from_trade():.4f}")
    with col2:
        st.metric("SPE: P1 Utility", f"{spe.player1_utility:.4f}")

    # Pareto frontier plot
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    u1s = [f[0] for f in frontier]
    u2s = [f[1] for f in frontier]
    ax5.plot(u1s, u2s, "b-", linewidth=2, label="Pareto Frontier")
    ax5.scatter(
        [spe.player1_utility], [spe.player2_utility],
        c="red", s=120, zorder=5, label=f"Rubinstein SPE", marker="*",
    )
    ax5.scatter(
        [efficient.player1_utility], [efficient.player2_utility],
        c="green", s=100, zorder=5, label="Efficient", marker="D",
    )

    # Equal split
    equal = np.full(n_issues, 0.5)
    eq_u1 = model.utility(equal, 1)
    eq_u2 = model.utility(equal, 2)
    ax5.scatter([eq_u1], [eq_u2], c="orange", s=80, zorder=5, label="Equal Split", marker="^")

    ax5.set_xlabel("Player 1 Utility")
    ax5.set_ylabel("Player 2 Utility")
    ax5.set_title("Pareto Frontier and Bargaining Outcomes")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)
    plt.close()

    # Comparative advantages
    st.subheader("Comparative Advantages")
    ca = model.comparative_advantage()
    for issue_idx, player in ca:
        if player == 0:
            st.text(f"Issue {issue_idx+1}: No advantage (equal valuations)")
        else:
            st.text(
                f"Issue {issue_idx+1}: Player {player} has comparative advantage "
                f"(v1={v1[issue_idx]:.2f}, v2={v2[issue_idx]:.2f})"
            )


# --- Nash Bargaining Solution ---
elif model_choice == "Nash Bargaining Solution":
    st.header("Nash Bargaining Solution")
    st.latex(
        r"\max_{(u_1, u_2) \in F} (u_1 - d_1)^{\alpha} (u_2 - d_2)^{1-\alpha}"
    )

    col1, col2 = st.columns(2)
    with col1:
        surplus = st.slider("Total Surplus", 0.1, 10.0, 1.0, 0.1)
        alpha = st.slider("Bargaining Power (alpha)", 0.01, 0.99, 0.5, 0.01)
    with col2:
        d1 = st.slider("Disagreement Payoff d1", 0.0, 0.45, 0.0, 0.01)
        d2 = st.slider("Disagreement Payoff d2", 0.0, 0.45, 0.0, 0.01)

    result = NashBargainingSolution.closed_form_linear(surplus, (d1, d2), alpha)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Player 1's Utility", f"{result.player1_utility:.4f}")
    with col2:
        st.metric("Player 2's Utility", f"{result.player2_utility:.4f}")
    with col3:
        st.metric("Nash Product", f"{result.nash_product:.6f}")

    # Visualization
    fig6, ax6 = plt.subplots(figsize=(8, 6))

    # Feasible set
    t_vals = np.linspace(0, 1, 200)
    u1_vals = t_vals * surplus
    u2_vals = (1 - t_vals) * surplus
    ax6.fill_between(u1_vals, 0, u2_vals, alpha=0.15, color="blue", label="Feasible Set")
    ax6.plot(u1_vals, u2_vals, "b-", linewidth=2, label="Frontier")

    # NBS point
    ax6.scatter(
        [result.player1_utility], [result.player2_utility],
        c="red", s=150, zorder=5, label="Nash Bargaining Solution", marker="*",
    )

    # Disagreement point
    ax6.scatter([d1], [d2], c="black", s=80, zorder=5, label=f"Disagreement ({d1}, {d2})", marker="x")

    # Iso-Nash-product curves
    if result.player1_utility > d1 and result.player2_utility > d2:
        u1_range = np.linspace(d1 + 0.001, surplus, 200)
        target = result.nash_product
        u2_curve = []
        for u1 in u1_range:
            excess1 = u1 - d1
            if excess1 > 0 and alpha < 1:
                try:
                    needed = target / (excess1 ** alpha)
                    u2 = needed ** (1 / (1 - alpha)) + d2
                    u2_curve.append(u2)
                except (OverflowError, ZeroDivisionError):
                    u2_curve.append(np.nan)
            else:
                u2_curve.append(np.nan)
        ax6.plot(u1_range, u2_curve, "r--", alpha=0.4, label="Iso-Nash-product curve")

    ax6.set_xlabel("Player 1 Utility")
    ax6.set_ylabel("Player 2 Utility")
    ax6.set_title("Nash Bargaining Solution")
    ax6.legend(fontsize=8)
    ax6.set_xlim(-0.1, surplus + 0.1)
    ax6.set_ylim(-0.1, surplus + 0.1)
    ax6.grid(True, alpha=0.3)
    st.pyplot(fig6)
    plt.close()

    # Axiom verification
    st.subheader("Axiom Verification")
    nbs = NashBargainingSolution.from_linear_frontier(surplus, surplus, (d1, d2), alpha)
    nbs_result = nbs.solve()
    axioms = nbs.verify_axioms(nbs_result)
    for axiom, check in axioms.items():
        status = "PASS" if check["satisfied"] else "FAIL"
        st.text(f"[{status}] {axiom}: {check['detail']}")


# --- Convergence ---
elif model_choice == "Convergence Analysis":
    st.header("Rubinstein -> Nash Bargaining Convergence")
    st.markdown(
        r"As $\delta \to 1$, the Rubinstein SPE converges to the Nash Bargaining Solution."
    )

    convergence = NashBargainingSolution.rubinstein_convergence()

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    deltas = [c[0] for c in convergence]
    rub_p1 = [c[1] for c in convergence]
    nash_p1 = [c[3] for c in convergence]

    ax7.plot(deltas, rub_p1, "o-", color="#2196F3", label="Rubinstein P1 Share", markersize=6)
    ax7.axhline(y=nash_p1[0], color="red", linestyle="--", linewidth=2,
                label=f"Nash BS P1 Share = {nash_p1[0]:.4f}")
    ax7.set_xlabel("Discount Factor (delta)")
    ax7.set_ylabel("Player 1's Share")
    ax7.set_title("Convergence of Rubinstein SPE to Nash Bargaining Solution")
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 1.02)
    ax7.set_ylim(0.4, 1.0)
    st.pyplot(fig7)
    plt.close()

    # Table
    st.subheader("Convergence Table")
    st.markdown("| delta | Rubinstein P1 | Nash BS P1 | Difference |")
    st.markdown("|-------|--------------|------------|------------|")
    for c in convergence:
        diff = abs(c[1] - c[3])
        st.markdown(f"| {c[0]:.4f} | {c[1]:.6f} | {c[3]:.6f} | {diff:.6f} |")

    # Asymmetric convergence
    st.subheader("Asymmetric Discount Factors")
    st.markdown(
        r"With asymmetric $\delta_1 \neq \delta_2$, the Rubinstein SPE "
        r"converges to the *asymmetric* NBS with bargaining powers "
        r"determined by the ratio of log discount factors."
    )

    delta1_conv = st.slider("delta_1", 0.5, 0.999, 0.95, 0.001, key="conv_d1")
    delta2_conv = st.slider("delta_2", 0.5, 0.999, 0.90, 0.001, key="conv_d2")

    model_conv = InfiniteHorizonBargaining(delta1_conv, delta2_conv)
    s1_conv, s2_conv = model_conv.spe_shares()

    # Asymmetric NBS: bargaining power proportional to ln(delta_j)/ln(delta_i)
    # As delta -> 1: alpha = ln(delta2)/( ln(delta1)+ln(delta2) )
    import math
    ln1 = math.log(delta1_conv)
    ln2 = math.log(delta2_conv)
    alpha_asymm = ln2 / (ln1 + ln2) if (ln1 + ln2) != 0 else 0.5
    nbs_asymm = NashBargainingSolution.closed_form_linear(1.0, (0, 0), alpha_asymm)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rubinstein P1 Share", f"{s1_conv:.6f}")
        st.metric("Rubinstein P2 Share", f"{s2_conv:.6f}")
    with col2:
        st.metric("Asymmetric NBS P1", f"{nbs_asymm.player1_utility:.6f}")
        st.metric("Asymmetric NBS P2", f"{nbs_asymm.player2_utility:.6f}")
    st.metric("Implied alpha (P1 power)", f"{alpha_asymm:.4f}")
