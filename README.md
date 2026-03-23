# Rubinstein Alternating-Offers Bargaining Model

A complete implementation of the **Rubinstein (1982) alternating-offers bargaining model** and its extensions, including finite-horizon bargaining, outside options, risk of breakdown, multi-issue bargaining, and the Nash Bargaining Solution with convergence analysis.

## The Bargaining Problem

Two players must agree on how to divide a surplus (e.g., a dollar). They alternate making proposals: Player 1 proposes, Player 2 can accept or reject; if rejected, Player 2 proposes, and so on. Both players discount the future -- a dollar tomorrow is worth less than a dollar today.

**Key insight**: Even though both players are fully rational and there is complete information, the equilibrium is *not* an even split. The outcome depends on the players' discount factors (patience), and there is a **first-mover advantage**.

## Mathematical Foundation

### Infinite-Horizon SPE

In the infinite-horizon game, the **Subgame Perfect Equilibrium** (SPE) has a closed-form solution. Player 1 proposes first, and both players have discount factors $\delta_1, \delta_2 \in (0, 1)$.

**Player 1's equilibrium share:**

$$x^* = \frac{1 - \delta_2}{1 - \delta_1 \delta_2}$$

**Player 2's equilibrium share:**

$$1 - x^* = \frac{\delta_2(1 - \delta_1)}{1 - \delta_1 \delta_2}$$

The equilibrium is derived from two indifference conditions:
- Player 2 is indifferent between accepting $x^*$ and waiting to propose next round
- Player 1 is indifferent between accepting Player 2's counter-offer and waiting

**Properties:**
- Agreement is **immediate** (no delay in equilibrium)
- As $\delta_1, \delta_2 \to 1$: the outcome approaches an **even split** (50/50)
- More patient player (higher $\delta$) gets a **larger share**
- First proposer has an advantage that vanishes as $\delta \to 1$

### Risk of Breakdown

With exogenous probability $p$ of breakdown each round, effective discount factors become $\hat{\delta}_i = \delta_i(1-p)$, and the SPE formula applies with the effective values.

### Finite-Horizon Bargaining

With $T$ rounds, solved by **backward induction**:
- Round $T$: proposer takes everything (responder accepts any non-negative offer)
- Round $t < T$: proposer offers responder exactly $\delta_r \times$ (responder's continuation value)

As $T \to \infty$, the finite-horizon outcome converges to the infinite-horizon SPE.

### Outside Options

Each player has a fallback payoff $\bar{u}_i$ they can take instead of bargaining. The outside option is **binding** if it exceeds the player's continuation value in the standard Rubinstein game.

Two variants:
1. **After-rejection**: can opt out only after rejecting an offer
2. **Always-available**: can opt out at any time

### Multi-Issue Bargaining

Players negotiate over $k$ divisible issues simultaneously. Player $i$ has valuation $v_i^j$ for issue $j$. The key insight: **gains from trade** arise when players value issues differently. The efficient allocation gives each issue primarily to the player who values it more.

### Nash Bargaining Solution (NBS)

The **axiomatic** approach to bargaining. The NBS maximizes the **Nash product**:

$$\max_{(u_1, u_2) \in F} (u_1 - d_1)^{\alpha} (u_2 - d_2)^{1-\alpha}$$

subject to feasibility and individual rationality, where:
- $F$ is the feasible set
- $(d_1, d_2)$ is the disagreement point
- $\alpha$ is Player 1's bargaining power

**The NBS satisfies four axioms:**
1. **Pareto efficiency** -- no feasible outcome makes both better off
2. **Symmetry** -- symmetric problems have symmetric solutions
3. **Independence of irrelevant alternatives** -- removing non-optimal options doesn't change the solution
4. **Invariance to affine transformations** -- solution is robust to rescaling utilities

### Rubinstein-Nash Convergence

**Rubinstein's key result**: As $\delta_1, \delta_2 \to 1$, the SPE converges to the **asymmetric NBS** with bargaining powers:

$$\alpha = \frac{\ln \delta_2}{\ln \delta_1 + \ln \delta_2}$$

For symmetric $\delta$, this gives $\alpha = 1/2$ (the symmetric NBS = even split).

## Architecture

```
src/
├── core/
│   ├── rubinstein.py       # Infinite-horizon SPE with breakdown risk
│   ├── finite_horizon.py   # Backward induction over T rounds
│   ├── outside_options.py  # After-rejection and always-available variants
│   ├── multi_issue.py      # k-issue bargaining, Pareto frontier
│   └── nash_bargaining.py  # Axiomatic NBS, convergence analysis
├── viz/
│   └── app.py              # Streamlit interactive dashboard
└── cli.py                  # Command-line interface
```

### Core Modules

| Module | Key Class | Description |
|--------|-----------|-------------|
| `rubinstein.py` | `InfiniteHorizonBargaining` | Closed-form SPE, breakdown risk, patience analysis |
| `finite_horizon.py` | `FiniteHorizonBargaining` | Backward induction, game tree, convergence to infinite |
| `outside_options.py` | `OutsideOptionBargaining` | Binding/non-binding outside options, sensitivity |
| `multi_issue.py` | `MultiIssueBargaining` | Pareto frontier, comparative advantage, gains from trade |
| `nash_bargaining.py` | `NashBargainingSolution` | Generalized Nash product, axiom verification, convergence |

## Installation

```bash
# Clone
git clone https://github.com/abhi-wadhwa/rubinstein-bargaining.git
cd rubinstein-bargaining

# Install with all dependencies
pip install -e ".[all]"

# Or minimal install
pip install -e .
```

## Usage

### Command Line

```bash
# Infinite horizon
python -m src.cli infinite --delta1 0.9 --delta2 0.8 -v

# Finite horizon
python -m src.cli finite --delta1 0.9 --delta2 0.8 --rounds 10 -v

# Nash bargaining
python -m src.cli nash --surplus 1.0 --alpha 0.6

# Convergence analysis
python -m src.cli convergence

# Multi-issue bargaining
python -m src.cli multi --v1 "0.8,0.3,0.6" --v2 "0.3,0.7,0.5"
```

### Python API

```python
from src.core.rubinstein import InfiniteHorizonBargaining
from src.core.finite_horizon import FiniteHorizonBargaining
from src.core.nash_bargaining import NashBargainingSolution

# Infinite horizon SPE
model = InfiniteHorizonBargaining(delta1=0.9, delta2=0.8)
s1, s2 = model.spe_shares()
print(f"Player 1: {s1:.4f}, Player 2: {s2:.4f}")

# Finite horizon
fin = FiniteHorizonBargaining(delta1=0.9, delta2=0.8, total_rounds=10)
for r in fin.backward_induction():
    print(r)

# Nash bargaining
nbs = NashBargainingSolution.closed_form_linear(
    surplus=1.0, disagreement=(0, 0), alpha=0.5
)
print(nbs)
```

### Interactive Dashboard

```bash
streamlit run src/viz/app.py
```

Features:
- **Offer timeline**: visual back-and-forth on a number line converging to agreement
- **Discount factor sliders**: watch the equilibrium shift in real time
- **Pareto frontier plot**: multi-issue bargaining outcomes
- **Finite-horizon game tree**: backward induction visualization
- **Convergence analysis**: Rubinstein to Nash as delta approaches 1

### Demo Script

```bash
python examples/demo.py
```

## Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Lint
make lint
```

## Docker

```bash
docker build -t rubinstein-bargaining .
docker run -p 8501:8501 rubinstein-bargaining
```

## References

- Rubinstein, A. (1982). "Perfect Equilibrium in a Bargaining Model." *Econometrica*, 50(1), 97-109.
- Nash, J. (1950). "The Bargaining Problem." *Econometrica*, 18(2), 155-162.
- Binmore, K., Rubinstein, A., & Wolinsky, A. (1986). "The Nash Bargaining Solution in Economic Modelling." *RAND Journal of Economics*, 17(2), 176-188.
- Osborne, M. J., & Rubinstein, A. (1990). *Bargaining and Markets*. Academic Press.
- Muthoo, A. (1999). *Bargaining Theory with Applications*. Cambridge University Press.
