# Reviewer #2, Comment 13 -- Monotonicity of realized risk in the drawdown budget

**Comment.** "In several tables, the 10% budget does not produce the lowest realized
maximum drawdown, and in some cases the 40% budget produces the lowest drawdown and
highest return. This may reflect forecast error, nonbinding constraints, model
instability, or the difference between predicted additive drawdown and realized wealth
drawdown. Analyze this pattern directly instead of describing the method as
consistently controlling drawdown."

**Response in one line.** The reviewer's observation about the tables is correct, but
it concerns the wrong statistic. At the level the constraint actually operates on --
the per-window, H-day, additive drawdown -- the ordering in n1 is monotone and highly
significant. The aggregate eight-year maximum drawdown is a different object, and it
does not inherit that ordering.

---

## 1. The phenomenon is real at the aggregate level

Mean eight-year maximum drawdown of DFL-MDD (TC=0, averaged over lambda and lookback):

| Setting | n1=0.1 | n1=0.2 | n1=0.3 | n1=0.4 | Range |
| --- | --- | --- | --- | --- | --- |
| 10 inds, H=126 | 35.80 | 34.88 | 34.98 | 34.72 | 1.07 pp |
| 10 inds, H=252 | 35.94 | 36.75 | 35.50 | 33.98 | 2.77 pp |
| 30 inds, H=126 | 37.18 | 35.65 | 37.91 | 35.33 | 2.58 pp |
| 30 inds, H=252 | 35.97 | 33.91 | 36.93 | 38.87 | 4.96 pp |

No ordering. Across the 24 (universe x lookback x lambda) cells at H=126, the Spearman
correlation between n1 and aggregate MDD averages -0.04, and n1=0.4 attains the lowest
drawdown in 9 of 24 cells against 7 for n1=0.1.

## 2. At the per-window level the ordering is monotone and significant

The constraint applies to `u_k - yhat_k' x <= n1 * C` over the H-day predicted path.
The matching realized quantity is `M_real`, the additive maximum drawdown over the
realized H-day path of the chosen portfolio -- the same quantity the training loss
penalises.

Within a fixed (universe, H, lookback, lambda), the four n1 runs share the same
rebalancing dates in the same order, so window i is one decision date evaluated under
four budgets. This is a repeated-measures design, tested with **Page's L** against the
ordered alternative `theta(0.1) <= theta(0.2) <= theta(0.3) <= theta(0.4)`.

**Mean per-window drawdown `M_real` (%), and trend test**

| Setting | n1=0.1 | n1=0.2 | n1=0.3 | n1=0.4 | Blocks | Page z | Page p | Paired t (0.4 vs 0.1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 inds, H=126 | 3.837 | 3.955 | 4.101 | 4.206 | 752 | +5.84 | 2.6e-09 | t=+6.21, p=4.3e-10 |
| 10 inds, H=252 | 3.834 | 3.933 | 3.963 | 4.021 | 696 | +3.18 | 7.4e-04 | t=+3.42, p=3.3e-04 |
| 30 inds, H=126 | 4.118 | 4.308 | 4.495 | 4.440 | 752 | +5.64 | 8.5e-09 | t=+3.98, p=3.7e-05 |
| 30 inds, H=252 | 4.118 | 4.252 | 4.401 | 4.548 | 696 | +5.40 | 3.4e-08 | t=+5.01, p=3.5e-07 |

The trend is increasing and significant in every configuration: loosening the budget
raises realized per-window drawdown, exactly as intended. Three of the four settings
are monotone across all four levels; in 30 inds at H=126 only the last step (0.3 to
0.4) inverts, while the endpoint contrast remains significant.

**Robustness: windows where the LP solved under every budget** (infeasible windows
dropped rather than carried forward)

| Setting | n1=0.1 | n1=0.4 | Blocks | Page z | Page p |
| --- | --- | --- | --- | --- | --- |
| 10 inds, H=126 | 3.847 | 4.196 | 494 | +4.59 | 2.2e-06 |
| 10 inds, H=252 | 3.995 | 4.068 | 417 | +2.04 | 2.1e-02 |
| 30 inds, H=126 | 4.098 | 4.458 | 665 | +5.17 | 1.2e-07 |
| 30 inds, H=252 | 4.022 | 4.413 | 561 | +4.96 | 3.6e-07 |

The trend survives in all four. It weakens for 10 industries at H=252, where the
endpoint contrast alone is no longer significant (p=0.14) although Page's L still is
(p=0.021); we report this rather than overstate the result.

## 3. Why the two levels disagree

**(a) The constraint is slack in realization.** Realized per-window drawdown averages
about 4%, while the budgets run from 10% to 40%. Even the tightest budget sits well
above what actually occurs, so it binds only in the tail.

Share of windows in which realized drawdown exceeds its own budget (10 inds, H=126,
pooled over lambda and lookback):

| n1 | Mean `M_real` | p90 | Violation rate |
| --- | --- | --- | --- |
| 0.1 | 3.84% | 7.45% | 4.9% |
| 0.2 | 3.96% | 7.49% | 1.1% |
| 0.3 | 4.10% | 8.04% | 0.5% |
| 0.4 | 4.21% | 8.55% | 0.0% |

This is the reviewer's "nonbinding constraints" hypothesis, and it is the dominant
mechanism. The budget separates the four models only through the minority of windows
where it actually bites, which is why the effect is real but small -- roughly 0.2 to
0.4 pp of per-window drawdown across the full range of n1.

**(b) The reported statistic is a single path extremum, not an average.** Aggregate
maximum drawdown is one realization of a compounded eight-year equity curve, dominated
by a handful of crisis episodes (chiefly March 2020). A systematic shift of 0.2 to
0.4 pp in the average window is second-order against the path noise that determines
which configuration happens to record the deepest trough. The per-window statistic
averages over 700-plus paired observations; the aggregate statistic has an effective
sample size of roughly one.

**(c) The remaining hypotheses are not supported.** The difference between additive
and compounded drawdown does not explain the pattern: `MDD` and `MDD_abs` both show a
Spearman correlation of about -0.05 against n1 at the aggregate level, so the two
definitions fail to order identically. Model instability is also not indicated, since
the per-window ordering is stable across two universes, two horizons, four lambda
values and two lookbacks.

---

## Reproduction

From the project root (`DFL-MDD/`):

```
python analyze_n1_monotonicity.py
python analyze_n1_monotonicity.py --feasible-only
python plot_n1_monotonicity.py
```

Tables land in `{10,30}_inds/results/` of this folder as
`{N}_inds_n1_monotonicity*.csv` (per-level means, Page's L, paired t and Wilcoxon for
the consecutive and endpoint contrasts) and `{N}_inds_n1_monotonicity*_violations.csv`
(realized drawdown quantiles and the budget violation rate). The figure is
`{N}_inds/plots/n1_monotonicity_{N}_inds.png`.
