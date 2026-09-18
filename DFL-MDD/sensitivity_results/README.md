# DFL-MDD Sensitivity Analysis Results

Sensitivity-analysis outputs for the 10- and 30-industry portfolio experiments.

- **Data**: Fama-French 10 / 30 Industry Portfolios, daily returns (2000-01-03 to 2025-12-31)
- **Backtest**: 8-fold walk-forward (test years 2018-2025), rebalanced every 21 trading days
- **Solver**: CLARABEL (zero numerical failures across all runs)
- **Carry-forward**: when the optimization is infeasible, the previous weights are held; if the very first window is infeasible, equal weights (EW) are used

---

## 1. Experiment grid

| Experiment | Parameter | Values | 10 inds | 30 inds |
| --- | --- | --- | --- | --- |
| Baseline | -- | H=126, LB={252, 504}, delta=20 | yes | yes |
| **Horizon H** | HORIZON | 126 -> **252** | yes | yes |
| **Lookback LB** | LOOKBACK | {252, 504} -> **add 1260** | yes | yes |
| **Transaction cost** | TC | 0, 5, 10, 20, 40 bps | yes | yes |
| **Risk aversion** | delta | 20 to 10,000 (9 levels) | yes | yes |
| **Weight cap** | x_max | 1.0, 0.6, 0.3, **0.2** | yes | not run |

Shared hyperparameters: lam in {0.3, 0.5, 0.7, 1.0}, n1 in {0.1, 0.2, 0.3, 0.4}

Models compared: **DFL-MDD** (proposed) / DFL-MVO / PTO-MDD / PTO-MVO / EW / GMV / hist-MVO

The weight-cap experiment was intentionally skipped for 30 industries.

---

## 2. Folder layout

```
sensitivity_results/
  README.md
  10_inds/
    results/     33 CSV
    plots/       70 PNG
  30_inds/
    results/     13 CSV
    plots/       62 PNG
  docs/
    delta_analysis.md    in-depth analysis of the delta sensitivity
```

An earlier CSV format (one file per lambda and per transaction-cost level, without
carry-forward) is excluded, because the current format below covers its content in full.

---

## 3. File reference

### 3-1. Performance and transaction cost

| File | Contents | Rows |
| --- | --- | --- |
| `{N}_inds_tc_cf_lam{L}.csv`<br>`{N}_inds_h126_tc_cf_lam{L}.csv` | **H=126**, all models, 5 TC levels | 125 |
| `{N}_inds_h252_tc_cf_lam{L}.csv` | **H=252**, same | 125 |
| `{N}_inds_tc_full_cf.csv`<br>`{N}_inds_h126_tc_full_cf.csv` | H=126, 4 lambda values combined | 500 |
| `{N}_inds_h252_tc_full_cf.csv` | H=252, 4 lambda values combined | 500 |

> **Naming note.** For H=126 the 10-industry files use the older untagged name
> (`10_inds_tc_cf_lam*.csv`) while the 30-industry files carry the tag
> (`30_inds_h126_tc_cf_lam*.csv`). **The contents are the same**; only the column
> difference noted below applies.

**Columns**

```
[H] [lam] tc_bps  group  label
Ann.Ret(%)  Sharpe  CVaR(5%)(%)  MDD(%)  MDD_abs(%)  Calmar  HHI  Turnover
```

- The `H` column is absent only in the two untagged 10-industry files
  (`10_inds_tc_cf_lam*.csv`, `10_inds_tc_full_cf.csv`). Those files are all H=126.
- The `lam` column appears only in `*_tc_full_cf.csv` (the combined files).
- `group` is one of DFL-MDD / DFL-MVO / PTO-MDD / PTO-MVO / Benchmark.
- `MDD` is the relative drawdown measured on the compounded equity curve.
- `MDD_abs` is the absolute drawdown on additive cumulative return, matching the
  definition used in the optimization constraint.
- `Turnover` is one-way turnover including drift.

### 3-2. Lookback comparison (includes LB=1260)

| File | Contents | Rows |
| --- | --- | --- |
| `{N}_inds_h126_LBcompare.csv` | DFL-MDD, LB {252, 504, 1260} x n1 (4) x lam (4) x TC (5) | 240 |

**Columns**: `lam`, `LB`, `n1`, `tc_bps`, `label`, `Ann.Ret`, `Sharpe`, `CVaR(5%)`,
`MDD`, `MDD_abs`, `Calmar`, `HHI`, `Turnover`

Unlike the files in 3-1, the metric names here carry no `(%)` suffix. The units are
the same (percent).

### 3-3. Statistical tests

| File | Contents | Rows |
| --- | --- | --- |
| `{N}_inds_h126_mdd_ttest.csv` | H=126, one-sided paired t-test on per-window MDD | 48 |
| `{N}_inds_h252_mdd_ttest.csv` | H=252, same | 48 |

- **H0**: mean drawdown of DFL-MDD >= comparison model. **H1**: strictly less.
- Paired over rebalancing dates common to all models (H=126: n=91, H=252: n=85).
- Three significance levels are reported side by side: 0.10, 0.05, 0.01.
- The Wilcoxon signed-rank p-value is reported alongside.

These CSV files carry **Korean column headers**. The mapping is:

| Column | Meaning |
| --- | --- |
| `비교대상` | comparison model |
| `DFL-MDD` | mean per-window MDD of DFL-MDD (%) |
| `비교모델` | mean per-window MDD of the comparison model (%) |
| `차이` | difference (DFL-MDD minus comparison) |
| `유의(0.10)` `유의(0.05)` `유의(0.01)` | significance at each level |

The significance cells hold one of three marks:

| Character | Meaning |
| --- | --- |
| check mark (U+2713) | DFL-MDD significantly better |
| en dash (U+2013) | no significant difference |
| ballot X (U+2717) | comparison model significantly better |

### 3-4. Weight cap (10 industries only)

| File | Contents |
| --- | --- |
| `10_inds_h126_xm{v}_tc_cf_lam{L}.csv` | all models, per x_max and per lambda (v = 1, 0.6, 0.3, 0.2) |
| `10_inds_h126_xmax_compare.csv` | combined table across all x_max |
| `10_inds_h126_xmax_tc_all.csv` | combined across x_max and TC |
| `10_inds_h126_xmax_ttest.csv` | t-tests per x_max (192 rows) |
| `10_inds_h126_mdd_vs_mvo_xmax.csv` | focused DFL-MDD vs DFL-MVO comparison |

**The same cap is applied to the benchmarks (GMV, hist-MVO)** so the comparison stays
fair. EW is 1/N by construction and is unaffected by the cap.

These files add `MaxW` (the realized maximum weight, for verifying the cap is
respected) to the 3-1 columns. `xmax_compare.csv` further adds `x_max` and `nActive`
(the number of assets with weight greater than zero).

### 3-5. Figures

| File | Contents |
| --- | --- |
| `ranked_MDD_{N}_inds_h{H}_lam{L}_tc0.png` | all configurations sorted by MDD, as a bar chart |
| `cumret_LBcompare_{N}_inds_h{H}_lam{L}.png` | cumulative return by LB. H=126 has 3 panels (LB 252/504/1260); **H=252 has 2 panels**, since LB=1260 was trained only at H=126 |
| `monthly_mdd_dist_{N}_inds_l{LB}_h{H}_lam{L}.png` | per-window MDD distribution (violin plus ECDF) |
| `mdd_dist_{N}_inds_h{H}_l{L}_CLARABEL_cf.png` | KDE of the MDD distribution per configuration |
| `dfl_mdd_{N}_inds_h{H}_l{L}_CLARABEL_cf.png` | cumulative PnL with drawdown |
| `infeasibility_timeline_{N}inds.png` | when optimization failures occur over time |
| `dfl_mvo_delta_sweep_{N}_inds_h126.png` | **delta sweep**: DFL-MVO concentration (HHI) against delta |
| `ranked_MDD_xmax_10_inds_h126_lam{L}.png` | the four x_max values side by side (10 inds) |
| `cumret_xmax_10_inds_h126_lam{L}.png` | cumulative return by x_max (10 inds) |

In the original `plots/` folder some H=126 figures were saved under an early naming
scheme with no horizon tag (for example `dfl_mdd_10_inds_0.3_CLARABEL_cf.png`). They
were copied here with an `_h126_` tag so that the two universes use symmetric names.
The image contents are identical to the originals.

---

## 4. Summary of findings

### 4-1. Horizon H

Extending H from 126 to 252 **strengthens the advantage over DFL-MVO**. For 30
industries at alpha = 0.05, the number of significant configurations rises from 3/8 to
6/8. At lam = 1.0 in particular, H=126 shows no significant difference at either
lookback, while H=252 is significant at alpha = 0.01 for both.

The gap against PTO-MDD widens as well (mean per-window MDD of PTO-MDD, 30 inds):

| Lookback | H=126 | H=252 |
| --- | --- | --- |
| LB=252 | 5.35 | 6.02 |
| LB=504 | 6.06 | 7.66 |

### 4-2. Lookback

LB=1260 yields the same 94 rebalancing windows per configuration as LB=252/504, and
its carry-forward fallback rate is the **lowest** of the three in both universes. A
longer lookback appears to stabilize the covariance estimate.

Fallback rate at H=126, aggregated over all lambda and n1 (1,504 windows each):

| Lookback | 10 inds | 30 inds |
| --- | --- | --- |
| LB=252 | 13.43% | 2.66% |
| LB=504 | 11.44% | 3.52% |
| LB=1260 | **8.84%** | **2.19%** |

### 4-3. Risk aversion delta

**delta is not an effective risk control.** For delta <= 2,000 the DFL-MVO portfolio
stays concentrated in a single asset (HHI close to 1.0) and does not move;
diversification only begins around delta = 5,000.

The cause is a unit mismatch in the objective: the return term is an N-day cumulative
value while the risk term uses a daily covariance, so delta must absorb a factor of
roughly N. Inflation of the predicted return scale accounts for the remainder. See
`docs/delta_analysis.md` for the derivation and the numbers.

### 4-4. Transaction cost

DFL-MDD turns over far more than the benchmarks and is therefore sensitive to
transaction cost. At 30 inds, H=126, TC=0 the mean turnover is **0.85** against
**0.32** for hist-MVO, about 2.6 times as high. The drawdown advantage survives out to
40 bps, but the advantage on Calmar relative to the benchmarks disappears somewhere
around 0 to 14 bps.

### 4-5. Weight cap (10 industries)

**Turnover falls by about 40% with no sacrifice in performance.**

| x_max | Turnover | HHI | Calmar (TC=0) | Calmar (TC=40bps) |
| --- | --- | --- | --- | --- |
| 1.0 | 0.600 | 0.472 | 0.233 | 0.144 |
| 0.6 | 0.516 | 0.413 | 0.256 | 0.179 |
| 0.3 | 0.359 | 0.265 | 0.287 | 0.231 |

At 40 bps the annualized return give-up shrinks from 3.04 pp to 1.86 pp, a **39%
reduction**.

One caveat: the cap also relieves DFL-MVO's concentration (HHI 1.00 -> 0.28), so
**DFL-MDD's relative drawdown advantage narrows** (7/8 -> 4/8 at alpha = 0.05). This
suggests that part of the original advantage came from the diversification the
drawdown constraint induces, rather than from drawdown control as such.

---

## 5. Reproduction

Every result can be regenerated from the checkpoints in the project root
(`checkpoint/*.pkl`).

```
train     run_dfl_mdd.py / run_dfl_mvo.py   (run in parallel via launch_*.py)
merge     merge_ckpt.py
analyze   10_inds.ipynb / 30_inds.ipynb / 10_inds_wcap.ipynb
```

Checkpoint naming:

```
dfl_mdd_{N}_inds_h{H}[_xm{cap}][_LB{lb}][_n1{n1}]_d{delta}_l{lam}_{solver}.pkl
```

Bracketed tags appear only for non-default values (x_max=1.0, LB in {252, 504}, and
the n1 tag is dropped after merging).

---

## 6. Not carried out

| Item | Reason |
| --- | --- |
| Weight cap for 30 industries | Skipped by the supervisor's decision |
| LB=1260 at H=252 | **Not trained.** LB=1260 checkpoints exist only for H=126, so the LB comparison at H=252 covers 252 and 504 only |
| delta sweep for DFL-MDD | Only delta=20 was run. The delta analysis is based on DFL-MVO; the two share an objective, so the unit-mismatch argument carries over, but the behavior was not verified directly |
| Direct measurement of the prediction spread | Model state_dict files were not saved, so the value is inferred backwards from the critical delta |
