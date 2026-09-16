# DFL-MDD 민감도 분석 결과

10개 / 30개 산업 포트폴리오에 대한 민감도 분석 산출물 모음.

- **데이터**: Fama-French 10 / 30 Industry Portfolios, 일별 수익률 (2000-01-03 ~ 2025-12-31)
- **백테스트**: 8-fold walk-forward (test 2018~2025), 21거래일 리밸런싱
- **solver**: CLARABEL (전 실행에서 수치적 실패 0건)
- **carry-forward**: 최적화 실패(infeasible) 시 직전 비중 유지, 첫 윈도우 실패 시 동일가중(EW)

---

## 1. 실험 구성

| 실험 | 변화 대상 | 값 | 10 inds | 30 inds |
| --- | --- | --- | --- | --- |
| 기본 | — | H=126, LB={252, 504}, δ=20 | ✓ | ✓ |
| **예측구간 H** | HORIZON | 126 → **252** | ✓ | ✓ |
| **Lookback LB** | LOOKBACK | {252, 504} → **1260 추가** | ✓ | ✓ |
| **거래비용** | TC | 0, 5, 10, 20, 40 bps | ✓ | ✓ |
| **위험회피 δ** | delta | 20 ~ 10,000 (9개) | ✓ | ✓ |
| **비중 상한** | x_max | 1.0, 0.6, 0.3, **0.2** | ✓ | — (미실시) |

공통 하이퍼파라미터: λ ∈ {0.3, 0.5, 0.7, 1.0}, n₁ ∈ {0.1, 0.2, 0.3, 0.4}

비교 모델: **DFL-MDD** (제안) / DFL-MVO / PTO-MDD / PTO-MVO / EW / GMV / hist-MVO

---

## 2. 폴더 구조

```
sensitivity_results/
├─ README.md
├─ 10_inds/
│   ├─ results/          CSV 33개
│   └─ plots/            PNG 70개
├─ 30_inds/
│   ├─ results/          CSV 13개
│   └─ plots/            PNG 62개
└─ docs/
    ├─ delta_analysis.md          δ 민감도 심층 분석 (LaTeX 수식)
    └─ delta_analysis_notion.md   동일 내용, 수식 텍스트화
```

초기 형식(λ·거래비용마다 파일 분리, carry-forward 미적용)의 CSV 는 아래 신형식이 내용상 완전히 포함하므로 제외했습니다.

---

## 3. 파일 설명

### 3-1. 성과 · 거래비용

| 파일 | 내용 | 행수 |
| --- | --- | --- |
| `{N}_inds_tc_cf_lam{λ}.csv`<br>`{N}_inds_h126_tc_cf_lam{λ}.csv` | **H=126** 전 모델 성과, tc 5수준 | 125 |
| `{N}_inds_h252_tc_cf_lam{λ}.csv` | **H=252** 동일 | 125 |
| `{N}_inds_tc_full_cf.csv`<br>`{N}_inds_h126_tc_full_cf.csv` | H=126 λ 4개 통합 | 500 |
| `{N}_inds_h252_tc_full_cf.csv` | H=252 λ 4개 통합 | 500 |

> 파일명 주의: H=126 에서 10 inds 는 `h126` 태그가 없는 구이름(`10_inds_tc_cf_lam*.csv`), 30 inds 는 태그가 붙은 이름(`30_inds_h126_tc_cf_lam*.csv`)을 씁니다. **내용은 동일**하며, 아래 컬럼 차이만 있습니다.

**컬럼**

```
[H] [lam] tc_bps  group  label
Ann.Ret(%)  Sharpe  CVaR(5%)(%)  MDD(%)  MDD_abs(%)  Calmar  HHI  Turnover
```

- `H` 컬럼은 10 inds 구이름 2개 파일(`10_inds_tc_cf_lam*.csv`, `10_inds_tc_full_cf.csv`)에만 없습니다. 이 파일들은 전부 H=126 입니다.
- `lam` 컬럼은 `*_tc_full_cf.csv` (λ 통합본)에만 있습니다.
- `group` = DFL-MDD / DFL-MVO / PTO-MDD / PTO-MVO / Benchmark
- `MDD` = 복리 자산곡선 기준 상대 낙폭
- `MDD_abs` = 가법 누적수익 기준 절대 낙폭 (최적화 제약과 동일 정의)
- `Turnover` = drift 반영 one-way 회전율

### 3-2. Lookback 비교 (LB=1260 포함)

| 파일 | 내용 | 행수 |
| --- | --- | --- |
| `{N}_inds_h126_LBcompare.csv` | DFL-MDD, LB {252, 504, 1260} × n₁ 4 × λ 4 × tc 5 | 240 |

**컬럼**: `λ`, `LB`, `n1`, `tc_bps`, `label`, `Ann.Ret`, `Sharpe`, `CVaR(5%)`, `MDD`, `MDD_abs`, `Calmar`, `HHI`, `Turnover`
(위 3-1 파일과 달리 지표명에 `(%)` 접미사가 없습니다. 단위는 동일하게 % 입니다.)

### 3-3. 통계 검정

| 파일 | 내용 | 행수 |
| --- | --- | --- |
| `{N}_inds_h126_mdd_ttest.csv` | H=126 per-window MDD 단측 대응 t-검정 | 48 |
| `{N}_inds_h252_mdd_ttest.csv` | H=252 동일 | 48 |

- **H₀**: DFL-MDD 평균 낙폭 ≥ 비교모델 / **H₁**: <
- 공통 리밸런싱 시점에서 짝지어 검정 (H=126 n=91, H=252 n=85)
- `유의(0.10)` `유의(0.05)` `유의(0.01)` 세 수준 병기
- **✓** DFL-MDD 우위 · **–** 유의차 없음 · **✗** 비교모델 우위
- Wilcoxon 부호순위 검정 p값 병기

### 3-4. 비중 상한 (10 inds 전용)

| 파일 | 내용 |
| --- | --- |
| `10_inds_h126_xm{v}_tc_cf_lam{λ}.csv` | x_max 별 · λ 별 전 모델 성과 (v = 1, 0.6, 0.3, 0.2) |
| `10_inds_h126_xmax_compare.csv` | 전 x_max 통합 성과표 |
| `10_inds_h126_xmax_tc_all.csv` | 전 x_max × tc 통합 |
| `10_inds_h126_xmax_ttest.csv` | x_max 별 t-검정 (192행) |
| `10_inds_h126_mdd_vs_mvo_xmax.csv` | DFL-MDD vs DFL-MVO 집중 비교 |

**벤치마크(GMV, hist-MVO)에도 동일한 상한을 적용**해 공정 비교했습니다. EW 는 1/N 이므로 상한과 무관합니다.

이 파일들은 3-1 컬럼에 더해 `MaxW`(실현 최대 비중 — 상한 준수 확인용)를 가지며,
`xmax_compare.csv` 는 추가로 `x_max`, `nActive`(비중 > 0 자산 수)를 가집니다.

### 3-5. 그림

| 파일 | 내용 |
| --- | --- |
| `ranked_MDD_{N}_inds_h{H}_lam{λ}_tc0.png` | 전 config 를 MDD 순으로 정렬한 막대그래프 |
| `cumret_LBcompare_{N}_inds_h{H}_lam{λ}.png` | LB별 누적수익 곡선. H=126 은 3패널(LB 252/504/1260), **H=252 는 2패널**(LB=1260 은 H=126 에서만 학습) |
| `monthly_mdd_dist_{N}_inds_l{LB}_h{H}_lam{λ}.png` | per-window MDD 분포 (violin + ECDF) |
| `mdd_dist_{N}_inds_h{H}_l{λ}_CLARABEL_cf.png` | config별 MDD 분포 KDE |
| `dfl_mdd_{N}_inds_h{H}_l{λ}_CLARABEL_cf.png` | 누적 PnL + drawdown |
| `infeasibility_timeline_{N}inds.png` | 최적화 실패 시점 분포 |
| `dfl_mvo_delta_sweep_{N}_inds_h126.png` | **δ sweep** — δ에 따른 DFL-MVO 집중도(HHI) 변화 |
| `ranked_MDD_xmax_10_inds_h126_lam{λ}.png` | x_max 4개 나란히 비교 (10 inds) |
| `cumret_xmax_10_inds_h126_lam{λ}.png` | x_max별 누적수익 (10 inds) |

원본 `plots/` 폴더의 일부 H=126 그림은 H 태그가 없던 초기 이름(예: `dfl_mdd_10_inds_0.3_CLARABEL_cf.png`)으로 저장되어 있습니다. 이 폴더에서는 두 자산군의 이름이 대칭이 되도록 `_h126_` 태그를 붙여 복사했습니다. 내용은 원본과 동일합니다.

---

## 4. 주요 결과 요약

### 4-1. 예측구간 H

H를 126 → 252 로 늘리면 **DFL-MVO 대비 우위가 강해집니다**. α=0.05 기준 유의한 조합이 30 inds 에서 3/8 → 7/8 로 증가하며, 특히 λ=1.0 에서 H=126 은 유의차가 없으나 H=252 는 α=0.01 로 유의합니다. PTO-MDD 와의 격차도 확대됩니다 (per-window MDD 5.35 → 6.02, LB=504 는 6.06 → 7.66).

### 4-2. Lookback

LB=1260 은 LB=252/504 와 동일한 94개 리밸런싱 윈도우를 가지며, fallback 비율이 1.3~2.9% 로 기본 설정(3.1%)보다 오히려 낮습니다. 긴 lookback 이 공분산 추정을 안정화하는 것으로 보입니다.

### 4-3. 위험회피 δ

**δ 는 실질적 위험 통제 수단이 아닙니다.** δ ≤ 2,000 구간에서 DFL-MVO 포트폴리오가 단일 자산에 집중(HHI ≈ 1.0)된 채 변하지 않으며, 분산은 δ ≈ 5,000 에서야 시작됩니다. 원인은 목적함수의 단위 불일치(일별 Σ vs N일 누적 ŷ, 약 N배)와 예측값 스케일 팽창입니다. 상세는 `docs/delta_analysis.md` 참조.

### 4-4. 거래비용

DFL-MDD 는 회전율이 높아(30 inds h126 기준 0.81 vs hist-MVO 0.41) 거래비용에 민감합니다. 낙폭 우위는 40 bps 까지 유지되나, Calmar 기준 우위는 벤치마크 대비 약 0~14 bps 에서 소멸합니다.

### 4-5. 비중 상한 (10 inds)

**회전율을 40% 감소시키면서 성과를 희생하지 않습니다.**

| x_max | Turnover | HHI | Calmar (tc=0) | Calmar (tc=40bps) |
| --- | --- | --- | --- | --- |
| 1.0 | 0.600 | 0.472 | 0.233 | 0.144 |
| 0.6 | 0.516 | 0.413 | 0.256 | 0.179 |
| 0.3 | 0.359 | 0.265 | 0.287 | 0.231 |

거래비용 40 bps 에서 연환산 수익 손실이 3.04%p → 1.86%p 로 **39% 축소**됩니다. 다만 상한은 DFL-MVO 의 집중(HHI 1.0 → 0.28)도 함께 완화하므로, **DFL-MDD 의 상대적 낙폭 우위는 축소**됩니다 (α=0.05 기준 7/8 → 4/8). 이는 기존 우위의 일부가 drawdown 제약이 유도한 분산 효과였음을 시사합니다.

---

## 5. 재현

전 결과는 프로젝트 루트의 체크포인트(`checkpoint/*.pkl`)에서 재생성 가능합니다.

```
학습    run_dfl_mdd.py / run_dfl_mvo.py   (launch_*.py 로 병렬 실행)
병합    merge_ckpt.py
분석    10_inds.ipynb / 30_inds.ipynb / 10_inds_wcap.ipynb
```

체크포인트 명명 규칙:

```
dfl_mdd_{N}_inds_h{H}[_xm{cap}][_LB{lb}][_n1{n1}]_d{δ}_l{λ}_{solver}.pkl
```

대괄호 태그는 기본값이 아닐 때만 붙습니다 (x_max=1.0, LB={252,504}, 병합 후 n1 태그 제거).

---

## 6. 미실시 항목

| 항목 | 사유 |
| --- | --- |
| 30 inds weight cap | 지도교수 판단으로 생략 |
| H=252 의 LB=1260 | **미학습.** LB=1260 체크포인트는 H=126 에만 존재하므로 H=252 의 LB 비교는 252/504 두 개로만 가능 |
| DFL-MDD δ sweep | δ=20 만 실행. δ 분석은 DFL-MVO 기준이며, 동일 목적함수이므로 단위 불일치 논리는 공유하나 거동은 미검증 |
| 예측값 Δŷ 직접 측정 | 모델 state_dict 미저장. 임계 δ 에서 역산한 추정치 사용 |
