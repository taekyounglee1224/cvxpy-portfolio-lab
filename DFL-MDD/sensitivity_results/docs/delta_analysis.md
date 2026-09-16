# δ(위험회피 계수)가 작동하지 않는 이유

> 30 industries, H=126, LB=252, λ=0.5 기준.
> 모든 수치는 `checkpoint/dfl_mvo_30_inds_h126_d*_l0.5_CLARABEL.pkl` 및 `csv/30_industry.csv`에서 직접 계산.

---

## 1. 관측된 현상

DFL-MVO에서 δ를 20부터 10,000까지 바꿔가며 실행한 결과입니다.

| δ | HHI | 평균 보유종목 | 최대 비중 |
| --- | --- | --- | --- |
| 20 | 1.0000 | 1.0 | 1.000 |
| 50 | 1.0000 | 1.0 | 1.000 |
| 100 | 0.9968 | 1.0 | 0.998 |
| 200 | 0.9932 | 1.0 | 0.995 |
| 500 | 1.0000 | 1.0 | 1.000 |
| 1,000 | 1.0000 | 1.0 | 1.000 |
| 2,000 | 0.9951 | 1.0 | 0.996 |
| **5,000** | **0.6815** | **1.9** | 0.736 |
| **10,000** | **0.4289** | **3.7** | 0.529 |

**δ ≤ 2,000에서는 한 종목에 100% 몰빵**입니다. δ를 100배 키워도 포트폴리오 구조가 변하지 않습니다. 분산이 시작되는 것은 δ=5,000부터입니다.

같은 시점에서 δ=20과 δ=10,000의 비중을 비교하면:

```
δ =    20 : Chems 1.000
δ = 10000 : Util 0.427, Smoke 0.289, Meals 0.146, Mines 0.138
```

---

## 2. 수식 유도 — 코너해 이탈 조건

### 2.1 문제 설정

최적화 레이어의 목적함수입니다 (`dfl_mvo.py: build_mvo_layer`).

$$\max_{x} \; f(x) = \hat{y}_N^\top x - \frac{\delta}{2} x^\top \Sigma x
\quad \text{s.t.} \quad \mathbf{1}^\top x = 1,\; x \ge 0$$

- $\hat{y}_N$ : 예측 누적수익 벡터 (N일 누적)
- $\Sigma$ : 공분산행렬 (일별)

### 2.2 코너해에서의 섭동

코너해 $x = e_i$ (자산 $i$에 100%)에서, 예산제약을 지키며 자산 $j$로 비중 $\varepsilon$을 옮깁니다.

$$x(\varepsilon) = (1-\varepsilon)e_i + \varepsilon e_j, \qquad \varepsilon \in [0,1]$$

**수익항**

$$\hat{y}^\top x(\varepsilon) = (1-\varepsilon)\hat{y}_i + \varepsilon \hat{y}_j = \hat{y}_i + \varepsilon(\hat{y}_j - \hat{y}_i)$$

**위험항**

$$x(\varepsilon)^\top \Sigma x(\varepsilon) = (1-\varepsilon)^2\Sigma_{ii} + 2\varepsilon(1-\varepsilon)\Sigma_{ij} + \varepsilon^2\Sigma_{jj}$$

$\varepsilon$으로 미분해 $\varepsilon = 0$에서 평가하면

$$\left.\frac{d}{d\varepsilon}\left[x^\top \Sigma x\right]\right|_{0} = -2\Sigma_{ii} + 2\Sigma_{ij}$$

### 2.3 유지 조건과 이탈 조건

$$\left.\frac{df}{d\varepsilon}\right|_{0}
= (\hat{y}_j - \hat{y}_i) - \frac{\delta}{2}\left(-2\Sigma_{ii} + 2\Sigma_{ij}\right)
= (\hat{y}_j - \hat{y}_i) + \delta\left(\Sigma_{ii} - \Sigma_{ij}\right)$$

이 값의 **부호**가 모든 것을 결정합니다.

| $df/d\varepsilon\rvert_0$ | 의미 | 결과 |
| --- | --- | --- |
| $\le 0$ | 움직이면 목적함수가 나빠짐 | **코너 유지** (몰빵) |
| $> 0$ | 움직이면 목적함수가 좋아짐 | **코너 이탈** (분산 시작) |

**코너 유지 조건** — $e_i$가 최적이려면 모든 $j$에 대해 $df/d\varepsilon|_0 \le 0$이어야 하므로

$$\boxed{\;\hat{y}_i - \hat{y}_j \;\ge\; \delta\left(\Sigma_{ii} - \Sigma_{ij}\right)\;}$$

좌변은 "i를 고집해서 얻는 수익 우위", 우변은 "j로 분산해서 줄어드는 위험"입니다. **수익 우위가 더 크면 몰빵을 유지**합니다.

**코너 이탈 조건** — 위 부등호가 깨지는 경우입니다.

$$\hat{y}_i - \hat{y}_j \;<\; \delta\left(\Sigma_{ii} - \Sigma_{ij}\right)$$

$\delta > 0$이고 $\Sigma$가 양반정치이므로 목적함수가 오목(concave)합니다. 따라서 이 1차 조건은 **전역 최적의 필요충분조건**입니다.

**이탈에 필요한 δ** — 이탈 조건을 $\delta$에 대해 풀면

$$\delta \;>\; \delta^{*} = \frac{\Delta\hat{y}}{\Sigma_{ii} - \Sigma_{ij}}, \qquad \Delta\hat{y} \equiv \hat{y}_i - \hat{y}_j$$

즉 **δ가 $\delta^{*}$를 넘어서야 분산이 시작**됩니다.

핵심은 분모가 $\Sigma_{ii}$가 아니라 $\Sigma_{ii} - \Sigma_{ij}$라는 점입니다. 두 자산이 같이 움직이는 부분($\Sigma_{ij}$)만큼은 갈아타도 위험이 줄지 않기 때문입니다.

---

## 3. 실제 데이터 계산

### 3.1 공분산 추정 방식

```python
z_b = z_raw[b].detach().numpy()          # (252, 30) 일별 수익률
S   = np.cov(z_b.T) + 1e-4 * np.eye(m)   # (30, 30)
```

lookback 252일의 **일별** 수익률 표본공분산에 ridge $10^{-4}I$를 더합니다.
테스트 구간 32개 리밸런싱 시점에서 각각 계산해 평균냈습니다.

### 3.2 분모 $\Sigma_{ii} - \Sigma_{ij}$

| | $\Sigma_{ii}$ | $\Sigma_{ij}$ | $\Sigma_{ii}-\Sigma_{ij}$ | 비율 | 평균 상관 $\rho$ |
| --- | --- | --- | --- | --- | --- |
| ridge 포함 (최적화가 쓰는 값) | 3.821e−4 | 1.496e−4 | **2.325e−4** | **60.8%** | 0.354 |
| ridge 제외 (원시 상관) | 2.821e−4 | 1.496e−4 | 1.325e−4 | 47.0% | **0.543** |

- 평균 상관 $\rho$는 $C_{ij} = \Sigma_{ij}/\sqrt{\Sigma_{ii}\Sigma_{jj}}$의 비대각 원소(30×29 = 870개) 평균
- **산업 간 실제 평균 상관은 0.543**이며, 0.354는 ridge가 대각을 26% 부풀린 결과
- 비율 검산: $1 - \Sigma_{ij}/\Sigma_{ii} = 1 - 0.392 = 0.608$ ✓
- 모든 분산이 같다면 이 비율은 정확히 $1-\rho = 0.646$. 자산별 분산 차이로 0.608

**상관 때문에 실효 곡률이 61%로 깎여, 같은 효과를 내려면 δ가 1.6배 더 커야 합니다.**

### 3.3 필요한 δ 역산

$$\delta^{*} = \frac{\Delta\hat{y}}{2.325\times10^{-4}}$$

| $\Delta\hat{y}$ (126일 누적) | 필요 δ |
| --- | --- |
| 0.01 (1%p) | 43 |
| 0.05 (5%p) | 215 |
| 0.10 (10%p) | 430 |
| 0.20 (20%p) | 860 |
| 0.30 (30%p) | 1,290 |

**관측된 임계 δ ≈ 5,000을 역산하면**

$$\Delta\hat{y} = 5000 \times 2.325\times10^{-4} \approx 1.16$$

126일 누적 기준 **116%p**입니다. δ=2,000에서도 코너가 유지됐으므로 최소 46%p 이상입니다.

### 3.4 단위 불일치 검증

목적함수 두 항의 실제 크기입니다.

```
수익항  ŷ·x    ≈ 0.036      (126일 누적, ~3.6%)
xᵀΣx           ≈ 8.96e-5    (일별 분산)
```

| δ | $(\delta/2)\,x^\top\Sigma x$ | 수익항 대비 |
| --- | --- | --- |
| 20 | 0.00090 | **2.5%** |
| 500 | 0.02241 | 62.3% |
| 803 | 0.036 | **100%** |
| 10,000 | 0.44813 | 1,246% |

교과서 MVO는 두 항을 **같은 기간 단위**로 맞춥니다. $\Sigma_N \approx N \cdot \Sigma_{\text{daily}}$이므로

$$\frac{\delta}{2}\Sigma_{\text{daily}} = \frac{\delta_{\text{std}}}{2}\Sigma_N
= \frac{\delta_{\text{std}}}{2}\cdot N \cdot\Sigma_{\text{daily}}
\;\Rightarrow\; \delta = \delta_{\text{std}} \times N$$

통상 $\delta_{\text{std}} \approx 5$이므로 $\delta \approx 5 \times 126 = 630$이 여기서의 "정상값"입니다.

---

## 4. 원인 분해

관측 임계 δ = 5,000을 세 요인으로 나누면

| 요인 | 기여 |
| --- | --- |
| ① **단위 불일치** — 일별 Σ vs 126일 누적 ŷ | ~126배 |
| ② **자산 간 상관** — 실효 곡률 61%로 축소 | ~1.6배 |
| ③ **예측값 스케일 팽창** | 나머지 **6~12배** |

①과 ②만으로는 $\delta \approx 630 \sim 860$이 설명됩니다. 현실적인 격차 $\Delta\hat{y} = 0.10 \sim 0.20$을 가정하면 필요 δ는 430~860입니다. **관측된 5,000과는 6~12배 차이**가 남습니다.

### ③의 정체 — task loss에 예측 정확도 항이 없음

$$\mathcal{L} = \lambda\,(-\text{Sharpe}) + (1-\lambda)\,\text{MDD}_{\text{real}}$$

이 loss는 $\hat{y}$에 **$x^*$를 통해서만** 영향을 받습니다. $x^*$는 $\hat{y}$의 **순서(ranking)**에만 의존하고 크기에는 무관합니다 — 코너해에서는 argmax만 중요하기 때문입니다.

따라서 **$\hat{y}$가 실제 수익률과 같은 스케일일 이유가 없고**, 학습이 진행되며 자유롭게 팽창합니다. 억제 요인은 δ가 붙은 위험항뿐인데, δ=20에서 그것은 수익항의 2.5%에 불과해 제동이 걸리지 않습니다.

대조군이 이를 뒷받침합니다. **PTO-MDD는 MSE로 학습**해 $\hat{y}$가 실제 수익률에 고정(anchor)되며, HHI가 0.82~0.90으로 DFL-MVO(1.0)보다 덜 극단적입니다.

---

## 5. 결론 — 쉬운 말로

**무슨 일이 일어났나.**
목적함수는 "수익은 크게, 위험은 작게"인데, **수익은 6개월치로 재고 위험은 하루치로 쟀습니다.** 저울 한쪽에 6개월을, 다른 쪽에 하루를 올려놓은 셈이라 위험 쪽이 126배 가볍습니다. δ는 그 저울추인데, 원래 5쯤이면 될 것이 **126배를 대신 떠안아 630 정도**가 되어야 균형이 맞습니다.

**왜 그마저도 부족했나.**
두 가지가 더 겹쳤습니다. 첫째, 산업들이 서로 비슷하게 움직여서(상관 0.54) **A에서 B로 갈아타도 위험이 기대만큼 줄지 않습니다.** 그만큼 저울추가 1.6배 더 무거워야 합니다. 둘째, 더 결정적으로 — **모델이 "이 자산이 저 자산보다 116%p 더 오른다"는 식의 과장된 예측**을 내놓습니다. 수익 쪽이 이렇게 무거우면 어지간한 저울추로는 움직이지 않습니다.

**왜 모델이 과장하나.**
학습 목표가 "예측을 맞혀라"가 아니라 "포트폴리오 성과를 높여라"이기 때문입니다. **어느 자산이 1등인지 순서만 맞으면 되고, 숫자의 크기는 아무 벌점도 받지 않습니다.** 그래서 크기가 제멋대로 커집니다. MSE로 학습하는 PTO에는 이 현상이 없습니다.

**그래서 무슨 뜻인가.**
위험항이 사실상 꺼져 있으면 문제가 선형계획이 되고, **선형계획의 답은 항상 꼭짓점 — 즉 1등 자산 몰빵**입니다. δ를 20에서 2,000까지 100배 올려도 "어느 종목에 몰빵할지"만 바뀔 뿐 분산은 일어나지 않습니다. 실제로 이 구간에서 Calmar가 0.154 → 0.401 → 0.077로 요동치는데, 이는 분산 효과가 아니라 **종목이 바뀐 데서 오는 노이즈**입니다.

**그럼 DFL-MDD는 왜 분산되어 있나.**
δ 때문이 아니라 **drawdown 제약** 때문입니다.

$$u_k - y_k^\top x \le n_1 C, \quad u_k \ge y_k^\top x, \quad u_k \ge u_{k-1}$$

이 제약들이 단체(simplex)의 꼭짓점을 잘라내서, 몰빵이 제약을 위반하면 내부해로 밀려납니다. DFL-MDD의 HHI가 0.16~0.69로 나오는 이유이며, **$n_1$이 작을수록 HHI가 낮아지는** 관측과 일치합니다.

**최종 결론.**
본 프레임워크에서 **δ는 실질적 위험 통제 수단이 아닙니다.** 그 역할은 drawdown 제약 $n_1$이 담당하며, 이것이 DFL-MDD가 DFL-MVO 대비 낙폭에서 우위를 보이는 메커니즘입니다. δ sweep 결과는 이 구조를 뒷받침하는 ablation 근거로 해석해야 합니다.

---

## 6. 남은 검증 과제

| 항목 | 현재 상태 | 확인 방법 |
| --- | --- | --- |
| $\Delta\hat{y} \approx 1.16$ | **추정값** (임계 δ에서 역산) | config 1개 재학습 후 $\hat{y}$ 분포 직접 측정 (~30분) |
| DFL-MDD의 δ 민감도 | **미검증** (δ=20만 실행) | λ=0.5 고정, δ ∈ {20, 500, 2000, 10000} sweep |

서술 시 **δ sweep은 DFL-MVO에서만 실증됐다**는 점을 명시해야 합니다. DFL-MDD는 목적함수가 같아 단위 불일치 논리는 동일하게 적용되지만, δ를 키웠을 때의 실제 거동은 확인되지 않았습니다.

---

## 7. Future work로 연결

진단이 자연스럽게 개선안을 시사합니다.

**(a) 스케일 정합** — $\Sigma_N = N\cdot\Sigma_{\text{daily}}$로 연율화하면 δ가 교과서 범위(2~10)로 복귀합니다. 코드 한 줄입니다.

```python
S = N * (np.cov(z_b.T) + 1e-4 * np.eye(m_dim))
```

**(b) 예측 스케일 고정** — loss에 MSE 항을 소량 섞어 $\hat{y}$를 실제 수익률에 anchor합니다.

$$\mathcal{L} = \lambda(-\text{Sharpe}) + (1-\lambda)\text{MDD} + \eta\,\text{MSE}(\hat{r}, r)$$

(b)가 근본 처방입니다. (a)만으로는 ③의 6~12배가 남기 때문입니다.

---

## 부록 — 재현 코드

### A. δ sweep 요약 (§1 표)

```python
import pickle, numpy as np
DS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
LAM, LB = 0.5, 252
for d in DS:
    ck = pickle.load(open(
        f"checkpoint/dfl_mvo_30_inds_h126_d{d}_l{LAM}_CLARABEL.pkl", "rb"))
    w = np.array([r["weights"] for r in ck["fold_results_map"][LB]], dtype=float)
    print(f"δ={d:>6}  HHI={(w**2).sum(1).mean():.4f}  "
          f"종목={(w > 1e-4).sum(1).mean():.1f}  최대={w.max(1).mean():.4f}")
```

### B. 공분산 통계 (§3.2 표)

```python
import numpy as np, pandas as pd
inds = pd.read_csv("csv/30_industry.csv"); inds["Date"] = pd.to_datetime(inds["Date"])
inds = inds.set_index("Date").sort_index()
inds = inds[~inds.index.duplicated(keep="first")] / 100.
X, m, LB = inds.values, 30, 252
t0 = max(inds.index.searchsorted(pd.Timestamp("2018-01-01")), LB)

for ridge in (1e-4, 0.0):
    dii, dij, gap, rho = [], [], [], []
    for k in range(0, 94, 3):
        i = t0 + k * 21
        if i > len(X): break
        S = np.cov(X[i-LB:i].T) + ridge * np.eye(m)
        d, M = np.diag(S), ~np.eye(m, dtype=bool)
        dii.append(d.mean()); dij.append(S[M].mean())
        gap.append((d[:, None] - S)[M].mean())
        rho.append((S / np.sqrt(np.outer(d, d)))[M].mean())
    a, b, g, p = map(np.mean, (dii, dij, gap, rho))
    print(f"ridge={ridge:g}  Σii={a:.3e}  Σij={b:.3e}  "
          f"Σii-Σij={g:.3e} ({g/a:.1%})  ρ={p:.3f}")
```
