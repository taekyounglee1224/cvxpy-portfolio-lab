"""
merge_ckpt.py
─────────────
run_dfl_mdd.py를 (lam, LOOKBACK, n1) shard로 쪼개 돌린 뒤,
노트북이 읽는 통짜 체크포인트 하나로 합친다.

  shard : dfl_mdd_30_inds_h126_n10.2_d20_l0.3_CLARABEL.pkl        (λ×n1 분할)
          dfl_mdd_30_inds_h126_LB504_d20_l0.3_CLARABEL.pkl        (λ×LB 분할)
          dfl_mdd_30_inds_h126_LB504_n10.2_d20_l0.3_CLARABEL.pkl  (λ×LB×n1 분할)
  통합  : dfl_mdd_30_inds_h126_d20_l0.3_CLARABEL.pkl
          (= 노트북 Cell 21이 찾는 이름)

--lb / --n1 에 없는 값이 붙은 체크포인트(예: _LB1260 같은 별도 실험)는
자동으로 제외한다.

fold_results_map / infeas_map은 (LOOKBACK, n1) 키로 갈라져 있으므로 단순 병합이며,
같은 키가 여러 shard에 중복되면 에러로 막는다.

사용법
------
  python merge_ckpt.py --data 30 --horizon 126
  python merge_ckpt.py --data 30 --horizon 252 --lam 0.3 0.5 --force
  python merge_ckpt.py --data 30 --horizon 126 --dry-run
"""

import argparse
import glob
import os
import pickle
import re

ap = argparse.ArgumentParser()
ap.add_argument("--data",    default="30", choices=["10", "30"])
ap.add_argument("--horizon", type=int, default=126)
ap.add_argument("--delta",   type=float, default=20.0)
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--lam",     type=float, nargs="+",
                default=[0.3, 0.5, 0.7, 1.0])
ap.add_argument("--lb", type=int, nargs="+", default=[252, 504],
                help="합칠 LOOKBACK 값. 여기 없는 LB가 붙은 shard는 제외")
ap.add_argument("--n1", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4],
                help="합칠 n1 값. 여기 없는 n1이 붙은 shard는 제외")
ap.add_argument("--xmax", type=float, default=1.0,
                help="비중 상한. 1.0이 아니면 _xm 태그가 붙은 체크포인트를 대상으로 한다")
ap.add_argument("--out-tag", default=None,
                help="병합 결과 파일명에 붙일 태그. 기본(auto): --lb 가 기본값"
                     "(252 504)이 아니면 _LB<값> 을 붙인다. "
                     "예) --lb 1260 → dfl_mdd_30_inds_h126_LB1260_d20_l0.3_CLARABEL.pkl "
                     "(노트북의 _load('_LB1260') 과 동일). 빈 문자열이면 태그 없음.")
ap.add_argument("--ckpt-dir", default="./checkpoint")
ap.add_argument("--force",   action="store_true",
                help="통합 파일이 이미 있어도 덮어쓴다")
ap.add_argument("--dry-run", action="store_true",
                help="합치지 않고 무엇을 합칠지만 출력")
args = ap.parse_args()

N_STOCKS = int(args.data)
DELTA    = int(args.delta) if float(args.delta).is_integer() else args.delta
_XMTAG   = "" if args.xmax >= 1.0 else f"_xm{args.xmax:g}"
BASE     = f"dfl_mdd_{N_STOCKS}_inds_h{args.horizon}{_XMTAG}"
OK_LB    = {float(v) for v in args.lb}
# 출력 태그: LB가 기본 그리드(252,504)와 다르면 파일명에 남겨 기존 결과와 분리
if args.out_tag is None:
    OUT_TAG = ("" if OK_LB == {252.0, 504.0}
               else f"_LB{'-'.join(str(int(v)) for v in sorted(args.lb))}")
else:
    OUT_TAG = args.out_tag
OK_N1    = {float(v) for v in args.n1}
TAG_RE   = re.compile(r"^(?:_LB([\d\-]+))?(?:_n1([\d.\-]+))?$")


# LB 그리드가 기본(252,504)이 아니면, LB 태그가 없는 shard(=기본 그리드로 돌린 것)는
# 대상이 아니다. 이 경우 _LB 태그를 필수로 요구한다.
LB_TAG_REQUIRED = OK_LB != {252.0, 504.0}


def shard_tag_ok(mid):
    """파일명 중간 태그가 --lb/--n1 범위 안의 shard 태그인지 판정."""
    if not mid:                       # 태그 없음 = 통짜 파일
        return False
    m = TAG_RE.match(mid)
    if not m or not (m.group(1) or m.group(2)):
        return False
    if LB_TAG_REQUIRED and not m.group(1):
        return False
    if m.group(1) and not {float(v) for v in m.group(1).split("-")} <= OK_LB:
        return False
    if m.group(2) and not {float(v) for v in m.group(2).split("-")} <= OK_N1:
        return False
    return True

n_ok = n_skip = 0
for lam in args.lam:
    suffix   = f"_d{DELTA}_l{lam}_{args.solver}.pkl"
    pattern  = os.path.join(args.ckpt_dir, f"{BASE}_*{suffix}")
    out_path = os.path.join(args.ckpt_dir, f"{BASE}{OUT_TAG}{suffix}")
    shards   = sorted(
        sp for sp in glob.glob(pattern)
        if os.path.abspath(sp) != os.path.abspath(out_path)
        and shard_tag_ok(os.path.basename(sp)[len(BASE):-len(suffix)]))

    if not shards:
        print(f"  - lam={lam}: shard 없음 ({os.path.basename(pattern)})")
        n_skip += 1
        continue

    if os.path.exists(out_path) and not args.force:
        print(f"  ! lam={lam}: {os.path.basename(out_path)} 이미 존재 "
              f"— --force 로 덮어쓰기")
        n_skip += 1
        continue

    fold_results_map, infeas_map = {}, {}
    completed, meta = [], {}
    for sp in shards:
        with open(sp, "rb") as f:
            ck = pickle.load(f)
        dup = set(ck["fold_results_map"]) & set(fold_results_map)
        if dup:
            raise SystemExit(
                f"config 키 중복 {sorted(dup)} — {os.path.basename(sp)}\n"
                f"shard 범위가 겹칩니다. 겹치는 체크포인트를 지우고 다시 돌리세요.")
        fold_results_map.update(ck["fold_results_map"])
        infeas_map.update(ck.get("infeas_map", {}))
        completed.append(ck["completed_fold"])
        meta = {"delta_val": ck.get("delta_val", DELTA),
                "lam_val"  : ck.get("lam_val", lam),
                "solver"   : ck.get("solver", args.solver)}

    lo, hi = min(completed), max(completed)
    keys   = sorted(fold_results_map)
    status = f"fold {lo}" + (f"~{hi} (미완 shard 있음)" if lo != hi else " 완료")
    print(f"  ✓ lam={lam}: shard {len(shards)}개 → config {len(keys)}개, {status}")
    print(f"      {keys}")
    expected = len(OK_LB) * len(OK_N1)
    if len(keys) != expected:
        print(f"      ! config {len(keys)}/{expected}개 — 누락된 shard가 있습니다")

    if args.dry_run:
        continue

    with open(out_path, "wb") as f:
        pickle.dump({"fold_results_map": fold_results_map,
                     "infeas_map"      : infeas_map,
                     "completed_fold"  : lo,   # 가장 덜 끝난 shard 기준
                     **meta}, f)
    print(f"      → {os.path.basename(out_path)}")
    n_ok += 1

print(f"\n병합 {n_ok}개, 건너뜀 {n_skip}개"
      + ("  (dry-run)" if args.dry_run else ""))
