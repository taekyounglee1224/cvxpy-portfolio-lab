"""
merge_ckpt.py
-------------
Merge the (lam, LOOKBACK, n1) shards produced by run_dfl_mdd.py into the single
checkpoint per lambda that the notebooks read.

  shard : dfl_mdd_30_inds_h126_n10.2_d20_l0.3_CLARABEL.pkl        (lam x n1)
          dfl_mdd_30_inds_h126_LB504_d20_l0.3_CLARABEL.pkl        (lam x LB)
          dfl_mdd_30_inds_h126_LB504_n10.2_d20_l0.3_CLARABEL.pkl  (lam x LB x n1)
  merged: dfl_mdd_30_inds_h126_d20_l0.3_CLARABEL.pkl
          (the name the analysis notebooks look for)

Checkpoints tagged with a value outside --lb / --n1 (a separate experiment such
as _LB1260) are excluded automatically.

fold_results_map and infeas_map are keyed by (LOOKBACK, n1), so merging is a plain
dict update; a key appearing in more than one shard raises an error.

Usage
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
                help="LOOKBACK values to merge; shards tagged with any other LB are skipped")
ap.add_argument("--n1", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4],
                help="n1 values to merge; shards tagged with any other n1 are skipped")
ap.add_argument("--xmax", type=float, default=1.0,
                help="weight cap; anything other than 1.0 targets the _xm-tagged checkpoints")
ap.add_argument("--out-tag", default=None,
                help="tag appended to the merged filename. With the default (auto) a "
                     "_LB<value> tag is added whenever --lb differs from the "
                     "default grid (252 504). For example --lb 1260 gives "
                     "dfl_mdd_30_inds_h126_LB1260_d20_l0.3_CLARABEL.pkl, matching "
                     "_load('_LB1260') in the notebooks. Pass an empty string for no tag.")
ap.add_argument("--ckpt-dir", default="./checkpoint")
ap.add_argument("--force",   action="store_true",
                help="overwrite the merged file if it already exists")
ap.add_argument("--dry-run", action="store_true",
                help="report what would be merged without writing anything")
args = ap.parse_args()

N_STOCKS = int(args.data)
DELTA    = int(args.delta) if float(args.delta).is_integer() else args.delta
_XMTAG   = "" if args.xmax >= 1.0 else f"_xm{args.xmax:g}"
BASE     = f"dfl_mdd_{N_STOCKS}_inds_h{args.horizon}{_XMTAG}"
OK_LB    = {float(v) for v in args.lb}
# output tag: when LB differs from the default grid (252, 504), keep it in the
# filename so the run stays separate from the existing results
if args.out_tag is None:
    OUT_TAG = ("" if OK_LB == {252.0, 504.0}
               else f"_LB{'-'.join(str(int(v)) for v in sorted(args.lb))}")
else:
    OUT_TAG = args.out_tag
OK_N1    = {float(v) for v in args.n1}
TAG_RE   = re.compile(r"^(?:_LB([\d\-]+))?(?:_n1([\d.\-]+))?$")


# When the LB grid is not the default (252, 504), an untagged shard belongs to the
# default grid and is not a target, so the _LB tag becomes mandatory.
LB_TAG_REQUIRED = OK_LB != {252.0, 504.0}


def shard_tag_ok(mid):
    """Decide whether the middle filename tag is a shard tag within --lb / --n1."""
    if not mid:                       # no tag means an already-merged file
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
        print(f"  - lam={lam}: no shards ({os.path.basename(pattern)})")
        n_skip += 1
        continue

    if os.path.exists(out_path) and not args.force:
        print(f"  ! lam={lam}: {os.path.basename(out_path)} already exists; "
              f"use --force to overwrite")
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
                f"duplicate config keys {sorted(dup)} in {os.path.basename(sp)}\n"
                f"The shard ranges overlap. Delete the overlapping checkpoint and re-run.")
        fold_results_map.update(ck["fold_results_map"])
        infeas_map.update(ck.get("infeas_map", {}))
        completed.append(ck["completed_fold"])
        meta = {"delta_val": ck.get("delta_val", DELTA),
                "lam_val"  : ck.get("lam_val", lam),
                "solver"   : ck.get("solver", args.solver)}

    lo, hi = min(completed), max(completed)
    keys   = sorted(fold_results_map)
    status = f"fold {lo}" + (f"-{hi} (some shards unfinished)" if lo != hi else " complete")
    print(f"  lam={lam}: {len(shards)} shards -> {len(keys)} configs, {status}")
    print(f"      {keys}")
    expected = len(OK_LB) * len(OK_N1)
    if len(keys) != expected:
        print(f"      ! {len(keys)}/{expected} configs -- some shards are missing")

    if args.dry_run:
        continue

    with open(out_path, "wb") as f:
        pickle.dump({"fold_results_map": fold_results_map,
                     "infeas_map"      : infeas_map,
                     "completed_fold"  : lo,   # based on the least advanced shard
                     **meta}, f)
    print(f"      -> {os.path.basename(out_path)}")
    n_ok += 1

print(f"\nmerged {n_ok}, skipped {n_skip}"
      + ("  (dry-run)" if args.dry_run else ""))
