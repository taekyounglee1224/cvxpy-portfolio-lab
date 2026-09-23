"""
run_xmax_sweep.py
-----------------
Run the weight-cap (x_max) experiment: one x_max at a time, parallel within each.

Per x_max
    1. DFL-MDD, 16 shards (4 lambda x 4 n1), --jobs at a time
    2. merge_ckpt.py builds one checkpoint per lambda
    3. DFL-MVO, 4 shards (4 lambda, delta=20), 4 at a time
  then move on to the next x_max

Because x_max=0.3 finishes completely before 0.6 starts, the 0.3 results can be
analysed while 0.6 is still running.

Usage
------
  python run_xmax_sweep.py --data 30 --horizon 126 --xmax 0.3 0.6 --jobs 16
  python run_xmax_sweep.py --data 30 --horizon 126 --xmax 0.3 0.6 --dry-run

Checkpoints carry an _xm tag, which keeps them separate from the uncapped
(x_max=1.0) results.
    dfl_mdd_30_inds_h126_xm0.3_d20_l0.5_CLARABEL.pkl
"""

import argparse
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--data",    default="30", choices=["10", "30"])
ap.add_argument("--horizon", type=int, default=126)
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--delta",   type=float, default=20.0)
ap.add_argument("--xmax",    type=float, nargs="+", default=[0.3, 0.6],
                help="weight caps to run, in order")
ap.add_argument("--jobs",    type=int, default=None,
                help="number of concurrent DFL-MDD jobs (default: cores-4, capped at 16)")
ap.add_argument("--mvo-jobs", type=int, default=4)
ap.add_argument("--skip-mvo", action="store_true", help="skip the DFL-MVO stage")
ap.add_argument("--python",  default=sys.executable)
ap.add_argument("--log-dir", default="./logs")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

JOBS = args.jobs or max(1, min(16, (os.cpu_count() or 8) - 4))
DELTA = int(args.delta) if float(args.delta).is_integer() else args.delta
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
os.makedirs(args.log_dir, exist_ok=True)
T0 = time.time()


def el():
    s = int(time.time() - T0)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def run(desc, cmd):
    """Run a subcommand and block. A failure is recorded but does not stop the sweep."""
    print(f"\n[+{el()}] start  {desc}", flush=True)
    print(f"          {' '.join(cmd[1:])}", flush=True)
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd, env=ENV).returncode
    mark = "ok" if rc == 0 else f"failed (rc={rc})"
    print(f"[+{el()}] done   {desc} -- {mark}", flush=True)
    return rc


print(f"x_max sweep starting -- {args.data} inds, h{args.horizon}, "
      f"x_max {args.xmax}, {JOBS} concurrent DFL-MDD jobs")

failed = []
for xm in args.xmax:
    tag = f"x_max={xm:g}"
    print(f"\n{'=' * 70}\n  {tag}\n{'=' * 70}", flush=True)

    # 1. DFL-MDD (4 lambda x 4 n1 = 16 shards)
    rc = run(f"[{tag}] DFL-MDD training",
             [args.python, "-u", "launch_dfl_mdd.py",
              "--data", args.data, "--horizon", str(args.horizon),
              "--solver", args.solver, "--jobs", str(JOBS),
              "--xmax", f"{xm:g}", "--python", args.python])
    if rc: failed.append(f"{tag} DFL-MDD")

    # 2. merge the _n1 shards into one checkpoint per lambda
    rc = run(f"[{tag}] DFL-MDD merge",
             [args.python, "merge_ckpt.py",
              "--data", args.data, "--horizon", str(args.horizon),
              "--delta", str(DELTA), "--solver", args.solver,
              "--xmax", f"{xm:g}"])
    if rc: failed.append(f"{tag} merge")

    # 3. DFL-MVO (4 lambda, delta fixed)
    if not args.skip_mvo:
        rc = run(f"[{tag}] DFL-MVO training",
                 [args.python, "-u", "launch_dfl_mvo.py",
                  "--data", args.data, "--horizon", str(args.horizon),
                  "--delta", str(DELTA), "--solver", args.solver,
                  "--jobs", str(args.mvo_jobs),
                  "--xmax", f"{xm:g}", "--python", args.python])
        if rc: failed.append(f"{tag} DFL-MVO")

    print(f"\n[+{el()}] {tag} complete", flush=True)

print(f"\n{'=' * 70}")
print(f"sweep finished -- total {el()}")
if failed:
    print(f"{len(failed)} stage(s) failed: {failed}")
    print("  check logs/ and re-run; shards that finished are skipped")
else:
    print("all stages completed")
    print(f"checkpoints: checkpoint/dfl_mdd_{args.data}_inds_h{args.horizon}_xm*_d{DELTA}_l*.pkl")
