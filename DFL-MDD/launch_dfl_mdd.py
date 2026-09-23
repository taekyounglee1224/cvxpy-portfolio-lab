"""
launch_dfl_mdd.py
-----------------
Expand DFL-MDD training into (lam, LOOKBACK, n1) shards and run them through a
process pool with a cap on concurrency.

Why shard
------------
dfl_mdd.solve_portfolio solves the LPs in a batch one at a time, and run_dfl_mdd.py
pins BLAS and torch to a single thread, so one process uses exactly one core.
Splitting on lambda alone gives 4 processes, which leaves 4 of 24 cores busy (17%).
Splitting on lam(4) x LB(2) x n1(4) gives 32 shards, enough to fill the machine.

Why a pool
---------------
Launching all 32 shards at once exhausts RAM first. For 30 industries at LB=504 in
the later folds each process holds about 1GB, peaking near 1.5GB where the float64
sample list, its np.array copy and the float32 tensor overlap. On a 31.5GB machine
that caps concurrency at roughly 18 to 20.

Having more shards than workers also absorbs stragglers caused by the speed
difference between performance and efficiency cores.

Heavy shards (LB=504) are dispatched first, longest-processing-time first, which
shortens the tail.

Usage
------
  python launch_dfl_mdd.py --data 30 --horizon 126 --jobs 18
  python launch_dfl_mdd.py --data 30 --horizon 252 --jobs 18 --dry-run

Always merge afterwards:
  python merge_ckpt.py --data 30 --horizon 126
"""

import argparse
import itertools
import os
import pickle
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--data",    default="30", choices=["10", "30"])
ap.add_argument("--horizon", type=int, default=126)
ap.add_argument("--delta",   type=float, default=20.0)
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--lam", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
ap.add_argument("--lb",  type=int,   nargs="+", default=[252, 504])
ap.add_argument("--n1",  type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
ap.add_argument("--jobs", type=int, default=None,
                help="number of concurrent processes (default: cores-4, capped at 18). "
                     "More than 18 is not recommended with 31.5GB of RAM at 30 industries")
ap.add_argument("--xmax", type=float, default=1.0,
                help="per-asset weight cap; anything other than 1.0 adds an _xm tag to the "
                     "checkpoint and log names")
ap.add_argument("--python", default=sys.executable,
                help="python executable to train with (default: the current interpreter)")
ap.add_argument("--log-dir", default="./logs")
ap.add_argument("--n-folds", type=int, default=8,
                help="fold count that counts as complete; finished shards are skipped")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

N_STOCKS = int(args.data)
DELTA    = int(args.delta) if float(args.delta).is_integer() else args.delta
JOBS     = args.jobs or max(1, min(18, (os.cpu_count() or 8) - 4))
XMTAG    = "" if args.xmax >= 1.0 else f"_xm{args.xmax:g}"

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs("./checkpoint", exist_ok=True)


def ckpt_path(lam, lb, n1):
    return os.path.join(
        "./checkpoint",
        f"dfl_mdd_{N_STOCKS}_inds_h{args.horizon}{XMTAG}_LB{lb}_n1{n1:g}"
        f"_d{DELTA}_l{lam}_{args.solver}.pkl")


def already_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            return pickle.load(f).get("completed_fold", 0) >= args.n_folds
    except Exception:
        return False


# ---- expand the shards, heaviest (largest LOOKBACK) first ----
shards, done = [], 0
for lb, lam, n1 in sorted(itertools.product(args.lb, args.lam, args.n1),
                          key=lambda t: -t[0]):
    if already_done(ckpt_path(lam, lb, n1)):
        done += 1
        continue
    shards.append((lam, lb, n1))

print(f"data={args.data} inds  h={args.horizon}  d={DELTA}  solver={args.solver}")
print(f"cores {os.cpu_count()}  concurrency {JOBS}")
print(f"running {len(shards)} of {len(shards) + done} shards "
      f"({done} already complete, skipped)\n")

if args.dry_run:
    for lam, lb, n1 in shards:
        print(f"  {args.python} -u run_dfl_mdd.py --data {args.data} "
              f"--horizon {args.horizon} --delta {DELTA} --solver {args.solver} "
              f"--lam {lam} --lb {lb} --n1 {n1:g} --xmax {args.xmax:g}")
    raise SystemExit(0)

pending = list(shards)
running = []          # [(name, Popen, filehandle, t0)]
t_start = time.time()
failed  = []


def spawn(shard):
    lam, lb, n1 = shard
    name = f"l{lam}_LB{lb}_n1{n1:g}"
    log_path = os.path.join(
        args.log_dir,
        f"mdd_{N_STOCKS}_h{args.horizon}{XMTAG}_LB{lb}_n1{n1:g}_l{lam}.txt")
    f = open(log_path, "w", encoding="utf-8")
    # The logs contain characters that cp949 (the default on Korean Windows) cannot
    # encode, so force the child process stdout to UTF-8.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen(
        [args.python, "-u", "run_dfl_mdd.py",
         "--data", args.data, "--horizon", str(args.horizon),
         "--delta", str(DELTA), "--solver", args.solver,
         "--lam", str(lam), "--lb", str(lb), "--n1", f"{n1:g}",
         "--xmax", str(args.xmax)],
        stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"[+{int(time.time() - t_start):5d}s] start {name} "
          f"(PID {p.pid}) -> {log_path}", flush=True)
    return (name, p, f, time.time())


try:
    while pending or running:
        while pending and len(running) < JOBS:
            running.append(spawn(pending.pop(0)))

        time.sleep(2)

        still = []
        for name, p, f, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((name, p, f, t0))
                continue
            f.close()
            mark = "ok" if rc == 0 else f"failed (rc={rc})"
            if rc != 0:
                failed.append(name)
            print(f"[+{int(time.time() - t_start):5d}s] {mark} {name} "
                  f"[{int(time.time() - t0)}s]  "
                  f"pending {len(pending)} / running {len(still)}", flush=True)
        running = still
except KeyboardInterrupt:
    print("\ninterrupted -- terminating the running processes. "
          "Checkpoints are written per fold, so re-running resumes where it stopped.")
    for _, p, f, _ in running:
        p.terminate(); f.close()
    raise SystemExit(130)

print(f"\nall done -- {int(time.time() - t_start)}s")
if failed:
    print(f"{len(failed)} failed: {failed}  (check logs/ and re-run to resume)")
else:
    print(f"next: python merge_ckpt.py --data {args.data} "
          f"--horizon {args.horizon} --delta {DELTA} --solver {args.solver}")
