"""
launch_dfl_mvo.py
-----------------
Expand the DFL-MVO delta sweep into (delta, lambda) shards and run them through a
process pool.

Why a pool
---------
The sweep cell in the notebooks stepped through delta sequentially and ran only the
four lambdas in parallel. With 9 deltas and 4 lambdas that is 36 jobs but only 4 at a
time, leaving 4 of 24 cores busy (17%). The deltas are independent and each
combination writes its own checkpoint (dfl_mvo_..._d{D}_l{lam}_...pkl), so all 36
can run at once.

Each shard is 2 lookbacks x 8 folds = 16 units, so all 36 shards are the same size.
At a concurrency of 18 that divides into exactly two rounds with no stragglers.
There is no reason to split further on LOOKBACK: the total time is unchanged and it
only multiplies the checkpoints.

Memory
------
With the float32 WindowSet change in run_dfl_mvo.py, a worker holds about 1.4GB at
30 industries, LB=504, fold 8. Eighteen workers come to roughly 25GB, which fits in
31.5GB. Without that change a worker holds about 1.8GB and 18 workers would exceed
32GB, so lower --jobs to 14.

Usage
------
  python launch_dfl_mvo.py --data 30 --horizon 126 --jobs 18
  python launch_dfl_mvo.py --data 30 --horizon 126 --jobs 18 --dry-run

Checkpoints already carry the names the notebooks read, so no merge step is needed.
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
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--delta", type=float, nargs="+",
                default=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
ap.add_argument("--lam",   type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
ap.add_argument("--jobs", type=int, default=None,
                help="number of concurrent processes (default: cores-6, capped at 18)")
ap.add_argument("--xmax", type=float, default=1.0,
                help="per-asset weight cap; anything other than 1.0 adds an _xm tag to the "
                     "checkpoint and log names")
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--log-dir", default="./logs")
ap.add_argument("--n-folds", type=int, default=8,
                help="fold count that counts as complete; finished shards are skipped")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

N_STOCKS = int(args.data)
JOBS     = args.jobs or max(1, min(18, (os.cpu_count() or 8) - 6))
XMTAG    = "" if args.xmax >= 1.0 else f"_xm{args.xmax:g}"
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs("./checkpoint", exist_ok=True)


def _d(v):
    return int(v) if float(v).is_integer() else v


def ckpt_path(delta, lam):
    return os.path.join(
        "./checkpoint",
        f"dfl_mvo_{N_STOCKS}_inds_h{args.horizon}{XMTAG}"
        f"_d{_d(delta)}_l{lam}_{args.solver}.pkl")


def already_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            return pickle.load(f).get("completed_fold", 0) >= args.n_folds
    except Exception:
        return False


shards, done = [], 0
for delta, lam in itertools.product(args.delta, args.lam):
    if already_done(ckpt_path(delta, lam)):
        done += 1
        continue
    shards.append((delta, lam))

print(f"data={args.data} inds  h={args.horizon}  solver={args.solver}")
print(f"cores {os.cpu_count()}  concurrency {JOBS}")
print(f"running {len(shards)} of {len(shards) + done} shards "
      f"({done} already complete, skipped)")
if shards:
    rounds = -(-len(shards) // JOBS)
    print(f"about {rounds} round(s) x 16 units\n")

if args.dry_run:
    for delta, lam in shards:
        print(f"  {args.python} -u run_dfl_mvo.py --data {args.data} "
              f"--horizon {args.horizon} --solver {args.solver} "
              f"--delta {_d(delta)} --lam {lam} --xmax {args.xmax:g}")
    raise SystemExit(0)

pending = list(shards)
running = []
t_start = time.time()
failed  = []


def spawn(shard):
    delta, lam = shard
    name = f"d{_d(delta)}_l{lam}"
    log_path = os.path.join(
        args.log_dir,
        f"dflmvo_{N_STOCKS}_h{args.horizon}{XMTAG}_d{_d(delta)}_l{lam}.txt")
    f = open(log_path, "w", encoding="utf-8")
    # The logs contain characters cp949 cannot encode, so force child stdout to UTF-8.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen(
        [args.python, "-u", "run_dfl_mvo.py",
         "--data", args.data, "--horizon", str(args.horizon),
         "--solver", args.solver,
         "--delta", str(_d(delta)), "--lam", str(lam),
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
            if rc != 0:
                failed.append(name)
            mark = "ok" if rc == 0 else f"failed (rc={rc})"
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
    print("Checkpoints already use the names the notebooks read; no merge needed.")
