"""
launch_dfl_mdd.py
─────────────────
DFL-MDD 학습을 (lam, LOOKBACK, n1) shard로 전개하고, 동시 실행 수를 제한한
프로세스 풀로 돌린다.

왜 shard인가
------------
dfl_mdd.solve_portfolio는 배치 안의 LP를 한 개씩 순차로 풀고, run_dfl_mdd.py는
BLAS/torch 스레드를 1로 고정한다. 즉 **프로세스 1개 = 코어 1개**다.
lambda로만 4분할하면 24코어 중 4개(17%)만 쓴다.
lam(4) × LB(2) × n1(4) = shard 32개로 쪼개면 코어를 채울 수 있다.

왜 풀(pool)인가
---------------
shard 32개를 한꺼번에 띄우면 RAM이 먼저 터진다. 30 inds / LB=504 / 뒤쪽 fold
기준 프로세스당 정상 ~1GB, 피크 ~1.5GB다 (train 샘플 리스트 float64 + np.array
복사본 + float32 텐서가 겹치는 구간). 31.5GB 머신이면 동시 18~20개가 상한이다.
또 shard 수 > 워커 수라서 P/E 코어 속도 차로 생기는 낙오(straggler)도 흡수된다.

무거운 shard(LB=504)를 먼저 던져(LPT) 꼬리 시간을 줄인다.

사용법
------
  python launch_dfl_mdd.py --data 30 --horizon 126 --jobs 18
  python launch_dfl_mdd.py --data 30 --horizon 252 --jobs 18 --dry-run

끝나면 반드시 병합:
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
                help="동시 실행 프로세스 수 (기본: 코어수-4, 최대 18). "
                     "RAM 31.5GB / 30 inds 기준 18 이상은 권장하지 않음")
ap.add_argument("--xmax", type=float, default=1.0,
                help="자산별 비중 상한. 1.0이 아니면 체크포인트/로그 이름에 _xm 태그")
ap.add_argument("--python", default=sys.executable,
                help="학습에 쓸 python 실행파일 (기본: 현재 인터프리터)")
ap.add_argument("--log-dir", default="./logs")
ap.add_argument("--n-folds", type=int, default=8,
                help="완료 판정 기준 fold 수 (이미 끝난 shard는 건너뜀)")
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


# ── shard 전개: 무거운 것(LOOKBACK 큰 것) 먼저 ──
shards, done = [], 0
for lb, lam, n1 in sorted(itertools.product(args.lb, args.lam, args.n1),
                          key=lambda t: -t[0]):
    if already_done(ckpt_path(lam, lb, n1)):
        done += 1
        continue
    shards.append((lam, lb, n1))

print(f"data={args.data} inds  h={args.horizon}  d={DELTA}  solver={args.solver}")
print(f"코어 {os.cpu_count()}  동시 실행 {JOBS}")
print(f"shard {len(shards) + done}개 중 실행 {len(shards)}개 "
      f"(이미 완료 {done}개 건너뜀)\n")

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
    # 로그에 '—' '═' '✓' 등이 있어 cp949(한글 Windows 기본)로는 인코딩 실패한다.
    # 자식 프로세스의 stdout 인코딩을 UTF-8로 강제.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen(
        [args.python, "-u", "run_dfl_mdd.py",
         "--data", args.data, "--horizon", str(args.horizon),
         "--delta", str(DELTA), "--solver", args.solver,
         "--lam", str(lam), "--lb", str(lb), "--n1", f"{n1:g}",
         "--xmax", str(args.xmax)],
        stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"[+{int(time.time() - t_start):5d}s] 시작 {name} "
          f"(PID {p.pid}) → {log_path}", flush=True)
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
            mark = "완료" if rc == 0 else f"실패(rc={rc})"
            if rc != 0:
                failed.append(name)
            print(f"[+{int(time.time() - t_start):5d}s] {mark} {name} "
                  f"[{int(time.time() - t0)}s]  "
                  f"남은 {len(pending)} / 실행중 {len(still)}", flush=True)
        running = still
except KeyboardInterrupt:
    print("\n중단 — 실행 중인 프로세스를 종료합니다 "
          "(체크포인트는 fold 단위로 저장되어 있으니 재실행하면 이어서 진행됩니다)")
    for _, p, f, _ in running:
        p.terminate(); f.close()
    raise SystemExit(130)

print(f"\n전체 완료 — {int(time.time() - t_start)}초")
if failed:
    print(f"실패 {len(failed)}개: {failed}  (logs/ 확인 후 재실행하면 이어서 진행)")
else:
    print(f"다음: python merge_ckpt.py --data {args.data} "
          f"--horizon {args.horizon} --delta {DELTA} --solver {args.solver}")
