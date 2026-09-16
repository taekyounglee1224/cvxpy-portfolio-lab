"""
launch_dfl_mvo.py
─────────────────
DFL-MVO delta sweep을 (delta, lambda) shard로 전개해 프로세스 풀로 돌린다.

왜 풀인가
---------
노트북의 기존 sweep 셀은 delta를 순차로 돌면서 delta마다 lambda 4개만 병렬로
띄웠다. delta 9개 × lambda 4개면 작업이 36개인데 동시 4개만 쓰니 24코어 중
4개(17%)만 돈다. delta끼리는 의존이 없고 체크포인트도
dfl_mvo_..._d{D}_l{lam}_...pkl 로 조합마다 갈리므로 36개를 한꺼번에 굴려도 된다.

shard 하나 = LOOKBACK 2개 × fold 8개 = 16 유닛으로 36개가 모두 같은 크기다.
따라서 동시 18개면 정확히 2라운드로 떨어지고 낙오(straggler)가 없다.
LOOKBACK으로 더 쪼갤 이유는 없다 (총 시간이 같고 체크포인트만 늘어난다).

메모리
------
run_dfl_mvo.py에 float32 WindowSet 수정이 적용된 뒤 기준으로 30 inds / LB=504 /
fold 8에서 워커당 ~1.4GB다. 동시 18개면 약 25GB로 31.5GB 안에 들어간다.
수정 전(워커당 ~1.8GB)이라면 18개는 32GB를 넘기니 --jobs 를 14로 낮출 것.

사용법
------
  python launch_dfl_mvo.py --data 30 --horizon 126 --jobs 18
  python launch_dfl_mvo.py --data 30 --horizon 126 --jobs 18 --dry-run

체크포인트는 노트북이 읽는 이름 그대로라 병합(merge)이 필요 없다.
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
                help="동시 실행 수 (기본: 코어수-6, 최대 18)")
ap.add_argument("--xmax", type=float, default=1.0,
                help="자산별 비중 상한. 1.0이 아니면 체크포인트/로그 이름에 _xm 태그")
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--log-dir", default="./logs")
ap.add_argument("--n-folds", type=int, default=8,
                help="완료 판정 기준 fold 수 (끝난 shard는 건너뜀)")
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
print(f"코어 {os.cpu_count()}  동시 실행 {JOBS}")
print(f"shard {len(shards) + done}개 중 실행 {len(shards)}개 "
      f"(이미 완료 {done}개 건너뜀)")
if shards:
    rounds = -(-len(shards) // JOBS)
    print(f"예상 {rounds}라운드 × 16유닛\n")

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
    # 로그에 '—' '═' 등이 있어 cp949로는 인코딩 실패 → 자식 stdout을 UTF-8로
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.Popen(
        [args.python, "-u", "run_dfl_mvo.py",
         "--data", args.data, "--horizon", str(args.horizon),
         "--solver", args.solver,
         "--delta", str(_d(delta)), "--lam", str(lam),
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
            if rc != 0:
                failed.append(name)
            mark = "완료" if rc == 0 else f"실패(rc={rc})"
            print(f"[+{int(time.time() - t_start):5d}s] {mark} {name} "
                  f"[{int(time.time() - t0)}s]  "
                  f"남은 {len(pending)} / 실행중 {len(still)}", flush=True)
        running = still
except KeyboardInterrupt:
    print("\n중단 — 실행 중인 프로세스를 종료합니다 "
          "(체크포인트는 fold 단위 저장이라 재실행하면 이어집니다)")
    for _, p, f, _ in running:
        p.terminate(); f.close()
    raise SystemExit(130)

print(f"\n전체 완료 — {int(time.time() - t_start)}초")
if failed:
    print(f"실패 {len(failed)}개: {failed}  (logs/ 확인 후 재실행하면 이어짐)")
else:
    print("체크포인트는 노트북이 읽는 이름 그대로입니다 (merge 불필요).")
