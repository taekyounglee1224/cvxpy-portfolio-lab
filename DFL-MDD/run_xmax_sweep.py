"""
run_xmax_sweep.py
─────────────────
weight cap(x_max) 실험을 x_max 값마다 순차로, 각 값 안에서는 병렬로 실행한다.

흐름 (x_max 하나당)
    1. DFL-MDD  shard 16개 (λ4 × n₁4)  — 동시 --jobs 개
    2. merge_ckpt.py 로 λ별 통짜 체크포인트 생성
    3. DFL-MVO  shard 4개 (λ4, δ=20)   — 동시 4개
  → 끝나면 다음 x_max 로 이동

x_max=0.3 이 완전히 끝난 뒤 0.6 이 시작되므로, 0.6 이 도는 동안 0.3 결과로
분석을 진행할 수 있다.

사용법
------
  python run_xmax_sweep.py --data 30 --horizon 126 --xmax 0.3 0.6 --jobs 16
  python run_xmax_sweep.py --data 30 --horizon 126 --xmax 0.3 0.6 --dry-run

체크포인트 이름에 _xm 태그가 붙어 기존(x_max=1.0) 결과와 분리된다.
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
                help="순서대로 실행할 비중 상한 목록")
ap.add_argument("--jobs",    type=int, default=None,
                help="DFL-MDD 동시 실행 수 (기본: 코어수-4, 최대 16)")
ap.add_argument("--mvo-jobs", type=int, default=4)
ap.add_argument("--skip-mvo", action="store_true", help="DFL-MVO 생략")
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
    """하위 명령을 블로킹 실행. 실패해도 다음 단계로 넘어가되 기록은 남긴다."""
    print(f"\n[+{el()}] ▶ {desc}", flush=True)
    print(f"          {' '.join(cmd[1:])}", flush=True)
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd, env=ENV).returncode
    mark = "완료" if rc == 0 else f"실패(rc={rc})"
    print(f"[+{el()}] ◀ {desc} — {mark}", flush=True)
    return rc


print(f"x_max sweep 시작 — {args.data} inds, h{args.horizon}, "
      f"x_max {args.xmax}, DFL-MDD 동시 {JOBS}개")

failed = []
for xm in args.xmax:
    tag = f"x_max={xm:g}"
    print(f"\n{'=' * 70}\n  {tag}\n{'=' * 70}", flush=True)

    # 1. DFL-MDD (λ4 × n₁4 = 16 shard)
    rc = run(f"[{tag}] DFL-MDD 학습",
             [args.python, "-u", "launch_dfl_mdd.py",
              "--data", args.data, "--horizon", str(args.horizon),
              "--solver", args.solver, "--jobs", str(JOBS),
              "--xmax", f"{xm:g}", "--python", args.python])
    if rc: failed.append(f"{tag} DFL-MDD")

    # 2. 병합 — shard(_n1 태그) → λ별 통짜
    rc = run(f"[{tag}] DFL-MDD 병합",
             [args.python, "merge_ckpt.py",
              "--data", args.data, "--horizon", str(args.horizon),
              "--delta", str(DELTA), "--solver", args.solver,
              "--xmax", f"{xm:g}"])
    if rc: failed.append(f"{tag} merge")

    # 3. DFL-MVO (λ4, δ 고정)
    if not args.skip_mvo:
        rc = run(f"[{tag}] DFL-MVO 학습",
                 [args.python, "-u", "launch_dfl_mvo.py",
                  "--data", args.data, "--horizon", str(args.horizon),
                  "--delta", str(DELTA), "--solver", args.solver,
                  "--jobs", str(args.mvo_jobs),
                  "--xmax", f"{xm:g}", "--python", args.python])
        if rc: failed.append(f"{tag} DFL-MVO")

    print(f"\n[+{el()}] ★ {tag} 전체 완료", flush=True)

print(f"\n{'=' * 70}")
print(f"sweep 종료 — 총 {el()}")
if failed:
    print(f"실패 단계 {len(failed)}개: {failed}")
    print("  → logs/ 확인 후 재실행하면 완료된 shard 는 건너뜁니다")
else:
    print("모든 단계 정상 완료")
    print(f"체크포인트: checkpoint/dfl_mdd_{args.data}_inds_h{args.horizon}_xm*_d{DELTA}_l*.pkl")
