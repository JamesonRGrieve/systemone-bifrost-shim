#!/usr/bin/env python3
"""Decision-tier benchmark: how many d1 decisions a GPU serves in parallel.

Stdlib only, so it runs as-is inside the llama.cpp CT. Two modes:

  load   Drive an already-running llama-server's /v1/systemone at a ladder of
         concurrencies and request sizes. Changes nothing on the server.

  sweep  Start a llama-server of its own on a spare port for each
         (--np, --ctx) pair, record its VRAM, run the same load ladder, stop
         it. The prod d1 server must be stopped first (it holds the VRAM), and
         the run must be agreed with the tier's users. A guard samples
         nvidia-smi every 0.5 s and kills the bench server the moment the GPU's
         used memory passes --vram-limit-mib (a GTX 970's fast segment ends at
         3584 MiB), or the host's MemAvailable drops under --host-mem-floor-mib.

Every result is one JSON line on stdout (and --out, if given); progress goes to
stderr.

  python3 decision_bench.py load --url http://127.0.0.1:8080
  python3 decision_bench.py sweep --model /opt/models/d1-2B-Q8_0.gguf \
      --mmproj /opt/models/mmproj-d1-2B-Q8_0.gguf --server /opt/llamacpp/llama-server \
      --np 2,4,6,8 --ctx 4096,8192,16384 --out /root/bench-970.jsonl
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = ("the captain watches the corridor while the servitor hums and the "
         "guard captain argues with the adept about the sealed door ").split()


def state_text(tokens):
    """About `tokens` tokens of plausible scene state (~0.75 words/token)."""
    n = max(1, int(tokens * 0.75))
    return " ".join(WORDS[i % len(WORDS)] for i in range(n))


def questions(n):
    """n questions, cycling the three d1 types the way rpg-ai mixes them."""
    out = {}
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        q = {"type": kind, "instructions": f"Question {i}: judge the scene."}
        if kind == "choice":
            q["criteria"] = {"A": "", "B": "", "C": ""}
        elif kind == "score":
            q["criteria"] = ["low", "mid", "high"]
        out[f"q{i}"] = q
    return out


def post(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return time.monotonic() - t0, r.status
    except urllib.error.HTTPError as e:
        return time.monotonic() - t0, e.code
    except Exception:
        return time.monotonic() - t0, 0


def gpu_used_mib(index):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def host_avail_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return None


class Guard(threading.Thread):
    """Samples VRAM and host memory; trips (and runs on_trip) past a limit."""

    def __init__(self, args, on_trip=None):
        super().__init__(daemon=True)
        self.args, self.on_trip = args, on_trip
        self.peak_vram, self.min_host = 0, None
        self.tripped, self.stop_ev = None, threading.Event()

    def run(self):
        while not self.stop_ev.wait(0.5):
            v = gpu_used_mib(self.args.gpu)
            h = host_avail_mib()
            if v is not None:
                self.peak_vram = max(self.peak_vram, v)
            if h is not None:
                self.min_host = h if self.min_host is None else min(self.min_host, h)
            reason = None
            if v is not None and v > self.args.vram_limit_mib:
                reason = f"vram {v} MiB > {self.args.vram_limit_mib}"
            elif h is not None and h < self.args.host_mem_floor_mib:
                reason = f"host MemAvailable {h} MiB < {self.args.host_mem_floor_mib}"
            if reason and not self.tripped:
                self.tripped = reason
                log(f"GUARD TRIPPED: {reason}")
                if self.on_trip:
                    self.on_trip()
                return

    def stop(self):
        self.stop_ev.set()


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
          file=sys.stderr, flush=True)


def emit(args, row):
    line = json.dumps(row)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


def run_cell(args, url, conc, nq, st, guard, extra):
    """Hold `conc` requests in flight for args.duration seconds."""
    body = {"state": state_text(st), "questions": questions(nq)}
    lats, codes, lock = [], {}, threading.Lock()
    deadline = time.monotonic() + args.duration

    def worker():
        while time.monotonic() < deadline and not (guard and guard.tripped):
            lat, code = post(url, body, args.timeout)
            with lock:
                codes[code] = codes.get(code, 0) + 1
                if code == 200:
                    lats.append(lat)

    t0 = time.monotonic()
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t0
    ok = len(lats)
    lats.sort()
    row = dict(extra, concurrency=conc, questions=nq, state_tokens=st,
               wall_s=round(wall, 2), ok=ok,
               errors={str(k): v for k, v in codes.items() if k != 200},
               decisions_per_s=round(ok / wall, 3) if wall else 0,
               answers_per_s=round(ok * nq / wall, 2) if wall else 0,
               p50_s=round(statistics.median(lats), 3) if lats else None,
               p95_s=round(lats[int(0.95 * (ok - 1))], 3) if lats else None,
               max_s=round(lats[-1], 3) if lats else None)
    if guard:
        row.update(peak_vram_mib=guard.peak_vram, min_host_avail_mib=guard.min_host,
                   guard=guard.tripped)
    emit(args, row)
    log(f"c={conc} q={nq} st={st}: {ok} ok, p50 {row['p50_s']} s, "
        f"{row['answers_per_s']} answers/s, errors {row['errors']}")


def ladder(args, url, guard, extra):
    for st in args.state_tokens:
        for nq in args.questions:
            for conc in args.concurrency:
                if guard and guard.tripped:
                    return
                run_cell(args, url, conc, nq, st, guard, extra)


def wait_health(url, proc, timeout=300):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def cmd_load(args):
    guard = Guard(args)
    guard.start()
    ladder(args, args.url, guard, {"mode": "load", "url": args.url})
    guard.stop()


def cmd_sweep(args):
    base = gpu_used_mib(args.gpu)
    log(f"GPU {args.gpu} used before sweep: {base} MiB")
    if base is not None and base > args.max_start_used_mib:
        sys.exit(f"refusing: GPU {args.gpu} already uses {base} MiB "
                 f"(> --max-start-used-mib {args.max_start_used_mib}); stop the prod d1 server first")
    url = f"http://127.0.0.1:{args.port}"
    for np_ in args.np:
        for ctx in args.ctx:
            cmd = [args.server, "-m", args.model, "--host", "127.0.0.1",
                   "--port", str(args.port), "-ngl", "99", "-c", str(ctx),
                   "-np", str(np_), "-b", str(min(ctx, 4096)), "-ub", str(min(ctx, 4096))]
            if args.mmproj:
                cmd += ["--mmproj", args.mmproj]
            cmd += args.extra.split() if args.extra else []
            extra = {"mode": "sweep", "np": np_, "ctx": ctx, "extra_args": args.extra,
                     "gpu_used_before_mib": base}
            log("start: " + " ".join(cmd))
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)

            def kill(p=proc):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            guard = Guard(args, on_trip=kill)
            guard.start()
            try:
                if not wait_health(url, proc):
                    emit(args, dict(extra, loaded=False, guard=guard.tripped,
                                    peak_vram_mib=guard.peak_vram,
                                    exit=proc.poll()))
                    continue
                time.sleep(2)
                loaded = gpu_used_mib(args.gpu)
                emit(args, dict(extra, loaded=True, vram_loaded_mib=loaded,
                                vram_model_mib=(loaded - base) if None not in (loaded, base) else None))
                ladder(args, url, guard, extra)
            finally:
                guard.stop()
                kill()
                proc.wait()
                time.sleep(3)


def ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("load", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--concurrency", type=ints, default=[1, 2, 4, 6, 8])
        s.add_argument("--questions", type=ints, default=[1, 12, 48])
        s.add_argument("--state-tokens", type=ints, default=[300, 1000])
        s.add_argument("--duration", type=int, default=45, help="seconds per cell")
        s.add_argument("--timeout", type=int, default=120, help="per request, like the shim")
        s.add_argument("--gpu", type=int, default=0)
        s.add_argument("--vram-limit-mib", type=int, default=3400)
        s.add_argument("--host-mem-floor-mib", type=int, default=2048)
        s.add_argument("--out")
        if name == "load":
            s.add_argument("--url", default="http://127.0.0.1:8080")
        else:
            s.add_argument("--server", required=True)
            s.add_argument("--model", required=True)
            s.add_argument("--mmproj")
            s.add_argument("--np", type=ints, default=[2, 4, 6, 8])
            s.add_argument("--ctx", type=ints, default=[4096, 8192, 16384])
            s.add_argument("--port", type=int, default=18080)
            s.add_argument("--extra", default="", help="more llama-server args, e.g. '--kv-unified -fa on'")
            s.add_argument("--max-start-used-mib", type=int, default=1200,
                           help="refuse to start if the GPU already uses more (prod d1 still up)")
    args = p.parse_args()
    (cmd_load if args.cmd == "load" else cmd_sweep)(args)


if __name__ == "__main__":
    main()
