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


def _png(w=448, h=448):
    """A small RGB gradient PNG (stdlib), as a data URL for vision decision models."""
    import base64, struct, zlib
    raw = b"".join(b"\x00" + b"".join(bytes((x * 255 // w, y * 255 // h, 128)) for x in range(w))
                   for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


TEST_IMAGE = _png()

BUF_RE = None


def parse_buffers(path):
    """Sum llama.cpp's '<dev> <kind> buffer size = N MiB' lines by kind."""
    import re
    global BUF_RE
    BUF_RE = BUF_RE or re.compile(r"(\S+)\s+(model|KV|RS|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
    out = {}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = BUF_RE.search(line)
                if m and not m.group(1).startswith("CPU"):
                    k = m.group(2)
                    out[k] = round(out.get(k, 0) + float(m.group(3)), 2)
    except OSError:
        pass
    return out


class Fitter:
    """Measures one model's device memory at (slots, per-slot context) points."""

    def __init__(self, args):
        self.args, self.cache = args, {}
        self.base = gpu_used_mib(args.gpu)
        self.url = f"http://127.0.0.1:{args.port}"
        self.logpath = (args.out or "fit") + ".server.log"
        self.resume()

    @staticmethod
    def cost_of(row):
        lanes = row.get("lanes")
        full = (row.get("loaded") and lanes is not None and lanes.get("ok") == row["np"]
                and not row.get("guard"))
        peak = row.get("peak_delta_mib")
        return peak if full and peak is not None else float("inf")

    def resume(self):
        """Reuse points an earlier, interrupted run of this model already measured."""
        if not self.args.out or not os.path.exists(self.args.out):
            return
        with open(self.args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("mode") == "fit" and not r.get("skipped"):
                    self.cache[(r["np"], r["ctx_per_slot"])] = self.cost_of(r)
        log(f"resumed {len(self.cache)} measured points from {self.args.out}")

    def measure(self, np_, ctx):
        key = (np_, ctx)
        if key in self.cache:
            return self.cache[key]
        a = self.args
        total = np_ * ctx if not a.unified else ctx
        if total > a.max_ctx_total:
            # Host safety (2026-11-25: 32 lanes x ~10K on the desktop 3090 ran with the host's
            # load average at 55): never load more total context than this; extrapolate past it.
            emit(a, {"mode": "fit", "model": os.path.basename(a.model), "np": np_,
                     "ctx_per_slot": ctx, "ctx_total": total, "skipped": "max-ctx-total"})
            self.cache[key] = float("inf")
            return float("inf")
        cmd = [a.server, "-m", a.model, "--host", "127.0.0.1", "--port", str(a.port),
               "-ngl", "99", "-c", str(total), "-np", str(np_), "--cache-ram", "0",
               "-b", str(a.batch or ctx), "-ub", str(a.batch or ctx)]  # decision prompts run in one ubatch: >= per-slot ctx
        if a.mmproj:
            cmd += ["--mmproj", a.mmproj]
        if a.device:
            cmd += ["--device", a.device]
        if a.unified:
            cmd += ["--kv-unified"]
        cmd += a.extra.split() if a.extra else []
        with open(self.logpath, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

        def kill(p=proc):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        guard = Guard(a, on_trip=kill)
        guard.start()
        lanes = None
        try:
            ok = wait_health(self.url, proc, timeout=a.load_timeout)
            time.sleep(1.5)
            used = gpu_used_mib(a.gpu) if ok else None
            if ok and not guard.tripped:
                lanes = self.fill_lanes(np_, ctx, guard)
        finally:
            guard.stop()
            kill()
            proc.wait()
        bufs = parse_buffers(self.logpath)
        dev_sum = round(sum(bufs.values()), 1) if bufs else None
        row = {"mode": "fit", "model": os.path.basename(a.model), "mmproj": bool(a.mmproj),
               "device": a.device, "np": np_, "ctx_per_slot": ctx, "ctx_total": total,
               "unified": a.unified, "loaded": ok, "guard": guard.tripped,
               "buffers_mib": bufs, "buffers_sum_mib": dev_sum,
               "gpu_used_delta_mib": (used - self.base) if ok and None not in (used, self.base) else None,
               "lanes": lanes,
               "peak_vram_mib": guard.peak_vram,
               "peak_delta_mib": (guard.peak_vram - self.base) if self.base is not None else None,
               "min_host_avail_mib": guard.min_host}
        emit(a, row)
        log(f"np={np_} ctx={ctx}: loaded={ok} idle={row['gpu_used_delta_mib']} "
            f"peak={row['peak_delta_mib']} lanes={lanes and lanes['ok']}/{np_} guard={guard.tripped}")
        # A point fits only if it loaded, every lane answered a full-context request
        # at once, and the peak device memory over load + that full load fits.
        cost = self.cost_of(row)
        self.cache[key] = cost
        time.sleep(a.pause)
        return cost

    def tokens(self, text):
        req = urllib.request.Request(self.url + "/tokenize", data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return len(json.load(r)["tokens"])

    def fill_lanes(self, np_, ctx, guard):
        """Send np_ concurrent decision requests, each sized to fill its slot's context."""
        a = self.args
        target = max(64, ctx - a.lane_margin)
        text = state_text(target)
        try:  # trim the state to the target by the server's own tokenizer
            n = self.tokens(text)
            while n > target and len(text) > 100:
                text = text[: int(len(text) * target / n * 0.98)]
                n = self.tokens(text)
        except Exception:
            n = None
        body = {"state": text, "questions": questions(a.lane_questions)}
        if a.mmproj:
            body["images"] = [TEST_IMAGE]
        results = [None] * np_

        def one(i):
            results[i] = post(self.url, body, a.lane_timeout)

        t0 = time.monotonic()
        ts = [threading.Thread(target=one, args=(i,)) for i in range(np_)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        codes = {}
        for r in results:
            c = r[1] if r else 0
            codes[str(c)] = codes.get(str(c), 0) + 1
        lats = sorted(r[0] for r in results if r and r[1] == 200)
        return {"state_tokens": n, "questions": a.lane_questions, "image": bool(a.mmproj),
                "ok": len(lats), "codes": codes, "wall_s": round(time.monotonic() - t0, 2),
                "max_latency_s": round(lats[-1], 2) if lats else None}

    def max_np(self, ctx, budget):
        """Largest slot count fitting `budget` at this per-slot context (binary search)."""
        if self.measure(1, ctx) > budget:
            return 0
        lo, hi = 1, self.args.np_max
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.measure(mid, ctx) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def max_ctx(self, np_, lo, budget):
        """Largest per-slot context (multiple of the fine step) fitting at np_."""
        step = self.args.ctx_fine
        hi = self.args.ctx_max
        lo_i, hi_i = lo // step, hi // step
        while lo_i < hi_i:
            mid = (lo_i + hi_i + 1) // 2
            if self.measure(np_, mid * step) <= budget:
                lo_i = mid
            else:
                hi_i = mid - 1
        return lo_i * step


def cmd_fit(args):
    f = Fitter(args)
    log(f"GPU {args.gpu} used before fit: {f.base} MiB; budgets {args.budgets_mib}")
    for budget in args.budgets_mib:
        frontier = {}
        ctx = args.ctx_coarse
        while ctx <= args.ctx_max:
            n = f.max_np(ctx, budget)
            if n == 0:
                break
            frontier[n] = ctx  # the largest coarse ctx at which n is still the max
            ctx += args.ctx_coarse
        for n, c in sorted(frontier.items()):
            best = f.max_ctx(n, c, budget)
            emit(args, {"mode": "fit-result", "model": os.path.basename(args.model),
                        "budget_mib": budget, "np": n, "max_ctx_per_slot": best})
            log(f"budget {budget}: np={n} -> max ctx/slot {best}")


def cmd_points(args):
    """Measure fixed (slots x per-slot context) points, e.g. to check another quant's offset."""
    f = Fitter(args)
    log(f"GPU {args.gpu} used before points: {f.base} MiB")
    for n in args.np_list:
        for c in args.ctx_list:
            f.measure(n, c)


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

    for name in ("fit", "points"):
        s = sub.add_parser(name, help="slot/context fit search against VRAM budgets" if name == "fit"
                           else "measure fixed slot x context points")

        if name == "points":
            s.add_argument("--np-list", type=ints, default=[1, 4, 8])
            s.add_argument("--ctx-list", type=ints, default=[2048, 8192])
        _fit_args(s)
    args = p.parse_args()
    {"load": cmd_load, "sweep": cmd_sweep, "fit": cmd_fit, "points": cmd_points}[args.cmd](args)


def _fit_args(s):
    s.add_argument("--server", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--mmproj")
    s.add_argument("--device", help="e.g. Vulkan0, as the 970 tier runs")
    s.add_argument("--budgets-mib", type=ints, default=[2048, 3584, 4096, 6144, 8192, 12288])
    s.add_argument("--ctx-coarse", type=int, default=2048)
    s.add_argument("--ctx-fine", type=int, default=256)
    s.add_argument("--ctx-max", type=int, default=32768)
    s.add_argument("--np-max", type=int, default=32)
    s.add_argument("--batch", type=int, default=0, help="-b/-ub; 0 = the per-slot context (a decision prompt must fit one ubatch)")
    s.add_argument("--unified", action="store_true", help="-c is the shared total, not per slot")
    s.add_argument("--extra", default="")
    s.add_argument("--port", type=int, default=18080)
    s.add_argument("--max-ctx-total", type=int, default=131072,
                   help="never load a point with more total context (host safety); extrapolate past it")
    s.add_argument("--pause", type=int, default=15, help="seconds between points")
    s.add_argument("--load-timeout", type=int, default=300)
    s.add_argument("--lane-margin", type=int, default=512,
                   help="tokens per slot left for questions, template and image")
    s.add_argument("--lane-questions", type=int, default=12)
    s.add_argument("--lane-timeout", type=int, default=600)
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--vram-limit-mib", type=int, default=24000)
    s.add_argument("--host-mem-floor-mib", type=int, default=6144)
    s.add_argument("--out")


if __name__ == "__main__":
    main()
