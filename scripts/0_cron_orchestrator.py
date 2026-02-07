from __future__ import annotations
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd

# -------- CONFIG (edit these values) --------
# Path to master CSV (will be created if missing)
MASTER_CSV = Path("assignments/zb_assgn/data/core/articles_master.csv")

# Location for previous snapshot(s) if you want to keep them
SNAPSHOT_DIR = Path("assignments/zb_assgn/data/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Job list: run commands in order. Each job must produce out_json (site map)
# out_json should be the exact path your crawler writes to.
# Update run_cmd to the exact invocation you use now to run each file.
JOB_CONFIGS = [
    {
        "name": "zipboard_help",
        "root_url": "https://help.zipboard.co",
        "out_json": "assignments/zb_assgn/data/site_map_zipboard.json",
        "run_cmd": ["python", "assignments/zb_assgn/scripts/crawl_job.py", "--root", "https://help.zipboard.co", "--out", "assignments/zb_assgn/data/site_map_zipboard.json"],
    },
    # Add more jobs here, they will run sequentially.
    # Example:
    # {
    #   "name": "other_docs",
    #   "root_url": "https://docs.example.com",
    #   "out_json": "assignments/zb_assgn/data/site_map_docs.json",
    #   "run_cmd": ["python", "assignments/zb_assgn/scripts/other_crawler.py", "--root", "https://docs.example.com", "--out", "assignments/zb_assgn/data/site_map_docs.json"],
    # },
]

# Optional: path for lock file to prevent overlapping runs
PIDFILE = Path("assignments/zb_assgn/data/cron_orchestrator.pid")

# CSV Columns schema
CSV_COLS = ["url", "title", "hash", "last_seen_at", "status", "site", "notes"]

# How many seconds to wait before giving up waiting for job-generated file (if job is long)
JOB_PRODUCE_TIMEOUT = 60 * 10  # 10 minutes


# -------- Helpers --------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def acquire_lock_or_exit():
    """Simple PID lock: prevents overlapping runs."""
    if PIDFILE.exists():
        try:
            pid = int(PIDFILE.read_text().strip())
        except Exception:
            pid = None
        # check if process still running
        if pid:
            if _pid_running(pid):
                print(f"[ERROR] Another run appears active (pid={pid}). Exiting.")
                sys.exit(1)
            else:
                # stale pid file
                PIDFILE.unlink(missing_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    PIDFILE.chmod(0o644)


def release_lock():
    try:
        if PIDFILE.exists():
            PIDFILE.unlink()
    except Exception:
        pass


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def load_json_safe(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ensure_master_csv():
    if not MASTER_CSV.exists():
        MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=CSV_COLS)
        df.to_csv(MASTER_CSV, index=False)


def read_master_df() -> pd.DataFrame:
    ensure_master_csv()
    df = pd.read_csv(MASTER_CSV, dtype=str).fillna("")
    # Ensure required columns exist
    for c in CSV_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[CSV_COLS].copy()


def atomic_write_csv(df: pd.DataFrame, target: Path):
    tmp = target.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    # optional backup
    bak = target.with_suffix(".bak")
    if target.exists():
        shutil.copy2(target, bak)
    tmp.replace(target)


def url_belongs_to_root(url: str, root_url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(root_url).netloc
    except Exception:
        return False


def wait_for_file(path: Path, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.5)
    return False


# -------- Core logic --------
def run_job_command(cmd: List[str], cwd: Path = Path(".")) -> int:
    """Run an external crawler command synchronously (blocking). Returns exit code."""
    print(f"[job] running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    rc = proc.wait()
    print(f"[job] exit code: {rc}")
    return rc


def process_job(job: Dict, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    job: dict with keys: name, root_url, out_json, run_cmd
    master_df: DataFrame read from master CSV (will be modified and returned)
    """
    name = job["name"]
    root_url = job["root_url"]
    out_json = Path(job["out_json"])

    # Step A: run the job command
    rc = run_job_command(job["run_cmd"])
    if rc != 0:
        print(f"[warning] job `{name}` exited with code {rc}. Continuing to next job.")
    else:
        print(f"[info] job `{name}` finished successfully.")

    # Step B: wait until out_json appears (some scripts write it after finishing)
    if not wait_for_file(out_json, JOB_PRODUCE_TIMEOUT):
        print(f"[warning] out_json {out_json} not present or empty after timeout.")
        site_map = {}
    else:
        site_map = load_json_safe(str(out_json))

    # optional: archive snapshot of site_map with timestamp
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_path = SNAPSHOT_DIR / f"{name}_snapshot_{ts}.json"
    try:
        snapshot_path.write_text(json.dumps(site_map, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # Flatten site_map to per-url dict -> expected format: site_map[url] = { "hash": "...", "title": "...", "links": [...] }
    # But we support flexible structure: try to read .get(url).get("hash") etc.
    new_entries = {}  # url -> {hash, title, site}
    for url, meta in site_map.items():
        if not url or not isinstance(meta, dict):
            continue
        content_hash = meta.get("hash") or meta.get("content_hash") or ""
        title = meta.get("title") or ""
        new_entries[url] = {"hash": content_hash, "title": title, "site": root_url}

    # Now apply changes to master_df
    now = now_iso()
    master_idx = {r["url"]: idx for idx, r in master_df.iterrows()}

    # Track urls seen in this job
    urls_seen = set(new_entries.keys())

    # 1. Handle additions and modifications & keep presence info
    for url, info in new_entries.items():
        if url in master_idx:
            idx = master_idx[url]
            existing_hash = master_df.at[idx, "hash"]
            if (existing_hash or "") != (info["hash"] or ""):
                # modified
                master_df.at[idx, "title"] = info["title"]
                master_df.at[idx, "hash"] = info["hash"]
                master_df.at[idx, "last_seen_at"] = now
                master_df.at[idx, "status"] = "Updated"
                master_df.at[idx, "site"] = info["site"]
            else:
                # unchanged: update last_seen_at and status=Active if previously removed?
                master_df.at[idx, "last_seen_at"] = now
                # if was removed earlier, bring back to Active
                if master_df.at[idx, "status"] in ("Removed", ""):
                    master_df.at[idx, "status"] = "Active"
        else:
            # new article
            row = {
                "url": url,
                "title": info["title"],
                "hash": info["hash"],
                "last_seen_at": now,
                "status": "Active",
                "site": info["site"],
                "notes": "",
            }
            master_df = pd.concat([master_df, pd.DataFrame([row])], ignore_index=True)
            # update index map for future iterations in same job
            master_idx[url] = len(master_df) - 1

    # 2. Handle removals: any row in master that belongs to this job's domain but is NOT seen in new_entries
    for idx, row in master_df.iterrows():
        url = row["url"]
        # skip blank urls
        if not url:
            continue
        if url_belongs_to_root(url, root_url):
            if url not in urls_seen and master_df.at[idx, "status"] != "Removed":
                master_df.at[idx, "status"] = "Removed"
                master_df.at[idx, "last_seen_at"] = now
                master_df.at[idx, "notes"] = f"Marked removed by job {name} at {now}"

    return master_df


def main():
    acquire_lock_or_exit()
    try:
        ensure_master_csv()
        master_df = read_master_df()

        for job in JOB_CONFIGS:
            print(f"\n--- Running job: {job['name']} ({job['root_url']}) ---")
            master_df = process_job(job, master_df)
            # short pause between jobs
            time.sleep(0.4)

        # final canonicalize: ensure column order and fill missing cols
        for c in CSV_COLS:
            if c not in master_df.columns:
                master_df[c] = ""
        master_df = master_df[CSV_COLS]

        # atomic write
        atomic_write_csv(master_df, MASTER_CSV)
        print(f"[done] master CSV updated: {MASTER_CSV} (rows={len(master_df)})")

    finally:
        release_lock()


if __name__ == "__main__":
    main()