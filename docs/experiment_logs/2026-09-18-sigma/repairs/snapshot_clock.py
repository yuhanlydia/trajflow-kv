"""Versioned infrastructure repair, invoked after snapshot ADB authorization.

No policy calls, UI actions, timezone changes, or task replay. The authorizer
is invoked synchronously by the existing snapshot initialization path, before
the policy receives its first observation. All changes have append-only logs.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

PROTOCOL = "snapshot_host_clock_v1"
RECEIPT = Path("/app/artifacts/sigma_snapshot_clock_v1/receipts.jsonl")


def append_receipt(record):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPT.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def synchronize(device, *, run=subprocess.run, now=time.time, emit=append_receipt):
    if device != "emulator-5554":
        raise ValueError("Unregistered clock repair target")
    prefix = ["adb", "-s", device, "shell"]

    def invoke(args):
        result = run(prefix + args, capture_output=True, text=True, timeout=8)
        if result.returncode:
            raise RuntimeError("ADB clock operation failed: " + str(result.returncode))
        return result.stdout.strip()

    before = int(invoke(["date", "+%s"]))
    host = now()
    receipt = dict(protocol=PROTOCOL, device=device, pid=os.getpid(),
                   parent_pid=os.getppid(), host_epoch=host, guest_epoch_before=before,
                   skew_before_seconds=before-host)
    if abs(before-host) <= 5:
        receipt.update(stage="already_synchronized", guest_epoch_after=before,
                       skew_after_seconds=before-host, changed=False)
        emit(receipt)
        return receipt
    # Persist the intent before making any environmental change.
    emit(dict(receipt, stage="intent"))
    target_ms = int(now()*1000)
    invoke(["cmd", "alarm", "set-time", str(target_ms)])
    after = int(invoke(["date", "+%s"]))
    skew = after-now()
    receipt.update(stage="synchronized" if abs(skew) <= 5 else "verification_failed",
                   target_epoch_ms=target_ms, guest_epoch_after=after,
                   skew_after_seconds=skew, changed=True)
    emit(receipt)
    if abs(skew) > 5:
        raise RuntimeError("Guest clock did not synchronize within five seconds")
    return receipt


def after_authorization(device):
    try:
        synchronize(device)
        return 0
    except Exception as exc:
        # A failed initialization must never silently claim a repaired clock.
        try:
            append_receipt(dict(protocol=PROTOCOL, stage="failed", device=device,
                                host_epoch=time.time(), error_type=type(exc).__name__,
                                error=str(exc)))
        finally:
            print("Snapshot clock synchronization failed: " + type(exc).__name__, flush=True)
        return 1
