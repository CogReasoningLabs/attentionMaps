#!/usr/bin/env python3
"""Run the dataset explorer with reliable process-tree shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "apps" / "dataset_explorer.py"
POLL_SECONDS = 0.1
INTERRUPT_GRACE_SECONDS = 3.0
TERMINATE_GRACE_SECONDS = 2.0


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | streamlit-supervisor | {message}", file=sys.stderr, flush=True)


@dataclass
class ShutdownRequest:
    """Signal state updated by the main-thread signal handler."""

    signum: int | None = None
    count: int = 0

    def handle(self, signum: int, _frame: FrameType | None) -> None:
        self.signum = signum
        self.count += 1

    @property
    def requested(self) -> bool:
        return self.signum is not None

    @property
    def force_requested(self) -> bool:
        return self.count > 1


def streamlit_command(arguments: Sequence[str]) -> list[str]:
    """Build the child command while preserving Streamlit CLI arguments."""

    forwarded = list(arguments)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        *forwarded,
    ]


def _signal_process_tree(process: subprocess.Popen[bytes], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            # start_new_session=True makes the child PID its process-group ID.
            os.killpg(process.pid, signum)
        else:  # pragma: no cover - the project currently runs on Linux
            process.send_signal(signum)
    except ProcessLookupError:
        return


def _wait_for_exit(
    process: subprocess.Popen[bytes],
    timeout: float,
    request: ShutdownRequest,
) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        if request.force_requested:
            return None
        time.sleep(POLL_SECONDS)
    return process.poll()


def _shutdown_process_tree(
    process: subprocess.Popen[bytes], request: ShutdownRequest
) -> int:
    requested_signal = request.signum or signal.SIGINT
    requested_name = signal.Signals(requested_signal).name
    _log(f"{requested_name} received; stopping Streamlit and active workers...")

    first_signal = signal.SIGINT if requested_signal == signal.SIGINT else signal.SIGTERM
    _signal_process_tree(process, first_signal)
    if _wait_for_exit(process, INTERRUPT_GRACE_SECONDS, request) is not None:
        _log("Streamlit stopped cleanly.")
        return 128 + requested_signal

    if not request.force_requested:
        _log("Graceful stop timed out; terminating the complete process tree...")
        _signal_process_tree(process, signal.SIGTERM)
        if _wait_for_exit(process, TERMINATE_GRACE_SECONDS, request) is not None:
            _log("Process tree terminated.")
            return 128 + requested_signal

    _log("Force-stopping the complete process tree.")
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    _signal_process_tree(process, kill_signal)
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return 128 + requested_signal


def _normal_exit_code(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def supervise(command: Sequence[str]) -> int:
    """Run one command until it exits or the user requests bounded shutdown."""

    request = ShutdownRequest()
    watched_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in watched_signals
    }
    for signum in watched_signals:
        signal.signal(signum, request.handle)

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            start_new_session=os.name == "posix",
        )
        _log(
            f"Streamlit started (PID {process.pid}). "
            "Press Ctrl+C to stop; press it again to force stop."
        )
        while True:
            returncode = process.poll()
            if returncode is not None:
                return _normal_exit_code(returncode)
            if request.requested:
                return _shutdown_process_tree(process, request)
            time.sleep(POLL_SECONDS)
    finally:
        if process is not None and process.poll() is None:
            _signal_process_tree(process, getattr(signal, "SIGKILL", signal.SIGTERM))
            process.wait()
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


def main(arguments: Sequence[str] | None = None) -> int:
    return supervise(streamlit_command(sys.argv[1:] if arguments is None else arguments))


if __name__ == "__main__":
    raise SystemExit(main())
