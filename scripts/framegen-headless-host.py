#!/usr/bin/env python3
"""Run a command nested in a private headless GNOME Shell, then tear it down.

For showcase runs in the background: the host is Wayland-only with one virtual
monitor, on a private bus and runtime directory, so the runs show no window
and neither the desktop's monitors nor its mouse and keyboard touch them. The
command gets the host's WAYLAND_DISPLAY and XDG_RUNTIME_DIR, no DISPLAY, and
HOST_DISPLAY="headless <mode>" (recorded by framegen-showcase.sh as context).
The private runtime directory is deleted on exit, so OUT defaults to
<your XDG_RUNTIME_DIR>/framegen-showcase/<time> and the showcase results
survive; an explicit OUT is kept. The host's and the command's process groups
are killed on exit, also on SIGTERM (kill, timeout).

The host renders on the GPU its EGL picks. It must be the GPU gamescope
presents with (--prefer-vk-device), or the host cannot import gamescope's
buffers and drops the connection; on a hybrid machine whose dGPU EGL loads
first, choose the host's EGL vendor, e.g.
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json.

  framegen-headless-host.py [--mode 2560x1440@60] -- command [args...]

Example:
  scripts/framegen-headless-host.py --mode 2560x1440@60 -- \\
      env HZ=60 scripts/framegen-showcase.sh 6 7
"""

import argparse
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The host is the smoke harness's, so the two never drift.
_spec = importlib.util.spec_from_file_location(
    "framegen_nested_smoke", Path(__file__).with_name("framegen-nested-smoke.py"))
smoke = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = smoke  # its dataclasses resolve their module by name
_spec.loader.exec_module(smoke)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="2560x1440@60", help="virtual monitor WxH@Hz (default: %(default)s)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command [args...]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    # SIGTERM takes the same cleanup path as Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    outer_runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    # Short and fixed under /tmp: Wayland and X socket paths must fit sun_path.
    runtime_dir = Path(tempfile.mkdtemp(prefix="fghost-", dir="/tmp"))
    state_dir = Path(tempfile.mkdtemp(prefix="fghost-state-"))
    host = child = None
    try:
        with open(state_dir / "host.log", "wb") as log:
            host = smoke.start_host(state_dir / "host-bus.conf", state_dir, runtime_dir,
                                    {**os.environ, "XDG_RUNTIME_DIR": str(runtime_dir)}, log, mode=args.mode)
            env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
            env.update(WAYLAND_DISPLAY=smoke.HOST_DISPLAY, XDG_RUNTIME_DIR=str(runtime_dir),
                       HOST_DISPLAY=f"headless {args.mode}")
            env.setdefault("OUT", f"{outer_runtime}/framegen-showcase/{time.strftime('%Y%m%d-%H%M%S')}")
            child = subprocess.Popen(command, env=env, start_new_session=True)
            return child.wait()
    except RuntimeError as e:
        print(f"framegen-headless-host: {e}", file=sys.stderr)
        return 2
    finally:
        if child is not None:
            smoke.terminate_group(child.pid)
        if host is not None:
            smoke.terminate_group(host.pid)
        shutil.rmtree(runtime_dir, ignore_errors=True)
        shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
