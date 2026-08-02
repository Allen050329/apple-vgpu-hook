#!/usr/bin/env python3
"""Synthesise a real mouse drag of a window, and say what was actually posted.

Goal 6 is "window-dragging performance of Safari should be a stable 120 fps".
Measuring that needs the drag to happen, and it has to be a *drag*: moving a
window by setting its accessibility position produces window-server work that
does not go through the same path as a pointer held down across a title bar.
So this posts real `kCGEventLeftMouseDown` / `Dragged` / `Up` events through
Quartz, which is the same event stream a hand produces.

Runs in the guest against the system Python's PyObjC. Prints one line of JSON so
the host can tell a slow device from a drag that never happened — the standing
lesson from `scripts/web-content-probe`, where a clean verdict came from a
stressor that produced nothing.
"""
import json
import sys
import time

from Quartz import (  # type: ignore[import-not-found]
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventSetType,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)


def post(kind, x, y):
    ev = CGEventCreateMouseEvent(None, kind, (x, y), kCGMouseButtonLeft)
    CGEventSetType(ev, kind)
    CGEventPost(kCGHIDEventTap, ev)


def main():
    x0, y0 = float(sys.argv[1]), float(sys.argv[2])
    seconds = float(sys.argv[3])
    hz = float(sys.argv[4])
    # Amplitude in pixels of the path traced. A drag that barely moves damages
    # almost nothing and would measure the idle device.
    ax, ay = float(sys.argv[5]), float(sys.argv[6])

    period = 1.0 / hz
    post(kCGEventLeftMouseDown, x0, y0)
    # Let the window server latch the drag before moving, or the first samples
    # measure a click.
    time.sleep(0.05)

    start = time.time()
    posted = 0
    late = 0
    worst = 0.0
    nxt = start
    while True:
        now = time.time()
        t = now - start
        if t >= seconds:
            break
        # A closed path so the window returns to where it started and the run is
        # repeatable; two frequencies so it is not a straight line, which some
        # compositors coalesce.
        import math

        x = x0 + ax * math.sin(2 * math.pi * 0.5 * t)
        y = y0 + ay * math.sin(2 * math.pi * 0.31 * t)
        post(kCGEventLeftMouseDragged, x, y)
        posted += 1
        nxt += period
        slack = nxt - time.time()
        if slack > 0:
            time.sleep(slack)
        else:
            # We could not keep up with the requested rate. Reported rather than
            # hidden: a drag posted at 40 Hz cannot show a 120 Hz device.
            late += 1
            worst = max(worst, -slack)
            nxt = time.time()

    elapsed = time.time() - start
    post(kCGEventLeftMouseUp, x0, y0)
    print(json.dumps({
        "posted": posted,
        "elapsed": round(elapsed, 3),
        "posted_hz": round(posted / elapsed, 1) if elapsed else 0.0,
        "late": late,
        "worst_late_s": round(worst, 4),
    }))


if __name__ == "__main__":
    main()
