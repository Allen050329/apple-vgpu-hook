#!/usr/bin/env python3
"""Whole-screen PNG through the `xdg-desktop-portal` Screenshot API.

The portal is the one capture interface that KDE, GNOME and the wlroots
compositors all implement, so it is what a host with no compositor-specific tool
is left with. It screenshots the **screen**: the API takes no window handle, so
unlike the KDE and macOS helpers this cannot single out the guest window, and a
caller that needs the window alone has to crop.

`interactive=false` asks for no picker. A portal backend may still show a consent
prompt the first time, and if nobody is at the keyboard the request simply never
returns — hence the deadline, which turns that into a named failure rather than a
run that hangs to its timeout.

Exit codes: 0 wrote the file, 2 the portal refused or the user dismissed it,
3 no response before the deadline, 4 the portal is not reachable at all.
"""

import os
import shutil
import sys
import urllib.parse
import urllib.request

import gi

gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib  # noqa: E402  (gi requires the version pin first)

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENSHOT_IFACE = "org.freedesktop.portal.Screenshot"
REQUEST_IFACE = "org.freedesktop.portal.Request"
DEADLINE_SECONDS = 30


def _request_path(bus: Gio.DBusConnection, token: str) -> str:
    """The object path the portal will emit this request's Response on.

    Subscribing before the call is what makes the exchange race-free, and the
    path is derivable rather than returned, so it has to be built the way the
    portal spec builds it: the caller's unique name with its leading colon
    dropped and dots turned into underscores.
    """
    unique = bus.get_unique_name()
    return f"{PORTAL_PATH}/request/{unique[1:].replace('.', '_')}/{token}"


def capture(destination: str) -> int:
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error as error:
        print(f"portal-screenshot: no session bus: {error.message}", file=sys.stderr)
        return 4

    token = f"applevgpuhook{os.getpid()}"
    loop = GLib.MainLoop()
    outcome: dict[str, object] = {}

    def on_response(_conn, _sender, _path, _iface, _signal, params):
        response, results = params.unpack()
        outcome["response"] = response
        outcome["uri"] = results.get("uri")
        loop.quit()

    subscription = bus.signal_subscribe(
        PORTAL_BUS,
        REQUEST_IFACE,
        "Response",
        _request_path(bus, token),
        None,
        Gio.DBusSignalFlags.NONE,
        on_response,
    )

    try:
        bus.call_sync(
            PORTAL_BUS,
            PORTAL_PATH,
            SCREENSHOT_IFACE,
            "Screenshot",
            GLib.Variant(
                "(sa{sv})",
                (
                    "",
                    {
                        "handle_token": GLib.Variant("s", token),
                        "interactive": GLib.Variant("b", False),
                    },
                ),
            ),
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )
    except GLib.Error as error:
        bus.signal_unsubscribe(subscription)
        print(f"portal-screenshot: portal call failed: {error.message}", file=sys.stderr)
        return 4

    def give_up() -> bool:
        outcome["timeout"] = True
        loop.quit()
        return False

    GLib.timeout_add_seconds(DEADLINE_SECONDS, give_up)
    loop.run()
    bus.signal_unsubscribe(subscription)

    if outcome.get("timeout"):
        print(
            f"portal-screenshot: no response in {DEADLINE_SECONDS}s — a consent "
            "prompt with nobody to answer it looks exactly like this",
            file=sys.stderr,
        )
        return 3
    if outcome.get("response") != 0 or not outcome.get("uri"):
        print(
            f"portal-screenshot: portal returned response={outcome.get('response')}",
            file=sys.stderr,
        )
        return 2

    parsed = urllib.parse.urlparse(str(outcome["uri"]))
    source = urllib.request.url2pathname(parsed.path)
    shutil.move(source, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: portal-screenshot.py OUT.png", file=sys.stderr)
        raise SystemExit(64)
    raise SystemExit(capture(sys.argv[1]))
