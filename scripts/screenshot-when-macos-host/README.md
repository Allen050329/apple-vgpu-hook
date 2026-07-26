# Reims vGPU screenshot (macOS)

`screenshot-when-macos-host.sh` captures the exact host window titled `Reims vGPU`
on a macOS host. It resolves the window through ScreenCaptureKit and uses a
desktop-independent window filter, so the capture includes compositor-owned
Metal content while the window is inactive or covered and after the guest
changes display resolution. It requires macOS 14 or later.

The script prefers a process whose name contains `qemu-system`, avoiding
terminal or editor windows that happen to mention the display title. Override
the process hint with `REIMS_PROCESS_HINT` when necessary.

## Usage

```sh
scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/reims.png
```

With no output argument, the script writes a timestamped PNG under `/tmp`.

The invoking terminal or application needs Screen & System Audio Recording
permission:

`System Settings → Privacy & Security → Screen & System Audio Recording`

This tool is for macOS hosts only. Linux hosts use
`scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh`.
