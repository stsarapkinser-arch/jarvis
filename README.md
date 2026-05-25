# Jarvis — Kali AR Overlay

Voice-first assistant fused with KDE Plasma 6 (Wayland) on Kali Linux. Designed
for the Intel N100 form factor: low-priority recon daemons, FFT-driven HUD,
and a single Ollama brain (qwen2.5-coder:3b) that streams `<thought>` tokens
into a live cognitive graph.

## Subsystems

### Aegis — full-screen AR HUD (`jarvis_hud.py`)

A borderless, click-through, fullscreen overlay (`Qt.WindowTransparentForInput
+ showFullScreen`) rendered over the live desktop.

Layers, back-to-front:

1. **Scan grid + corner brackets** — base cyberpunk frame.
2. **App glow** — Iron-Man-style neon rectangles + corner brackets around
   windows we are observing (`KWinOrchestrator.find_window_rects` resolves
   geometry; `JarvisHUD.draw_app_glow` paints).
3. **Port matrix** — when `nmap` runs, port hits stream into a 32×8 grid via
   `nmap_stream.stream_nmap`. Open ports glow green, filtered amber, closed
   red. A progress bar shows the scan completion percentage.
4. **Nav graph** — five anchor nodes (CORE, MEMORY, HARDWARE, PIXEL,
   OLLAMA) connected by animated energy lines. Each LLM token pulse routes
   one micro-line; each `<thought>` triggers a labeled pulse.
5. **Pixel AR card** — perspective-skewed notification card on the right
   edge driven by `PIXEL_EVENT` and `HUD_OVERLAY:pixel_projection`.
6. **Core sphere** — wireframe sphere with 24 longitude bars and 9 latitude
   rings whose radius is modulated by the FFT bands of the Piper PCM
   stream (`audio_fft.PiperFFTPump`). Particle ring at the equator.
7. **Recon halo** — yellow wash + blinking border for Wi-Fi vulnerabilities,
   red wash for intrusion events. Decays over 5 s.
8. **Token ticker** — scrolling Ollama output at the bottom edge.

### Wraith — recon daemon (`recon_daemon.py`)

Singleton that watches the local environment for Wi-Fi vulnerabilities and
intrusion signals. Runs under `nice -n 19` so a heavy tshark capture cannot
starve Ollama.

* **Wi-Fi**
  * If an interface is already in monitor mode (`*mon`) and `tshark` is
    installed, listens for WEP beacons and EAPOL handshakes.
  * Otherwise issues a passive `iw dev <iface> scan` every 45 s and parses
    the output for WEP / WPS-enabled BSSIDs.
* **Intrusion**
  * Tails `/var/log/auth.log` (or `secure`) for `Failed password`,
    `invalid user`, and friends.
  * Follows `dmesg` for port-scan and USB-injection signatures.
  * Coincident hits (≥4 in 30 s) escalate to an `intrusion_storm` finding.

Findings land on the bus as `RECON_ALERT` with a `color` (`yellow|red`).
Red findings carry a `ufw_suggest` field; the core wires it through the
standard destructive-command voice gate before any rule is applied.

### Digital Navigator — visual feedback (`jarvis_hud.py` + `core.py`)

* Token-level pulses on the nav graph during `THINKING` / `SPEAKING`.
* Pixel notifications (clipboard, call, quick command) project onto the
  AR card automatically — no extra wiring beyond the existing
  `PIXEL_EVENT` stream.
* When a bash command fails on a known window (`code`, `konsole`, …),
  `Jarvis._highlight_error_window` paints a neon bracket around the
  matching window via `HUD_OVERLAY:app_glow`.

## Event bus extensions

| Type           | Producer            | Consumer        |
|----------------|---------------------|-----------------|
| `AUDIO_FFT`    | `PiperFFTPump`      | HUD core sphere |
| `RECON_ALERT`  | `ReconDaemon`       | HUD + `Jarvis.on_recon_alert` |
| `NMAP_SCAN`    | `nmap_stream`       | HUD port matrix |
| `HUD_OVERLAY`  | core / future tools | HUD             |

## Protocols (`commands.txt`)

The `PROTOCOLS` section contains composite one-line bash chains for
high-stakes flows:

* **Battle mode** (`режим боя` / `battle mode`) — UFW deny-incoming + log
  dump + Plasma notification.
* **Stealth mode** (`режим невидимости` / `stealth mode`) — Tor service
  start + MAC randomization (macchanger) on all `wl*` and `en*` ifaces +
  shell history clear.
* **Disengage variants** for both, plus a manual visor toggle.

Each protocol is a single `&&` pipeline so the destructive-command gate
and the QuickPatcher healer treat it as one operation.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 start_jarvis.py
```

Optional system tooling for full Wraith coverage:

```bash
sudo apt install tshark iw aircrack-ng ufw macchanger tor
```

## Tests

```bash
pip install pytest
python3 -m pytest tests/
```

The HUD tests use `QT_QPA_PLATFORM=offscreen` and exercise the Qt signal
plumbing without rendering, so they pass on any headless box.
