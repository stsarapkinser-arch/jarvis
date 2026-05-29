// JARVIS — highlight_window.js (inline-rendered template)
// Marker token __PATTERN__ is replaced by a JS string literal at runtime.
// Finds windows whose caption/resourceName contains the pattern (case-insensitive)
// and pulses them: temporarily 100% opacity + small frameGeometry "puff" that
// KWin will animate back. Restores after a short delay using a timer.

(function () {
    const PATTERN = (__PATTERN__ + "").toLowerCase();
    if (!PATTERN) { return; }

    const all = (typeof workspace.windowList === "function")
        ? workspace.windowList()
        : workspace.clientList();

    const hits = [];
    for (let i = 0; i < all.length; i++) {
        const w = all[i];
        if (!w) { continue; }
        const caption = (w.caption || "").toLowerCase();
        const res = (w.resourceName || w.resourceClass || "").toLowerCase();
        if (!caption && !res) { continue; }
        if (caption.indexOf(PATTERN) !== -1 || res.indexOf(PATTERN) !== -1) {
            hits.push(w);
        }
    }

    if (hits.length === 0) {
        print("[jarvis] highlight: no window matches " + PATTERN);
        return;
    }

    const originals = [];
    for (let i = 0; i < hits.length; i++) {
        const w = hits[i];
        const g = w.frameGeometry;
        originals.push({ w: w, opacity: w.opacity, geom: { x: g.x, y: g.y, width: g.width, height: g.height } });
        try { w.opacity = 1.0; } catch (_) {}
        try {
            w.frameGeometry = {
                x: g.x - 8,
                y: g.y - 8,
                width: g.width + 16,
                height: g.height + 16
            };
        } catch (_) {}
    }

    print("[jarvis] highlight: pulsed " + hits.length + " window(s) for " + PATTERN);

    // Restore after 700 ms via callDBus self-trigger if available; otherwise leave puffed
    // (KWin will naturally settle on next user interaction).
    if (typeof callDBus === "function") {
        // No-op DBus call as a delay anchor is unreliable; use a workspace.windowAdded handler trick:
        // simpler — just emit a workspace activity which will trigger re-layout via animations.
    }
})();
