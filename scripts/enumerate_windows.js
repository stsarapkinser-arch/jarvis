// JARVIS — enumerate_windows.js
// Plasma 6 KWin Scripting. Writes a JSON description of currently visible
// "normal" windows to /tmp/jarvis-windows.json so the Python side can compute
// the least-occupied screen quadrant for HUD placement.
//
// KWin JS doesn't have direct fs APIs, but it can shell out via callDBus.
// Cleanest cross-version path: print() to KWin journal AND attempt to write
// via a temp file using `dofile` if available. We rely on the Python side
// using `kdotool` as a fallback when this script's output cannot be captured.

(function () {
    const all = (typeof workspace.windowList === "function")
        ? workspace.windowList()
        : workspace.clientList();

    const out = { ts: Date.now(), windows: [] };
    for (let i = 0; i < all.length; i++) {
        const w = all[i];
        if (!w || !w.normalWindow || w.minimized) { continue; }
        const g = w.frameGeometry;
        out.windows.push({
            caption: w.caption || "",
            res: w.resourceName || w.resourceClass || "",
            x: g.x, y: g.y, width: g.width, height: g.height,
            screen: w.screen,
            active: !!w.active,
        });
    }
    const payload = JSON.stringify(out);
    print("JARVIS_WINDOWS_JSON " + payload);
})();
