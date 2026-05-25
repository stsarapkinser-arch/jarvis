// JARVIS — prepare_workspace.js
// Plasma 6 / KWin Scripting API.
// Раскладывает нормальные окна на активном экране в сетку, понижает прозрачность.

(function () {
    const all = (typeof workspace.windowList === "function")
        ? workspace.windowList()
        : workspace.clientList();

    const windows = all.filter(function (w) {
        return w && w.normalWindow && !w.minimized;
    });

    if (windows.length === 0) {
        print("[jarvis] prepare_workspace: no windows");
        return;
    }

    const ref = workspace.activeWindow || windows[0];
    let area;
    try {
        area = workspace.clientArea(KWin.MaximizeArea, ref);
    } catch (e) {
        area = workspace.workArea
            ? workspace.workArea(workspace.activeScreen, workspace.currentDesktop)
            : { x: 0, y: 0, width: 1920, height: 1080 };
    }

    const cols = Math.ceil(Math.sqrt(windows.length));
    const rows = Math.ceil(windows.length / cols);
    const w = Math.floor(area.width / cols);
    const h = Math.floor(area.height / rows);

    for (let i = 0; i < windows.length; i++) {
        const win = windows[i];
        const col = i % cols;
        const row = Math.floor(i / cols);
        const geom = {
            x: area.x + col * w,
            y: area.y + row * h,
            width: w,
            height: h
        };
        try {
            win.frameGeometry = geom;
        } catch (e) {
            try { win.geometry = geom; } catch (_) {}
        }
        try { win.opacity = 0.95; } catch (_) {}
    }
    print("[jarvis] prepare_workspace: tiled " + windows.length + " windows");
})();
