const target = "JarvisHUD";
const wins = workspace.windowList ? workspace.windowList()
    : (workspace.clientList ? workspace.clientList() : []);
let pinned = 0;
for (let i = 0; i < wins.length; i++) {
    const w = wins[i];
    if (!w || !w.caption) continue;
    if (w.caption.indexOf(target) === -1) continue;
    try { w.keepAbove = true; } catch (e) {}
    try { w.skipTaskbar = true; } catch (e) {}
    try { w.skipPager = true; } catch (e) {}
    try { w.skipSwitcher = true; } catch (e) {}
    pinned++;
}
print('jarvis-hud pin: ' + pinned + ' window(s)');
