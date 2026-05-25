import difflib
from pathlib import Path


class CommandBook:
    """Tier 1 fuzzy matcher backed by commands.txt (format: phrase => bash)."""

    def __init__(self, path: str = "./commands.txt", cutoff: float = 0.75):
        self.path = Path(path)
        self.cutoff = cutoff
        self.macros: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self.macros.clear()
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=>" not in line:
                continue
            phrase, _, bash = line.partition("=>")
            phrase = phrase.strip().lower()
            bash = bash.strip()
            if phrase and bash:
                self.macros[phrase] = bash

    def match(self, text: str) -> str | None:
        if not self.macros:
            return None
        query = text.strip().lower()
        if query in self.macros:
            return self.macros[query]
        candidates = difflib.get_close_matches(query, self.macros.keys(), n=1, cutoff=self.cutoff)
        if candidates:
            return self.macros[candidates[0]]
        return None
