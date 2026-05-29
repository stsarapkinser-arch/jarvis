# Project Structure Migration Guide

## 🔄 What Changed

The Jarvis project was reorganized from a flat structure with files scattered in the root directory to a well-organized hierarchical structure with clear separation of concerns.

## 📂 File Mapping

### Core & Orchestration
- `core.py` → `src/core/orchestrator.py`
- `main.py` → `src/core/entry_point.py`
- `start_jarvis.py` → `src/core/bootstrap.py`

### Services & Daemons
- `daemon_swarm.py` → `src/services/daemon_swarm.py`
- `recon_daemon.py` → `src/services/recon_daemon.py`
- `deep_watch.py` → `src/services/watch_service.py`
- `sentinel.py` → `src/services/sentinel.py`

### Memory & State Management
- `memory_engine.py` → `src/memory/engine.py`
- `mnemosyne.py` → `src/memory/storage.py`
- `ephemeral.py` → `src/memory/ephemeral.py`
- `state_snapshot.py` → `src/memory/snapshot.py`

### User Interface
- `jarvis_hud.py` → `src/ui/hud.py`
- `kwin.py` → `src/ui/window_manager.py`
- `pixel.py` → `src/ui/pixel_renderer.py`

### Audio Processing
- `audio_fft.py` → `src/audio/fft_analyzer.py`

### Network Tools
- `nmap_stream.py` → `src/network/scanner.py`

### Security & Execution
- `shadow_exec.py` → `src/security/execution.py`

### Utilities
- `event_bus.py` → `src/common/event_bus.py`
- `parser.py` → `src/common/parser.py`
- `singleton.py` → `src/common/singleton.py`
- `repair.py` → `src/common/repair.py`

### Configuration Files
- `system_prompt` → `config/system_prompt`
- `jarvis.service` → `config/jarvis.service`
- `requirements.txt` → `config/requirements.txt`
- `pyproject.toml` → `config/pyproject.toml`

### Scripts
- All `.sh` files moved to `scripts/` directory
- JavaScript files (`enumerate_windows.js`, etc.) already in `scripts/`

## 🔗 Import Updates

### Before
```python
from core import CoreEngine
from event_bus import EventBus
from memory_engine import ChronoMemory
from kwin import KWinOrchestrator
```

### After
```python
from src.core.orchestrator import CoreEngine
from src.common.event_bus import EventBus
from src.memory.engine import ChronoMemory
from src.ui.window_manager import KWinOrchestrator
```

## 📍 Path Resolution

### PROJECT_ROOT Changes

**Before:**
```python
_PROJECT_ROOT = Path(__file__).resolve().parent
```

**After (in src/core/ and src/core/entry_point.py):**
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
```

This is because files moved 3 levels deeper:
- `orchestrator.py` is now at `src/core/orchestrator.py` (3 levels: core → src → root)

## ✅ Verification

To verify the structure is working:

```bash
# Run imports test
python -c "
from src.core.orchestrator import *
from src.services.daemon_swarm import DaemonSwarm
from src.memory.engine import ChronoMemory
print('✓ All imports working')
"

# Run tests
pytest tests/ -v

# Run linter
ruff check .
```

## 🚀 Running the Application

```bash
# Entry point is now in src/core/bootstrap.py
python -m src.core.bootstrap

# Or with absolute import
python src/core/bootstrap.py
```

## 📝 Development Guidelines

1. **Adding New Modules**
   - Create files in the appropriate `src/` subdirectory
   - Use snake_case for filenames
   - Use PascalCase for classes
   - Add docstrings to modules and classes

2. **Naming Conventions**
   - Module names describe their main responsibility
   - Avoid generic names like `utils.py` (prefer `event_bus.py`)
   - Related functionality grouped in same directory

3. **Import Order**
   - Standard library imports first
   - Third-party imports second
   - Local imports last
   - All imports at top of file (except circular dependency cases)

## 🔗 Cross-Module Dependencies

**Key module relationships:**

```
orchestrator (core)
├─→ event_bus (common)
├─→ memory/engine (memory)
├─→ services/* (services)
└─→ ui/* (ui)

entry_point (core)
├─→ event_bus (common)
└─→ singleton (common)

daemon_swarm (services)
├─→ event_bus (common)
└─→ singleton (common)
```

See `src/README.md` for detailed module documentation.
