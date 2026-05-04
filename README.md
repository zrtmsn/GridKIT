# GridKIT

Simulation of smart low-voltage grids with multi-agent reinforcement learning (MARL).

---

## Project Structure

```
src/GridKIT/
├── core/          # Shared data models, config, and interfaces — maintained by Artem
├── grid_model/    # Power grid simulation (pandapower, §14a EnWG logic)
├── rl_engine/     # MARL agents and training (Gymnasium, RLlib / SB3)
├── map_ui/        # OSM data loading and grid topology via GridCreator
└── dashboard/     # Streamlit visualization and export
```

### Key principle

**Modules never import from each other directly — only from `core`.**

```
map_ui    ──┐
grid_model ─┼──► core (models, config, interfaces)
rl_engine  ─┘
dashboard  ─┘
```

This allows parallel development without merge conflicts. Each module uses stub data from `core` while other modules are still being built.

---

## Before You Start

### 1. Clone the repo

```bash
git clone <repo-url>
cd GridKIT
```

### 2. Install dependencies and create virtual environment

```bash
uv sync
```

This creates `.venv` and installs all dependencies from `pyproject.toml` automatically.

### 3. Activate the environment

```bash
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

After this you can import from any module:

```python
from GridKIT.core.models import GridNetwork
```

### 4. Check that tests pass

```bash
pytest
```

---

## Development Rules

**1. Freeze `core` first.**
Before writing any module code, the team agrees on the data models in `core/models.py`. These are the shared language — do not change them without telling everyone.

**2. Import only from `core`.**
Your module should never do `from GridKIT.grid_model import ...` inside `rl_engine`. If you need something from another module, it belongs in `core`.

**3. External libraries stay inside one module.**
For example, `GridCreator` is only imported inside `map_ui/`. The result is converted to `core.models.GridNetwork` before leaving the module. This way the rest of the project is not affected if the library changes.

**4. Write tests for your own module.**
Add them to `tests/` with the prefix `test_<your_module>_`. At minimum: does your module accept the `core` models it's supposed to receive, and does it return the `core` models it's supposed to return?

**5. Branch naming.**
Work on feature branches, open a PR into `main`. Artem reviews PRs that touch `core`.

---

## Module Responsibilities

| Module | Owner | Inputs (from core) | Outputs (to core) |
|---|---|---|---|
| `map_ui` | — | — | `GridNetwork` |
| `grid_model` | — | `GridNetwork`, `Action` | `Observation`, `SimResult` |
| `rl_engine` | — | `Observation`, `Reward` | `Action` |
| `dashboard` | — | `SimResult` | — |

---

## External Libraries (vendor)

| Library | Used in | Purpose |
|---|---|---|
| [GridCreator](https://github.com/INATECHCIG/GridCreator) | `map_ui` | Generate LV grid topology from OSM data |
| [pandapower](https://pandapower.readthedocs.io) | `grid_model` | Power flow simulation |
| [Gymnasium](https://gymnasium.farama.org) | `rl_engine` | RL environment interface |
| [RLlib](https://docs.ray.io/en/latest/rllib/) or [SB3](https://stable-baselines3.readthedocs.io) | `rl_engine` | MARL agents |
| [Streamlit](https://streamlit.io) | `dashboard` | UI and visualization |