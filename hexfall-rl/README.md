# hexfall-rl

Reinforcement-learning simulator for Hex Fall (Paxie Games). Implements the game as a partially observable MDP: a hex-stack field is cleared by selecting colored buckets from a reserve into a 5-slot buffer. This repo provides the core data types, level loader, and (in later sessions) the simulator and Gymnasium environment used for RL training and automated difficulty assessment.

## Setup

1. Create venv:
   ```
   python -m venv .venv
   ```

2. Activate venv:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. **Manual CUDA torch install (Windows, required before `pip install -e .`):**
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   Note: the default PyPI torch wheel on Windows is CPU-only and will silently make a CUDA GPU sit idle. This manual install is required to use the GPU.

4. Editable install with dev extras:
   ```
   pip install -e .[dev]
   ```

5. Verify install:
   - `python -c "import torch; print(torch.cuda.is_available())"` → `True`
   - `pytest` → 86 passing

### Vendored dependencies

`vendor/cleanrl_ppo_reference.py` is the reference PPO implementation from CleanRL, used as the basis for the training entry point that will live at `scripts/train_ppo.py` (not yet present — to be added in a later session).

- Source: CleanRL `cleanrl/ppo.py`
- Commit: `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`
- URL: https://raw.githubusercontent.com/vwxyzjn/cleanrl/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py

The file is intentionally unmodified for byte-level diffability against upstream.
