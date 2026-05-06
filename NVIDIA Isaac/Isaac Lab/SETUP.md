# Isaac Lab — Lunabotics Training Setup

## First-time setup on a new machine

### 1. Configure Isaac Lab path

`isaaclab_path.cfg` is the **only path you need to set**. Everything else (USD file, repo structure) is resolved automatically.

```
cd "NVIDIA Isaac\Isaac Lab\lunabotics"
copy isaaclab_path.cfg.example isaaclab_path.cfg
```

Open `isaaclab_path.cfg` and set your local Isaac Lab installation path:

```
# Windows
ISAACLAB_PATH=E:\IsaacLab

# Linux
ISAACLAB_PATH=/home/user/IsaacLab
```

### 2. Install the package

Run once from the `lunabotics/` directory. This registers the `lunabotics` module with Isaac Lab's Python environment — **required before training or playing**.

```bat
# Windows
isaaclab.bat -p -m pip install -e source\lunabotics
```
```bash
# Linux
./isaaclab.sh -p -m pip install -e source/lunabotics
```

---

## Training

Run from the `lunabotics/` directory so logs are saved inside the project.

```bat
# Windows
isaaclab.bat -p scripts\rsl_rl\train.py --task Template-Lunabotics-Direct-v0

# Headless (no GUI)
isaaclab.bat -p scripts\rsl_rl\train.py --task Template-Lunabotics-Direct-v0 --headless

# Resume from a previous run
isaaclab.bat -p scripts\rsl_rl\train.py --task Template-Lunabotics-Direct-v0 --resume --load_run 2026-03-15_10-00-00
```

```bash
# Linux
./isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Lunabotics-Direct-v0

# Headless
./isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Lunabotics-Direct-v0 --headless

# Resume
./isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Lunabotics-Direct-v0 --resume --load_run 2026-03-15_10-00-00
```

Logs and checkpoints are saved to:
```
lunabotics/logs/rsl_rl/lunabotics_direct/<timestamp>/
```
`.pt` model files are tracked via Git LFS.

---

## Playing a checkpoint

```bat
# Windows — latest checkpoint
isaaclab.bat -p scripts\rsl_rl\play.py --task Template-Lunabotics-Direct-Play-v0 --num_envs 16

# Specific run and checkpoint
isaaclab.bat -p scripts\rsl_rl\play.py --task Template-Lunabotics-Direct-Play-v0 --num_envs 16 --load_run 2026-03-15_10-00-00 --checkpoint model_1000.pt
```

```bash
# Linux — latest checkpoint
./isaaclab.sh -p scripts/rsl_rl/play.py --task Template-Lunabotics-Direct-Play-v0 --num_envs 16

# Specific run and checkpoint
./isaaclab.sh -p scripts/rsl_rl/play.py --task Template-Lunabotics-Direct-Play-v0 --num_envs 16 --load_run 2026-03-15_10-00-00 --checkpoint model_1000.pt
```

---

## Key files

| File | Purpose |
|------|---------|
| `lunabotics/isaaclab_path.cfg` | Machine-local Isaac Lab path (gitignored) |
| `lunabotics/isaaclab_path.cfg.example` | Template — copy to `isaaclab_path.cfg` |
| `lunabotics/isaaclab.bat` / `.sh` | Wrappers that forward to the real Isaac Lab scripts |
| `lunabotics/source/lunabotics/lunabotics/assets/lunabotics.py` | Robot config (ArticulationCfg), USD path auto-resolved |
| `lunabotics/source/lunabotics/lunabotics/tasks/direct/lunabotics/lunabotics_env_cfg.py` | Reward scales, curriculum thresholds, physics params |
| `lunabotics/source/lunabotics/lunabotics/tasks/direct/lunabotics/lunabotics_env.py` | Environment logic |
| `lunabotics/source/lunabotics/lunabotics/tasks/direct/lunabotics/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
