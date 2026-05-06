# CSU Lunabotics — Isaac Lab External Project

External Isaac Lab training project for the CSU Lunabotics 6-wheel skid-steer rover.

## Setup

### 1. Configure Isaac Lab path

```
cp isaaclab_path.cfg.example isaaclab_path.cfg
# Edit isaaclab_path.cfg and set ISAACLAB_PATH to your local Isaac Lab installation
```

Windows: `ISAACLAB_PATH=E:\IsaacLab`
Linux:   `ISAACLAB_PATH=/home/user/IsaacLab`

### 2. Install the package

```bat
# Windows — run from this folder (lunabotics/)
isaaclab.bat -p -m pip install -e source\lunabotics
```

```bash
# Linux
./isaaclab.sh -p -m pip install -e source/lunabotics
```

### 3. Run training

```bat
# Windows — run from lunabotics/ project root so logs land here
isaaclab.bat -p scripts\rsl_rl\train.py --task Template-Lunabotics-Direct-v0
```

```bash
# Linux
./isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Lunabotics-Direct-v0
```

Training logs (including `.pt` model checkpoints) are saved to:
```
logs/rsl_rl/lunabotics_direct/<timestamp>/
```

`.pt` files are tracked via Git LFS (see `.gitattributes`).

### 4. Play a trained checkpoint

```bat
isaaclab.bat -p scripts\rsl_rl\play.py --task Template-Lunabotics-Direct-Play-v0 --num_envs 16
```

### 5. List registered environments

```bat
isaaclab.bat -p scripts\list_envs.py
```

---

## Project structure

```
lunabotics/
├── isaaclab.bat / isaaclab.sh     ← wrapper — reads ISAACLAB_PATH from isaaclab_path.cfg
├── isaaclab_path.cfg              ← gitignored, machine-local path
├── isaaclab_path.cfg.example      ← committed template
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   └── rsl_rl/
│       ├── train.py
│       ├── play.py
│       └── cli_args.py
└── source/lunabotics/
    ├── setup.py
    └── lunabotics/
        ├── assets/
        │   └── lunabotics.py      ← ArticulationCfg, USD path resolved relative to repo
        └── tasks/direct/lunabotics/
            ├── lunabotics_env.py      ← DirectRLEnv implementation
            ├── lunabotics_env_cfg.py  ← env config (rewards, curriculum, physics)
            └── agents/
                └── rsl_rl_ppo_cfg.py  ← PPO hyperparameters
```

## Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `action_scale` | 3.67 rad/s | 35 RPM max wheel speed |
| `wheel_radius` | 0.91 m | 18.2 mm USD × scale=0.1 |
| `env_spacing` | 12.0 m | robot ~10× real size at scale=0.1 |
| `num_envs` (train) | 1024 | |
| `num_envs` (play) | 16 | |
| `curriculum_phase1` | 16 000 steps | straight only |
| `curriculum_phase2` | 36 000 steps | + turning |

## Cross-platform notes

- The `isaaclab.bat` / `isaaclab.sh` wrappers read `ISAACLAB_PATH` from `isaaclab_path.cfg`.
- The USD path in `assets/lunabotics.py` is resolved at import time relative to the file location — it does **not** need to be edited when cloning to a new machine, as long as the repository structure is preserved.
