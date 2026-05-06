"""Direct RL environments for the CSU Lunabotics 6-wheel skid-steer rover."""

import gymnasium as gym

from . import agents

gym.register(
    id="Template-Lunabotics-Direct-v0",
    entry_point=f"{__name__}.lunabotics_env:LunaboticsDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lunabotics_env_cfg:LunaboticsDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LunaboticsDirectPPORunnerCfg",
    },
)

gym.register(
    id="Template-Lunabotics-Direct-Play-v0",
    entry_point=f"{__name__}.lunabotics_env:LunaboticsDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lunabotics_env_cfg:LunaboticsDirectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LunaboticsDirectPPORunnerCfg",
    },
)
