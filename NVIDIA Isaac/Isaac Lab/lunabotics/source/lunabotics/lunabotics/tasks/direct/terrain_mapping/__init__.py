import gymnasium as gym

gym.register(
    id="Template-TerrainCollection-v0",
    entry_point=f"{__name__}.terrain_collection_env:TerrainCollectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.terrain_collection_env_cfg:TerrainCollectionEnvCfg",
    },
)
