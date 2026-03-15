# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to list registered Lunabotics task environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List registered Lunabotics environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import lunabotics.tasks  # noqa: F401


def main():
    """Print all registered Template- environments."""
    envs = [env_spec for env_spec in gym.envs.registry if env_spec.startswith("Template-")]
    print(f"\nRegistered Lunabotics environments ({len(envs)}):")
    for env in sorted(envs):
        print(f"  {env}")
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
