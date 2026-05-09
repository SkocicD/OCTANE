"""
Deactivate all OmniGraph nodes in a USD.

OmniGraph nodes (ROS2 publishers, differential drive controllers, camera
pipelines, etc.) are authored for Isaac Sim interactive use.  When Isaac Lab
loads the same USD for RL training they still execute, competing with Python
actuators and printing CUDA-tensor conversion warnings.

This script sets every OmniGraph prim to inactive (prim.SetActive(False)) so
they are ignored by the sim without being permanently deleted.  The physics
articulation, joints, and collision geometry are left completely untouched.

Usage:
    .\isaaclab.bat -p "C:/path/to/Tools/deactivate_omnigraphs.py" ^
        --usd "path/to/robot.usd" --headless

    --dry-run   Print what would be deactivated without saving.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deactivate OmniGraph nodes in a robot USD.")
parser.add_argument("--usd",     type=str, required=True)
parser.add_argument("--dry-run", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd

stage = Usd.Stage.Open(args_cli.usd)
assert stage, f"Could not open: {args_cli.usd}"

print(f"\n--- Deactivating OmniGraph prims ---")
print(f"  USD:      {args_cli.usd}")
print(f"  Dry-run:  {args_cli.dry_run}\n")

# Type names that identify OmniGraph prims
OMNI_TYPES = {
    "OmniGraph",
    "ComputeGraph",
    "OmniGraphNode",
}

# USD API schemas that indicate an OmniGraph prim
OMNI_SCHEMAS = {
    "OmniGraphAPI",
    "NodeGraphNodeAPI",
}

deactivated = []
already_inactive = []
saved_layers = set()

for prim in stage.Traverse():
    type_name = prim.GetTypeName()

    is_graph = (
        type_name in OMNI_TYPES
        or any(api in prim.GetAppliedSchemas() for api in OMNI_SCHEMAS)
        or type_name.startswith("omni.graph")
        or "OmniGraph" in type_name
    )

    if not is_graph:
        continue

    if not prim.IsActive():
        already_inactive.append(str(prim.GetPath()))
        print(f"  SKIP (already inactive): {prim.GetPath()}")
        continue

    print(f"  DEACTIVATE: {prim.GetPath()}  (type: {type_name})")
    deactivated.append(str(prim.GetPath()))

    if not args_cli.dry_run:
        prim.SetActive(False)
        for spec in prim.GetPrimStack():
            saved_layers.add(spec.layer)

# Also deactivate any prim named "Graphs" (common container scope in Isaac Sim)
for prim in stage.Traverse():
    if prim.GetName() == "Graphs" and prim.IsActive():
        print(f"  DEACTIVATE scope: {prim.GetPath()}")
        deactivated.append(str(prim.GetPath()))
        if not args_cli.dry_run:
            prim.SetActive(False)
            for spec in prim.GetPrimStack():
                saved_layers.add(spec.layer)

print(f"\n--- Summary ---")
print(f"  Deactivated:       {len(deactivated)}")
print(f"  Already inactive:  {len(already_inactive)}")

if args_cli.dry_run:
    print("  (dry-run — nothing saved)")
elif saved_layers:
    print(f"\n--- Saving {len(saved_layers)} layer(s) ---")
    for layer in saved_layers:
        layer.Save()
        print(f"  Saved: {layer.identifier}")
    print("\nDone. OmniGraph nodes will be ignored at next sim load.")
elif deactivated:
    print("  Warning: prims were found but no layers were dirty — check USD structure.")
else:
    print("  No OmniGraph prims found (already clean or different structure).")

simulation_app.close()
