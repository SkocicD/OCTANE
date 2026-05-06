"""
Disable collision on all rigid bodies EXCEPT the chassis and wheels.

Arm/excavation body collision geometry that extends below the ground plane
at spawn causes explosive depenetration launches. Since driving-policy
training doesn't need arm collision, this script disables it.

Usage:
    .\isaaclab.bat -p "C:/path/to/Tools/disable_nonwheel_collision.py" ^
        --usd  "path/to/robot.usd"      ^
        --keep "chassis_body,Wheel1,Wheel2,..."  ^
        --headless

    --keep  comma-separated list of rigid body prim NAMES to keep collision ON
            (chassis + all wheel bodies). Everything else gets disabled.
            Example: --keep "tn__base_link1_wJ,tn__Wheel11_i7t6,tn__Wheel21_i7t6"

    --dry-run   Print what would be disabled without saving.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Disable non-wheel/chassis collision in a robot USD.")
parser.add_argument("--usd",     type=str, required=True,
                    help="Path to the USD file to modify")
parser.add_argument("--keep",    type=str, default="",
                    help="Comma-separated rigid body prim names to keep collision ON")
parser.add_argument("--dry-run", action="store_true",
                    help="Report what would change without saving")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdPhysics

keep_set = {n.strip() for n in args_cli.keep.split(",") if n.strip()}

def rigid_body_ancestor(prim):
    p = prim.GetParent()
    while p and p.GetPath() != Sdf.Path("/"):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None

stage = Usd.Stage.Open(args_cli.usd)
assert stage, f"Could not open: {args_cli.usd}"

print(f"\n--- Disabling non-wheel collision ---")
print(f"  USD:       {args_cli.usd}")
print(f"  Keep ON:   {keep_set if keep_set else '(none specified — all arm bodies will be disabled)'}")
print(f"  Dry-run:   {args_cli.dry_run}\n")

disabled = []
saved_layers = set()

for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue

    rb  = rigid_body_ancestor(prim)
    name = rb.GetName() if rb else None

    if name in keep_set:
        print(f"  KEEP : {prim.GetPath()}  (rb: {name})")
        continue

    col = UsdPhysics.CollisionAPI(prim)
    currently_enabled = col.GetCollisionEnabledAttr().Get()
    if currently_enabled is False:
        print(f"  SKIP : {prim.GetPath()}  (already disabled)")
        continue

    print(f"  DISABLE: {prim.GetPath()}  (rb: {name})")
    disabled.append(str(prim.GetPath()))

    if not args_cli.dry_run:
        col.GetCollisionEnabledAttr().Set(False)
        for spec in prim.GetPrimStack():
            saved_layers.add(spec.layer)

print(f"\n--- Summary ---")
print(f"  Collision shapes to disable: {len(disabled)}")

if args_cli.dry_run:
    print("  (dry-run — nothing saved)")
elif saved_layers:
    print(f"\n--- Saving {len(saved_layers)} layer(s) ---")
    for layer in saved_layers:
        layer.Save()
        print(f"  Saved: {layer.identifier}")
else:
    print("  Nothing to save (no changes made).")

simulation_app.close()
