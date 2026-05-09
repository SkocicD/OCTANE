"""
Replace arm joint drives with position-hold drives.

Joints you DON'T control from Python should hold their resting pose rather
than hanging freely under gravity. This script sets a position drive
(high stiffness, moderate damping, zero velocity) on every joint EXCEPT
the ones you specify as wheel/actuated joints.

Why not just zero drives?
  Zeroing drives removes the restoring force that keeps structural joints
  (e.g. bucket pivot) in place. The joint then hangs under gravity, breaking
  the arm geometry and potentially causing solver instability.

Usage:
    .\isaaclab.bat -p "C:/path/to/Tools/lock_arm_joints.py" ^
        --usd   "path/to/robot.usd"                          ^
        --skip  "Left_Front_Wheel,Right_Front_Wheel,..."     ^
        --headless

    --skip        comma-separated joint prim NAMES to leave untouched
                  (your wheel joints / any joint with a Python actuator)
    --stiffness   Nm/rad, default 1000
    --damping     Nm*s/rad, default 100
    --max-force   Nm, default 500
    --dry-run     Print what would change without saving
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Set position-hold drives on non-wheel joints.")
parser.add_argument("--usd",        type=str,   required=True)
parser.add_argument("--skip",       type=str,   default="",
                    help="Comma-separated joint names to skip (wheel joints)")
parser.add_argument("--stiffness",  type=float, default=1000.0)
parser.add_argument("--damping",    type=float, default=100.0)
parser.add_argument("--max-force",  type=float, default=500.0,  dest="max_force")
parser.add_argument("--dry-run",    action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

skip_set = {n.strip() for n in args_cli.skip.split(",") if n.strip()}

stage = Usd.Stage.Open(args_cli.usd)
assert stage, f"Could not open: {args_cli.usd}"

print(f"\n--- Lock arm joints to position-hold ---")
print(f"  USD:        {args_cli.usd}")
print(f"  Skip:       {skip_set if skip_set else '(none — all joints will be locked)'}")
print(f"  Stiffness:  {args_cli.stiffness} Nm/rad")
print(f"  Damping:    {args_cli.damping} Nm*s/rad")
print(f"  Max force:  {args_cli.max_force} Nm")
print(f"  Dry-run:    {args_cli.dry_run}\n")

saved_layers = set()
locked, skipped = [], []

for prim in stage.Traverse():
    is_rev = prim.IsA(UsdPhysics.RevoluteJoint)
    is_pri = prim.IsA(UsdPhysics.PrismaticJoint)
    if not (is_rev or is_pri):
        continue

    name = prim.GetName()
    if name in skip_set:
        skipped.append(name)
        print(f"  SKIP  : {name}")
        continue

    drive_type = "angular" if is_rev else "linear"

    print(f"  LOCK  : {name}  [{drive_type}]  "
          f"stiffness={args_cli.stiffness}, damping={args_cli.damping}, maxForce={args_cli.max_force}")
    locked.append(name)

    if not args_cli.dry_run:
        if not prim.HasAPI(UsdPhysics.DriveAPI, drive_type):
            UsdPhysics.DriveAPI.Apply(prim, drive_type)
        drive = UsdPhysics.DriveAPI(prim, drive_type)
        drive.GetStiffnessAttr().Set(args_cli.stiffness)
        drive.GetDampingAttr().Set(args_cli.damping)
        drive.GetMaxForceAttr().Set(args_cli.max_force)
        drive.GetTargetVelocityAttr().Set(0.0)
        drive.GetTargetPositionAttr().Set(0.0)
        for spec in prim.GetPrimStack():
            saved_layers.add(spec.layer)

print(f"\n--- Summary ---")
print(f"  Joints locked: {len(locked)}")
print(f"  Joints skipped (wheels/actuated): {len(skipped)}")

if args_cli.dry_run:
    print("  (dry-run — nothing saved)")
elif saved_layers:
    print(f"\n--- Saving {len(saved_layers)} layer(s) ---")
    for layer in saved_layers:
        layer.Save()
        print(f"  Saved: {layer.identifier}")
else:
    print("  Nothing to save.")

simulation_app.close()
