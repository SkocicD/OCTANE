"""Quick structure inspection of a robot USD for Isaac Lab compatibility."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, required=True)
parser.add_argument("--out", type=str, default="C:/Users/adam.carbone/usd_inspect_out.txt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Sdf

OUT = open(args_cli.out, "w")
def p(s=""): OUT.write(s + "\n"); OUT.flush()

stage = Usd.Stage.Open(args_cli.usd)
assert stage, "Could not open USD"
p("USD: " + args_cli.usd)

p("\n=== ARTICULATION ROOTS ===")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        fix = prim.GetAttribute("physics:fixedBase").Get()
        p("  " + str(prim.GetPath()) + "  fixedBase=" + str(fix))

p("\n=== JOINTS ===")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() in (
        "PhysicsRevoluteJoint","PhysicsPrismaticJoint","PhysicsFixedJoint",
        "RevoluteJoint","PrismaticJoint","FixedJoint"):
        drives = []
        for ax in ("angular", "linear"):
            d = UsdPhysics.DriveAPI.Get(prim, ax)
            if d:
                stiff = prim.GetAttribute("drive:" + ax + ":physics:stiffness").Get()
                damp  = prim.GetAttribute("drive:" + ax + ":physics:damping").Get()
                drives.append(ax + "(stiff=" + str(stiff) + ",damp=" + str(damp) + ")")
        p("  %-40s type=%-30s drives=%s" % (prim.GetName(), prim.GetTypeName(), drives))

p("\n=== RIGID BODIES ===")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        enabled   = prim.GetAttribute("physics:rigidBodyEnabled").Get()
        kinematic = prim.GetAttribute("physics:kinematicEnabled").Get()
        mass_attr = prim.GetAttribute("physics:mass")
        mass = mass_attr.Get() if mass_attr else None
        p("  %-40s enabled=%-6s kinematic=%-6s mass=%s" % (prim.GetPath().name, str(enabled), str(kinematic), str(mass)))

p("\n=== OMNIGRAPH PRIMS ===")
found_graphs = []
for prim in stage.Traverse():
    tn = prim.GetTypeName()
    if "OmniGraph" in tn or "ComputeGraph" in tn or tn.startswith("omni.graph") or prim.GetName() in ("Graphs","ActionGraph"):
        found_graphs.append("  " + str(prim.GetPath()) + "  active=" + str(prim.IsActive()))
if found_graphs:
    for g in found_graphs: p(g)
else:
    p("  (none found)")

p("\nDone.")
OUT.close()
simulation_app.close()
