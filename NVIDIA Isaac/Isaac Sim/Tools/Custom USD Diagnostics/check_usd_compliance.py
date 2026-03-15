"""
USD Isaac Lab Compliance Checker

Validates a robot USD for common issues before Isaac Lab training.
Produces a PASS / WARN / FAIL report with fix suggestions.

Usage:
    .\isaaclab.bat -p "C:/path/to/Tools/check_usd_compliance.py" --usd "path/to/robot.usd" --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check a robot USD for Isaac Lab compliance.")
parser.add_argument("--usd", type=str, required=True, help="Path to the USD file to check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

issues_critical = []
issues_warn = []

def passed(msg):  print(f"  {PASS} {msg}")
def warn(msg, fix=""):
    print(f"  {WARN} {msg}")
    if fix: print(f"         FIX: {fix}")
    issues_warn.append(msg)
def fail(msg, fix=""):
    print(f"  {FAIL} {msg}")
    if fix: print(f"         FIX: {fix}")
    issues_critical.append(msg)
def info(msg):    print(f"  {INFO} {msg}")

def rigid_body_ancestor(prim):
    p = prim.GetParent()
    while p and p.GetPath() != Sdf.Path("/"):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  USD COMPLIANCE CHECK")
print(f"  {args_cli.usd}")
print(f"{'='*60}\n")

stage = Usd.Stage.Open(args_cli.usd)
if not stage:
    print(f"{FAIL} Could not open USD: {args_cli.usd}")
    simulation_app.close()
    exit(1)

# ── 1. UNITS & STAGE ─────────────────────────────────────────────────────────
print("━━━ 1. UNITS & STAGE ━━━")
mpu = UsdGeom.GetStageMetersPerUnit(stage)
up  = UsdGeom.GetStageUpAxis(stage)

if   abs(mpu - 1.0)   < 1e-6: passed(f"metersPerUnit = {mpu} (meters — correct)")
elif abs(mpu - 0.01)  < 1e-6: fail(
    f"metersPerUnit = {mpu} (centimeters)",
    "Re-export in meters OR add scale=(0.01,0.01,0.01) to UsdFileCfg. "
    "Without correction auto-computed masses are 10^6x too large and wheel geometry "
    "is 100x the actual radius, causing perpetual upward drift from depenetration.")
elif abs(mpu - 0.001) < 1e-6: fail(
    f"metersPerUnit = {mpu} (millimeters)",
    "Re-export in meters OR add scale=(0.001,0.001,0.001) to UsdFileCfg.")
else: warn(f"metersPerUnit = {mpu} — verify this is intentional")

if up == "Z": passed(f"upAxis = {up}")
else:         fail(f"upAxis = {up}", "Isaac Lab expects Z-up. Re-export with Z as up axis.")

# ── 2. ARTICULATION ROOT ──────────────────────────────────────────────────────
print("\n━━━ 2. ARTICULATION ROOT ━━━")
art_roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]

if not art_roots:
    fail("No ArticulationRootAPI found.",
         "Add ArticulationRootAPI to the chassis body in Isaac Sim's physics panel.")
elif len(art_roots) > 1:
    warn(f"Multiple ArticulationRootAPI prims ({len(art_roots)}): "
         + ", ".join(str(p.GetPath()) for p in art_roots),
         "Only one expected — remove extras.")
else:
    art_root = art_roots[0]
    passed(f"ArticulationRootAPI: {art_root.GetPath()}")

    if art_root.HasAPI(UsdPhysics.RigidBodyAPI):
        passed("Root is a RigidBody")
        rb = UsdPhysics.RigidBodyAPI(art_root)
        if rb.GetRigidBodyEnabledAttr().Get() == False:
            fail("rigidBodyEnabled = False", "Set physics:rigidBodyEnabled = True on chassis.")
        else:
            passed("rigidBodyEnabled = True")
        km = art_root.GetAttribute("physics:kinematicEnabled")
        if km and km.HasValue() and km.Get():
            fail("kinematicEnabled = True — robot is locked in place!",
                 "Set physics:kinematicEnabled = False on chassis.")
        else:
            passed("kinematicEnabled = False")
    else:
        warn(f"{art_root.GetPath()} is not a RigidBody (virtual/xform root). "
             "Isaac Lab prefers the articulation root to be the chassis rigid body.")

    # World fixed joints
    world_fixed = []
    for fj in [p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]:
        j  = UsdPhysics.Joint(fj)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            world_fixed.append(fj)
    if world_fixed:
        fail("FixedJoint(s) connecting robot to world: "
             + ", ".join(str(f.GetPath()) for f in world_fixed),
             "Remove these or set ArticulationRootPropertiesCfg(fix_root_link=False).")
    else:
        passed("No world-fixed joints")

    if art_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
        fb = art_root.GetAttribute("physxArticulation:fixedBase")
        if fb and fb.HasValue() and fb.Get():
            fail("physxArticulation:fixedBase = True",
                 "Set fixedBase = False in Isaac Sim physics panel.")
        else:
            passed("physxArticulation:fixedBase = False")

# ── 3. JOINTS ─────────────────────────────────────────────────────────────────
print("\n━━━ 3. JOINTS ━━━")
rev_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
pri_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
all_joints = rev_joints + pri_joints

info(f"{len(rev_joints)} revolute, {len(pri_joints)} prismatic joints")

# Broken joint references
broken = []
for jp in all_joints:
    j = UsdPhysics.Joint(jp)
    for rel, lbl in [(j.GetBody0Rel(), "body0"), (j.GetBody1Rel(), "body1")]:
        for t in rel.GetTargets():
            if not stage.GetPrimAtPath(t).IsValid():
                broken.append((jp.GetPath(), lbl, t))
if broken:
    fail(f"{len(broken)} broken joint body reference(s):",
         "Fix or remove joints with missing body targets.")
    for path, lbl, t in broken: print(f"         {path} → {lbl} = {t} (NOT FOUND)")
else:
    passed("All joint body references valid")

# Joint axes
print()
axes = {}
for jp in rev_joints:
    j = UsdPhysics.RevoluteJoint(jp)
    ax = j.GetAxisAttr().Get() if j.GetAxisAttr() else "X"
    axes.setdefault(ax, []).append(jp.GetName())
for ax, names in sorted(axes.items()):
    info(f"Revolute axis={ax}: {names}")

# Active drives
print()
DRIVE_ATTRS = [
    "drive:angular:physics:targetVelocity", "drive:angular:physics:damping",
    "drive:angular:physics:stiffness",      "drive:angular:physics:maxForce",
    "drive:linear:physics:targetVelocity",  "drive:linear:physics:damping",
    "drive:linear:physics:stiffness",       "drive:linear:physics:maxForce",
]
active_drives = {}
for jp in all_joints:
    for an in DRIVE_ATTRS:
        a = jp.GetAttribute(an)
        if a and a.HasValue():
            v = a.Get()
            if v is not None and abs(float(v)) > 1e-6:
                active_drives.setdefault(jp.GetName(), []).append((an.split(":")[-1], v))

if active_drives:
    warn(f"Non-zero drives on {len(active_drives)} joint(s):",
         "Joints you don't control via Python should have drives zeroed, OR "
         "replaced with a position-hold (stiffness>0, damping>0, targetVelocity=0) "
         "to prevent them hanging freely under gravity. "
         "Use lock_arm_joints.py for the second approach.")
    for jn, attrs in active_drives.items():
        print(f"         {jn}: " + ", ".join(f"{k}={v:.2f}" for k, v in attrs))
else:
    passed("All joint drives are zero or unset")

# ── 4. COLLISION ──────────────────────────────────────────────────────────────
print("\n━━━ 4. COLLISION GEOMETRY ━━━")
all_rb = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]

with_col, without_col = [], []
for rb in all_rb:
    has = False
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        anc = rigid_body_ancestor(prim)
        if anc and anc.GetPath() == rb.GetPath():
            col = UsdPhysics.CollisionAPI(prim)
            en  = col.GetCollisionEnabledAttr().Get()
            if en is None or en:
                has = True; break
    if not has and rb.HasAPI(UsdPhysics.CollisionAPI):
        col = UsdPhysics.CollisionAPI(rb)
        en  = col.GetCollisionEnabledAttr().Get()
        if en is None or en: has = True
    (with_col if has else without_col).append(rb.GetName())

info(f"Bodies WITH  active collision: {len(with_col)}  → {with_col}")
info(f"Bodies WITHOUT collision:      {len(without_col)}  → {without_col}")

if without_col:
    warn(f"{len(without_col)} rigid body(s) have no active collision shapes. "
         "If these are structural parts (arm, bucket) this may be intentional. "
         "Wheels and chassis must have collision or the robot will fall through the ground.")

# ── 5. MASS / INERTIA ─────────────────────────────────────────────────────────
print("\n━━━ 5. MASS / INERTIA ━━━")
with_mass, without_mass, total = [], [], 0.0
for rb in all_rb:
    if rb.HasAPI(UsdPhysics.MassAPI):
        a = UsdPhysics.MassAPI(rb).GetMassAttr()
        if a and a.HasValue() and a.Get() > 0:
            with_mass.append((rb.GetName(), a.Get()))
            total += a.Get()
            continue
    without_mass.append(rb.GetName())

if with_mass:
    passed(f"{len(with_mass)} bodies have explicit mass:")
    for n, m in with_mass: print(f"         {n}: {m:.4f} kg")
    info(f"Sum of explicit masses: {total:.3f} kg")

if without_mass:
    if abs(mpu - 1.0) > 1e-6:
        fail(
            f"{len(without_mass)} body(s) use auto-computed mass with metersPerUnit={mpu} "
            f"→ masses will be {(1.0/mpu)**3:.0f}x too large:",
            "Set metersPerUnit=1.0 at export, OR add physics:mass overrides via MassAPI, "
            "OR set a compensating density on a PhysicsMaterialAPI.")
        for n in without_mass: print(f"         {n}")
    else:
        info(f"{len(without_mass)} bodies use auto-computed mass (metersPerUnit=1.0 — OK).")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY — {len(issues_critical)} critical  |  {len(issues_warn)} warnings")
print(f"{'='*60}")
if not issues_critical and not issues_warn:
    print(f"  {PASS} No issues found — USD looks compliant!")
else:
    if issues_critical:
        print(f"\n  Critical:")
        for i, m in enumerate(issues_critical, 1): print(f"    {i}. {m}")
    if issues_warn:
        print(f"\n  Warnings:")
        for i, m in enumerate(issues_warn, 1): print(f"    {i}. {m}")

print()
simulation_app.close()
