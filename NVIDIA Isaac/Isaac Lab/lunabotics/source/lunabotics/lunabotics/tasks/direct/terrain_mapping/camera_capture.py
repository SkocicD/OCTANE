"""Capture camera frames as numpy arrays using omni.replicator annotators.

No ROS, no DDS, no bridge.  Frames are returned as plain numpy arrays and
saved directly into the episode NPZ files alongside ground-truth terrain data.
A separate offline ROS node can replay them later.

Outputs per capture() call:
    {
        "left_front":    (360, 480, 3)  uint8  RGB
        "left_side":     (360, 480, 3)  uint8  RGB
        "right_front":   (360, 480, 3)  uint8  RGB
        "right_side":    (360, 480, 3)  uint8  RGB
        "back_rear":     (360, 480, 3)  uint8  RGB
        "depth_cam_rgb": (480, 640, 3)  uint8  RGB
        "depth_cam_d":   (480, 640)     float32  metres
    }
"""

from __future__ import annotations
import numpy as np

# Map: prim-name-fragment → (serial, annotator_type, width, height)
_CAM_MAP: dict[str, tuple] = {
    "LEFT_FRONT":           ("left_front",    "rgb",   480, 360),
    "LEFT_SIDE":            ("left_side",     "rgb",   480, 360),
    "RIGHT_FRONT":          ("right_front",   "rgb",   480, 360),
    "RIGHT_SIDE":           ("right_side",    "rgb",   480, 360),
    "BACK_REAR":            ("back_rear",     "rgb",   480, 360),
    "Orbbec_Astra_Pro_RGB": ("depth_cam_rgb", "rgb",   640, 480),
    "Orbbec_Astra_Pro_D":   ("depth_cam_d",   "depth", 640, 480),
}


def _find_camera_prims() -> dict[str, str]:
    """Traverse the USD stage and return {key: prim_path} for known cameras."""
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:
        return {}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {}

    found: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path     = str(prim.GetPath())
        prim_name = prim.GetName()
        for key in _CAM_MAP:
            if key in prim_name or key in path:
                if key not in found:
                    found[key] = path
                break

    return found


class CameraCapture:
    """Attaches omni.replicator annotators to all robot cameras and captures frames."""

    def __init__(self):
        self._ready   = False
        self._cameras: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────
    def setup(self) -> None:
        """Create render products and attach annotators.  Call once after scene loads."""
        print("[CameraCapture] setup() starting...")

        try:
            import omni.replicator.core as rep
        except ImportError as e:
            print(f"[CameraCapture] FATAL: omni.replicator not available: {e}")
            return

        cam_prims = _find_camera_prims()
        print(f"[CameraCapture] stage scan: {len(cam_prims)}/{len(_CAM_MAP)} camera prims found")
        for k, p in cam_prims.items():
            print(f"  {k:30s} → {p}")
        if not cam_prims:
            print("[CameraCapture] WARNING: no camera prims found in stage — "
                  "run scripts/find_camera_prims.py in the Script Editor to inspect paths")

        for key, (serial, cam_type, w, h) in _CAM_MAP.items():
            prim = cam_prims.get(key)
            if prim is None:
                print(f"[CameraCapture] SKIP {key}: not in stage")
                continue

            try:
                rp    = rep.create.render_product(prim, resolution=(w, h))
                annot = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane" if cam_type == "depth" else "rgb"
                )
                annot.attach([rp])
                self._cameras.append({
                    "key": key, "serial": serial,
                    "cam_type": cam_type, "w": w, "h": h,
                    "annotator": annot,
                })
                print(f"[CameraCapture] attached: {key}")
            except Exception as e:
                print(f"[CameraCapture] WARNING: could not attach {key}: {e}")

        self._ready = bool(self._cameras)
        print(f"[CameraCapture] {len(self._cameras)}/{len(_CAM_MAP)} cameras ready")

    # ──────────────────────────────────────────────────────────────────────
    def capture(self) -> dict[str, np.ndarray]:
        """Read latest rendered frames.  Returns {serial: ndarray}, skips empty."""
        if not self._ready:
            return {}

        frames: dict[str, np.ndarray] = {}
        for cam in self._cameras:
            data = cam["annotator"].get_data()
            if data is None or data.size == 0:
                continue
            w, h = cam["w"], cam["h"]
            if cam["cam_type"] == "depth":
                frames[cam["serial"]] = data.reshape(h, w).astype(np.float32)
            else:
                rgba = data.reshape(h, w, 4)
                frames[cam["serial"]] = rgba[:, :, :3].copy()  # drop alpha → RGB uint8

        return frames

    # ──────────────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        self._ready   = False
        self._cameras = []
