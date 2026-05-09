"""Capture camera frames as numpy arrays using omni.replicator annotators.

No ROS, no DDS, no bridge.  Frames are returned as plain numpy arrays and
saved directly into the episode NPZ files alongside ground-truth terrain data.
A separate offline ROS node can replay them later.

Outputs per capture() call:
    {
        env_idx: {
            "left_front":    (360, 480, 3)  uint8  RGB
            "left_side":     (360, 480, 3)  uint8  RGB
            "right_front":   (360, 480, 3)  uint8  RGB
            "right_side":    (360, 480, 3)  uint8  RGB
            "back_rear":     (360, 480, 3)  uint8  RGB
            "depth_cam_rgb": (480, 640, 3)  uint8  RGB
            "depth_cam_d":   (480, 640)     float32  metres
        }
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


def _known_paths(env_idx: int) -> dict[str, str]:
    base = f"/World/envs/env_{env_idx}/Robot/tn__base_link1_wJ/tn__Cameras1_XG"
    return {
        "LEFT_FRONT":           f"{base}/Innomaker_RGB_130_LEFT_FRONT",
        "LEFT_SIDE":            f"{base}/Innomaker_RGB_130_LEFT_SIDE",
        "RIGHT_FRONT":          f"{base}/Innomaker_RGB_130_RIGHT_FRONT",
        "RIGHT_SIDE":           f"{base}/Innomaker_RGB_130_RIGHT_SIDE",
        "BACK_REAR":            f"{base}/Innomaker_RGB_130_BACK_REAR",
        "Orbbec_Astra_Pro_RGB": f"{base}/Orbbec_Astra_Pro_RGB",
        "Orbbec_Astra_Pro_D":   f"{base}/Orbbec_Astra_Pro_D",
    }


def _find_camera_prims(env_idx: int = 0) -> dict[str, str]:
    """Traverse the USD stage and return {key: prim_path} for env_idx's cameras."""
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:
        return {}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {}

    env_prefix = f"/World/envs/env_{env_idx}/"
    found: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path      = str(prim.GetPath())
        if env_prefix not in path:
            continue
        prim_name = prim.GetName()
        for key in _CAM_MAP:
            if key in prim_name or key in path:
                if key not in found:
                    found[key] = path
                break

    # Fallback: use known static paths for any camera the traversal missed.
    for key, path in _known_paths(env_idx).items():
        if key not in found:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                found[key] = path
                print(f"[CameraCapture] env_{env_idx} fallback path used for {key}: {path}")

    return found


class CameraCapture:
    """Attaches omni.replicator annotators to all robot cameras and captures frames."""

    def __init__(self, num_envs: int = 1):
        self._ready            = False
        self._num_envs         = num_envs
        self._cameras_per_env: list[list[dict]] = [[] for _ in range(num_envs)]

    # ──────────────────────────────────────────────────────────────────────
    def setup(self) -> None:
        """Create render products and attach annotators.  Call once after scene loads."""
        print(f"[CameraCapture] setup() starting for {self._num_envs} env(s)...")

        try:
            import omni.replicator.core as rep
        except ImportError as e:
            print(f"[CameraCapture] FATAL: omni.replicator not available: {e}")
            return

        # Disable AA/DLSS — reduces internal render resolution below the housing
        # near-clip threshold, causing depth_cam_rgb to go black intermittently.
        try:
            import carb.settings as _cs
            _cs.get_settings().set("/rtx/post/aa/op", 0)
        except Exception:
            pass

        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            _stage = omni.usd.get_context().get_stage()
        except Exception:
            _stage = None

        total_attached = 0
        for env_idx in range(self._num_envs):
            cam_prims = _find_camera_prims(env_idx)
            print(f"[CameraCapture] env_{env_idx}: {len(cam_prims)}/{len(_CAM_MAP)} camera prims found")
            for k, p in cam_prims.items():
                print(f"  env_{env_idx}  {k:30s} → {p}")
            if not cam_prims:
                print(f"[CameraCapture] WARNING: no camera prims found for env_{env_idx}")

            for key, (serial, cam_type, w, h) in _CAM_MAP.items():
                prim = cam_prims.get(key)
                if prim is None:
                    print(f"[CameraCapture] SKIP env_{env_idx}/{key}: not in stage")
                    continue

                if _stage is not None:
                    try:
                        cam_prim  = UsdGeom.Camera(_stage.GetPrimAtPath(prim))
                        clip_attr = cam_prim.GetClippingRangeAttr()
                        existing  = clip_attr.Get() if clip_attr else None
                        near = float(existing[0]) if existing else None
                        far  = float(existing[1]) if existing else None

                        near_target = 0.20 if "Orbbec" in key else 0.01
                        if near is None or near != near_target or (far is not None and far < 20.0):
                            clip_attr.Set(Gf.Vec2f(near_target, 150.0))
                            print(f"[CameraCapture]   env_{env_idx}/{key}: clipping fixed → ({near_target}, 150)")
                        else:
                            print(f"[CameraCapture]   env_{env_idx}/{key}: clipping OK near={near:.4f} far={far:.1f}")

                        xf  = UsdGeom.Xformable(_stage.GetPrimAtPath(prim))
                        mat = xf.ComputeLocalToWorldTransform(0)
                        fwd = mat.TransformDir(Gf.Vec3d(0, 0, -1))
                        fwd = fwd.GetNormalized()
                        print(f"[CameraCapture]   env_{env_idx}/{key}: world forward "
                              f"({fwd[0]:+.2f}, {fwd[1]:+.2f}, {fwd[2]:+.2f})"
                              f"  {'*** pointing UP ***' if fwd[2] > 0.7 else ''}")
                    except Exception as ce:
                        print(f"[CameraCapture]   env_{env_idx}/{key}: could not inspect clipping — {ce}")

                try:
                    rp    = rep.create.render_product(prim, resolution=(w, h))
                    annot = rep.AnnotatorRegistry.get_annotator(
                        "distance_to_image_plane" if cam_type == "depth" else "rgb"
                    )
                    annot.attach([rp])
                    self._cameras_per_env[env_idx].append({
                        "key": key, "serial": serial, "prim": prim,
                        "cam_type": cam_type, "w": w, "h": h,
                        "annotator": annot, "render_product": rp,
                    })
                    total_attached += 1
                    print(f"[CameraCapture] attached: env_{env_idx}/{key}")
                except Exception as e:
                    print(f"[CameraCapture] WARNING: could not attach env_{env_idx}/{key}: {e}")

        self._ready = any(len(c) > 0 for c in self._cameras_per_env)
        print(f"[CameraCapture] {total_attached}/{len(_CAM_MAP) * self._num_envs} cameras ready "
              f"across {self._num_envs} env(s)")

        # If RTX was requested via --/rtx/rendermode, the render products above
        # captured their renderer context.  Push the *viewport* back to the
        # interactive (Storm) renderer so the GUI stays fast to monitor.
        try:
            import carb.settings as _cs
            _mode = _cs.get_settings().get("/rtx/rendermode") or ""
            if "Ray" in _mode or "Path" in _mode:
                import omni.kit.viewport.utility as _vpu
                _vp = _vpu.get_active_viewport()
                if _vp is not None:
                    _vp.set_hd_engine("HdStormRendererPlugin")
                    print("[CameraCapture] Viewport reset to Interactive (cameras use RTX)")
        except Exception:
            pass  # headless or viewport not available — no action needed

    # ──────────────────────────────────────────────────────────────────────
    def capture(self) -> dict[int, dict[str, np.ndarray]]:
        """Read latest rendered frames.  Returns {env_idx: {serial: ndarray}}, skips empty."""
        if not self._ready:
            return {}

        result: dict[int, dict[str, np.ndarray]] = {}
        for env_idx, cameras in enumerate(self._cameras_per_env):
            frames: dict[str, np.ndarray] = {}
            for cam in cameras:
                data = cam["annotator"].get_data()
                if data is None or data.size == 0:
                    continue
                w, h = cam["w"], cam["h"]
                if cam["cam_type"] == "depth":
                    frames[cam["serial"]] = data.reshape(h, w).astype(np.float32)
                else:
                    rgba = data.reshape(h, w, 4)
                    frames[cam["serial"]] = rgba[:, :, :3].copy()
            result[env_idx] = frames
        return result

    # ──────────────────────────────────────────────────────────────────────
    def reinitialize_camera(self, env_idx: int, serial: str) -> bool:
        """Destroy and recreate the render product + annotator for one camera by env and serial."""
        import omni.replicator.core as rep
        cameras = self._cameras_per_env[env_idx]
        for i, cam in enumerate(cameras):
            if cam["serial"] != serial:
                continue
            print(f"[CameraCapture] Reinitializing env_{env_idx}/{cam['key']} ({serial})...")
            try:
                cam["annotator"].detach([cam["render_product"]])
            except Exception:
                pass
            try:
                cam["render_product"].destroy()
            except Exception:
                pass
            try:
                rp    = rep.create.render_product(cam["prim"], resolution=(cam["w"], cam["h"]))
                annot = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane" if cam["cam_type"] == "depth" else "rgb"
                )
                annot.attach([rp])
                cameras[i]["render_product"] = rp
                cameras[i]["annotator"]      = annot
                print(f"[CameraCapture] Reinitialized: env_{env_idx}/{cam['key']}")
                return True
            except Exception as e:
                print(f"[CameraCapture] Reinitialize FAILED for env_{env_idx}/{serial}: {e}")
                return False
        return False

    # ──────────────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        for cameras in self._cameras_per_env:
            for cam in cameras:
                try:
                    cam["annotator"].detach()
                except Exception:
                    pass
                try:
                    cam["render_product"].destroy()
                except Exception:
                    pass
        self._ready            = False
        self._cameras_per_env  = [[] for _ in range(self._num_envs)]
