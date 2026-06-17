"""Model-free geometry self-test for carpet_search.rectify (no SAM3 / HF token needed).

Projects a rectangle of KNOWN physical aspect through a real pinhole camera at a known tilt,
fills the projected quad as a mask, then checks that:
  - mask_to_quad recovers a convex 4-corner quad,
  - recover_wh_ratio (Zhang & He) recovers the true width:height within tolerance,
  - _should_passthrough is True for an un-tilted (top-down) rug and False for a tilted one.

    python -m scripts.selftest_rectify
"""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np

from carpet_search.rectify import (
    _should_passthrough,
    _zigzag_from_ordered,
    mask_to_quad,
    recover_wh_ratio,
)

W, H, F, D = 800, 600, 900.0, 5.0
TOL = 0.15  # Zhang-He is corner-sensitive; 15% is a fair bar for a rasterised mask


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project_rect(true_wh: float, pitch: float, yaw: float = 0.0, roll: float = 0.0):
    """Project a true_wh-aspect planar rectangle through a pinhole camera at a given pose.

    A generic pose needs BOTH pitch and yaw so both edge-pairs converge to finite vanishing
    points — pure pitch (yaw=0) leaves one pair parallel, the degenerate single-VP case where
    Zhang-He cannot recover the focal length and the code falls back to a side-length estimate.
    """
    long = 1.2
    hw, hh = (long, long / true_wh) if true_wh >= 1 else (long * true_wh, long)
    corners = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], float)
    R = _rz(math.radians(roll)) @ _rx(math.radians(pitch)) @ _ry(math.radians(yaw))
    cx, cy = W / 2.0, H / 2.0
    pts = []
    for P in corners:
        xc = R @ P + np.array([0, 0, D])
        pts.append([F * xc[0] / xc[2] + cx, F * xc[1] / xc[2] + cy])
    return np.array(pts, np.float32)


def make_mask(pts: np.ndarray) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 255)
    return mask


def main() -> None:
    failures = []
    from carpet_search.rectify import WH_MAX, WH_MIN, _order_points

    # --- accuracy: generic poses (pitch + yaw) where Zhang-He is valid -------------------
    print("Zhang-He accuracy (generic pitch+yaw):")
    print(f"  {'true w:h':>9} {'pose':>14} {'recovered':>10} {'err%':>6}  result")
    for true_wh in (0.5, 1.0, 1.6, 2.5):
        for pitch, yaw, roll in ((25.0, 20.0, 0.0), (35.0, 25.0, 5.0)):
            quad = mask_to_quad(make_mask(project_rect(true_wh, pitch, yaw, roll)))
            if quad is None:
                print(f"  {true_wh:9.2f} {f'p{pitch:.0f},y{yaw:.0f}':>14} {'-':>10} {'-':>6}  FAIL no quad")
                failures.append(f"wh={true_wh} p{pitch},y{yaw}: mask_to_quad returned None")
                continue
            wh = recover_wh_ratio(_zigzag_from_ordered(quad), W, H)
            err = abs(wh - true_wh) / true_wh
            ok = err < TOL
            print(f"  {true_wh:9.2f} {f'p{pitch:.0f},y{yaw:.0f}':>14} {wh:10.3f} {err*100:6.1f}  "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"wh={true_wh} p{pitch},y{yaw}: recovered {wh:.3f} (err {err*100:.1f}%)")

    # --- degenerate: pure pitch (single VP) must fall back gracefully, not crash/explode --
    print("\nDegenerate single-VP (pure pitch) graceful fallback:")
    for true_wh in (0.5, 1.6):
        quad = mask_to_quad(make_mask(project_rect(true_wh, 30.0, 0.0)))
        wh = recover_wh_ratio(_zigzag_from_ordered(quad), W, H) if quad is not None else None
        ok = wh is not None and WH_MIN <= wh <= WH_MAX
        shown = f"{wh:.3f}" if wh is not None else "None"
        print(f"  true {true_wh}: fallback={shown}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"degenerate wh={true_wh}: returned {wh} (want finite, in range)")

    # --- passthrough gate: top-down rug should skip, tilted rug should not ----------------
    flat = _order_points(project_rect(1.6, 0.0, 0.0))
    tilt = _order_points(project_rect(1.6, 30.0, 22.0, 4.0))
    if not _should_passthrough(flat, W, H):
        failures.append("passthrough: top-down rug was NOT detected as passthrough")
    if _should_passthrough(tilt, W, H):
        failures.append("passthrough: tilted rug was wrongly detected as passthrough")
    print(f"\npassthrough(top-down)={_should_passthrough(flat, W, H)}  "
          f"passthrough(tilted)={_should_passthrough(tilt, W, H)}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nAll geometry self-tests passed.")


if __name__ == "__main__":
    main()
