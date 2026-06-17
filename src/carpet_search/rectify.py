"""Classical perspective rectification of a carpet from its segmentation mask.

Given a (clean) boolean carpet mask — e.g. from SAM3 — this module recovers a
fronto-parallel, metrically-correct top-down view of the rug:

  mask -> 4-corner quad (Phase 2) -> true width:height via Zhang & He (Phase 3)
       -> homography + Lanczos warp (Phases 4-5).

It is deliberately classical (OpenCV only; project non-goal: no deep dewarping nets) and
NEVER raises: every stage falls back to returning the input unchanged when detection is
unreliable, matching the fallback philosophy of preprocess.py. A SAFETY PASSTHROUGH skips
the warp entirely for images that are already (near) top-down, so the step is safe to leave
ON for clean uploads while still correcting genuinely distorted phone photos.

References: the perspective-rectification report (Zhang & He closed form for aspect ratio
from a single uncalibrated view); see the project plan.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- safety-passthrough tunables (skip rectification only when clearly already top-down) ---
# We bias toward rectifying: warping a near-flat rug just deskews it (harmless), whereas a
# missed correction is the visible failure. So we passthrough ONLY when the rug fills the frame
# or is almost pixel-perfectly axis-aligned AND balanced.
PASSTHRU_FILL = 0.92        # quad covers >= this fraction of the frame -> nothing to crop/warp
PASSTHRU_ANGLE_DEG = 2.0    # edges this close to axis-aligned -> already upright
PASSTHRU_EDGE_TOL = 0.04    # opposite edges within this relative length -> already rectangular

# whRatio is clamped to this range (a rug far outside it means bad corners)
WH_MIN, WH_MAX = 0.2, 5.0


# ---------------------------------------------------------------- point ordering
def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (float32).

    Uses a top-two / bottom-two split (by y, then x), which is robust to in-plane rotation up
    to ~45deg. The older x+y / y-x heuristic collapsed two corners to the same slot for a rug
    photographed at a diagonal (a 'diamond' quad) — the bug that made rotated rugs fall back to
    a min-area rectangle instead of their true trapezoid corners.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:                               # rare: order by angle around centroid
        cen = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0]))]
        return pts.astype(np.float32)
    ys = pts[np.argsort(pts[:, 1], kind="stable")]      # sort by y -> two topmost, two bottommost
    top, bot = ys[:2], ys[2:]
    tl, tr = top[np.argsort(top[:, 0], kind="stable")]  # within each pair, smaller x is left
    bl, br = bot[np.argsort(bot[:, 0], kind="stable")]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _zigzag_from_ordered(rect: np.ndarray) -> np.ndarray:
    """TL,TR,BR,BL (from _order_points) -> zig-zag m1=TL, m2=TR, m3=BL, m4=BR (Zhang-He order)."""
    tl, tr, br, bl = rect
    return np.float32([tl, tr, bl, br])


# ---------------------------------------------------------------- geometry helpers
def _poly_area(q: np.ndarray) -> float:
    """Shoelace area of an ordered quad."""
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _edge_lens(rect: np.ndarray):
    """[top, bottom, left, right] edge lengths for a TL,TR,BR,BL quad."""
    tl, tr, br, bl = rect
    return [float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)),
            float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr))]


def _vec_angle_deg(v) -> float:
    """Orientation of a vector in [0,180)."""
    return math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0


def _horiz_dev(v) -> float:
    a = _vec_angle_deg(v)
    return min(a, 180.0 - a)


def _vert_dev(v) -> float:
    return abs(90.0 - _vec_angle_deg(v))


def _angle_between(u, v) -> float:
    d = abs(_vec_angle_deg(u) - _vec_angle_deg(v))
    return min(d, 180.0 - d)


# ---------------------------------------------------------------- Phase 2: corners
def _validate(quad, mask_area: float, loose: bool = False) -> bool:
    """Quad sanity: 4 distinct, convex, area within tolerance of the mask area."""
    try:
        q = _order_points(np.asarray(quad, np.float32))
    except Exception:
        return False
    if q.shape[0] != 4 or not np.isfinite(q).all():
        return False
    diag = float(np.linalg.norm(q.max(0) - q.min(0)))
    if diag <= 0:
        return False
    min_sep = (0.01 if loose else 0.02) * diag
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(q[i] - q[j]) < min_sep:
                return False
    if not cv2.isContourConvex(q.astype(np.int32).reshape(-1, 1, 2)) and not loose:
        return False
    if mask_area <= 0:
        return False
    lo, hi = (0.30, 1.30) if loose else (0.50, 1.15)
    return lo <= (_poly_area(q) / mask_area) <= hi


def _intersect(l1, l2):
    """Intersection of two (rho, theta) Hough lines, or None if near-parallel."""
    r1, t1 = l1
    r2, t2 = l2
    A = np.array([[math.cos(t1), math.sin(t1)], [math.cos(t2), math.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x, y = np.linalg.solve(A, np.array([r1, r2]))
    return [float(x), float(y)]


def _hough_quad(mask_bin: np.ndarray):
    """Fit a quad via the 4 dominant boundary lines (robust to localized mask defects)."""
    edges = cv2.Canny((mask_bin * 255).astype(np.uint8), 50, 150)
    h, w = mask_bin.shape
    base = int(0.3 * max(h, w))
    lines = None
    for thr in (base, base // 2, max(20, base // 4)):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, max(10, thr))
        if lines is not None and len(lines) >= 4:
            break
    if lines is None or len(lines) < 4:
        return None
    L = lines[:, 0, :]  # (N, 2) rho, theta
    theta0 = float(L[0, 1])

    def adiff(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    near = np.array([adiff(t, theta0) < np.pi / 4 for t in L[:, 1]])
    g1, g2 = L[near], L[~near]
    if len(g1) < 2 or len(g2) < 2:
        return None

    def extremes(g):
        idx = np.argsort(g[:, 0])
        return g[idx[0]], g[idx[-1]]

    a1, b1 = extremes(g1)
    a2, b2 = extremes(g2)
    pts = []
    for la in (a1, b1):
        for lb in (a2, b2):
            p = _intersect(la, lb)
            if p is not None:
                pts.append(p)
    if len(pts) != 4:
        return None
    return np.array(pts, dtype=np.float32)


def mask_to_quad(mask):
    """Fit a 4-corner quad (TL,TR,BR,BL float32) to a boolean carpet mask, or None.

    On a clean SAM mask the contour is smooth enough that approxPolyDP is reliable (unlike
    on raw Canny edges of a textured rug). Method ladder, first valid wins:
      (a) epsilon-sweep approxPolyDP on the convex hull -> exactly 4 convex vertices;
      (b) Hough-line intersection (the report's preferred, defect-robust method);
      (c) cv2.minAreaRect enclosing quad (last resort).
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2 or int(m.sum()) == 0:
        print("[mask_to_quad] empty mask")
        return None
    H, W = m.shape
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[mask_to_quad] no contours")
        return None
    c = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(c)) < 0.02 * H * W:        # too small to be the rug
        print(f"[mask_to_quad] largest contour too small ({cv2.contourArea(c)/(H*W):.3f} of frame)")
        return None

    # We fit the quad to the CONVEX HULL, so validate against the HULL area (not the raw,
    # possibly-concave contour) — otherwise fringe/notches inflate hull/contour and the
    # area check rejects every candidate.
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    peri = cv2.arcLength(hull, True)

    # Primary: approxPolyN forces exactly 4 vertices (note the keyword args — its 3rd positional
    # is an OUTPUT array, not epsilon). Degenerate results are caught by _validate below.
    if hasattr(cv2, "approxPolyN"):
        try:
            pn = cv2.approxPolyN(hull, 4, epsilon_percentage=-1.0, ensure_convex=True)
            q = np.asarray(pn, np.float32).reshape(-1, 2)
            if q.shape[0] == 4 and _validate(q, hull_area):
                return _order_points(q)
        except (cv2.error, TypeError):
            pass

    for eps in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = approx.reshape(4, 2).astype(np.float32)
            if _validate(q, hull_area):
                return _order_points(q)

    q = _hough_quad(m)
    if q is not None and _validate(q, hull_area):
        return _order_points(q)

    # Last resort: the min-area enclosing rectangle of the largest blob. Always a sane 4-gon,
    # so we never silently fail to attempt rectification on a real rug mask.
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    print(f"[mask_to_quad] fell back to minAreaRect (cover={cv2.contourArea(c)/(H*W):.2f}, "
          f"hull/box={hull_area/max(1.0, _poly_area(_order_points(box))):.2f})")
    return _order_points(box)


# ---------------------------------------------------------------- Phase 3: aspect ratio
def _euclidean_wh_ratio(q: np.ndarray) -> float:
    """Mean width edges / mean height edges (q in zig-zag order m1=TL,m2=TR,m3=BL,m4=BR)."""
    m1, m2, m3, m4 = q
    width = (np.linalg.norm(m1 - m2) + np.linalg.norm(m3 - m4)) / 2.0
    height = (np.linalg.norm(m1 - m3) + np.linalg.norm(m2 - m4)) / 2.0
    if height <= 0:
        return 1.0
    return float(min(WH_MAX, max(WH_MIN, width / height)))


def _max_edge_wh_ratio(q: np.ndarray) -> float:
    """max width edge / max height edge — reproduces the legacy rectify() sizing heuristic."""
    m1, m2, m3, m4 = q
    width = max(np.linalg.norm(m1 - m2), np.linalg.norm(m3 - m4))
    height = max(np.linalg.norm(m1 - m3), np.linalg.norm(m2 - m4))
    if height <= 0:
        return 1.0
    return float(min(WH_MAX, max(WH_MIN, width / height)))


def recover_wh_ratio(quad_zigzag, W: int, H: int, k_eps: float = 0.02):
    """Recover true width/height from a perspective quad (Zhang & He closed form).

    quad_zigzag is in ZIG-ZAG order m1=TL, m2=TR, m3=BL, m4=BR. Principal point is the image
    centre; coords are shifted by (-W/2, -H/2). Returns whRatio clamped to [WH_MIN, WH_MAX].
    Degenerate handling: if EITHER edge-pair is ~parallel (|k-1| tiny) its vanishing point is at
    infinity and the focal length is unrecoverable -> fall back to the Euclidean side-length
    ratio; imaginary focal (f^2 <= 0) -> max-edge ratio (legacy behaviour).

    Note: sqrt(num/den) is ALREADY width/height (verified against synthetic pinhole projections
    in scripts/selftest_rectify) — there is no final reciprocal.
    """
    q = np.asarray(quad_zigzag, dtype=np.float64).reshape(4, 2) - np.array([W / 2.0, H / 2.0])
    (m1x, m1y), (m2x, m2y), (m3x, m3y), (m4x, m4y) = q

    k2_den = (m2y - m4y) * m3x - (m2x - m4x) * m3y + m2x * m4y - m2y * m4x
    k3_den = (m3y - m4y) * m2x - (m3x - m4x) * m2y + m3x * m4y - m3y * m4x
    if abs(k2_den) < 1e-9 or abs(k3_den) < 1e-9:
        return _euclidean_wh_ratio(q)
    k2 = ((m1y - m4y) * m3x - (m1x - m4x) * m3y + m1x * m4y - m1y * m4x) / k2_den
    k3 = ((m1y - m4y) * m2x - (m1x - m4x) * m2y + m1x * m4y - m1y * m4x) / k3_den

    if abs(k2 - 1) < k_eps or abs(k3 - 1) < k_eps:   # a vanishing point at infinity -> focal unrecoverable
        return _euclidean_wh_ratio(q)

    denom = (k3 - 1) * (k2 - 1)
    if abs(denom) < 1e-12:
        return _max_edge_wh_ratio(q)
    f2 = -((k3 * m3y - m1y) * (k2 * m2y - m1y) + (k3 * m3x - m1x) * (k2 * m2x - m1x)) / denom
    if not np.isfinite(f2) or f2 <= 0:                # imaginary focal -> bad corners
        return _max_edge_wh_ratio(q)

    num = (k2 - 1) ** 2 + (k2 * m2y - m1y) ** 2 / f2 + (k2 * m2x - m1x) ** 2 / f2
    den = (k3 - 1) ** 2 + (k3 * m3y - m1y) ** 2 / f2 + (k3 * m3x - m1x) ** 2 / f2
    if den <= 0:
        return _max_edge_wh_ratio(q)
    wh = math.sqrt(num / den)                          # = width / height directly (no reciprocal)
    if not np.isfinite(wh) or wh <= 0:
        return _max_edge_wh_ratio(q)
    return float(min(WH_MAX, max(WH_MIN, wh)))


# ---------------------------------------------------------------- Phases 4-5: warp
def _target_size(rect: np.ndarray, wh: float):
    """Destination (w,h) locked to wh, bounded by the source quad's max pixel span (Phase 5)."""
    top, bottom, left, right = _edge_lens(rect)
    maxspan = max(max(top, bottom), max(left, right))
    if wh >= 1.0:
        out_w = int(round(maxspan))
        out_h = int(round(out_w / wh))
    else:
        out_h = int(round(maxspan))
        out_w = int(round(out_h * wh))
    return max(1, out_w), max(1, out_h)


def _should_passthrough(rect: np.ndarray, W: int, H: int) -> bool:
    """True only when the rug is clearly already top-down, so rectification is skipped.

    We deliberately keep this STRICT (bias toward rectifying): passthrough only when the quad
    fills the frame, OR it is almost pixel-perfectly axis-aligned AND its opposite edges are
    near-equal. A flat-but-rotated rug is NOT skipped — warping it just deskews to upright,
    which is harmless. This avoids the failure mode where a mildly-angled photo is wrongly
    declared "already top-down" and left uncorrected.
    """
    if _poly_area(rect) >= PASSTHRU_FILL * W * H:
        return True

    tl, tr, br, bl = rect
    top, bottom, left, right = tr - tl, br - bl, bl - tl, br - tr
    tl_, bl_, ll_, rl_ = _edge_lens(rect)  # top,bottom,left,right lengths
    balanced = (abs(tl_ - bl_) / max(tl_, bl_, 1e-6) < PASSTHRU_EDGE_TOL
                and abs(ll_ - rl_) / max(ll_, rl_, 1e-6) < PASSTHRU_EDGE_TOL)
    upright = (_horiz_dev(top) < PASSTHRU_ANGLE_DEG and _horiz_dev(bottom) < PASSTHRU_ANGLE_DEG
               and _vert_dev(left) < PASSTHRU_ANGLE_DEG and _vert_dev(right) < PASSTHRU_ANGLE_DEG)
    return bool(balanced and upright)


def _dump_debug(img: Image.Image, quad=None, mask=None) -> None:
    """Prototype diagnostics: save the photo (+ mask/quad overlay) to debug/ for inspection.
    Off by default; set RECTIFY_DUMP=1 to re-enable when debugging a bad mask."""
    if os.environ.get("RECTIFY_DUMP", "0") != "1":
        return
    try:
        dbg = Path(__file__).resolve().parents[2] / "debug"
        dbg.mkdir(exist_ok=True)
        img.convert("RGB").save(dbg / "last_query.png")
        ov = np.asarray(img.convert("RGB")).copy()
        if mask is not None:
            mb = (np.asarray(mask) > 0).astype(np.uint8) * 255
            Image.fromarray(mb).save(dbg / "last_mask.png")
            ov[mb > 0] = (0.5 * ov[mb > 0] + np.array([0, 128, 0])).astype(np.uint8)
        if quad is not None:
            cv2.polylines(ov, [_order_points(quad).astype(np.int32)], True, (0, 255, 0), 2)
        Image.fromarray(ov).save(dbg / "last_overlay.png")
    except Exception as e:
        print(f"[rectify] dump skipped: {e}")


def rectify_from_quad(img: Image.Image, quad, settings=None, tag: str = "quad") -> Image.Image:
    """Warp a carpet to a fronto-parallel, correct-aspect top-down view from 4 corners (any order).

    Never raises: returns the input unchanged on a degenerate quad, near-top-down (safety
    passthrough), or any failure.
    """
    if quad is None:
        return img
    rgb = np.asarray(img.convert("RGB"))
    H, W = rgb.shape[:2]
    quad = _order_points(np.asarray(quad, np.float32))      # -> TL,TR,BR,BL
    if quad.shape[0] != 4 or not np.isfinite(quad).all():
        return img
    pt = _should_passthrough(quad, W, H)
    wh = None if pt else recover_wh_ratio(_zigzag_from_ordered(quad), W, H)
    print(f"[rectify:{tag}] quad={quad.astype(int).tolist()} passthrough={pt} wh={wh}")
    if pt or wh is None or not np.isfinite(wh) or wh <= 0:
        return img
    out_w, out_h = _target_size(quad, wh)
    if out_w < 10 or out_h < 10:
        return img
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    try:
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        work = rgb
        if max(out_w, out_h) < 0.8 * max(_edge_lens(quad)):   # downsampling -> anti-alias
            work = cv2.GaussianBlur(rgb, (3, 3), 0)
        warped = cv2.warpPerspective(work, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4)
    except cv2.error:
        return img
    return Image.fromarray(warped)


def rectify_from_mask(img: Image.Image, mask, settings=None) -> Image.Image:
    """Warp using a carpet MASK (local SAM3/GrabCut): mask -> quad -> rectify."""
    if mask is None:
        return img
    quad = mask_to_quad(mask)                        # TL,TR,BR,BL or None
    _dump_debug(img, quad=quad, mask=mask)
    if quad is None:
        print("[rectify] no 4-corner quad fit from mask -> passthrough")
        return img
    return rectify_from_quad(img, quad, settings, tag="mask")


def rectify_from_corners(img: Image.Image, corners, settings=None) -> Image.Image:
    """Warp using 4 corners supplied by an external service (e.g. the SAM3 corner API).

    `corners` is any 4 (x,y) in the image's own pixel frame; we re-order them ourselves.
    """
    if corners is None:
        return img
    c = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if c.shape[0] != 4 or not np.isfinite(c).all():
        return img
    _dump_debug(img, quad=c)
    return rectify_from_quad(img, c, settings, tag="api")
