#!/usr/bin/env python3
"""
STEP 5 - Tori v2: final layers, part positions, pivots and hierarchy (rig definition).

Input : art/v2/tori/step4-filled/<part>.png + optional/{armL,armR,collar-front}.png  (RGBA 1024x1024)
Output: art/v2/tori/step5-rig/
          layers/<part>.png    final full-canvas layers (1024x1024, same coords as the original)
          rig.json             rig definition (also copied to public/assets/characters/tori-v2/rig.json)
          pivots.png           rest pose at 2x, pivots (cross + label) and parent links
          range-test.png       every joint at min / mid / max of its limits + combined poses,
                               over #6BBF59 and #202020
          tuft-check.png       tuft before / after (flat, behind the head) + sway test
          rest-check.json      rest-pose numbers (also inside rig.json -> "checks")
        public/assets/characters/tori-v2/<part>.png   cropped layers (placed back at x,y)

Usage
  python3 scripts/v2/step5_rig.py                 build everything
  python3 scripts/v2/step5_rig.py render pose.json out.png [--bg 6BBF59] [--scale 1]
      pose.json = {"armR": -100, "head": {"rot": 6}, "torso": {"sy": 1.02, "anchor": "breath"},
                   "root": {"dx": 0, "dy": -10}}

Changes vs step 4 (user decisions)
  * arms: the OPTIONAL stack (arms with rounded shoulder ends + collar_front drawn over the arms).
  * tuft: flattened to one flat fur colour (the head's own flat tan, sampled on the head top away
    from the stripes); silhouette + anti-aliased outer alpha kept; the layer moves BEHIND the head
    and its base is extended downward (rounded) where the head is fully opaque, so the extension is
    hidden at rest and the tuft can sway without a gap.
  * head: the strip of the head top that was under the tuft (step-4 fill, now visible because the
    head is in front) is re-coloured so it continues the head pixels right below it (no seam line):
    per class (tan / stripe) normalised-convolution continuation of the visible colours, mixed by
    each pixel's own stripe coverage (its AA is kept).  Alpha is unchanged.

Transform model (identical in the compositor below and meant to be mirrored in step 6)
  All coordinates are 1024-canvas pixels, y down.  Angles in degrees, + = clockwise on screen
  (same sign as CSS rotate()).  For a part with parent chain root -> ... -> parent -> part:
      M(part) = M(parent) . T(d) . T(a) S(sx,sy) T(-a) . T(p) R(deg) T(-p)
  p = the part's pivot (rest canvas coords), a = scale anchor (default = pivot, or a named anchor),
  d = (dx,dy) translation.  A child therefore first rotates about its own pivot, then follows its
  parent's transform (it rotates with the parent about the parent's pivot).  root's p/a = rig.root.
  CSS (flat list, all parts in one 1024x1024 box, so z can interleave across the hierarchy):
      <img left=x top=y width=w height=h style="transform-origin: -x px -y px;
           transform: <root> <...> <parent> <part>"> with each level written as
      translate(dx,dy) translate(ax,ay) scale(sx,sy) translate(-ax,-ay) translate(px,py) rotate(deg) translate(-px,-py)
"""
import json, pathlib, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

ROOT = pathlib.Path(__file__).resolve().parents[2]
IN = ROOT / "art/v2/tori/step4-filled"
OUT = ROOT / "art/v2/tori/step5-rig"
PUB = ROOT / "public/assets/characters/tori-v2"
H = W = 1024
GREEN = (0x6B, 0xBF, 0x59)
DARK = (0x20, 0x20, 0x20)
PAD = 2                                   # transparent margin around each crop (browser edge AA)

ORDER = ["tail", "legL", "legR", "torso", "pendant", "armL", "armR", "collar_front",
         "earL", "earR", "tuft", "head"]
STEP4_ORDER = ["tail", "legL", "legR", "torso", "pendant", "armL", "armR", "earL", "earR", "head", "tuft"]
STEP4_OPT_ORDER = ["tail", "legL", "legR", "torso", "pendant", "armL", "armR", "collar_front",
                   "earL", "earR", "head", "tuft"]
PARENT = {"torso": "root", "legL": "root", "legR": "root",
          "tail": "torso", "pendant": "torso", "armL": "torso", "armR": "torso", "collar_front": "torso",
          "head": "torso", "earL": "head", "earR": "head", "tuft": "head"}

HEAD_TAN_SEED = np.array([236, 191, 131], float)
HEAD_STRIPE_SEED = np.array([115, 80, 58], float)


# ============================================================== helpers
def load_step4():
    L = {}
    for p in ["tail", "legL", "legR", "torso", "pendant", "earL", "earR", "head", "tuft"]:
        L[p] = np.array(Image.open(IN / f"{p}.png").convert("RGBA"))
    L4 = dict(L)
    L4["armL"] = np.array(Image.open(IN / "armL.png").convert("RGBA"))
    L4["armR"] = np.array(Image.open(IN / "armR.png").convert("RGBA"))
    opt = dict(L)
    opt["armL"] = np.array(Image.open(IN / "optional/armL.png").convert("RGBA"))
    opt["armR"] = np.array(Image.open(IN / "optional/armR.png").convert("RGBA"))
    opt["collar_front"] = np.array(Image.open(IN / "optional/collar-front.png").convert("RGBA"))
    return L4, opt


def bbox_of(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def premul(rgba):
    f = rgba.astype(np.float64)
    a = f[..., 3:4] / 255.0
    return np.concatenate([f[..., :3] * a, a], -1)


def compose_full(layers, order, bg):
    out = np.empty((H, W, 3)); out[:] = bg
    acc = np.zeros((H, W))
    for p in order:
        pm = premul(layers[p])
        out = pm[..., :3] + out * (1 - pm[..., 3:4])
        acc = pm[..., 3] + acc * (1 - pm[..., 3])
    return out, acc


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def norm_conv(vals, known, sigmas=(2.0, 4.0, 8.0, 16.0)):
    """fill every pixel from known ones: Gaussian normalised convolution, coarser where needed."""
    out = np.zeros(vals.shape)
    done = np.zeros(vals.shape[:2], bool)
    k = known.astype(float)
    for s in sigmas:
        wsum = ndi.gaussian_filter(k, s)
        ok = (wsum > 1e-3) & ~done
        for c in range(vals.shape[2]):
            num = ndi.gaussian_filter(vals[..., c] * k, s)
            out[..., c][ok] = num[ok] / wsum[ok]
        done |= ok
    return out


def fit_ellipse(x, y):
    """Fitzgibbon direct least-squares ellipse (as in step4_fill.py) -> F(X,Y) <= 0 inside."""
    mx, my, s = x.mean(), y.mean(), max(x.std(), y.std())
    xn, yn = (x - mx) / s, (y - my) / s
    D = np.c_[xn ** 2, xn * yn, yn ** 2, xn, yn, np.ones_like(xn)]
    C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    ev, evec = np.linalg.eig(np.linalg.solve(D.T @ D, C))
    ev, evec = np.real(ev), np.real(evec)
    a = evec[:, np.argmax(ev > 0) if (ev > 0).any() else np.argmax(ev)]

    def F(X, Y):
        Xn, Yn = (X - mx) / s, (Y - my) / s
        return a[0] * Xn ** 2 + a[1] * Xn * Yn + a[2] * Yn ** 2 + a[3] * Xn + a[4] * Yn + a[5]
    sign = -1.0 if F(np.array([mx]), np.array([my]))[0] > 0 else 1.0
    return lambda X, Y: sign * F(X, Y)


# ============================================================== tuft + head top
def make_tuft_and_head(head, tuft):
    """returns new head, new tuft, info dict, masks."""
    ha, ta = head[..., 3], tuft[..., 3]
    info = {}
    # --- flat colour = head's flat tan, sampled on the head top (y 170-260) away from the stripes
    reg = head[170:260, 300:720].reshape(-1, 4)
    reg = reg[reg[:, 3] == 255][:, :3].astype(float)
    sel = reg[np.abs(reg - HEAD_TAN_SEED).max(1) < 12]
    flat = np.round(np.median(sel, 0)).astype(np.uint8)
    info["flat_colour"] = [int(v) for v in flat]
    info["flat_colour_sample"] = "median of %d opaque head px in x300-720,y170-260 within 12 of tan (stripes excluded)" % len(sel)

    # --- zone Z: head pixels that were under the tuft (now visible because head is in front)
    Z = (ha > 0) & (ta > 0)
    info["head_px_newly_visible"] = int(Z.sum())

    # --- head re-colour in Z: continuation of the visible head pixels below, per class
    rgb = head[..., :3].astype(float)
    d = HEAD_STRIPE_SEED - HEAD_TAN_SEED
    f = np.clip(((rgb - HEAD_TAN_SEED) @ d) / (d @ d), 0, 1)          # stripe coverage 0..1
    band = ndi.binary_dilation(Z, iterations=14) & ~Z & (ha == 255)
    # class by the nearer flat colour (shadowed tan stays tan); skip the 1-px AA rim between classes
    is_str = f > 0.5
    rim = ndi.binary_dilation(is_str) & ndi.binary_dilation(~is_str)
    vis_tan = band & ~is_str & ~rim
    vis_str = band & is_str & ~rim
    T = norm_conv(rgb, vis_tan, (1.5, 3.0, 6.0, 12.0))
    Sf = norm_conv(rgb, vis_str, (1.5, 3.0, 6.0, 12.0))
    newrgb = (1 - f[..., None]) * T + f[..., None] * Sf
    head2 = head.copy()
    head2[Z, :3] = np.clip(np.round(newrgb[Z]), 0, 255).astype(np.uint8)

    # --- tuft flat (alpha kept)
    tuft2 = tuft.copy()
    tuft2[ta > 0, :3] = flat
    tuft2[ta == 0, :3] = 0

    # --- base extension (hidden under the head): convex hull of the two side lobes' completed ellipses
    #     (fitted to their visible outer silhouette) and a rounded U below the contact span.  The hull's
    #     sides are tangent to the lobes, so what rises above the head outline when the tuft sways is a
    #     smooth continuation of the lobe contour (no straight vertical edge); 4x4 supersampled.
    zx = np.nonzero(Z.any(0))[0]
    x0c, x1c = float(zx.min()), float(zx.max())                       # contact span
    cx = (x0c + x1c) / 2
    ys_, xs_ = np.nonzero((ta >= 40) & (ta <= 215) & (ha == 0))
    xs_, ys_ = xs_.astype(float), ys_.astype(float)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    pts = []
    lobes = {}
    for name, m in (("left", xs_ < cx - 50), ("right", xs_ > cx + 40)):
        F = fit_ellipse(xs_[m], ys_[m])
        ins = F(xx, yy) <= 0
        lobes[name] = bbox_of(ins)
        py, px = np.nonzero(ins)
        pts.append(np.c_[px, py])
    uw, y_rect, ub = 44.0, 192.0, 24.0                                   # U: half-width, straight to, round
    U = (np.abs(xx - cx) <= uw) & (yy >= 150) & ((yy <= y_rect) | (((xx - cx) / uw) ** 2 + ((yy - y_rect) / ub) ** 2 <= 1))
    py, px = np.nonzero(U)
    pts.append(np.c_[px, py])
    hull = ConvexHull(np.concatenate(pts).astype(float))
    eq = hull.equations                                                   # a x + b y + c <= 0 inside
    SS = 4
    y0b, y1b, x0b, x1b = 60, 240, int(x0c) - 40, int(x1c) + 40
    cov = np.zeros((H, W))
    gy, gx = np.mgrid[y0b:y1b, x0b:x1b].astype(float)
    for sy in range(SS):
        for sx in range(SS):
            X = gx + (sx + 0.5) / SS - 0.5
            Y = gy + (sy + 0.5) / SS - 0.5
            inside = np.all(eq[:, 0, None, None] * X + eq[:, 1, None, None] * Y + eq[:, 2, None, None] <= 0, 0) \
                & (X >= x0c - 1.5) & (X <= x1c + 1.5)       # sides start exactly at the contact ends (no tooth)
            cov[y0b:y1b, x0b:x1b] += inside
    cov /= SS * SS
    allowed = (ha == 255) | (Z & (ha > 0))
    ext_a = np.where(allowed, np.round(cov * 255), 0).astype(np.uint8)
    new_a = np.maximum(ta, ext_a)
    added = (new_a > ta)
    tuft2[added, :3] = flat
    tuft2[..., 3] = new_a
    info["base_extension"] = {
        "shape": ("convex hull of the completed left/right lobe ellipses (bbox %s / %s) and a U (x %.0f..%.0f, "
                  "straight to y=%.0f, round bottom to y=%.0f), clipped to the contact span x %.1f..%.1f, kept only "
                  "where the head is opaque (hidden)")
                 % (lobes["left"], lobes["right"], cx - uw, cx + uw, y_rect, y_rect + ub, x0c - 1.5, x1c + 1.5),
        "px_alpha_raised": int(added.sum()),
        "new_px_outside_step4_tuft": int((added & (ta == 0)).sum()),
        "new_px_outside_step4_tuft_not_under_opaque_head": int((added & (ta == 0) & (ha < 255)).sum()),
        "bbox": bbox_of(new_a > 0),
    }
    info["contact_span_x"] = [x0c, x1c]
    return head2, tuft2, info, Z, (ta > 0)


def unmix_arm_collar(final):
    """The step-4 shoulder caps were added ABOVE each arm's old top edge, whose anti-aliasing (alpha<255,
    torso collar showing through) is now INSIDE the arm (a dotted line that shows whatever is behind as
    soon as the arm moves).  Collar_front is drawn over the arms, so that AA belongs to collar_front:
      arm px -> opaque, colour = clean local arm colour (median of opaque arm px in a 5x5 window);
      collar_front px -> alpha b = 1 - arm alpha, colour solved so b*c + (1-b)*arm = rest colour
      (c comes out ~ the torso collar colour there).
    Only where the torso is opaque (rest colour is background-independent)."""
    info = {}
    below = ["tail", "legL", "legR", "torso", "pendant"]
    cf = final["collar_front"].copy()
    for arm in ("armL", "armR"):
        A = final[arm].copy()
        a = A[..., 3]
        full = a == 255
        closed = ndi.binary_closing(full, structure=np.ones((3, 3)), iterations=1)
        P = closed & (a > 0) & (a < 255) & (final["torso"][..., 3] == 255) & (cf[..., 3] == 0)
        lay = below + (["armL"] if arm == "armR" else [])
        Bc, _ = compose_full({k: final[k] for k in lay}, lay, np.zeros(3))
        ys, xs = np.nonzero(P)
        res = []
        for y, x in zip(ys, xs):
            al = a[y, x] / 255.0
            C = al * A[y, x, :3].astype(float) + (1 - al) * Bc[y, x]
            win = A[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            ok = win[..., 3] == 255
            ac = np.median(win[ok][:, :3].astype(float), 0) if ok.sum() >= 3 else A[y, x, :3].astype(float)
            b8 = 255 - int(a[y, x])                      # collar_front takes exactly the arm's missing coverage
            b = b8 / 255.0
            c = np.clip(np.round((C - (1 - b) * ac) / b), 0, 255)
            A[y, x, :3] = np.round(ac).astype(np.uint8); A[y, x, 3] = 255
            cf[y, x, :3] = c.astype(np.uint8); cf[y, x, 3] = b8
            res.append(float(np.abs(b * c + (1 - b) * np.round(ac) - C).max()))
        # pass 2: opaque arm px along the old top edge still carrying the dark collar AA in their own
        # colour (a dark dotted line that would travel with the arm) -> clean arm colour, the darkness
        # moved into collar_front (colour of the nearest opaque collar_front px, alpha by least squares)
        near = ndi.binary_dilation(cf[..., 3] > 0, iterations=2) & (cf[..., 3] < 255) & (A[..., 3] == 255) \
            & (final["torso"][..., 3] == 255)
        _, (iy, ix) = ndi.distance_transform_edt(cf[..., 3] < 255, return_indices=True)
        n2, res2 = 0, []
        for y, x in zip(*np.nonzero(near)):
            win = A[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            ok = win[..., 3] == 255
            ac = np.median(win[ok][:, :3].astype(float), 0)
            al0 = cf[y, x, 3] / 255.0
            C = al0 * cf[y, x, :3].astype(float) + (1 - al0) * A[y, x, :3].astype(float)   # rest colour
            if np.abs(A[y, x, :3].astype(float) - ac).max() <= 6:
                continue
            c = cf[iy[y, x], ix[y, x], :3].astype(float)
            d = c - ac
            b = float(np.clip(np.dot(C - ac, d) / max(np.dot(d, d), 1e-6), 0, 1))
            b8 = int(round(b * 255)); b = b8 / 255.0
            cc = np.clip(np.round((C - (1 - b) * ac) / b), 0, 255) if b8 > 0 else c
            r = float(np.abs(b * cc + (1 - b) * np.round(ac) - C).max())
            if r > 2.0:
                continue
            A[y, x, :3] = np.round(ac).astype(np.uint8)
            cf[y, x, :3] = cc.astype(np.uint8); cf[y, x, 3] = b8
            n2 += 1; res2.append(r)
        final[arm] = A
        info[arm] = {"interior_alpha_px": int(P.sum()), "max_rest_residual": round(max(res), 2) if res else 0,
                     "dark_edge_px": n2, "dark_edge_max_rest_residual": round(max(res2), 2) if res2 else 0}
    final["collar_front"] = cf
    return info


def unmix_torso_head(final):
    """Along the head's bottom edge the step-3 cut left the head's light rim AA inside TORSO pixels
    (head alpha jumps 255 -> 0).  Those light / dark-blended torso px sit on the collar and show as a
    white dash + dotted line as soon as the head tilts.  Un-mix them into the head (drawn in front):
      torso px -> clean collar colour (median of the torso px 3-7 rows below, same column +-1),
      head px  -> soft edge: colour solved, alpha b by least squares toward the nearest opaque head
                  colour, so b*head + (1-b)*torso_clean = rest colour (residual reported)."""
    T, Hd = final["torso"].copy(), final["head"].copy()
    front = np.zeros((H, W), bool)
    for k in ("pendant", "armL", "armR", "collar_front", "earL", "earR", "tuft", "head"):
        front |= final[k][..., 3] > 0
    near = ndi.binary_dilation(Hd[..., 3] == 255, iterations=2) & (T[..., 3] == 255) & ~front
    far = ~ndi.binary_dilation(Hd[..., 3] > 0, iterations=2) & (T[..., 3] == 255)
    _, (iy, ix) = ndi.distance_transform_edt(Hd[..., 3] < 255, return_indices=True)
    n, res = 0, []
    for y, x in zip(*np.nonzero(near)):
        win = T[y + 3:y + 8, x - 1:x + 2]
        ok = far[y + 3:y + 8, x - 1:x + 2]
        if ok.sum() < 4:
            continue
        tc = np.median(win[ok][:, :3].astype(float), 0)
        C = T[y, x, :3].astype(float)
        if np.abs(C - tc).max() <= 5:
            continue
        hc = Hd[iy[y, x], ix[y, x], :3].astype(float)
        d = hc - tc
        b = float(np.clip(np.dot(C - tc, d) / max(np.dot(d, d), 1e-6), 0, 1))
        b8 = int(round(b * 255))
        if b8 == 0:
            cc, b = hc, 0.0
        else:
            b = b8 / 255.0
            cc = np.clip(np.round((C - (1 - b) * np.round(tc)) / b), 0, 255)
        r = float(np.abs(b * cc + (1 - b) * np.round(tc) - C).max())
        if r > 3.0:
            continue
        T[y, x, :3] = np.round(tc).astype(np.uint8)
        if b8 > 0:
            Hd[y, x, :3] = cc.astype(np.uint8); Hd[y, x, 3] = b8
        n += 1; res.append(r)
    final["torso"], final["head"] = T, Hd
    return {"px": n, "max_rest_residual": round(max(res), 2) if res else 0}


# ============================================================== rig math / compositor
def mat_T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def mat_R(deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)      # y down: + = clockwise on screen


def mat_S(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)


def norm_pose(v):
    if v is None:
        return {}
    if isinstance(v, (int, float)):
        return {"rot": float(v)}
    return v


def local_matrix(rig, name, pv):
    pv = norm_pose(pv)
    if name == "root":
        p = rig["root"]["origin"]
    else:
        p = rig["parts"][name]["pivot"]
    anc = pv.get("anchor")
    a = rig["anchors"][anc]["point"] if isinstance(anc, str) else (anc or p)
    M = mat_T(pv.get("dx", 0), pv.get("dy", 0))
    sx = pv.get("sx", pv.get("scale", 1)); sy = pv.get("sy", pv.get("scale", 1))
    M = M @ mat_T(*a) @ mat_S(sx, sy) @ mat_T(-a[0], -a[1])
    M = M @ mat_T(*p) @ mat_R(pv.get("rot", 0)) @ mat_T(-p[0], -p[1])
    return M


def world_matrices(rig, pose):
    Ms = {"root": local_matrix(rig, "root", pose.get("root"))}

    def get(n):
        if n not in Ms:
            par = rig["parts"][n]["parent"]
            Ms[n] = get(par) @ local_matrix(rig, n, pose.get(n))
        return Ms[n]
    for n in rig["parts"]:
        get(n)
    return Ms


class Rig:
    def __init__(self, rig_path=PUB / "rig.json", asset_dir=None):
        self.rig = json.loads(pathlib.Path(rig_path).read_text())
        d = pathlib.Path(asset_dir or pathlib.Path(rig_path).parent)
        self.pm = {}
        for n, p in self.rig["parts"].items():
            im = np.array(Image.open(d / p["file"]).convert("RGBA"))
            assert im.shape[1] == p["w"] and im.shape[0] == p["h"], n
            self.pm[n] = premul(im)
        self.order = sorted(self.rig["parts"], key=lambda n: self.rig["parts"][n]["z"])

    def render(self, pose=None, bg=(255, 255, 255), box=(0, 0, W, H), scale=1.0, only=None):
        """composite (float RGB 0..255, alpha) of the posed rig inside canvas box, at `scale`."""
        pose = pose or {}
        Ms = world_matrices(self.rig, pose)
        x0, y0, x1, y1 = box
        ow, oh = int(round((x1 - x0) * scale)), int(round((y1 - y0) * scale))
        out = np.empty((oh, ow, 3)); out[:] = bg
        acc = np.zeros((oh, ow))
        for n in self.order:
            if only is not None and n not in only:
                continue
            p = self.rig["parts"][n]
            pm = self.pm[n]
            M = Ms[n]
            # output pixel (i,j) -> canvas point
            corners = np.array([[p["x"], p["y"], 1], [p["x"] + p["w"], p["y"], 1],
                                [p["x"], p["y"] + p["h"], 1], [p["x"] + p["w"], p["y"] + p["h"], 1]], float).T
            wc = M @ corners
            bx0 = max(0, int(np.floor((wc[0].min() - x0) * scale)) - 2)
            by0 = max(0, int(np.floor((wc[1].min() - y0) * scale)) - 2)
            bx1 = min(ow, int(np.ceil((wc[0].max() - x0) * scale)) + 2)
            by1 = min(oh, int(np.ceil((wc[1].max() - y0) * scale)) + 2)
            if bx1 <= bx0 or by1 <= by0:
                continue
            jj, ii = np.mgrid[by0:by1, bx0:bx1].astype(np.float64)
            # pixel centre in canvas coords (pixel k covers [k, k+1); centre k+0.5)
            cxs = (ii + 0.5) / scale + x0 - 0.5
            cys = (jj + 0.5) / scale + y0 - 0.5
            Mi = np.linalg.inv(M)
            sx = Mi[0, 0] * cxs + Mi[0, 1] * cys + Mi[0, 2] - p["x"]
            sy = Mi[1, 0] * cxs + Mi[1, 1] * cys + Mi[1, 2] - p["y"]
            lay = np.stack([ndi.map_coordinates(pm[..., k], [sy, sx], order=1, mode="constant", cval=0.0)
                            for k in range(4)], -1)
            sl = (slice(by0, by1), slice(bx0, bx1))
            out[sl] = lay[..., :3] + out[sl] * (1 - lay[..., 3:4])
            acc[sl] = lay[..., 3] + acc[sl] * (1 - lay[..., 3])
        return out, acc


def to_img(rgb):
    return Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8))


# ============================================================== gap metrics
def crack_mask(acc, r=4):
    """thin background cracks: transparent px (alpha<0.5) that a closing (disk r) of the opaque
    mask fills."""
    m = acc >= 0.5
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r * r
    closed = ndi.binary_closing(np.pad(m, r + 1), structure=disk)[r + 1:-r - 1, r + 1:-r - 1]
    return closed & ~m


def enclosed_holes(acc):
    t = acc < 0.9
    lab, n = ndi.label(t)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return t & ~np.isin(lab, list(border))


def new_holes(a0, a1, mats=()):
    """new enclosed transparent components in pose a1 vs rest a0.  A component that is mostly
    (>50%) background at rest too - either in place or carried back through the world matrix of one
    of the moved parts (`mats`) - is a pre-existing background pocket that the pose merely moved /
    closed off (e.g. the triangle between head, collar and tail when the torso leans) -> 'pocket'."""
    h0, h1 = enclosed_holes(a0), enclosed_holes(a1)
    m = h1 & ~h0
    lab, n = ndi.label(m)
    hole = np.zeros_like(m); pocket = 0
    for i in range(1, n + 1):
        c = lab == i
        ys, xs = np.nonzero(c)
        fr = [(a0[c] < 0.5).mean()]
        for M in mats:
            Mi = np.linalg.inv(M)
            sx = np.clip(np.round(Mi[0, 0] * xs + Mi[0, 1] * ys + Mi[0, 2]).astype(int), 0, W - 1)
            sy = np.clip(np.round(Mi[1, 0] * xs + Mi[1, 1] * ys + Mi[1, 2]).astype(int), 0, H - 1)
            fr.append((a0[sy, sx] < 0.5).mean())
        if max(fr) > 0.5:
            pocket += int(c.sum())
        else:
            hole |= c
    return hole, pocket


def joint_gap(R, pose, joint, radius):
    """new thin cracks (inside a disk around the posed joint pivot) + new enclosed holes vs rest."""
    _, a0 = R.render({})
    _, a1 = R.render(pose)
    Ms = world_matrices(R.rig, pose)
    pv = R.rig["parts"][joint]["pivot"]
    pw = Ms[R.rig["parts"][joint]["parent"]] @ np.array([pv[0], pv[1], 1.0])
    yy, xx = np.mgrid[0:H, 0:W]
    disk = (xx - pw[0]) ** 2 + (yy - pw[1]) ** 2 <= radius ** 2
    c0, c1 = crack_mask(a0) & disk, crack_mask(a1) & disk
    hole, pocket = new_holes(a0, a1, [Ms[joint]])
    newc = c1 & ~ndi.binary_dilation(c0, iterations=2)
    return {"new_crack_px": int(newc.sum()), "new_hole_px": int(hole.sum()),
            "closed_off_background_pocket_px": pocket}, newc | hole


# ============================================================== rig definition (pivots, limits)
# pivots: canvas px.  Chosen in step 5 (see PIVOT_NOTES; tests in range-test.png / rig.json checks)
PIVOTS = {
    "torso": [512.0, 884.0],
    "legL": [409.0, 884.0],
    "legR": [615.5, 884.0],
    "tail": [705.0, 825.0],
    "pendant": [513.0, 662.0],
    "armL": [362.6, 643.4],
    "armR": [660.8, 643.8],
    "collar_front": [512.0, 884.0],
    "head": [512.0, 620.0],
    "earL": [356.2, 206.0],
    "earR": [666.7, 205.0],
    "tuft": [511.0, 168.0],
}
# + = clockwise on screen.  L/R = screen left / screen right.
LIMITS = {
    "torso": [-5, 5],
    "legL": [-20, 20],
    "legR": [-20, 20],
    "tail": [-8, 8],
    "pendant": [-10, 10],
    "armL": [-15, 75],
    "armR": [-75, 15],
    "collar_front": [0, 0],
    "head": [-8, 8],
    "earL": [-15, 15],
    "earR": [-15, 15],
    "tuft": [-8, 8],
}
PIVOT_NOTES = {
    "torso": "hip centre between the two hip caps (lean / sway); legs are root children so the feet stay planted",
    "legL": "centre of the hip-cap half-ellipse filled under the torso in step 4 (hip joint)",
    "legR": "centre of the hip-cap half-ellipse filled under the torso in step 4 (hip joint)",
    "tail": "tail root behind the torso (step-4 root fill), so the curl swings while the root stays hidden",
    "pendant": "top of the acorn stem where it hangs from the collar",
    "armL": "centre of the rounded shoulder end (between the cap-circle centre and the old top-edge midpoint), "
            "hidden under collar_front: the arm turns like a ball joint",
    "armR": "centre of the rounded shoulder end (mirror of armL), hidden under collar_front",
    "collar_front": "rigid part of the torso (same pivot, never rotated on its own; limits 0)",
    "head": "neck: under the head's bottom edge, inside the step-4 neck/collar fill",
    "earL": "ear base where the ear enters the head outline (step-4 ear-bottom fill is behind the head)",
    "earR": "ear base where the ear enters the head outline",
    "tuft": "tuft root ~14 px below the head outline at the centre of the contact span (base hidden behind the head)",
}
LIMIT_NOTES = {
    "armL": "+ = raise outward. ~+75 is the useful maximum: beyond ~80 the arm goes behind the head's cheek. "
            "Inward (-) kept to 15: further, the shoulder bulge leaves the collar end and a background notch opens.",
    "armR": "- = raise outward (wave: -60..-75). Inward (+) kept to 15 (background notch under the collar end beyond). "
            "Raising past ~-45 reveals the tail stem behind the arm (step-4 fill) and a small armpit opening "
            "between body and tail (~100 px at -75) - background seen between body and tail, not a seam.",
    "tuft": "+-5 fully clean; +-8 shows only a 1-2 px straight sliver of the hidden base at the lifting side; "
            "at +-10 that sliver is clearly visible (2-4 px) -> limit 8, idle sway 4-6 recommended.",
    "head": "+-8 clean (step-4 neck/collar fill shows a few px of straight collar end above the rounded end); "
            "at +-10 small corner gaps between head edge and collar fill.",
    "tail": "+-8: beyond, the root area between arm and torso shows the step-4 stem fill shapes.",
    "earL": "+-15 (+-20 works but the inner-ear fill under the head shows a small notch).",
    "earR": "+-15 (+-20 works but the inner-ear fill under the head shows a small notch).",
    "legL": "+-20 clean (hip caps stay under the torso).",
    "legR": "+-20 clean (hip caps stay under the torso).",
    "torso": "+-5 lean about the hips with the legs planted; children follow.",
    "pendant": "+-10 clean (the hole behind it was filled in step 4).",
}


def feet_line(layers):
    a = np.maximum(layers["legL"][..., 3], layers["legR"][..., 3])
    return int(np.nonzero(a.any(1))[0].max()) + 1          # first row below the feet


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layers").mkdir(exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    L4, opt = load_step4()

    # ---------------- final layers
    head2, tuft2, tinfo, Z, tuft_area = make_tuft_and_head(opt["head"], opt["tuft"])
    final = dict(opt)
    final["head"], final["tuft"] = head2, tuft2
    arm_fix = unmix_arm_collar(final)
    arm_fix["torso_head_unmix"] = unmix_torso_head(final)

    for p in ORDER:
        Image.fromarray(final[p], "RGBA").save(OUT / "layers" / f"{p}.png")

    # ---------------- crops
    parts = {}
    for z, p in enumerate(ORDER):
        bb = bbox_of(final[p][..., 3] > 0)
        x0, y0 = max(0, bb[0] - PAD), max(0, bb[1] - PAD)
        x1, y1 = min(W, bb[2] + PAD), min(H, bb[3] + PAD)
        fn = f"{p}.png"
        Image.fromarray(final[p][y0:y1, x0:x1], "RGBA").save(PUB / fn, optimize=True)
        parts[p] = {"file": fn, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                    "pivot": [round(float(v), 1) for v in PIVOTS[p]], "parent": PARENT[p], "z": z,
                    "limits": {"minDeg": LIMITS[p][0], "maxDeg": LIMITS[p][1]}}
        if p in PIVOT_NOTES:
            parts[p]["pivotNote"] = PIVOT_NOTES[p]
        if p in LIMIT_NOTES:
            parts[p]["limitsNote"] = LIMIT_NOTES[p]

    fy = feet_line(final)
    allm = np.zeros((H, W), bool)
    for p in ORDER:
        allm |= final[p][..., 3] > 0
    bb = bbox_of(allm)
    top = bbox_of(np.any([final[p][..., 3] >= 128 for p in ORDER], 0))[1]
    rig = {
        "name": "tori-v2",
        "version": 1,
        "canvas": [W, H],
        "units": "px on the 1024x1024 source canvas (y down)",
        "angleConvention": "degrees, + = clockwise on screen (same sign as CSS rotate()); L/R = screen left/right",
        "transformModel": ("M(part) = M(parent) . T(dx,dy) . T(a) S(sx,sy) T(-a) . T(pivot) R(deg) T(-pivot); "
                           "a = scale anchor (default pivot, or an entry of 'anchors'); a child first rotates "
                           "about its own pivot and then follows its parent. Draw parts as one flat list sorted "
                           "by z (z interleaves across the hierarchy, e.g. tail behind legs). CSS per part: "
                           "left=x, top=y, width=w, height=h, transform-origin: -x px -y px, transform = "
                           "<root> <...ancestors> <part>, each level = translate(dx,dy) translate(ax,ay) "
                           "scale(sx,sy) translate(-ax,-ay) translate(px,py) rotate(deg) translate(-px,-py). "
                           "Reference implementation: scripts/v2/step5_rig.py (world_matrices / Rig.render)."),
        "root": {"origin": [float(PIVOTS["torso"][0]), float(fy)],
                 "originNote": "bottom-centre between the feet (x = torso/hip centre, y = first row below the soles); "
                               "place this point on the scene ground line",
                 "height": int(fy - top), "heightNote": "feet line to the top of the tuft (alpha>=128)",
                 "bbox": bb},
        "drawOrder": ORDER,
        "parts": parts,
        "anchors": {
            "breath": {"point": [float(PIVOTS["torso"][0]), float(fy)], "target": "torso",
                       "use": "torso scaleY (breathing / squash) anchored at the feet line; children (head, arms, "
                              "tail, pendant, collar_front) follow; legs are root children so they stay planted",
                       "range": {"sy": [0.97, 1.03], "sx": [0.99, 1.02]}},
            "lean": {"point": list(PIVOTS["torso"]), "target": "torso",
                     "use": "body lean / sway = torso rotation about the hip centre (feet stay planted)"},
            "headBob": {"point": list(PIVOTS["head"]), "target": "head",
                        "use": "head squash (scaleY ~0.97-1.03) anchored at the neck"},
        },
    }
    rig["_tuft"] = tinfo
    rig["_armFix"] = arm_fix
    (OUT / "rig.json").write_text(json.dumps(rig, indent=1, ensure_ascii=False))
    shutil.copy(OUT / "rig.json", PUB / "rig.json")
    return L4, opt, final, Z, tuft_area, tinfo, arm_fix


# ============================================================== verification sheets
BOXES = {
    "torso": (190, 40, 960, 1000), "legL": (300, 780, 730, 1000), "legR": (300, 780, 730, 1000),
    "tail": (560, 520, 960, 920), "pendant": (420, 610, 610, 800), "armL": (180, 520, 480, 860),
    "armR": (550, 520, 850, 860), "head": (150, 30, 880, 780), "earL": (250, 90, 470, 300),
    "earR": (560, 90, 780, 300), "tuft": (390, 40, 640, 240),
}
FULL = (150, 30, 980, 1000)


def combined_poses(rig):
    lim = {n: rig["parts"][n]["limits"] for n in rig["parts"]}
    return [
        ("wave: armR max raise (%d), armL %d, head +5, earR -8, tuft +6" % (lim["armR"]["minDeg"], 10),
         {"armR": lim["armR"]["minDeg"], "armL": 10, "head": 5, "earR": -8, "tuft": 6}),
        ("wave: armL max raise (%d), head -5" % lim["armL"]["maxDeg"],
         {"armL": lim["armL"]["maxDeg"], "head": -5, "earL": 8}),
        ("head tilt min (%d) + ears (earL +15, earR +15) + tuft max" % lim["head"]["minDeg"],
         {"head": lim["head"]["minDeg"], "earL": 15, "earR": 15, "tuft": lim["tuft"]["maxDeg"]}),
        ("head tilt max (%d) + ears (earL -15, earR -15) + tuft min" % lim["head"]["maxDeg"],
         {"head": lim["head"]["maxDeg"], "earL": -15, "earR": -15, "tuft": lim["tuft"]["minDeg"]}),
        ("walk A: legL +15 legR -15, armL -15 armR -15, torso +2, tail +8",
         {"legL": 15, "legR": -15, "armL": -15, "armR": -15, "torso": 2, "tail": 8, "pendant": -5}),
        ("walk B: legL -15 legR +15, armL +15 armR +15, torso -2, tail -8",
         {"legL": -15, "legR": 15, "armL": 15, "armR": 15, "torso": -2, "tail": -8, "pendant": 5}),
        ("hop squash: torso sy 0.97 sx 1.02 (breath anchor), legs +-8 out, arms out 30",
         {"torso": {"sy": 0.97, "sx": 1.02, "anchor": "breath"}, "legL": 8, "legR": -8, "armL": 30, "armR": -30}),
        ("breath in: torso sy 1.03 (anchor breath = feet line), head -3, tuft +5",
         {"torso": {"sy": 1.03, "sx": 1.0, "anchor": "breath"}, "head": -3, "tuft": 5}),
        ("lean: torso -5, head +6, legs 0 (planted), arms +-20",
         {"torso": -5, "head": 6, "armL": 20, "armR": -20, "tail": 10}),
    ]


def thumb(R, pose, box, tw, bg):
    o, a = R.render(pose, bg, box, scale=tw / (box[2] - box[0]))
    return to_img(o), a


def range_sheet(R, path):
    rig = R.rig
    TW = 250
    rows, stats = [], {}
    joints = [n for n in ORDER if rig["parts"][n]["limits"]["minDeg"] != rig["parts"][n]["limits"]["maxDeg"]]
    for j in joints:
        lo, hi = rig["parts"][j]["limits"]["minDeg"], rig["parts"][j]["limits"]["maxDeg"]
        mid = round((lo + hi) / 2, 1)
        ims, st = [], {}
        for deg in (lo, mid, hi):
            pose = {j: deg}
            g, _ = thumb(R, pose, BOXES[j], TW, GREEN)
            d, _ = thumb(R, pose, BOXES[j], TW, DARK)
            ims += [g, d]
            if deg != 0:
                s, _ = joint_gap(R, pose, j, 90 if j not in ("head", "torso") else 320)
                st[str(deg)] = s
        stats[j] = st
        txt = "  ".join("%s: crack+%d hole+%d" % (k, v["new_crack_px"], v["new_hole_px"]) for k, v in st.items())
        rows.append((f"{j}  pivot {rig['parts'][j]['pivot']}  parent {rig['parts'][j]['parent']}  "
                     f"min {lo} / mid {mid} / max {hi}    [{txt}]", ims))
    combo_stats = []
    for name, pose in combined_poses(rig):
        g, _ = thumb(R, pose, FULL, 2 * TW + 5, GREEN)
        d, _ = thumb(R, pose, FULL, 2 * TW + 5, DARK)
        _, a = R.render(pose)
        _, a0 = R.render({})
        Ms = world_matrices(rig, pose)
        hole, pocket = new_holes(a0, a, [Ms[k] for k in pose if k != "root"])
        nh = int(hole.sum())
        combo_stats.append({"pose": name, "new_hole_px": nh, "closed_off_background_pocket_px": pocket})
        rows.append((f"{name}   [new holes {nh} px]", [g, d]))
    W6 = 6 * (TW + 5) + 10
    hgt = 60 + sum(max(im.height for im in r[1]) + 30 for r in rows)
    sheet = Image.new("RGB", (W6, hgt), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), "Tori v2 step 5 - range test: each joint at min / mid / max (children follow parents); "
                      "pairs = over #6BBF59 | #202020;  + = clockwise", fill=(0, 0, 0), font=font(15))
    dr.text((10, 32), "crack = new thin background crack (<=8px) within the joint radius vs rest; "
                      "hole = new enclosed background px where the body was opaque at rest", fill=(60, 60, 60), font=font(13))
    y = 60
    for label, ims in rows:
        dr.text((10, y), label, fill=(0, 0, 0), font=font(13))
        x = 5
        for im in ims:
            sheet.paste(im, (x, y + 20)); x += im.width + 5
        y += max(im.height for im in ims) + 30
    sheet.save(path)
    return stats, combo_stats


def pivots_sheet(R, path):
    rig = R.rig
    o, _ = R.render({}, (255, 255, 255), (0, 0, W, H), scale=2)
    im = to_img(o).convert("RGB")
    d = ImageDraw.Draw(im)
    pts = {n: rig["parts"][n]["pivot"] for n in rig["parts"]}
    pts["root"] = rig["root"]["origin"]
    cols = {"root": (0, 0, 0), "torso": (200, 0, 0), "head": (0, 90, 220)}
    for n, p in rig["parts"].items():                          # parent links
        q = pts[p["parent"]]
        d.line((p["pivot"][0] * 2, p["pivot"][1] * 2, q[0] * 2, q[1] * 2), fill=(255, 0, 200), width=3)
    f = font(22)
    seen = {}
    for n, p in sorted(pts.items(), key=lambda kv: kv[0]):
        if n == "collar_front":                                  # same pivot as torso, fixed
            continue
        x, y = p[0] * 2, p[1] * 2
        c = cols.get(n, (230, 0, 120))
        d.line((x - 14, y, x + 14, y), fill=c, width=4); d.line((x, y - 14, x, y + 14), fill=c, width=4)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 255), width=2)
        key = (round(x / 40), round(y / 40))
        off = seen.get(key, 0); seen[key] = off + 1
        lab = f"{n} ({p[0]:.0f},{p[1]:.0f})"
        tx, ty = x + 16, y - 26 + off * 26
        d.rectangle((tx - 2, ty, tx + d.textlength(lab, font=f) + 2, ty + 24), fill=(255, 255, 255))
        d.text((tx, ty), lab, fill=c, font=f)
    # anchors
    a = rig["anchors"]["breath"]["point"]
    d.line((a[0] * 2 - 300, a[1] * 2, a[0] * 2 + 300, a[1] * 2), fill=(0, 160, 0), width=2)
    d.text((a[0] * 2 + 310, a[1] * 2 - 12), "feet line / breath anchor / root", fill=(0, 120, 0), font=font(20))
    d.text((20, 20), "Tori v2 rig - rest pose x2: pivots (cross) + parent links (magenta)", fill=(0, 0, 0), font=font(28))
    im.save(path)


def tuft_sheet(R, opt, path):
    """step-4 tuft (in front, shaded) vs step-5 tuft (flat, behind the head) and sways."""
    box = (380, 40, 650, 260)
    TW = 330
    rows = []
    for bgname, bg in (("green", GREEN), ("dark", DARK)):
        o4, _ = compose_full(opt, STEP4_OPT_ORDER, np.array(bg, float))
        ims = [to_img(o4[box[1]:box[3], box[0]:box[2]]).resize((TW, int(TW * 220 / 270)), Image.LANCZOS)]
        st = []
        for deg in (0, -5, 5, -10, 10, -15, 15):
            im, a = thumb(R, {"tuft": deg}, box, TW, bg)
            ims.append(im)
        rows.append((f"over {bgname}: step 4 (front, shaded) | step 5 rest | tuft -5 | +5 | -10 | +10 | -15 | +15", ims))
    sheet = Image.new("RGB", (8 * (TW + 5) + 10, 40 + 2 * (int(TW * 220 / 270) + 30)), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), "tuft: flat head-tan, layer behind the head, base extended (hidden) - sway test about pivot %s"
            % R.rig["parts"]["tuft"]["pivot"], fill=(0, 0, 0), font=font(15))
    y = 32
    for label, ims in rows:
        dr.text((10, y), label, fill=(0, 0, 0), font=font(13))
        x = 5
        for im in ims:
            sheet.paste(im, (x, y + 18)); x += im.width + 5
        y += ims[0].height + 30
    sheet.save(path)


def rest_checks(R, L4, opt, final, tuft_area):
    res = {}
    for name, bg in (("white", (255, 255, 255)), ("green", GREEN), ("dark", DARK)):
        bgv = np.array(bg, float)
        rig_o, rig_a = R.render({}, bg)
        fin_o, _ = compose_full(final, ORDER, bgv)
        s4_o, _ = compose_full(L4, STEP4_ORDER, bgv)
        s4o_o, _ = compose_full(opt, STEP4_OPT_ORDER, bgv)
        r8, f8, s8, so8 = (np.round(x) for x in (rig_o, fin_o, s4_o, s4o_o))
        d_rig = np.abs(r8 - f8).max(-1)
        d4 = np.abs(r8 - s8).max(-1)
        d4o = np.abs(r8 - so8).max(-1)
        out = ~tuft_area
        res[name] = {
            "rig_render_from_crops_vs_final_full_layers": {"max": int(d_rig.max()), "px_diff": int((d_rig > 0).sum())},
            "vs_step4_rest(main stack)": {
                "outside_tuft_area": {"max": int(d4[out].max()), "px_diff": int((d4[out] > 0).sum()),
                                      "px_diff_gt2": int((d4[out] > 2).sum())},
                "inside_tuft_area": {"px": int(tuft_area.sum()), "px_diff": int((d4[tuft_area] > 0).sum()),
                                     "max": int(d4[tuft_area].max())}},
            "vs_step4_rest(optional stack)": {
                "outside_tuft_area": {"max": int(d4o[out].max()), "px_diff": int((d4o[out] > 0).sum()),
                                      "px_diff_gt2": int((d4o[out] > 2).sum())}},
        }
        if name == "white":
            res["_diff_img"] = (d4, d4o)
    return res


def main():
    L4, opt, final, Z, tuft_area, tinfo, arm_fix = build()
    R = Rig(PUB / "rig.json")
    checks = rest_checks(R, L4, opt, final, tuft_area)
    d4, d4o = checks.pop("_diff_img")
    # diff image (white bg): red = differs from step-4 rest, blue outline = step-4 tuft area
    base, _ = R.render({})
    im = (base * 0.35 + 255 * 0.65)
    im[d4o > 0] = [255, 0, 0]
    edge = tuft_area & ~ndi.binary_erosion(tuft_area)
    im[edge] = [0, 0, 255]
    to_img(im[40:260, 380:650]).resize((810, 660), Image.NEAREST).save(OUT / "rest-diff-tuft.png")
    pivots_sheet(R, OUT / "pivots.png")
    tuft_sheet(R, opt, OUT / "tuft-check.png")
    tuft_sway = {}
    for deg in (-15, -10, -5, 5, 10, 15):
        s, _ = joint_gap(R, {"tuft": deg}, "tuft", 90)
        tuft_sway[str(deg)] = s
    stats, combo = range_sheet(R, OUT / "range-test.png")
    rig = json.loads((OUT / "rig.json").read_text())
    rig["checks"] = {"rest": checks, "tuft": tinfo, "tuft_sway": tuft_sway, "arm_collar_unmix": arm_fix,
                     "range_test": stats, "combined_poses": combo}
    rig.pop("_tuft", None); rig.pop("_armFix", None)
    (OUT / "rig.json").write_text(json.dumps(rig, indent=1, ensure_ascii=False))
    shutil.copy(OUT / "rig.json", PUB / "rig.json")
    (OUT / "rest-check.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    print(json.dumps(tuft_sway, indent=1))


def cli_render(argv):
    pose = json.loads(pathlib.Path(argv[0]).read_text())
    bg = (255, 255, 255)
    scale = 1.0
    if "--bg" in argv:
        h = argv[argv.index("--bg") + 1]; bg = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if "--scale" in argv:
        scale = float(argv[argv.index("--scale") + 1])
    o, _ = Rig().render(pose, bg, scale=scale)
    to_img(o).save(argv[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        cli_render(sys.argv[2:])
    else:
        main()
