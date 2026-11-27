#!/usr/bin/env python3
"""
STEP 4 - Tori v2: fill occluded / hidden joint areas (minimal, design-preserving).

Input : art/v2/tori/step3-layers/<part>.png   (RGBA 1024x1024, visible pixels only)
Output: art/v2/tori/step4-filled/
          <part>.png            step-3 layer + hidden fill (every step-3 pixel with alpha>0 is
                                byte-identical except the listed seam un-mix exception, see
                                Fill.seam_unmix; new pixels only where the layers in FRONT of it are
                                fully opaque, so the rest pose recomposite is unchanged)
          filled-<part>.png     diagnostic: added hidden pixels tinted magenta, seam-exception pixels
                                cyan (layer on a checkerboard)
          motion-test.png       test poses from the step-4 layers (over #6BBF59 and #202020)
          motion-test-before.png the same poses from the step-3 layers
          fill-report.json      per part: added px, bbox, method, visible-pixel changes
          optional/             shoulder extensions for the arms + a "collar-front" split
                                (only usable if step 5 draws the collar over the arms; see
                                fill-report.json -> "optional")
No AI images are used (all fills are deterministic geometry); work/ is kept empty.

Principles
  * A pixel may be added to layer P only where P's step-3 alpha is 0 AND some layer drawn in front
    of P has alpha 255 there ("hidden").  Hence over any background the rest pose composite is
    bit-identical to step 3.
  * Flat-vector continuation, no new design: every new pixel is one of the layer's own flat
    region colours (median of that region's opaque pixels); region boundaries are continued
    geometrically:
      - "harmonic SDF": each colour region's signed distance field is solved into the hidden zone
        with Laplace's equation (Dirichlet = the layer's own visible opaque pixels next to the zone,
        Neumann elsewhere); label = argmax.  Straight/curved boundaries that cross a hole are
        continued smoothly; bands that run into an extension keep running.
      - "harmonic ordinal" (tail stem): the bands across the tail stem are ordered
        (outer orange < dark stripe < inner orange < cream underside); that ordinal value is solved
        harmonically so each band boundary entering the hidden zone is joined to the matching one
        on the other side.
      - silhouettes of extensions come from curves fitted to the visible silhouette
        (circle / ellipse / quadratic), rendered with 4x4 supersampled coverage = clean AA edge,
        colour un-premultiplied (no white fringe).
"""
import json, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import splu

ROOT = pathlib.Path(__file__).resolve().parents[2]
IN = ROOT / "art/v2/tori/step3-layers"
OUT = ROOT / "art/v2/tori/step4-filled"
STACK = ["tail", "legL", "legR", "torso", "pendant", "armL", "armR", "earL", "earR", "head", "tuft"]
H = W = 1024
SS = 4                                  # supersampling for coverage / labels
GREEN = (0x6B, 0xBF, 0x59)
DARK = (0x20, 0x20, 0x20)

# flat region colours of each layer (seeds; refined to the median of that region's opaque px)
PALETTES = {
    "torso": {"cream": (251, 243, 220), "tan": (235, 190, 131), "collar_dark": (138, 91, 60),
              "collar_light": (158, 114, 80), "beige": (234, 203, 172), "shade": (206, 147, 98)},
    "tail": {"tan": (236, 190, 131), "shade": (207, 147, 97), "dark": (113, 76, 54),
             "cream": (236, 205, 177)},
    "legL": {"tan": (235, 190, 131), "shade": (208, 148, 98)},
    "legR": {"tan": (235, 190, 131), "shade": (208, 148, 98)},
    "armL": {"tan": (235, 190, 131), "shade": (207, 147, 97)},
    "armR": {"tan": (235, 190, 131), "shade": (207, 147, 97)},
    "earL": {"tan": (235, 190, 131), "inner": (210, 157, 110)},
    "earR": {"tan": (235, 190, 131), "inner": (210, 157, 110)},
    "head": {"tan": (235, 191, 131), "stripe": (115, 80, 58), "cream": (251, 243, 221),
             "band": (239, 201, 178), "eye": (79, 35, 19)},
}


# ============================================================== basic helpers
def load_layers(folder):
    return {p: np.array(Image.open(folder / f"{p}.png").convert("RGBA")) for p in STACK}


def bbox_of(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def hidden_zone(layers, p):
    """pixels where p is transparent and some layer in front of p is fully opaque."""
    i = STACK.index(p)
    front = np.zeros((H, W), bool)
    for q in STACK[i + 1:]:
        front |= layers[q][..., 3] == 255
    return front & (layers[p][..., 3] == 0)


def classify(layer, pal):
    """label per opaque pixel = nearest flat palette colour (-1 where alpha < 255)."""
    names = list(pal)
    P = np.array([pal[n] for n in names], float)
    rgb = layer[..., :3].astype(float)
    lab = ((rgb[..., None, :] - P) ** 2).sum(-1).argmin(-1)
    lab[layer[..., 3] < 255] = -1
    cols = []
    for k in range(len(names)):
        m = lab == k
        deep = m & ndi.binary_erosion(m, iterations=2)
        src = deep if deep.sum() > 30 else m
        cols.append(np.median(rgb[src], 0) if src.any() else P[k])
    return names, lab, np.array(cols)


def sdf_fields(lab, n):
    """per-label signed distance (+ inside), computed on the known (labelled) pixels."""
    known = lab >= 0
    F = np.zeros((n, H, W))
    for k in range(n):
        ins = lab == k
        oth = known & ~ins
        d_in = ndi.distance_transform_edt(~oth) if oth.any() else np.full((H, W), 99.0)
        d_out = ndi.distance_transform_edt(~ins) if ins.any() else np.full((H, W), 99.0)
        F[k] = np.where(ins, d_in - 0.5, -(d_out - 0.5))
        F[k] = np.clip(F[k], -40, 40)
    return F


def harmonic(vals, known, zone):
    """Solve Laplace(u)=0 on `zone` for each field in vals (F,H,W); Dirichlet = known pixels
    4-adjacent to the zone, Neumann on every other zone border.  Components without any Dirichlet
    neighbour are tied (weakly) to the nearest known value."""
    ys, xs = np.nonzero(zone)
    n = len(ys)
    out = vals.copy()
    if n == 0:
        return out
    idx = -np.ones((H, W), int)
    idx[ys, xs] = np.arange(n)
    nf = vals.shape[0]
    rows, cols = [], []
    deg = np.zeros(n)
    b = np.zeros((n, nf))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny = np.clip(ys + dy, 0, H - 1)
        nx = np.clip(xs + dx, 0, W - 1)
        inz = zone[ny, nx]
        ink = known[ny, nx] & ~inz
        rows.append(np.nonzero(inz)[0]); cols.append(idx[ny[inz], nx[inz]])
        deg += inz | ink
        b[ink] += vals[:, ny[ink], nx[ink]].T
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = coo_matrix((-np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    eps = 1e-5
    _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
    fb = vals[:, iy[ys, xs], ix[ys, xs]].T
    A = A + coo_matrix((deg + eps, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    b += eps * fb
    sol = splu(A.tocsc()).solve(b)
    out[:, ys, xs] = sol.T
    return out


def sub_coords(box):
    """subpixel sample coordinates for bbox (x0,y0,x1,y1): arrays (h*SS, w*SS)."""
    x0, y0, x1, y1 = box
    o = (np.arange(SS) + 0.5) / SS - 0.5
    X = (np.arange(x0, x1)[:, None] + o[None, :]).ravel()
    Y = (np.arange(y0, y1)[:, None] + o[None, :]).ravel()
    return np.meshgrid(X, Y)


def pool(a, h, w):
    return a.reshape(h, SS, w, SS, *a.shape[2:]).mean((1, 3))


def sample_fields(F, defined, X, Y):
    """bilinear sample of fields F (defined where `defined`) at subpixel coords."""
    _, (iy, ix) = ndi.distance_transform_edt(~defined, return_indices=True)
    Ff = F[:, iy, ix]
    return np.stack([ndi.map_coordinates(f, [Y, X], order=1, mode="nearest") for f in Ff])


def render(zone, shape_fn, label_fn, colors):
    """New pixels on `zone`: coverage of shape_fn (None = whole pixel), colour = mean of the
    palette colours of the covered subsamples (un-premultiplied).  Returns (alpha8, rgb8) full
    canvas arrays, non-zero only on zone."""
    a8 = np.zeros((H, W), np.uint8)
    rgb8 = np.zeros((H, W, 3), np.uint8)
    bb = bbox_of(zone)
    if bb is None:
        return a8, rgb8
    x0, y0, x1, y1 = bb
    x0, y0, x1, y1 = max(x0 - 1, 0), max(y0 - 1, 0), min(x1 + 1, W), min(y1 + 1, H)
    h, w = y1 - y0, x1 - x0
    X, Y = sub_coords((x0, y0, x1, y1))
    inside = np.ones(X.shape, bool) if shape_fn is None else shape_fn(X, Y)
    lab = label_fn(X, Y)
    col = colors[lab]
    cov = pool(inside.astype(float), h, w)
    csum = pool(col * inside[..., None], h, w)
    rgb = csum / np.maximum(cov, 1e-9)[..., None]
    z = zone[y0:y1, x0:x1]
    al = np.round(cov * 255).astype(np.uint8)
    al[~z] = 0
    rg = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
    rg[al == 0] = 0
    a8[y0:y1, x0:x1] = al
    rgb8[y0:y1, x0:x1] = rg
    return a8, rgb8


def label_fn_from_fields(F, defined):
    def fn(X, Y):
        return sample_fields(F, defined, X, Y).argmax(0)
    return fn


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy, c = sol
    return cx, cy, np.sqrt(c + cx ** 2 + cy ** 2)


def fit_ellipse(x, y):
    """Fitzgibbon direct least squares ellipse fit -> conic coefficients, F<0 inside."""
    mx, my, s = x.mean(), y.mean(), max(x.std(), y.std())
    xn, yn = (x - mx) / s, (y - my) / s
    D = np.c_[xn ** 2, xn * yn, yn ** 2, xn, yn, np.ones_like(xn)]
    S = D.T @ D
    C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    ev, evec = np.linalg.eig(np.linalg.solve(S, C))
    ev = np.real(ev); evec = np.real(evec)
    k = np.argmax(ev > 0) if (ev > 0).any() else np.argmax(ev)
    a = evec[:, k]

    def F(X, Y):
        Xn, Yn = (X - mx) / s, (Y - my) / s
        v = a[0] * Xn ** 2 + a[1] * Xn * Yn + a[2] * Yn ** 2 + a[3] * Xn + a[4] * Yn + a[5]
        return v
    sign = -1.0 if F(np.array([mx]), np.array([my]))[0] > 0 else 1.0
    return lambda X, Y: sign * F(X, Y)


def silhouette_pts(alpha, lo=40, hi=215):
    ys, xs = np.nonzero((alpha >= lo) & (alpha <= hi))
    return xs.astype(float), ys.astype(float)


def label_boundary_pts(lab, k, others):
    """midpoints between 4-adjacent pixels with label k and a label in `others`."""
    pts = []
    for dy, dx in ((0, 1), (1, 0)):
        a = lab[:H - dy, :W - dx]; b = lab[dy:, dx:]
        m = ((a == k) & np.isin(b, others)) | ((b == k) & np.isin(a, others))
        ys, xs = np.nonzero(m)
        pts.append(np.c_[xs + dx / 2, ys + dy / 2])
    p = np.concatenate(pts)
    return p[:, 0], p[:, 1]


def near(mask, d):
    return ndi.binary_dilation(mask, iterations=d)


# ============================================================== per-part fills
class Fill:
    def __init__(self, layers):
        self.L = layers
        self.new = {p: layers[p].copy() for p in STACK}
        self.added = {p: np.zeros((H, W), bool) for p in STACK}
        self.notes = {p: [] for p in STACK}
        self.extra = {}

    def put(self, p, a8, rgb8, note):
        """write new pixels: only where the layer is still transparent and the pixel is hidden."""
        zone = hidden_zone(self.L, p) & (self.new[p][..., 3] == 0) & (a8 > 0)
        self.new[p][zone, :3] = rgb8[zone]
        self.new[p][zone, 3] = a8[zone]
        self.added[p] |= zone
        self.notes[p].append(f"{note} [{int(zone.sum())} px]")
        return zone

    # ---------------------------------------------------------------- torso
    def torso(self):
        p = "torso"
        L = self.L[p]
        names, lab, cols = classify(L, PALETTES[p])
        hid = hidden_zone(self.L, p)
        known = lab >= 0
        F = sdf_fields(lab, len(names))

        # (1) hole under the pendant: the collar-light / beige-band / cream-belly boundaries are
        #     near-horizontal curves crossing the hole -> fit y = quadratic(x) to each boundary on
        #     both sides of the hole (>= 8 px away from it, where step 3 left clean flat colours) and
        #     label the hole by which side of each curve a subsample lies.
        hole = hid & (self.L["pendant"][..., 3] == 255)
        hx0, hy0, hx1, hy1 = bbox_of(hole)
        far = ~near(hole, 8)
        order = ["collar_dark", "collar_light", "beige", "cream"]
        ki = [names.index(n) for n in order]
        curves = []
        for a_, b_ in zip(ki[:-1], ki[1:]):
            bx, by = label_boundary_pts(lab, a_, [b_])
            sel = (bx > hx0 - 70) & (bx < hx1 + 70) & (by > hy0 - 40) & (by < hy1 + 20)
            sel &= far[np.round(by).astype(int), np.round(bx).astype(int)]
            curves.append(np.polyfit(bx[sel], by[sel], 2) if sel.sum() > 10 else None)
        self.geom["pendant_hole_band_curves"] = [None if c is None else [float(v) for v in c] for c in curves]

        def hole_lab(X, Y):
            idx = np.zeros(X.shape, int)
            for c in curves:
                if c is not None:
                    idx += Y > np.polyval(c, X)
            return np.array(ki)[idx]
        a8, rgb8 = render(hole, None, hole_lab, cols)
        self.put(p, a8, rgb8, "pendant hole: collar-dark|collar-light, collar-light|beige-band and beige|cream "
                              "boundaries fitted (quadratic) on both sides of the hole and continued across it; "
                              "flat region colours, 4x4 supersampled AA")

        # (2) under the arms: torso side silhouette continued under each arm.
        #     Left: visible side below the arm is vertical at its widest point (x~331 @ y~795-805);
        #     top anchor = just inside the collar's rounded end (x~350 @ y~643, where the arm top
        #     meets the collar).  x(y) = xb + (xt-xb)*((yb-y)/(yb-yt))^2  (vertical tangent at yb).
        al = self.L["torso"][..., 3]
        def side_x(side, y_lo, y_hi):
            xs = []
            for y in range(y_lo, y_hi):
                row = np.nonzero(al[y] >= 128)[0]
                xs.append(row.min() if side == "L" else row.max())
            return float(np.median(xs))
        xbL = side_x("L", 790, 812) - 0.0
        xbR = side_x("R", 790, 812) + 0.0
        yb, yt = 800.0, 643.0
        xtL, xtR = 347.0, 679.0
        self.geom = {"torso_side_L": [xtL, yt, xbL, yb], "torso_side_R": [xtR, yt, xbR, yb]}

        def curve(xt, xb):
            return lambda Y: xb + (xt - xb) * (np.clip(yb - Y, 0, None) / (yb - yt)) ** 2
        cL, cR = curve(xtL, xbL), curve(xtR, xbR)
        for arm, fn in (("armL", lambda X, Y: X >= cL(Y) - 0.5), ("armR", lambda X, Y: X <= cR(Y) + 0.5)):
            zone = hid & (self.L[arm][..., 3] == 255)
            yy, xx = np.mgrid[0:H, 0:W]
            zone &= fn(xx + 0.5 * np.sign(1), yy) | fn(xx - 0.5, yy)   # any coverage
            zone &= yy >= 630
            Fz = harmonic(F, known, zone)
            a8, rgb8 = render(zone, fn, label_fn_from_fields(Fz, known | zone), cols)
            self.put(p, a8, rgb8, f"under {arm}: torso side continued as a smooth quadratic from the collar end "
                                  f"to the visible side below the arm; collar-shadow band / tan by harmonic SDF")

        # (3) under the head: collar band continued upward + tan neck above it
        self.neck(names, lab, cols)

    def neck(self, names, lab, cols):
        p = "torso"
        L = self.new[p]
        a = L[..., 3]
        hid = hidden_zone(self.L, p) & (self.new[p][..., 3] == 0)
        ci = [names.index("collar_dark"), names.index("collar_light")]
        collar = np.isin(lab, ci)
        # collar bottom edge per column (lowest collar px above y 700), smoothed by a quartic fit
        xs_b, ys_b = [], []
        top_vis = {}
        for x in range(330, 700):
            col = np.nonzero(a[600:700, x] > 0)[0]
            if len(col) == 0:
                continue
            top_vis[x] = 600 + col.min()
            cc = np.nonzero(collar[600:700, x])[0]
            if len(cc):
                xs_b.append(x); ys_b.append(600 + cc.max() + 0.5)
        xs_b, ys_b = np.array(xs_b, float), np.array(ys_b, float)
        poly = np.polyfit(xs_b, ys_b, 4)
        ybot = lambda X: np.polyval(poly, X)
        thick = max(ybot(x) - top_vis[x] for x in top_vis if 360 <= x <= 665)
        T = float(np.ceil(thick)) + 14.0          # collar thickness: visible max + 14 px (hidden)
        # collar ends = leftmost / rightmost columns with collar in the top visible rows
        xle = min(x for x in top_vis if collar[top_vis[x]:top_vis[x] + 4, x].any())
        xre = max(x for x in top_vis if collar[top_vis[x]:top_vis[x] + 4, x].any())
        xle, xre = xle - 0.5, xre + 0.5
        NECK_IN, NECK_H, RAD = 12.0, 40.0, 18.0
        self.geom["collar"] = {"bottom_poly": [float(v) for v in poly], "thickness": T,
                               "ends": [xle, xre], "neck_inset": NECK_IN, "neck_height": NECK_H}
        ytop = lambda X: ybot(X) - T
        # column colour = torso pixel 2 px below the top visible row (collar shading carried up)
        #     (median of the column's opaque collar px 3..10 below the top visible row, then a
        #      9-px running median along x, so soft/AA rows never get stretched into streaks)
        colcol = np.zeros((W, 3)); have = np.zeros(W, bool)
        for x, y in top_vis.items():
            seg = L[y + 3:y + 11, x]
            ok = (seg[:, 3] == 255) & collar[y + 3:y + 11, x]
            if ok.sum() >= 2:
                colcol[x] = np.median(seg[ok, :3].astype(float), 0); have[x] = True
        xs_h = np.nonzero(have)[0]
        for x in range(W):
            if not have[x] and len(xs_h):
                colcol[x] = colcol[xs_h[np.argmin(np.abs(xs_h - x))]]
        colcol = ndi.median_filter(colcol, size=(9, 1), mode="nearest")
        self.collar_col = colcol
        tan = cols[names.index("tan")]

        ys, xs = np.nonzero(hid)
        sel = (ys < 660) & (ys > 540) & (xs > 320) & (xs < 710)
        zone = np.zeros((H, W), bool); zone[ys[sel], xs[sel]] = True
        x0, y0, x1, y1 = bbox_of(zone)
        X, Y = sub_coords((x0, y0, x1, y1))
        in_collar = (X >= xle) & (X <= xre) & (Y >= ytop(X))
        neck_top = ytop(np.clip(X, xle, xre)) - NECK_H
        in_neck = (X >= xle + NECK_IN) & (X <= xre - NECK_IN) & (Y >= neck_top) & (Y < ytop(X))
        # rounded top corners of the neck
        for cx in (xle + NECK_IN + RAD, xre - NECK_IN - RAD):
            cy = ytop(cx) - NECK_H + RAD
            out_corner = (Y < cy) & (((X < cx) & (cx < 512)) | ((X > cx) & (cx > 512))) & \
                         ((X - cx) ** 2 + (Y - cy) ** 2 > RAD ** 2)
            in_neck &= ~out_corner
        inside = in_collar | in_neck
        xi = np.clip(np.round(X).astype(int), 0, W - 1)
        sub_col = np.where(in_collar[..., None], colcol[xi], tan)
        h, w = y1 - y0, x1 - x0
        cov = pool(inside.astype(float), h, w)
        rgb = pool(sub_col * inside[..., None], h, w) / np.maximum(cov, 1e-9)[..., None]
        a8 = np.zeros((H, W), np.uint8); rgb8 = np.zeros((H, W, 3), np.uint8)
        al = np.round(cov * 255).astype(np.uint8); al[~zone[y0:y1, x0:x1]] = 0
        a8[y0:y1, x0:x1] = al
        rgb8[y0:y1, x0:x1] = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        self.put(p, a8, rgb8, f"under the head: collar band carried straight up column-wise (each column keeps "
                              f"its own visible collar colour) to a collar top = fitted collar bottom curve - {T:.0f} px; "
                              f"tan neck ({NECK_H:.0f} px, inset {NECK_IN:.0f} px, rounded) above it")

    # ---------------------------------------------------------------- legs
    def leg(self, p):
        L = self.L[p]
        names, lab, cols = classify(L, PALETTES[p])
        hid = hidden_zone(self.L, p)
        al = L[..., 3]
        # visible leg sides: outer side (silhouette) and inner side, rows 884..912
        rows = {y: np.nonzero(al[y] >= 128)[0] for y in range(884, 913)}
        left = np.array([[y, r.min()] for y, r in rows.items() if len(r)], float)
        right = np.array([[y, r.max()] for y, r in rows.items() if len(r)], float)
        # the side that touches the crotch band only becomes silhouette lower down (y >= 900)
        if p == "legL":
            xl = left[left[:, 0] <= 888, 1].min() - 0.5      # outer side at the cut
            xr = right[right[:, 0] >= 900, 1].max() + 0.5     # inner side (vertical)
        else:
            xl = left[left[:, 0] >= 900, 1].min() - 0.5
            xr = right[right[:, 0] <= 888, 1].max() + 0.5
        yc = 884.0
        cx, ax, by = (xl + xr) / 2, (xr - xl) / 2, 40.0
        self.geom[p] = {"hip_cap_ellipse": [cx, yc, ax, by]}

        def shape(X, Y):
            cap = (Y <= yc) & (((X - cx) / ax) ** 2 + ((Y - yc) / by) ** 2 <= 1)
            body = (Y > yc) & (Y <= 915) & (X >= xl) & (X <= xr)
            return cap | body
        yy, xx = np.mgrid[0:H, 0:W]
        zone = hid & (shape(xx - 0.5, yy) | shape(xx + 0.5, yy) | shape(xx, yy - 0.5) | shape(xx, yy + 0.5))
        # colour regions carried straight up column-wise: each column keeps the region of its
        # topmost visible opaque pixel (+2 px), so the inner-thigh shading band runs on up into the
        # hip (it meets the torso's crotch band of the same colour) instead of ending in a blob
        tan_k = names.index("tan")
        col_lab = np.full(W, tan_k)
        for xc in range(int(xl), int(xr) + 2):
            ys_ = np.nonzero(lab[:, xc] >= 0)[0]
            ys_ = ys_[ys_ > 870]
            if len(ys_):
                col_lab[xc] = lab[min(ys_.min() + 2, H - 1), xc] if lab[min(ys_.min() + 2, H - 1), xc] >= 0 \
                    else lab[ys_.min(), xc]
        col_lab = ndi.median_filter(col_lab, size=5, mode="nearest")
        labf = lambda X, Y: col_lab[np.clip(np.round(X).astype(int), 0, W - 1)]
        a8, rgb8 = render(zone, shape, labf, cols)
        self.put(p, a8, rgb8, "hip end: visible leg sides continued straight up to the top cut and closed by a "
                              f"half-ellipse cap (center {cx:.0f},{yc:.0f}, a={ax:.0f}, b={by:.0f}) under the torso; "
                              "tan / inner shading band carried straight up column-wise")

    # ---------------------------------------------------------------- ears
    def ear(self, p):
        L = self.L[p]
        names, lab, cols = classify(L, PALETTES[p])
        hid = hidden_zone(self.L, p)
        head = self.L["head"][..., 3] > 0
        x, y = silhouette_pts(L[..., 3])
        keep = ~near(head, 6)[y.astype(int), x.astype(int)]
        outer = fit_ellipse(x[keep], y[keep])
        ki, kt = names.index("inner"), names.index("tan")
        xi, yi = label_boundary_pts(lab, ki, [kt])
        keep = ~near(head, 4)[np.round(yi).astype(int), np.round(xi).astype(int)]
        inner = fit_ellipse(xi[keep], yi[keep])
        yy, xx = np.mgrid[0:H, 0:W]
        zone = hid & (outer(xx, yy) < 0.02)
        shape = lambda X, Y: outer(X, Y) < 0
        labf = lambda X, Y: np.where(inner(X, Y) < 0, ki, kt)
        a8, rgb8 = render(zone, shape, labf, cols)
        self.put(p, a8, rgb8, "ear bottom completed behind the head: ellipse fitted to the visible ear silhouette "
                              "(away from the head) and ellipse fitted to the visible inner-ear boundary; "
                              "4x4 supersampled AA edge")

    # ---------------------------------------------------------------- head (under the tuft)
    def head(self):
        p = "head"
        L = self.L[p]
        names, lab, cols = classify(L, PALETTES[p])
        hid = hidden_zone(self.L, p)
        tuft = self.L["tuft"][..., 3] > 0
        ears = (self.L["earL"][..., 3] > 0) | (self.L["earR"][..., 3] > 0)
        x, y = silhouette_pts(L[..., 3])
        sel = (y < 215) & (((x > 370) & (x < 425)) | ((x > 590) & (x < 645)))
        sel &= ~near(tuft | ears, 6)[y.astype(int), x.astype(int)]
        cx, cy, r = fit_circle(x[sel], y[sel])
        self.geom["head_top_circle"] = [cx, cy, r]
        shape = lambda X, Y: (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        yy, xx = np.mgrid[0:H, 0:W]
        zone = hid & tuft & ((xx - cx) ** 2 + (yy - cy) ** 2 <= (r + 1) ** 2)
        known = lab >= 0
        F = sdf_fields(lab, len(names))
        Fz = harmonic(F, known, zone)
        a8, rgb8 = render(zone, shape, label_fn_from_fields(Fz, known | zone), cols)
        self.put(p, a8, rgb8, f"head top under the tuft: head outline = circle fitted to the visible head top "
                              f"left/right of the tuft (c=({cx:.1f},{cy:.1f}), r={r:.1f}); the three stripes run "
                              f"on up to that outline (harmonic SDF)")

    # ---------------------------------------------------------------- tail
    def tail(self):
        p = "tail"
        L = self.L[p]
        names, lab, cols = classify(L, PALETTES[p])
        hid = hidden_zone(self.L, p)
        al = L[..., 3]
        # outer (upper-left) silhouette of the tail just above the right arm -> circle
        x, y = silhouette_pts(al)
        s1 = (y >= 575) & (y <= 650) & (x < 760) & (x > 690)
        c1 = fit_circle(x[s1], y[s1])
        # lower silhouette (cream underside) near the body -> circle
        s2 = (x >= 690) & (x <= 780) & (y > 840)
        c2 = fit_circle(x[s2], y[s2])
        XR = 640.0                       # root extends ~50 px behind the torso
        self.geom["tail"] = {"outer_circle": [float(v) for v in c1], "lower_circle": [float(v) for v in c2],
                             "root_x": XR}

        # below the arm top the tail's outer edge is led from the fitted circle (y 656) onto the
        # torso side curve (y 674, 6 px behind it), i.e. the tail root tucks in behind the body
        # instead of leaving a body/tail pocket open when the right arm swings in.
        xtR_, yt_, xbR_, yb_ = self.geom["torso_side_R"]
        cR = lambda Y: xbR_ + (xtR_ - xbR_) * (np.clip(yb_ - Y, 0, None) / (yb_ - yt_)) ** 2
        Y0, Y1 = 656.0, 674.0
        xa = c1[0] - np.sqrt(c1[2] ** 2 - (Y0 - c1[1]) ** 2)
        xb_ = cR(Y1) - 6.0
        self.geom["tail"]["tuck_line"] = [float(xa), Y0, float(xb_), Y1]

        def shape(X, Y):
            circ = ((X - c1[0]) ** 2 + (Y - c1[1]) ** 2 <= c1[2] ** 2)
            t = np.clip((Y - Y0) / (Y1 - Y0), 0, 1)
            tuck = X >= xa + (xb_ - xa) * t
            in1 = np.where(Y <= Y0, circ, np.where(Y < Y1, tuck, True))
            in2 = (X - c2[0]) ** 2 + (Y - c2[1]) ** 2 <= c2[2] ** 2
            return in1 & in2 & (X >= XR) & (Y >= 630)
        yy, xx = np.mgrid[0:H, 0:W]
        zone = hid & (shape(xx - 0.5, yy) | shape(xx + 0.5, yy) | shape(xx, yy - 0.5) | shape(xx, yy + 0.5))
        zone &= (yy < 900) & (xx < 800)
        # ordinal band value across the stem: outer orange 0 < dark 1 < inner orange 2 < cream 3
        kt, ks, kd, kc = (names.index(n) for n in ("tan", "shade", "dark", "cream"))
        orange = np.isin(lab, [kt, ks])
        win = np.zeros((H, W), bool); win[600:900, 640:800] = True
        cc, n = ndi.label(orange & win)
        sizes = ndi.sum(np.ones((H, W)), cc, index=np.arange(1, n + 1))
        # the outer band is the component touching the tail's upper-left silhouette above the arm
        ys_, xs_ = np.nonzero(orange & win & (np.arange(H)[:, None] < 670) & (np.arange(W)[None, :] < 712))
        outer_id = np.bincount(cc[ys_, xs_][cc[ys_, xs_] > 0]).argmax()
        U = np.full((H, W), np.nan)
        U[orange & win] = 2.0
        U[cc == outer_id] = 0.0
        U[lab == kd] = 1.0
        U[lab == kc] = 3.0
        known = ~np.isnan(U)
        U0 = np.nan_to_num(U)[None]
        Uz = harmonic(U0, known, zone)[0]
        defined = known | zone
        band_lab = np.array([ks, kd, ks, kc])

        def labf(X, Y):
            u = sample_fields(Uz[None], defined, X, Y)[0]
            return band_lab[np.clip(np.round(u), 0, 3).astype(int)]
        a8, rgb8 = render(zone, shape, labf, cols)
        self.put(p, a8, rgb8, "tail stem behind the right arm + root ~50 px behind the torso: silhouette = circle "
                              "fits of the visible upper-left edge and of the cream underside edge; bands "
                              "(outer orange / dark stripe / inner orange / cream) by harmonic ordinal field, "
                              "so the dark stripe entering under the arm joins the dark wedge between arm and body")

    # ---------------------------------------------------------------- seam exception
    SEAM_PAIRS = [("torso", "head"), ("tail", "armR"), ("tail", "torso"), ("head", "tuft"),
                  ("earL", "head"), ("earR", "head"), ("torso", "armL"), ("torso", "armR"),
                  ("legL", "torso"), ("legR", "torso"), ("torso", "pendant")]
    SEAM_TOL = 12

    def seam_unmix(self):
        """EXCEPTION (reported): step 2/3 cut every internal part/part boundary hard, so the BACK
        part's boundary row still holds the anti-aliasing mix with the FRONT part's colour (e.g. a
        light row of head-band colour on top of the brown collar, arm-tan dots on the tail's dark
        stripe).  When the front part moves, that row stays behind as a dotted / light seam line.
        Fix that keeps the rest pose: for each back-part boundary pixel (alpha 255, 4-adjacent to an
        opaque front-part pixel) whose colour is > SEAM_TOL off the clean local colour of its own
        region (median of same-region non-boundary px within 3 px), the back pixel gets that clean
        colour and the FRONT layer gets a soft edge pixel there: alpha = projection of the mix on the
        (front colour - clean colour) line, colour un-mixed exactly for that alpha, so front-over-
        back reproduces the original pixel (verified: rest recomposite diff below)."""
        self.seam_mask = {p: np.zeros((H, W), bool) for p in STACK}
        self.exceptions = {}
        for back, front in self.SEAM_PAIRS:
            if back not in PALETTES:
                continue
            Bl, Fl = self.L[back], self.L[front]
            B, Fn = self.new[back], self.new[front]
            names, lab, cols = classify(Bl, PALETTES[back])
            fa = Fl[..., 3] == 255
            adj = np.zeros((H, W), bool)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                adj |= np.roll(fa, (dy, dx), (0, 1))
            bnd = (Bl[..., 3] == 255) & adj & (Fl[..., 3] == 0)
            if not bnd.any():
                continue
            rgb = Bl[..., :3].astype(float)
            clean_px = (lab >= 0) & ~ndi.binary_dilation(bnd, iterations=1)
            ys, xs = np.nonzero(bnd)
            changed = np.zeros((H, W), bool)
            resid = []
            for y, x in zip(ys, xs):
                if any(self.seam_mask[q][y, x] for q in STACK):
                    continue                       # one un-mix per pixel
                y0, y1, x0, x1 = max(y - 3, 0), y + 4, max(x - 3, 0), x + 4
                cl = clean_px[y0:y1, x0:x1]
                if cl.sum() < 3:
                    continue
                # own region = the region most of the clean neighbours belong to (the mixed pixel's
                # own nearest-palette label is unreliable)
                k = np.bincount(lab[y0:y1, x0:x1][cl]).argmax()
                m = cl & (lab[y0:y1, x0:x1] == k)
                if m.sum() < 3:
                    continue
                Cc = np.round(np.median(rgb[y0:y1, x0:x1][m], 0))
                M = rgb[y, x]
                if np.abs(M - Cc).max() <= self.SEAM_TOL:
                    continue
                # front colour = mean of the opaque front neighbours
                nb = [Fl[y + dy, x + dx, :3].astype(float) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= y + dy < H and 0 <= x + dx < W and Fl[y + dy, x + dx, 3] == 255]
                Hc = np.mean(nb, 0)
                d = Hc - Cc
                if np.dot(d, d) < 100:
                    continue
                a = float(np.clip(np.dot(M - Cc, d) / np.dot(d, d), 0, 1))
                a8 = int(round(a * 255))
                if a8 < 8:
                    continue
                Fc = np.round(np.clip((M - (1 - a8 / 255) * Cc) / (a8 / 255), 0, 255))
                comp = a8 / 255 * Fc + (1 - a8 / 255) * Cc
                if np.abs(comp - M).max() > 1.0:
                    continue                       # mix not explained by these two colours: leave it
                B[y, x, :3] = Cc.astype(np.uint8)
                Fn[y, x, :3] = Fc.astype(np.uint8)
                Fn[y, x, 3] = a8
                resid.append(np.abs(comp - M).max())
                changed[y, x] = True
            n = int(changed.sum())
            if n == 0:
                continue
            self.seam_mask[back] |= changed
            self.seam_mask[front] |= changed
            self.exceptions[f"{back}|{front}"] = {
                "back_visible_px_recoloured": n, "front_soft_edge_px_added": n, "bbox": bbox_of(changed),
                "unmix_residual_max": round(float(max(resid)), 2)}
            self.notes[back].append(f"EXCEPTION seam un-mix vs {front}: {n} visible boundary px set to the clean "
                                    f"local colour (their AA moved into {front} as a soft edge)")
            self.notes[front].append(f"EXCEPTION seam un-mix over {back}: {n} soft edge px added (AA taken from "
                                     f"{back}'s boundary row; rest pose unchanged)")
        self.exceptions["why"] = " ".join(self.seam_unmix.__doc__.split())

    def run(self):
        self.geom = {}
        self.exceptions = {}
        self.torso()
        self.seam_unmix()
        self.leg("legL"); self.leg("legR")
        self.ear("earL"); self.ear("earR")
        self.head()
        self.tail()
        for p in STACK:
            if not self.notes[p]:
                self.notes[p].append("nothing hidden to fill (no layer in front covers a joint of this part) "
                                     "-> copied unchanged")


# ============================================================== compositing / motion test
def premul(rgba):
    f = rgba.astype(np.float64)
    a = f[..., 3:4] / 255.0
    return np.concatenate([f[..., :3] * a, a], -1)


def rotate_pm(pm, pivot, deg):
    """rotate a premultiplied float layer (H,W,4) by deg (counter-clockwise on screen) about pivot."""
    if deg == 0:
        return pm
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    px, py = pivot
    # output (x,y) samples input at R^-1 (p - pivot) + pivot ; screen y down -> ccw uses these signs
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - px, yy - py
    sx = c * dx - s * dy + px
    sy = s * dx + c * dy + py
    return np.stack([ndi.map_coordinates(pm[..., k], [sy, sx], order=1, mode="constant", cval=0)
                     for k in range(4)], -1)


def compose(pms, order, bg):
    out = np.empty((H, W, 3)); out[:] = bg
    acc_a = np.zeros((H, W))
    for p in order:
        pm = pms[p]
        out = pm[..., :3] + out * (1 - pm[..., 3:4])
        acc_a = pm[..., 3] + acc_a * (1 - pm[..., 3])
    return out, acc_a


def holes(acc_a):
    """transparent pixels (alpha<0.9) enclosed by the character (not connected to the border)."""
    t = acc_a < 0.9
    lab, n = ndi.label(t)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    enclosed = t & ~np.isin(lab, list(border))
    return int(enclosed.sum()), enclosed


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def checker(h, w, s=16):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // s + xx // s) % 2)[..., None] == 0, 200, 235).astype(np.float64).repeat(3, 2)


def poses(piv):
    """(title, {part: deg}, crop box) ; head tilt carries ears + tuft about the head pivot."""
    hd = lambda d: {"head": d}
    AR, HD, EA, TU, LG, TL, PE = (230, 570, 800, 860), (250, 120, 780, 700), (270, 90, 760, 290), \
        (400, 50, 620, 230), (310, 800, 720, 1000), (600, 540, 960, 910), (430, 620, 600, 790)
    return [
        ("rest", {}, (190, 50, 950, 990)),
        ("arms out 35 (armL -35, armR +35)", {"armL": -35, "armR": 35}, AR),
        ("arms in 35 (armL +35, armR -35)", {"armL": 35, "armR": -35}, AR),
        ("head +8 (ccw; ears+tuft follow)", hd(8), HD),
        ("head -8 (cw; ears+tuft follow)", hd(-8), HD),
        ("ears out 20 (earL +20, earR -20)", {"earL": 20, "earR": -20}, EA),
        ("ears in 20 (earL -20, earR +20)", {"earL": -20, "earR": 20}, EA),
        ("tuft +10", {"tuft": 10}, TU),
        ("tuft -10", {"tuft": -10}, TU),
        ("legs out 15 (legL -15, legR +15)", {"legL": -15, "legR": 15}, LG),
        ("legs in 15 (legL +15, legR -15)", {"legL": 15, "legR": -15}, LG),
        ("tail +10 (ccw)", {"tail": 10}, TL),
        ("tail -10 (cw)", {"tail": -10}, TL),
        ("pendant +6", {"pendant": 6}, PE),
        ("pendant -6", {"pendant": -6}, PE),
    ]


def pose_layers(pms, piv, rot, cache, stack=STACK):
    cur = {}
    for p in stack:
        if "head" in rot and p in ("earL", "earR", "tuft"):
            cur[p] = rotate_pm(pms[p], piv["head"], rot["head"])
        else:
            key = (p, rot.get(p, 0))
            if key not in cache:
                cache[key] = rotate_pm(pms[p], piv.get(p, (512, 512)), rot.get(p, 0))
            cur[p] = cache[key]
    return cur


def motion_sheet(layers, piv, path, title, stack=STACK, pose_list=None):
    pms = {p: premul(layers[p]) for p in stack}
    cache, rows, stats = {}, [], []
    TW = 560
    _, acc_rest = compose(pms, stack, np.zeros(3))
    _, hole_rest = holes(acc_rest)
    for name, rot, box in (pose_list or poses(piv)):
        cur = pose_layers(pms, piv, rot, cache, stack)
        ims = []
        for bg in (GREEN, DARK):
            img, acc = compose(cur, stack, np.array(bg, float))
            x0, y0, x1, y1 = box
            im = Image.fromarray(np.round(img[y0:y1, x0:x1]).clip(0, 255).astype(np.uint8))
            sc = TW / im.width
            ims.append(im.resize((TW, int(im.height * sc)), Image.LANCZOS))
        _, hm = holes(acc)
        nh = int((hm & ~hole_rest).sum())
        stats.append({"pose": name, "new_enclosed_hole_px": nh})
        rows.append((f"{name}   [new enclosed holes vs rest: {nh} px]", ims))
    hgt = sum(r[1][0].height + 34 for r in rows) + 50
    sheet = Image.new("RGB", (2 * TW + 30, hgt), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), title, fill=(0, 0, 0), font=font(20))
    y = 50
    for label, ims in rows:
        d.text((10, y), label, fill=(0, 0, 0), font=font(15))
        for j, im in enumerate(ims):
            sheet.paste(im, (10 + j * (TW + 10), y + 22))
        y += ims[0].height + 34
    sheet.save(path)
    return stats


def pivots(layers):
    a = {p: layers[p][..., 3] for p in STACK}
    piv = {}
    # arm: middle of the arm's top edge (arm px 4-adjacent to the torso collar), 6 px into the arm
    for arm in ("armL", "armR"):
        m = a[arm] > 0
        t = a["torso"] > 0
        edge = m & (np.roll(t, 1, 0) | np.roll(t, 1, 1) | np.roll(t, -1, 1)) & (np.arange(H)[:, None] < 670)
        ys, xs = np.nonzero(edge)
        piv[arm] = (float(xs.mean()), float(ys.mean()) + 6)
    hb = np.nonzero(a["head"][:, 512] > 0)[0].max()
    piv["head"] = (512.0, float(hb) - 8)
    for ear in ("earL", "earR"):
        m = a[ear] > 0
        hd = a["head"] > 0
        edge = m & ndi.binary_dilation(hd, iterations=1)
        ys, xs = np.nonzero(edge)
        piv[ear] = (float(xs.mean()), float(ys.mean()))
    m = a["tuft"] > 0
    edge = m & ndi.binary_dilation(a["head"] > 0, iterations=1)
    ys, xs = np.nonzero(edge)
    piv["tuft"] = (float(xs.mean()), float(ys.max()))
    piv["legL"] = (409.0, 884.0)
    piv["legR"] = (615.0, 884.0)
    piv["tail"] = (705.0, 825.0)
    ys, xs = np.nonzero(a["pendant"] > 0)
    top = ys.min()
    piv["pendant"] = (float(xs[ys == top].mean()), float(top) + 2)
    piv["torso"] = (512.0, 800.0)
    return {k: (round(v[0], 1), round(v[1], 1)) for k, v in piv.items()}


# ============================================================== diagnostics
def diag_sheet(layer, added, path, seam=None):
    bb = bbox_of(layer[..., 3] > 0)
    x0, y0, x1, y1 = bb
    x0, y0, x1, y1 = max(0, x0 - 10), max(0, y0 - 10), min(W, x1 + 10), min(H, y1 + 10)
    f = layer[y0:y1, x0:x1].astype(float)
    a = f[..., 3:4] / 255
    bg = checker(y1 - y0, x1 - x0, 8)
    img = f[..., :3] * a + bg * (1 - a)
    add = added[y0:y1, x0:x1]
    img[add] = img[add] * 0.45 + np.array([255, 0, 255]) * 0.55
    if seam is not None:
        sm = seam[y0:y1, x0:x1]
        img[sm] = np.array([0, 230, 255])         # listed exception px (cyan)
    im = Image.fromarray(np.round(img).astype(np.uint8))
    sc = max(2, int(900 / max(im.size)))
    if sc > 1:
        im = im.resize((im.width * sc, im.height * sc), Image.NEAREST)
    im.save(path)


# ============================================================== optional: shoulder extensions
def optional_shoulders(L, new, geom, outdir, seam_torso):
    """Arms are drawn IN FRONT of the torso incl. its collar (step-3 order), and each arm's top
    edge touches the collar, so no hidden pixel exists above an arm: a shoulder extension can not
    live in the main layers without changing the rest pose.  Offered here for step 5:
      collar-front.png  = the torso's own opaque (alpha 255) collar pixels at the two shoulders,
                          byte-identical copies (no new art) - to be drawn right above the arms;
      armL.png / armR.png = step-4 arm + rounded shoulder end that lies only under collar-front
                          or under the head.
    Stack with them: tail legL legR torso pendant armL* armR* collar-front earL earR head tuft."""
    outdir.mkdir(parents=True, exist_ok=True)
    names, lab, cols = classify(L["torso"], PALETTES["torso"])
    collar = np.isin(lab, [names.index("collar_dark"), names.index("collar_light")])
    xx = np.arange(W)[None, :]
    cf_mask = collar & ((xx < 430) | (xx > 594)) & (L["torso"][..., 3] == 255) & ~seam_torso
    cf = np.zeros_like(L["torso"])
    cf[cf_mask] = new["torso"][cf_mask]          # step-4 torso pixels (after the seam un-mix)
    Image.fromarray(cf, "RGBA").save(outdir / "collar-front.png")
    cover = cf_mask | (L["head"][..., 3] == 255) | (L["earL"][..., 3] == 255) | (L["earR"][..., 3] == 255) \
        | (L["tuft"][..., 3] == 255)
    res = {"collar-front_px": int(cf_mask.sum())}
    arms = {}
    for arm in ("armL", "armR"):
        A = new[arm]
        names_a, lab_a, cols_a = classify(A, PALETTES[arm])
        m = A[..., 3] > 0
        t = L["torso"][..., 3] > 0
        edge = m & (np.roll(t, 1, 0) | np.roll(t, 1, 1) | np.roll(t, -1, 1)) & (np.arange(H)[:, None] < 670)
        ys, xs = np.nonzero(edge)
        # top-edge endpoints and arm axis
        i0, i1 = np.argmin(xs), np.argmax(xs)
        P0, P1 = np.array([xs[i0], ys[i0]], float), np.array([xs[i1], ys[i1]], float)
        mid = (P0 + P1) / 2
        half = np.linalg.norm(P1 - P0) / 2 + 1.0
        tang = (P1 - P0) / np.linalg.norm(P1 - P0)
        nrm = np.array([tang[1], -tang[0]])
        if nrm[1] > 0:
            nrm = -nrm                         # pointing up (out of the arm, into the collar)
        depth = 22.0

        def shape(X, Y, mid=mid, tang=tang, nrm=nrm, half=half):
            u = (X - mid[0]) * tang[0] + (Y - mid[1]) * tang[1]
            v = (X - mid[0]) * nrm[0] + (Y - mid[1]) * nrm[1]
            return (v >= -3) & ((u / half) ** 2 + (np.clip(v, 0, None) / depth) ** 2 <= 1)
        yy, xg = np.mgrid[0:H, 0:W]
        zone = cover & (A[..., 3] == 0) & (shape(xg, yy) | shape(xg + 0.5, yy) | shape(xg - 0.5, yy))
        known = lab_a >= 0
        F = sdf_fields(lab_a, len(names_a))
        Fz = harmonic(F, known, zone)
        a8, rgb8 = render(zone, shape, label_fn_from_fields(Fz, known | zone), cols_a)
        out = A.copy()
        z = zone & (a8 > 0)
        out[z, :3] = rgb8[z]; out[z, 3] = a8[z]
        Image.fromarray(out, "RGBA").save(outdir / f"{arm}.png")
        arms[arm] = out
        res[arm] = {"added_px": int(z.sum()), "bbox": bbox_of(z), "shoulder_ellipse": {
            "center": [round(float(v), 1) for v in mid], "half_width": round(float(half), 1), "depth": depth}}
    return cf, arms, res


# ============================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "work").mkdir(exist_ok=True)
    L = load_layers(IN)
    fill = Fill(L)
    fill.run()
    new = fill.new

    report = {"source": "art/v2/tori/step3-layers", "canvas": [W, H], "draw_order_back_to_front": STACK,
              "rule": "pixels are added only where the part is transparent in step 3 and some layer in front "
                      "of it is fully opaque (alpha 255) -> invisible in the rest pose", "parts": {}}
    for p in STACK:
        Image.fromarray(new[p], "RGBA").save(OUT / f"{p}.png")
        diag_sheet(new[p], fill.added[p], OUT / f"filled-{p}.png", fill.seam_mask[p])
        vis = L[p][..., 3] > 0
        changed = vis & (np.abs(new[p].astype(int) - L[p].astype(int)).max(-1) > 0)
        add = fill.added[p]
        report["parts"][p] = {
            "added_px": int(add.sum()),
            "added_bbox": bbox_of(add),
            "added_px_alpha_lt_255": int((add & (new[p][..., 3] < 255)).sum()),
            "method": fill.notes[p],
            "visible_px_changed": int(changed.sum()),
            "visible_px_changed_outside_listed_exceptions": int((changed & ~fill.seam_mask[p]).sum()),
            "hidden_check_added_px_not_fully_covered": int((add & ~hidden_zone(L, p)).sum()),
        }
        print(f"{p:8s} added={int(add.sum()):6d} visible_changed={int(changed.sum())}")

    report["exceptions"] = fill.exceptions
    # rest-pose recomposite check
    ref = np.array(Image.open(IN / "recomposite-white.png").convert("RGB")).astype(float)
    rest = {}
    for name, bg in (("white", (255, 255, 255)), ("green", GREEN), ("dark", DARK)):
        a, _ = compose({p: premul(L[p]) for p in STACK}, STACK, np.array(bg, float))
        b, _ = compose({p: premul(new[p]) for p in STACK}, STACK, np.array(bg, float))
        d8 = np.abs(np.round(a) - np.round(b))
        rest[name] = {"max": int(d8.max()), "mean": float(d8.mean()), "px_diff": int((d8.max(-1) > 0).sum())}
        if name == "white":
            dref = np.abs(np.round(b) - ref)
            rest["white_vs_step3_recomposite_png"] = {"max": int(dref.max()), "mean": float(dref.mean())}
    report["rest_pose_recomposite_diff_step4_vs_step3"] = rest
    print("rest:", json.dumps(rest))

    piv = pivots(L)
    report["provisional_pivots"] = piv
    report["geometry"] = json.loads(json.dumps(fill.geom, default=float))
    before = motion_sheet(L, piv, OUT / "motion-test-before.png", "STEP 3 layers (before fill) - motion test")
    after = motion_sheet(new, piv, OUT / "motion-test.png", "STEP 4 layers (filled) - motion test")
    report["motion_test"] = {"poses": [{"pose": a["pose"], "new_enclosed_hole_px_before": b["new_enclosed_hole_px"],
                                        "new_enclosed_hole_px_after": a["new_enclosed_hole_px"]} for a, b in zip(after, before)],
                             "note": "angles: + = counter-clockwise on screen; head tilt carries ears + tuft about "
                                     "the head pivot; enclosed holes = composite alpha < 0.9 not connected to "
                                     "the canvas border (gaps open to the outside are judged visually)"}

    # optional shoulder set
    cf, arms, ores = optional_shoulders(L, new, fill.geom, OUT / "optional", fill.seam_mask["torso"])
    stack_opt = ["tail", "legL", "legR", "torso", "pendant", "armL", "armR", "collar_front", "earL", "earR", "head", "tuft"]
    lay_opt = dict(new); lay_opt.update(arms); lay_opt["collar_front"] = cf
    a, _ = compose({p: premul(L[p]) for p in STACK}, STACK, np.array((255, 255, 255), float))
    b, _ = compose({p: premul(lay_opt[p]) for p in stack_opt}, stack_opt, np.array((255, 255, 255), float))
    d8 = np.abs(np.round(a) - np.round(b))
    ores["rest_pose_diff_vs_step3_with_optional_stack"] = {"max": int(d8.max()), "mean": float(d8.mean())}
    ores["stack"] = stack_opt
    ores["why"] = optional_shoulders.__doc__.strip()
    report["optional"] = ores
    opt_stats = motion_sheet_opt(lay_opt, stack_opt, piv, OUT / "optional" / "motion-test-arms-optional.png")
    report["optional"]["motion_test"] = opt_stats
    (OUT / "fill-report.json").write_text(json.dumps(report, indent=1))


def motion_sheet_opt(layers, stack, piv, path):
    """arm poses only, with the optional collar-front stack."""
    AR = (230, 570, 800, 860)
    return motion_sheet(layers, piv, path, "OPTIONAL stack: arms with shoulder ends + collar-front over the arms",
                        stack=stack, pose_list=[("rest", {}, AR),
                                                ("arms out 35", {"armL": -35, "armR": 35}, AR),
                                                ("arms in 35", {"armL": 35, "armR": -35}, AR)])


if __name__ == "__main__":
    main()
