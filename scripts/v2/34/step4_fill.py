#!/usr/bin/env python3
"""
STEP 4 (3/4 view) - Tori v2: fill hidden joint areas for motion (minimal, design-preserving).
Mirrors scripts/v2/step4_fill.py (front); its helpers are imported and reused (classify, harmonic SDF,
supersampled render, ellipse/circle fits, seam un-mix, motion-test compositor).

Input : art/v2/tori-34/step3-layers/<part>.png
Output: art/v2/tori-34/step4-filled/
          <part>.png (+ collar_front.png)   final step-4 layers, 1024x1024
          filled-<part>.png                 diagnostic (added px magenta, seam-exception px cyan)
          motion-test-before.png / motion-test.png   test poses from step-3 / step-4 layers
          fill-report.json

Stack (back -> front):
  tail legFar legNear armFar torso pendant armNear collar_front earFar earNear head tuft
  collar_front (front lesson, adopted directly here): byte copies of the torso's own opaque collar px
  over the NEAR shoulder, drawn right above armNear, so the near arm's rounded shoulder end can tuck
  under the collar.  Rest pose unchanged (copies of pixels that are visible anyway).

Principles (as front)
  * a pixel is added to layer P only where P is transparent and a layer in front of P is opaque at
    that pixel -> the rest composite is unchanged (verified: rest diff below);
  * new pixels use the layer's own flat region colours; region boundaries continued by harmonic SDF;
    silhouettes from fitted curves, 4x4 supersampled AA.
  * EXCEPTION (front lesson, reported): seam anti-alias mixes on internal boundaries are moved from the
    BACK part into the FRONT part (soft edge) so no dotted line stays behind when parts move.
Fills (3/4 specific geometry, see Fill34)
  torso   : pendant hole (harmonic SDF across collar/beige/cream bands); torso's near side continued
            under the near arm (quadratic from the collar's left end to the visible side below the arm,
            vertical tangent there); collar band carried up + tan neck under the head (front method).
  armFar  : (far arm is BEHIND the torso) a 14 px band behind the torso along its inner cut + rounded
            shoulder above its top edge, under torso / collar / head (distance-field silhouette, AA).
  armNear : rounded shoulder end under collar_front / head (half-ellipse cap on the top edge).
  legs    : hip end under the torso: visible sides continued straight up + half-ellipse cap (front).
  earNear : ear bottom completed behind the head (ellipse fits of silhouette + inner oval, front).
  earFar  : solid disc completed behind the head (ellipse fit of the visible silhouette; 2-tone SDF).
  head    : head top under the tuft (circle fit left/right of the tuft; stripes run on by SDF).
  tail    : stem behind the near arm + root behind the torso: convex hull of the visible stem pieces
            (left of the arm, the sliver under the hand, the cream underside) limited to 40 px behind
            the torso side; bands by harmonic SDF so the dark stripe joins across the arm.
"""
import importlib.util, json, pathlib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("front4", ROOT / "scripts/v2/step4_fill.py")
F4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F4)

IN = ROOT / "art/v2/tori-34/step3-layers"
OUT = ROOT / "art/v2/tori-34/step4-filled"
H = W = 1024
STACK3 = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "earFar", "earNear", "head", "tuft"]
STACK = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "collar_front",
         "earFar", "earNear", "head", "tuft"]
GREEN, DARK = F4.GREEN, F4.DARK
TAN, SHADE = (234, 190, 129), (206, 147, 98)
PALETTES = {
    "torso": {"cream": (249, 243, 218), "tan": TAN, "collar_dark": (140, 92, 62),
              "collar_light": (153, 112, 80), "beige": (232, 197, 167), "shade": SHADE},
    "tail": {"tan": TAN, "shade": SHADE, "dark": (115, 79, 57), "cream": (236, 205, 177)},
    "legNear": {"tan": TAN, "shade": SHADE}, "legFar": {"tan": TAN, "shade": SHADE},
    "armNear": {"tan": TAN, "shade": SHADE}, "armFar": {"tan": TAN, "shade": SHADE},
    "earNear": {"tan": TAN, "inner": (209, 155, 108)},
    "earFar": {"main": (209, 155, 108), "crescent": (204, 149, 103)},
    "head": {"tan": TAN, "stripe": (117, 82, 60), "cream": (249, 243, 218), "band": (236, 203, 176),
             "eye": (79, 35, 19)},
}
# the front module's helpers read these globals
F4.STACK = STACK
F4.PALETTES = PALETTES
yy, xx = np.mgrid[0:H, 0:W]


def load_layers():
    L = {p: np.array(Image.open(IN / f"{p}.png").convert("RGBA")) for p in STACK3}
    return L


def make_collar_front(L3, torso_new, seam_torso):
    """byte copies of the torso's opaque collar px over the near shoulder (after the seam un-mix, so
    collar_front holds the same clean pixels the torso shows there)."""
    names, lab, cols = F4.classify(L3["torso"], PALETTES["torso"])
    collar = np.isin(lab, [names.index("collar_dark"), names.index("collar_light")])
    cf_mask = collar & (xx < 470) & (L3["torso"][..., 3] == 255) & ~seam_torso   # (front: same exclusion)
    cf = np.zeros_like(L3["torso"])
    cf[cf_mask] = torso_new[cf_mask]
    # the collar's rounded left END keeps its anti-aliased outline: its soft (0 < alpha < 255) silhouette
    # px are MOVED from the torso into collar_front (nothing lies between torso and collar_front there at
    # rest -> rest composite unchanged), otherwise a raised near arm would show a hard, jagged collar end
    ta = torso_new[..., 3]
    soft = (ta > 0) & (ta < 255) & ndi.binary_dilation(cf_mask, iterations=2) & (xx < 470) & (yy > 605) & \
        (yy < 670) & ~seam_torso
    cf[soft] = torso_new[soft]
    return cf, soft


def dist_render(zone, base_mask, R, label_fn, colors):
    """extension = pixels within R px of base_mask; coverage from the distance field (AA ~1 px)."""
    d = ndi.distance_transform_edt(~base_mask)
    cov = np.clip(R + 0.5 - d, 0, 1)
    a8 = np.zeros((H, W), np.uint8); rgb8 = np.zeros((H, W, 3), np.uint8)
    z = zone & (cov > 0)
    ys, xs = np.nonzero(z)
    lab = label_fn(xs.astype(float), ys.astype(float))
    a8[z] = np.round(cov[z] * 255).astype(np.uint8)
    rgb8[z] = np.round(colors[lab]).astype(np.uint8)
    return a8, rgb8


class Fill34(F4.Fill):
    SEAM_PAIRS = [("torso", "head"), ("tail", "armNear"), ("tail", "torso"), ("head", "tuft"),
                  ("earNear", "head"), ("earFar", "head"), ("torso", "armNear"), ("armFar", "torso"),
                  ("armFar", "head"), ("legNear", "torso"), ("legFar", "torso"), ("torso", "pendant")]

    # ------------------------------------------------------------ torso
    def torso(self):
        p = "torso"
        T = self.L[p]
        names, lab, cols = F4.classify(T, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        known = lab >= 0
        F = F4.sdf_fields(lab, len(names))
        # (1) pendant hole: harmonic SDF (band boundaries continued across the hole)
        hole = hid & (self.L["pendant"][..., 3] == 255)
        Fz = F4.harmonic(F, known, hole)
        a8, rgb8 = F4.render(hole, None, F4.label_fn_from_fields(Fz, known | hole), cols)
        self.put(p, a8, rgb8, "pendant hole: collar / beige / cream band boundaries continued across the hole "
                              "(harmonic SDF), flat region colours, 4x4 AA")
        # (2) under the near arm: torso near side continued from the collar's left end to the visible side
        al = T[..., 3]
        rows = range(826, 842)
        xb = float(np.median([np.nonzero(al[y] >= 128)[0].min() for y in rows])) - 0.5
        yb = 834.0
        crow = [y for y in range(616, 660) if (al[y] >= 128).any()]
        # collar left end: leftmost opaque torso px in the collar rows; top anchor just inside it at
        # the collar's bottom-left
        xt = float(min(np.nonzero(al[y] >= 128)[0].min() for y in crow)) + 4.0
        yt = 648.0
        self.geom = {"torso_side_near": [xt, yt, xb, yb]}
        cN = lambda Y: xb + (xt - xb) * (np.clip(yb - Y, 0, None) / (yb - yt)) ** 2
        fn = lambda X, Y: X >= cN(Y) - 0.5
        zone = hid & (self.L["armNear"][..., 3] == 255) & (fn(xx + 0.5, yy) | fn(xx - 0.5, yy)) & (yy >= 640)
        Fz = F4.harmonic(F, known, zone)
        a8, rgb8 = F4.render(zone, fn, F4.label_fn_from_fields(Fz, known | zone), cols)
        self.put(p, a8, rgb8, f"under armNear: torso near side continued as a quadratic from the collar's left end "
                              f"({xt:.0f},{yt:.0f}) to the visible side below the arm ({xb:.1f},{yb:.0f}, vertical "
                              f"tangent); collar-shadow / tan regions by harmonic SDF")
        # (3) under the head: collar band carried up + neck (front method, 3/4 ranges)
        self.neck34(names, lab, cols)

    def neck34(self, names, lab, cols):
        p = "torso"
        T = self.new[p]
        a = T[..., 3]
        hid = F4.hidden_zone(self.L, p) & (T[..., 3] == 0)
        collar = np.isin(lab, [names.index("collar_dark"), names.index("collar_light")])
        xs_b, ys_b, top_vis = [], [], {}
        for x in range(368, 685):
            col = np.nonzero(a[600:700, x] > 0)[0]
            if len(col) == 0:
                continue
            top_vis[x] = 600 + col.min()
            cc = np.nonzero(collar[600:700, x])[0]
            if len(cc):
                xs_b.append(x); ys_b.append(600 + cc.max() + 0.5)
        poly = np.polyfit(np.array(xs_b, float), np.array(ys_b, float), 4)
        ybot = lambda X: np.polyval(poly, X)
        thick = max(ybot(x) - top_vis[x] for x in top_vis if 395 <= x <= 655)
        Tk = float(np.ceil(thick)) + 14.0
        xle = min(x for x in top_vis if collar[top_vis[x]:top_vis[x] + 4, x].any()) - 0.5
        xre = max(x for x in top_vis if collar[top_vis[x]:top_vis[x] + 4, x].any()) + 0.5
        NECK_IN, NECK_H, RAD = 12.0, 40.0, 18.0
        self.geom["collar"] = {"bottom_poly": [float(v) for v in poly], "thickness": Tk, "ends": [xle, xre],
                               "neck_inset": NECK_IN, "neck_height": NECK_H}
        ytop = lambda X: ybot(X) - Tk
        colcol = np.zeros((W, 3)); have = np.zeros(W, bool)
        for x, y in top_vis.items():
            seg = T[y + 3:y + 11, x]
            ok = (seg[:, 3] == 255) & collar[y + 3:y + 11, x]
            if ok.sum() >= 2:
                colcol[x] = np.median(seg[ok, :3].astype(float), 0); have[x] = True
        xs_h = np.nonzero(have)[0]
        for x in range(W):
            if not have[x]:
                colcol[x] = colcol[xs_h[np.argmin(np.abs(xs_h - x))]]
        colcol = ndi.median_filter(colcol, size=(9, 1), mode="nearest")
        tan = cols[names.index("tan")]
        ys, xs = np.nonzero(hid)
        sel = (ys < 665) & (ys > 540) & (xs > 340) & (xs < 710)
        zone = np.zeros((H, W), bool); zone[ys[sel], xs[sel]] = True
        x0, y0, x1, y1 = F4.bbox_of(zone)
        X, Y = F4.sub_coords((x0, y0, x1, y1))
        in_collar = (X >= xle) & (X <= xre) & (Y >= ytop(X))
        neck_top = ytop(np.clip(X, xle, xre)) - NECK_H
        in_neck = (X >= xle + NECK_IN) & (X <= xre - NECK_IN) & (Y >= neck_top) & (Y < ytop(X))
        mid = (xle + xre) / 2
        for cx in (xle + NECK_IN + RAD, xre - NECK_IN - RAD):
            cy = ytop(cx) - NECK_H + RAD
            out_corner = (Y < cy) & (((X < cx) & (cx < mid)) | ((X > cx) & (cx > mid))) & \
                         ((X - cx) ** 2 + (Y - cy) ** 2 > RAD ** 2)
            in_neck &= ~out_corner
        inside = in_collar | in_neck
        xi = np.clip(np.round(X).astype(int), 0, W - 1)
        sub_col = np.where(in_collar[..., None], colcol[xi], tan)
        h, w = y1 - y0, x1 - x0
        cov = F4.pool(inside.astype(float), h, w)
        rgb = F4.pool(sub_col * inside[..., None], h, w) / np.maximum(cov, 1e-9)[..., None]
        a8 = np.zeros((H, W), np.uint8); rgb8 = np.zeros((H, W, 3), np.uint8)
        al = np.round(cov * 255).astype(np.uint8); al[~zone[y0:y1, x0:x1]] = 0
        a8[y0:y1, x0:x1] = al
        rgb8[y0:y1, x0:x1] = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        self.put(p, a8, rgb8, f"under the head: collar band carried straight up column-wise to collar top = fitted "
                              f"collar bottom - {Tk:.0f} px; tan neck ({NECK_H:.0f} px, inset {NECK_IN:.0f}, rounded)")

    # ------------------------------------------------------------ arms
    def arm_far(self):
        p = "armFar"
        A = self.L[p]
        names, lab, cols = F4.classify(A, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        m = A[..., 3] > 0
        known = lab >= 0
        F = F4.sdf_fields(lab, len(names))
        R = 16.0
        # bottom limit = the hand's bottom contour continued straight to the left (line fitted to the
        # lowest opaque px of the columns just right of the inner cut), so the extension ends flush with
        # the hand instead of leaving a square corner / a lump below it
        al = A[..., 3]
        cx0 = int(np.nonzero((al[796:806] >= 128).any(0))[0].min())
        bx = np.arange(cx0 + 2, cx0 + 22)
        by = np.array([np.nonzero(al[:, x] >= 128)[0].max() + 0.5 for x in bx], float)
        lc = np.polyfit(bx, by, 1)
        self.geom["armFar_band"] = {"R": R, "hand_bottom_line": [float(v) for v in lc]}
        d = ndi.distance_transform_edt(~m)
        lim = np.polyval(lc, xx)
        # band width tapers to 0 over the last R px above that line (quarter-ellipse corner, no square end)
        t = np.clip((lim - yy) / R, 0, 1)
        Reff = R * np.sqrt(1 - (1 - t) ** 2)
        zone = hid & (d <= Reff + 0.5) & (yy <= lim + 0.5) & (yy > 600)
        Fz = F4.harmonic(F, known, zone)
        lf = F4.label_fn_from_fields(Fz, known | zone)
        a8, rgb8 = dist_render(zone, m, R, lf, cols)
        cov = np.clip(Reff + 0.5 - d, 0, 1)
        a8 = np.where(zone, np.minimum(a8, np.round(cov * 255)), a8).astype(np.uint8)
        rgb8[a8 == 0] = 0
        self.put(p, a8, rgb8, f"far arm (behind the torso): {R:.0f} px band behind the torso along its inner cut "
                              "and over its top edge (under collar / head), distance-field silhouette with ~1 px AA; "
                              "inner shading stripe / tan continued by harmonic SDF")

    def shoulder_near(self):
        p = "armNear"
        A = self.L[p]
        names, lab, cols = F4.classify(A, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        m = A[..., 3] > 0
        cover = (self.L["collar_front"][..., 3] == 255) | (self.L["head"][..., 3] == 255)
        top = m & ndi.binary_dilation(cover, iterations=1) & (yy < 680)
        ys, xs = np.nonzero(top)
        i0, i1 = np.argmin(xs), np.argmax(xs)
        P0, P1 = np.array([xs[i0], ys[i0]], float), np.array([xs[i1], ys[i1]], float)
        mid = (P0 + P1) / 2
        half = np.linalg.norm(P1 - P0) / 2 + 1.0
        tang = (P1 - P0) / np.linalg.norm(P1 - P0)
        nrm = np.array([tang[1], -tang[0]])
        if nrm[1] > 0:
            nrm = -nrm
        depth = 22.0
        self.geom["armNear_shoulder"] = {"center": mid.round(1).tolist(), "half_width": round(float(half), 1),
                                         "depth": depth, "edge_ends": [P0.tolist(), P1.tolist()]}

        def shape(X, Y):
            u = (X - mid[0]) * tang[0] + (Y - mid[1]) * tang[1]
            v = (X - mid[0]) * nrm[0] + (Y - mid[1]) * nrm[1]
            return (v >= -3) & ((u / half) ** 2 + (np.clip(v, 0, None) / depth) ** 2 <= 1)
        zone = hid & cover & (shape(xx, yy) | shape(xx + 0.5, yy) | shape(xx - 0.5, yy))
        known = lab >= 0
        F = F4.sdf_fields(lab, len(names))
        Fz = F4.harmonic(F, known, zone)
        a8, rgb8 = F4.render(zone, shape, F4.label_fn_from_fields(Fz, known | zone), cols)
        self.put(p, a8, rgb8, f"near arm: rounded shoulder end (half-ellipse, half-width {half:.0f}, depth {depth:.0f}) "
                              "on its top edge, only under collar_front / head; colours by harmonic SDF")

    # ------------------------------------------------------------ legs
    def leg34(self, p):
        L = self.L[p]
        names, lab, cols = F4.classify(L, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        al = L[..., 3]
        ys_ = np.nonzero((al >= 128).any(1))[0]
        ytop = ys_.min()
        rows = {y: np.nonzero(al[y] >= 128)[0] for y in range(ytop, ytop + 40)}
        if p == "legNear":
            # outer side = silhouette at the top of the cut; inner side = gap edge lower down
            xl = min(r.min() for y, r in rows.items() if y <= ytop + 3 and len(r)) - 0.5
            xr = max(r.max() for y, r in rows.items() if y >= ytop + 22 and len(r)) + 0.5
        else:
            xl = min(r.min() for y, r in rows.items() if y >= ytop + 34 and len(r)) - 0.5
            xr = max(r.max() for y, r in rows.items() if y <= ytop + 3 and len(r)) + 0.5
        yc = float(ytop)
        cx, ax, by = (xl + xr) / 2, (xr - xl) / 2, 40.0
        self.geom[p] = {"hip_cap_ellipse": [cx, yc, ax, by], "sides": [xl, xr]}

        def shape(X, Y):
            cap = (Y <= yc) & (((X - cx) / ax) ** 2 + ((Y - yc) / by) ** 2 <= 1)
            body = (Y > yc) & (Y <= 935) & (X >= xl) & (X <= xr)
            return cap | body
        zone = hid & (shape(xx - 0.5, yy) | shape(xx + 0.5, yy) | shape(xx, yy - 0.5) | shape(xx, yy + 0.5))
        tan_k = names.index("tan")
        col_lab = np.full(W, tan_k)
        for xc in range(int(xl), int(xr) + 2):
            ys2 = np.nonzero(lab[:, xc] >= 0)[0]
            ys2 = ys2[ys2 > 860]
            if len(ys2):
                y2 = min(ys2.min() + 2, H - 1)
                col_lab[xc] = lab[y2, xc] if lab[y2, xc] >= 0 else lab[ys2.min(), xc]
        col_lab = ndi.median_filter(col_lab, size=5, mode="nearest")
        labf = lambda X, Y: col_lab[np.clip(np.round(X).astype(int), 0, W - 1)]
        a8, rgb8 = F4.render(zone, shape, labf, cols)
        self.put(p, a8, rgb8, f"hip end: visible leg sides continued straight up and closed by a half-ellipse cap "
                              f"(center {cx:.0f},{yc:.0f}, a={ax:.0f}, b={by:.0f}) under the torso; tan / shading carried "
                              "up column-wise")

    # ------------------------------------------------------------ ears
    def ear_far(self):
        p = "earFar"
        L = self.L[p]
        names, lab, cols = F4.classify(L, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        head = self.L["head"][..., 3] > 0
        x, y = F4.silhouette_pts(L[..., 3])
        keep = ~F4.near(head, 6)[y.astype(int), x.astype(int)]
        outer = F4.fit_ellipse(x[keep], y[keep])
        zone = hid & (outer(xx, yy) < 0.02)
        known = lab >= 0
        F = F4.sdf_fields(lab, len(names))
        Fz = F4.harmonic(F, known, zone)
        a8, rgb8 = F4.render(zone, lambda X, Y: outer(X, Y) < 0, F4.label_fn_from_fields(Fz, known | zone), cols)
        self.put(p, a8, rgb8, "far ear disc completed behind the head: ellipse fitted to the visible silhouette (away "
                              "from the head); main / crescent tone by harmonic SDF; 4x4 AA")

    # ------------------------------------------------------------ head under the tuft
    def head34(self):
        p = "head"
        L = self.L[p]
        names, lab, cols = F4.classify(L, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        tuft = self.L["tuft"][..., 3] > 0
        ears = (self.L["earNear"][..., 3] > 0) | (self.L["earFar"][..., 3] > 0)
        x, y = F4.silhouette_pts(L[..., 3])
        tx = np.nonzero(tuft.any(0))[0]
        sel = (y < 215) & (((x > tx.min() - 45) & (x < tx.min() - 2)) | ((x > tx.max() + 2) & (x < tx.max() + 45)))
        sel &= ~F4.near(tuft | ears, 6)[y.astype(int), x.astype(int)]
        cx, cy, r = F4.fit_circle(x[sel], y[sel])
        self.geom["head_top_circle"] = [cx, cy, r]
        shape = lambda X, Y: (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
        zone = hid & tuft & ((xx - cx) ** 2 + (yy - cy) ** 2 <= (r + 1) ** 2)
        known = lab >= 0
        F = F4.sdf_fields(lab, len(names))
        Fz = F4.harmonic(F, known, zone)
        a8, rgb8 = F4.render(zone, shape, F4.label_fn_from_fields(Fz, known | zone), cols)
        self.put(p, a8, rgb8, f"head top under the tuft: circle fitted to the visible head top left/right of the tuft "
                              f"(c=({cx:.1f},{cy:.1f}), r={r:.1f}); stripes run on by harmonic SDF")

    # ------------------------------------------------------------ tail
    def tail34(self):
        p = "tail"
        L = self.L[p]
        names, lab, cols = F4.classify(L, PALETTES[p])
        hid = F4.hidden_zone(self.L, p)
        al = L[..., 3]
        win = (xx > 255) & (xx < 400) & (yy > 600) & (yy < 890)
        ys, xs = np.nonzero((al > 0) & win)
        hull = ConvexHull(np.c_[xs, ys].astype(float))
        eq = hull.equations
        tside = self.geom["torso_side_near"]
        xt, yt, xb, yb = tside
        cN = lambda Y: xb + (xt - xb) * (np.clip(yb - Y, 0, None) / (yb - yt)) ** 2
        ROOT_IN = 40.0

        def shape(X, Y):
            inside = np.all(eq[:, 0, None] * X.ravel() + eq[:, 1, None] * Y.ravel() + eq[:, 2, None] <= 0, 0)
            inside = inside.reshape(X.shape)
            # root: band behind the torso's near side over the height where the tail meets the body
            # (from y 680 down: the stem passes straight behind the body, no background slit between
            #  stem and body when the near arm lifts)
            root = (X >= 300) & (X <= cN(Y) + ROOT_IN) & (Y >= 680) & (Y <= 882)
            return (inside & (X <= cN(Y) + ROOT_IN)) | root
        zone = hid & (shape(xx - 0.5, yy) | shape(xx + 0.5, yy) | shape(xx, yy - 0.5) | shape(xx, yy + 0.5)) & win
        # ordinal band value across the stem (front method): outer orange (between the dark stripe and the
        # arm / body) 0 < dark stripe 1 < inner orange 2 < cream underside 3; solved harmonically so the
        # dark stripe left of the arm joins the dark sliver under the hand
        kt, ks, kd, kc = (names.index(n) for n in ("tan", "shade", "dark", "cream"))
        orange = np.isin(lab, [kt, ks])
        cc, n = ndi.label(orange & win)
        # outer band = orange component right of the visible dark stripe, next to the arm, above the hand
        probe = orange & win & (yy < 720) & (xx > 312) & (xx < 350)
        outer_id = np.bincount(cc[probe][cc[probe] > 0]).argmax()
        U = np.full((H, W), np.nan)
        U[orange & win] = 2.0
        U[cc == outer_id] = 0.0
        U[(lab == kd) & win] = 1.0
        U[(lab == kc) & win] = 3.0
        known = ~np.isnan(U)
        Uz = F4.harmonic(np.nan_to_num(U)[None], known, zone)[0]
        defined = known | zone
        band_lab = np.array([ks, kd, ks, kc])

        def labf(X, Y):
            u = F4.sample_fields(Uz[None], defined, X, Y)[0]
            return band_lab[np.clip(np.round(u), 0, 3).astype(int)]
        a8, rgb8 = F4.render(zone, shape, labf, cols)
        self.geom["tail"] = {"hull_window": [255, 600, 400, 890], "root_behind_torso_px": ROOT_IN}
        self.put(p, a8, rgb8, "tail stem behind the near arm + root 40 px behind the torso's near side: convex hull of "
                              "the visible stem pieces around the arm; bands by harmonic ORDINAL field (outer orange < "
                              "dark stripe < inner orange < cream, as front) so the dark stripe left of the arm joins "
                              "the dark sliver under the hand")

    def run(self):
        self.geom = {}
        self.exceptions = {}
        self.torso()
        self.seam_unmix()
        cf, moved = make_collar_front(self.L, self.new["torso"], self.seam_mask["torso"])
        self.L["collar_front"] = cf
        self.new["collar_front"] = cf.copy()
        self.new["torso"][moved] = 0
        self.seam_mask["torso"] |= moved
        self.seam_mask["collar_front"] |= moved
        self.exceptions["torso->collar_front"] = {
            "px_moved": int(moved.sum()), "bbox": F4.bbox_of(moved),
            "why": "soft anti-aliased outline px of the collar's left end moved from torso to collar_front (rest "
                   "composite identical: no layer between them there)"}
        self.arm_far()
        self.shoulder_near()
        self.leg34("legNear"); self.leg34("legFar")
        self.ear("earNear"); self.ear_far()
        self.head34()
        self.tail34()
        for p in STACK:
            if not self.notes[p]:
                self.notes[p].append("nothing hidden to fill -> copied unchanged")


# ============================================================== motion test
def pivots(L, geom):
    a = {p: L[p][..., 3] for p in STACK}
    piv = {}
    for arm, cov in (("armNear", (a["collar_front"] == 255) | (a["head"] == 255)),
                     ("armFar", (a["torso"] == 255) | (a["head"] == 255))):
        m = a[arm] > 0
        edge = m & ndi.binary_dilation(cov, iterations=1) & (yy < 672)
        ys, xs = np.nonzero(edge)
        piv[arm] = (float(xs.mean()), float(ys.mean()) + 8)
    hx = 526
    hb = np.nonzero(a["head"][:, hx] > 0)[0].max()
    piv["head"] = (float(hx), float(hb) - 8)
    for ear in ("earNear", "earFar"):
        m = a[ear] > 0
        edge = m & ndi.binary_dilation(a["head"] > 0, iterations=1)
        ys, xs = np.nonzero(edge)
        piv[ear] = (float(xs.mean()), float(ys.mean()))
    m = a["tuft"] > 0
    edge = m & ndi.binary_dilation(a["head"] > 0, iterations=1)
    ys, xs = np.nonzero(edge)
    piv["tuft"] = (float(xs.mean()), float(ys.max()))
    for leg in ("legNear", "legFar"):
        cx, yc, _, _ = geom[leg]["hip_cap_ellipse"]
        piv[leg] = (float(cx), float(yc))
    piv["tail"] = (385.0, 835.0)
    ys, xs = np.nonzero(a["pendant"] > 0)
    top = ys.min()
    piv["pendant"] = (float(xs[ys == top].mean()), float(top) + 2)
    piv["torso"] = (526.0, 890.0)
    piv["collar_front"] = piv["torso"]
    return {k: (round(v[0], 1), round(v[1], 1)) for k, v in piv.items()}


def poses():
    AR, HD, EA, TU, LG, TL, PE = (250, 580, 800, 870), (230, 50, 830, 700), (260, 90, 760, 290), \
        (400, 50, 620, 230), (320, 820, 730, 1000), (60, 520, 480, 920), (520, 600, 690, 800)
    # step-4 motion convention (front step 4): + = counter-clockwise on screen
    return [
        ("rest", {}, (60, 50, 830, 990)),
        ("arms out 35 (armNear ccw-35 = out, armFar +35 = out)", {"armNear": -35, "armFar": 35}, AR),
        ("arms in 35 (armNear +35, armFar -35)", {"armNear": 35, "armFar": -35}, AR),
        ("head +8 (ccw; ears+tuft follow)", {"head": 8}, HD),
        ("head -8 (cw; ears+tuft follow)", {"head": -8}, HD),
        ("ears +15", {"earNear": 15, "earFar": 15}, EA),
        ("ears -15", {"earNear": -15, "earFar": -15}, EA),
        ("tuft +10", {"tuft": 10}, TU),
        ("tuft -10", {"tuft": -10}, TU),
        ("legs apart 18 (legNear -18, legFar +18)", {"legNear": -18, "legFar": 18}, LG),
        ("legs together 18 (legNear +18, legFar -18)", {"legNear": 18, "legFar": -18}, LG),
        ("tail +8 (ccw)", {"tail": 8}, TL),
        ("tail -8 (cw)", {"tail": -8}, TL),
        ("pendant +10", {"pendant": 10}, PE),
        ("pendant -10", {"pendant": -10}, PE),
    ]


def pose_layers(pms, piv, rot, cache, stack):
    cur = {}
    for p in stack:
        if "head" in rot and p in ("earNear", "earFar", "tuft"):
            pm = pms[p]
            if rot.get(p, 0):
                pm = F4.rotate_pm(pm, piv[p], rot[p])
            cur[p] = F4.rotate_pm(pm, piv["head"], rot["head"])
        else:
            key = (p, rot.get(p, 0))
            if key not in cache:
                cache[key] = F4.rotate_pm(pms[p], piv.get(p, (512, 512)), rot.get(p, 0))
            cur[p] = cache[key]
    return cur


def motion_sheet(layers, piv, path, title, stack):
    pms = {p: F4.premul(layers[p]) for p in stack}
    cache, rows, stats = {}, [], []
    TW = 560
    _, acc_rest = F4.compose(pms, stack, np.zeros(3))
    _, hole_rest = F4.holes(acc_rest)
    for name, rot, box in poses():
        cur = pose_layers(pms, piv, rot, cache, stack)
        ims = []
        for bg in (GREEN, DARK):
            img, acc = F4.compose(cur, stack, np.array(bg, float))
            x0, y0, x1, y1 = box
            im = Image.fromarray(np.round(img[y0:y1, x0:x1]).clip(0, 255).astype(np.uint8))
            ims.append(im.resize((TW, int(im.height * TW / im.width)), Image.LANCZOS))
        _, hm = F4.holes(acc)
        nh = int((hm & ~hole_rest).sum())
        stats.append({"pose": name, "new_enclosed_hole_px": nh})
        rows.append((f"{name}   [new enclosed holes vs rest: {nh} px]", ims))
    hgt = sum(r[1][0].height + 34 for r in rows) + 50
    sheet = Image.new("RGB", (2 * TW + 30, hgt), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), title, fill=(0, 0, 0), font=F4.font(20))
    y = 50
    for label, ims in rows:
        d.text((10, y), label, fill=(0, 0, 0), font=F4.font(15))
        for j, im in enumerate(ims):
            sheet.paste(im, (10 + j * (TW + 10), y + 22))
        y += ims[0].height + 34
    sheet.save(path)
    return stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L3 = load_layers()
    L = dict(L3)
    L["collar_front"] = np.zeros_like(L3["torso"])      # built inside Fill34.run (after the seam un-mix)
    fill = Fill34(L)
    fill.run()
    new = fill.new
    report = {"source": "art/v2/tori-34/step3-layers", "canvas": [W, H], "draw_order_back_to_front": STACK,
              "rule": "pixels are added only where the part is transparent in step 3 and some layer in front of it "
                      "is fully opaque (alpha 255) -> invisible in the rest pose",
              "collar_front": "byte copies of the torso's opaque collar px with x < 470 (near shoulder), drawn above "
                              "armNear (front lesson); %d px" % int((L['collar_front'][..., 3] > 0).sum()),
              "parts": {}}
    for p in STACK:
        Image.fromarray(new[p], "RGBA").save(OUT / f"{p}.png")
        F4.diag_sheet(new[p], fill.added[p], OUT / f"filled-{p}.png", fill.seam_mask[p])
        vis = L[p][..., 3] > 0
        changed = vis & (np.abs(new[p].astype(int) - L[p].astype(int)).max(-1) > 0)
        add = fill.added[p]
        report["parts"][p] = {
            "added_px": int(add.sum()), "added_bbox": F4.bbox_of(add),
            "method": fill.notes[p],
            "visible_px_changed": int(changed.sum()),
            "visible_px_changed_outside_listed_exceptions": int((changed & ~fill.seam_mask[p]).sum()),
            "hidden_check_added_px_not_fully_covered": int((add & ~F4.hidden_zone(L, p)).sum()),
        }
        print(f"{p:12s} added={int(add.sum()):6d} visible_changed={int(changed.sum())}")
    report["exceptions"] = fill.exceptions
    ref = np.array(Image.open(IN / "recomposite-white.png").convert("RGB")).astype(float)
    rest = {}
    for name, bg in (("white", (255, 255, 255)), ("green", GREEN), ("dark", DARK)):
        a, _ = F4.compose({p: F4.premul(L3[p]) for p in STACK3}, STACK3, np.array(bg, float))
        b, _ = F4.compose({p: F4.premul(new[p]) for p in STACK}, STACK, np.array(bg, float))
        d8 = np.abs(np.round(a) - np.round(b))
        rest[name] = {"max": int(d8.max()), "mean": float(d8.mean()), "px_diff": int((d8.max(-1) > 0).sum())}
        if name == "white":
            dref = np.abs(np.round(b) - ref)
            rest["white_vs_step3_recomposite_png"] = {"max": int(dref.max()), "mean": float(dref.mean())}
    report["rest_pose_recomposite_diff_step4_vs_step3"] = rest
    print("rest:", json.dumps(rest))
    piv = pivots(new, fill.geom)
    report["provisional_pivots"] = piv
    report["geometry"] = json.loads(json.dumps(fill.geom, default=float))
    Lb = dict(L3); Lb["collar_front"] = np.zeros_like(L3["torso"])
    before = motion_sheet(Lb, piv, OUT / "motion-test-before.png", "3/4 STEP 3 layers (before fill) - motion test", STACK)
    after = motion_sheet(new, piv, OUT / "motion-test.png", "3/4 STEP 4 layers (filled) - motion test", STACK)
    report["motion_test"] = {"poses": [{"pose": a["pose"], "new_enclosed_hole_px_before": b["new_enclosed_hole_px"],
                                        "new_enclosed_hole_px_after": a["new_enclosed_hole_px"]} for a, b in zip(after, before)],
                             "note": "step-4 test convention: + = counter-clockwise on screen (step 5 rig: + = clockwise)"}
    (OUT / "fill-report.json").write_text(json.dumps(report, indent=1))
    for a, b in zip(after, before):
        print(f"  {a['pose'][:50]:50s} holes before={b['new_enclosed_hole_px']:6d} after={a['new_enclosed_hole_px']}")


if __name__ == "__main__":
    main()
