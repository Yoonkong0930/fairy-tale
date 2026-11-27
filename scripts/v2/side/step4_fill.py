#!/usr/bin/env python3
"""
STEP 4 - Tori v2 SIDE view: fill hidden joint areas for motion (walking), minimal & design-preserving.

Input : art/v2/tori-side/step3-layers/<part>.png, art/v2/tori-side/step1-fix/work/body_under_arm.png
        (+ the step-1 arm renderer, scripts/v2/side/arm_render.py, for the arm's shoulder end)
Output: art/v2/tori-side/step4-filled/
          <part>.png (incl. collar_front.png)  filled layers, 1024 canvas, same coordinates
          filled-<part>.png  diagnostic: added hidden px magenta, seam-exception px cyan
          motion-test-before.png / motion-test.png   poses from step-3 / step-4 layers (#6BBF59 | #202020)
          fill-report.json
Front principles kept (scripts/v2/step4_fill.py, helpers imported): new pixels only where the part is
transparent and a layer in FRONT of it is opaque (alpha 255) -> rest pose unchanged; flat region
colours of the layer itself; region boundaries continued by harmonic SDF; silhouettes from fitted
curves; 4x4 supersampled AA.  Front lessons applied:
  * collar_front (torso's own collar pixels over the arm's shoulder, byte copies; the collar's bottom
    AA row un-mixed into it) so the arm gets a rounded, hidden shoulder end;
  * seam un-mix: AA mixes left in a BACK part's boundary row are moved into the FRONT part as a soft
    edge (back px -> its clean local colour), so moving parts leave no dotted line;
  * tuft: one flat fur colour, drawn behind the head, base extended (hidden) under the head.
Fills  (v2: near arm forward over the belly + far arm behind the torso)
  torso   : under the near arm = the arm-less torso (step-3 work/torso_full.png, de-whitened step-1 body);
            under the head = collar band carried up column-wise + rounded tan neck (front method);
            under the acorn = harmonic SDF of the torso's regions incl. a background class, so the
            belly front silhouette and the collar end continue under the acorn.
  armNear : shoulder end = the same AI arm silhouette rendered without the collar clip (rounded top),
            only where collar_front / head are opaque.
  armFar  : its hidden shoulder / upper arm behind the torso and the acorn = the complete step-1 far-arm render
            (own rounded top -> no cut base when it turns).
  legNear / legFar : rounded hip = convex hull of the leg's top rows and a disk around the hip pivot,
            under the torso (and, for the far leg, under the near leg); leg's own flat colour.
  tail    : root behind the torso: convex hull of the tail's cross-section at the junction and a disk about
            the tail pivot; bands by harmonic SDF.
  head    : under the ear: head ellipse interior, stripe / fur by harmonic SDF.
  tuft    : flat head-tan; base = convex hull of its bottom contact rows swept 26 px down, under the head.
"""
import importlib.util, json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("front4", ROOT / "scripts/v2/step4_fill.py")
F4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(F4)
from arm_render import render_arm  # noqa: E402,F401
from arm_render2 import render_arm2  # noqa: E402

IN = ROOT / "art/v2/tori-side/step3-layers"
S1 = ROOT / "art/v2/tori-side/step1-fix"
OUT = ROOT / "art/v2/tori-side/step4-filled"
H = W = 1024
STACK3 = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "tuft", "head", "earNear"]
STACK = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "collar_front", "tuft", "head", "earNear"]
GREEN, DARK = (0x6B, 0xBF, 0x59), (0x20, 0x20, 0x20)
PAL = {
    "torso": {"tan": (235, 191, 130), "cream": (249, 241, 220), "collar_light": (154, 113, 81),
              "collar_dark": (137, 92, 60), "band": (205, 148, 100), "beige": (234, 203, 174)},
    "tail": {"tan": (235, 190, 130), "shade": (205, 146, 98), "cream": (234, 205, 177), "dark": (112, 76, 54)},
    "head": {"tan": (235, 191, 130), "stripe": (113, 80, 58), "cream": (249, 242, 215), "band": (234, 203, 174),
             "shade": (207, 146, 98)},
    "legNear": {"tan": (235, 191, 130)}, "legFar": {"shade": (205, 147, 99)},
    "earNear": {"tan": (235, 191, 130), "inner": (207, 150, 104)},
    "pendant": {"olive": (111, 112, 68), "mint": (160, 225, 190)},
    "armNear": {"tan": (235, 191, 130), "shade": (207, 147, 97)},
    "armFar": {"fill": (205, 148, 100), "shade": (188, 132, 88)},
    "tuft": {"tan": (235, 191, 130)},
}
# provisional pivots (final ones in step 5)
PIV = {"legNear": (490.0, 896.0), "legFar": (628.0, 893.0), "tail": (458.0, 823.0), "armNear": (606.0, 660.0), "armFar": (690.0, 702.0),
       "head": (565.0, 618.0), "earNear": (478.0, 236.0), "tuft": (590.0, 170.0), "pendant": (679.0, 643.0),
       "torso": (560.0, 880.0)}
LEG_DISKS = {"legNear": [((490.0, 896.0), 54.0), ((505.0, 896.0), 44.0)], "legFar": [((628.0, 893.0), 48.0), ((650.0, 880.0), 38.0)]}
ROOT_R = 50.0
TUFT_SWEEP = 26.0


def hidden_zone(layers, stack, p):
    i = stack.index(p)
    front = np.zeros((H, W), bool)
    for q in stack[i + 1:]:
        front |= layers[q][..., 3] == 255
    return front & (layers[p][..., 3] == 0)


def hull_fn(pts):
    hull = ConvexHull(np.asarray(pts, float))
    eq = hull.equations

    def f(X, Y):
        return np.all(eq[:, 0, None, None] * X[None] + eq[:, 1, None, None] * Y[None] + eq[:, 2, None, None] <= 0, 0)
    return f


def disk_pts(c, r, n=72):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def any_cov(fn):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return fn(xx - 0.5, yy) | fn(xx + 0.5, yy) | fn(xx, yy - 0.5) | fn(xx, yy + 0.5) | fn(xx, yy)


class Fill:
    def __init__(self, L):
        L["collar_front"] = np.zeros_like(L["torso"])
        self.L = L
        self.new = {p: L[p].copy() for p in L}
        self.added = {p: np.zeros((H, W), bool) for p in L}
        self.seam = {p: np.zeros((H, W), bool) for p in L}
        self.notes = {p: [] for p in L}
        self.geom = {}
        self.exceptions = {}

    def hz(self, p):
        return hidden_zone(self.L, STACK, p) & (self.new[p][..., 3] == 0)

    def put(self, p, a8, rgb8, note):
        z = self.hz(p) & (a8 > 0)
        self.new[p][z, :3] = rgb8[z]; self.new[p][z, 3] = a8[z]
        self.added[p] |= z
        self.notes[p].append(f"{note} [{int(z.sum())} px]")
        return z

    # ------------------------------------------------------------ collar_front + arm shoulder
    def near_arm_full(self):
        """the near arm rendered WITHOUT the collar clip (its own rounded top), as in step 1 otherwise."""
        rep = json.loads((S1 / "fix-report.json").read_text())["arm"]
        pb = rep["band_bottom_poly"]
        tan = np.array(rep["colours"]["tan"], float); band = np.array(rep["colours"]["shade/band"], float)
        shade = np.array(rep["colours"].get("shade", [207, 147, 97]), float)
        params = dict(rep["transform"]); params["src"] = tuple(params["src"]); params["dst"] = tuple(params["dst"])
        A, C, _ = render_arm2(params, lambda X: np.full(X.shape, -1e9), lambda X: np.polyval(pb, np.clip(X, 458, 600)),
                              {"fill": tan, "shade": shade, "band": band}, (450, 560, 800, 900))
        return A, C

    def collar_front(self):
        T = self.L["torso"]
        Tf = np.asarray(Image.open(IN / "work/torso_full.png").convert("RGBA")).astype(float)
        names, lab, cols = F4.classify(T, PAL["torso"])
        collar = np.isin(lab, [names.index("collar_dark"), names.index("collar_light")])
        xx = np.arange(W)[None, :].repeat(H, 0); yy = np.arange(H)[:, None].repeat(W, 1)
        self.armfull = self.near_arm_full()
        Afull, Cfull = self.armfull
        cols_arm = np.nonzero((Afull[:700] > 0).any(0))[0]
        x0s, x1s = int(cols_arm.min()) - 12, int(cols_arm.max()) + 12
        span = (xx >= x0s) & (xx <= x1s) & (yy > 600) & (yy < 700)
        armA = self.L["armNear"][..., 3].astype(float) / 255
        core = collar & span & (T[..., 3] == 255)
        pure = core & (armA == 0)
        cf = np.zeros_like(T)
        cf[pure] = T[pure]
        # collar px partly covered by the step-1 arm + the collar's bottom AA rows: rest colour C (arm over torso,
        # step 3) = b * collar + (1-b) * U, U = what lies under collar_front once the arm has its full shoulder
        # (full arm over the arm-less torso); collar_front gets (c, b); torso px -> arm-less torso colour and the
        # arm px -> full arm (seam exceptions, rest preserved)
        Crest = armA[..., None] * self.L["armNear"][..., :3] + (1 - armA[..., None]) * T[..., :3]
        U = Afull[..., None] * Cfull + (1 - Afull[..., None]) * Tf[..., :3]
        below = span & ~core & (T[..., 3] == 255) & ndi.binary_dilation(core, iterations=3) & (yy > 640)
        aa = ((core & (armA > 0)) | below) & (Tf[..., 3] == 255)
        n, res = 0, []
        for y, x in zip(*np.nonzero(aa)):
            win = cf[max(0, y - 6):y + 1, max(0, x - 6):x + 7]
            ok = win[..., 3] == 255
            if ok.sum() < 2:
                win = T[max(0, y - 8):y + 1, max(0, x - 8):x + 9]
                ok = pure[max(0, y - 8):y + 1, max(0, x - 8):x + 9]
                if ok.sum() < 2:
                    continue
            cc = np.median(win[ok][:, :3].astype(float), 0)
            C = Crest[y, x]; u = U[y, x]
            d = cc - u
            bval = float(np.clip(np.dot(C - u, d) / max(np.dot(d, d), 1e-6), 0, 1))
            b8 = int(round(bval * 255))
            if b8 < 4:
                if np.abs(C - u).max() > 3:
                    continue
            else:
                bb = b8 / 255
                col = np.clip(np.round((C - (1 - bb) * u) / bb), 0, 255)
                cf[y, x, :3] = col; cf[y, x, 3] = b8
                self.seam["collar_front"][y, x] = True
                res.append(float(np.abs(bb * col + (1 - bb) * u - C).max()))
            self.new["torso"][y, x] = np.round(Tf[y, x]).astype(np.uint8)
            self.seam["torso"][y, x] = True
            if Afull[y, x] > 0:
                self.new["armNear"][y, x, :3] = np.round(Cfull[y, x]).clip(0, 255).astype(np.uint8)
                self.new["armNear"][y, x, 3] = int(round(Afull[y, x] * 255))
                self.seam["armNear"][y, x] = True
            n += 1
        self.L["collar_front"] = cf.copy()
        self.new["collar_front"] = cf
        self.added["collar_front"] = np.zeros((H, W), bool)
        self.notes["collar_front"] = [f"torso's own opaque collar px over the near arm's shoulder (x {x0s}-{x1s}), byte "
                                      f"copies [{int(pure.sum())} px] + {n} collar-bottom AA px un-mixed (alpha by "
                                      f"projection collar -> what lies under it; torso px there -> arm-less torso, arm px "
                                      f"-> full arm)"]
        self.exceptions["torso|collar_front"] = {"px": n, "max_rest_residual": round(max(res), 2) if res else 0,
                                                 "why": "collar bottom AA moved into collar_front so the arm's shoulder "
                                                        "can pass under it without a hard edge"}

    def arm(self):
        A, C = self.armfull
        a8 = np.round(A * 255).astype(np.uint8)
        rgb8 = np.round(C).clip(0, 255).astype(np.uint8)
        self.put("armNear", a8, rgb8, "rounded shoulder end: the step-1 AI arm silhouette rendered WITHOUT the collar "
                                      "clip (its own rounded top), band colour, only under collar_front / head")
        far = np.asarray(Image.open(S1 / "work/farArm_layer.png").convert("RGBA"))
        self.put("armFar", far[..., 3], far[..., :3], "hidden part of the far arm (shoulder / upper arm behind the "
                                                      "torso and the acorn) = the complete step-1 far-arm render: its "
                                                      "own rounded top, no cut base")

    # ------------------------------------------------------------ torso
    def torso(self):
        p = "torso"
        Tf = np.asarray(Image.open(IN / "work/torso_full.png").convert("RGBA"))
        z = self.hz(p) & (self.L["armNear"][..., 3] == 255) & (Tf[..., 3] > 0)
        a8 = np.where(z, Tf[..., 3], 0).astype(np.uint8)
        self.put(p, a8, Tf[..., :3], "under the near arm: the arm-less torso (step-3 work/torso_full.png = the step-1 "
                                     "body before the arms, de-whitened) - exact, incl. its belly-front silhouette")
        self.neck()
        self.under_pendant()

    def neck(self):
        p = "torso"
        T = self.new[p]
        names, lab, cols = F4.classify(self.L[p], PAL[p])
        collar = np.isin(lab, [names.index("collar_dark"), names.index("collar_light")])
        a = T[..., 3]
        top, colc = {}, {}
        for x in range(430, 720):
            c = np.nonzero(a[590:700, x] > 0)[0]
            if len(c) == 0:
                continue
            y0 = 590 + c.min()
            seg = T[y0 + 3:y0 + 11, x]
            ok = (seg[:, 3] == 255) & collar[y0 + 3:y0 + 11, x]
            if ok.sum() >= 2:
                top[x] = y0; colc[x] = np.median(seg[ok, :3].astype(float), 0)
        xs = np.array(sorted(colc))
        xle, xre = float(xs.min()) - 0.5, float(xs.max()) + 0.5
        cc = np.array([colc[x] for x in xs])
        cc = ndi.median_filter(cc, size=(9, 1), mode="nearest")
        colmap = {x: cc[i] for i, x in enumerate(xs)}
        ptop = np.polyfit(xs, [top[x] for x in xs], 2)
        T_UP, NECK_H, NECK_IN, RAD = 14.0, 40.0, 14.0, 18.0
        tan = cols[names.index("tan")]
        hid = self.hz(p) & (self.L["head"][..., 3] == 255)
        hid[640:, :] = False
        bb = F4.bbox_of(hid)
        if bb is None:
            return
        x0, y0, x1, y1 = bb
        X, Y = F4.sub_coords((x0, y0, x1, y1))
        ytop = np.polyval(ptop, np.clip(X, xle, xre))
        in_col = (X >= xle) & (X <= xre) & (Y >= ytop - T_UP)
        ntop = ytop - T_UP - NECK_H
        in_neck = (X >= xle + NECK_IN) & (X <= xre - NECK_IN) & (Y >= ntop) & (Y < ytop - T_UP + 1)
        for cx in (xle + NECK_IN + RAD, xre - NECK_IN - RAD):
            cy = np.polyval(ptop, cx) - T_UP - NECK_H + RAD
            side = (X < cx) if cx < (xle + xre) / 2 else (X > cx)
            in_neck &= ~((Y < cy) & side & ((X - cx) ** 2 + (Y - cy) ** 2 > RAD ** 2))
        inside = in_col | in_neck
        xi = np.clip(np.round(X).astype(int), int(xs.min()), int(xs.max()))
        lut = np.array([colmap.get(x, colmap[xs[np.argmin(np.abs(xs - x))]]) for x in range(W)])
        sub = np.where(in_col[..., None], lut[xi], tan)
        h, w = y1 - y0, x1 - x0
        cov = F4.pool(inside.astype(float), h, w)
        rgb = F4.pool(sub * inside[..., None], h, w) / np.maximum(cov, 1e-9)[..., None]
        a8 = np.zeros((H, W), np.uint8); rgb8 = np.zeros((H, W, 3), np.uint8)
        al = np.round(cov * 255).astype(np.uint8); al[~hid[y0:y1, x0:x1]] = 0
        a8[y0:y1, x0:x1] = al; rgb8[y0:y1, x0:x1] = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        self.geom["neck"] = {"collar_ends": [xle, xre], "collar_top_poly": [float(v) for v in ptop],
                             "collar_up_px": T_UP, "neck_h": NECK_H, "neck_inset": NECK_IN}
        self.put("torso", a8, rgb8, "under the head: collar band carried up column-wise (each column its own collar "
                                    f"colour) {T_UP:.0f} px + rounded tan neck ({NECK_H:.0f} px, inset {NECK_IN:.0f})")

    def under_pendant(self):
        p = "torso"
        # v2: classified on the arm-less torso (step-3 work/torso_full.png), so the region under the near arm
        # is torso, not background
        T = np.asarray(Image.open(IN / "work/torso_full.png").convert("RGBA"))
        pal = dict(PAL[p]); pal["bg"] = (255, 255, 255)
        names, lab, cols = F4.classify(T, PAL[p])
        kb = len(names)
        lab2 = lab.copy()
        lab2[(T[..., 3] == 0) & (self.L["pendant"][..., 3] == 0)] = kb      # background as a class
        # torso px within 3 px of the acorn may carry acorn AA: treat as unknown
        near = ndi.binary_dilation(self.L["pendant"][..., 3] > 0, iterations=3) & (T[..., 3] > 0)
        lab2[near] = -1
        zone = self.hz(p) & (self.L["pendant"][..., 3] == 255)
        zone2 = zone
        known = lab2 >= 0
        Fd = F4.sdf_fields(lab2, kb + 1)
        unk = ~known
        win = np.zeros((H, W), bool); win[600:760, 630:760] = True
        Fz = F4.harmonic(Fd, known, unk & win)
        defined = known | (unk & win)
        colors = np.vstack([cols, [[255, 255, 255]]])
        labf = F4.label_fn_from_fields(Fz, defined)
        shape = lambda X, Y: labf(X, Y) != kb
        a8, rgb8 = F4.render(zone2, shape, labf, colors)
        self.put(p, a8, rgb8, "under the acorn: torso regions incl. a background class continued by harmonic SDF "
                              "(belly front silhouette, collar end, cream, band), flat colours, AA coverage")

    # ------------------------------------------------------------ legs
    def leg(self, p):
        L = self.L[p]
        al = L[..., 3]
        ys, xs = np.nonzero(al >= 128)
        ytop = ys.min()
        topsel = ys <= ytop + 30
        pts = np.r_[np.c_[xs[topsel], ys[topsel]], np.vstack([disk_pts(c, r) for c, r in LEG_DISKS[p]])]
        fn = hull_fn(pts)
        zone = self.hz(p) & any_cov(fn)
        names, lab, cols = F4.classify(L, PAL[p])
        labf = lambda X, Y: np.zeros(X.shape, int)
        a8, rgb8 = F4.render(zone, fn, labf, cols[:1])
        self.geom[p] = {"hip_pivot": list(PIV[p]), "hip_disks": LEG_DISKS[p], "top_rows_from": int(ytop)}
        self.put(p, a8, rgb8, f"rounded hip: convex hull of the leg's top 30 rows and disk(s) {LEG_DISKS[p]} (first = "
                              f"about the hip pivot), kept inside the body; the leg's own flat colour")

    # ------------------------------------------------------------ tail root
    def tail(self):
        p = "tail"
        L = self.L[p]
        al = L[..., 3]
        names, lab, cols = F4.classify(L, PAL[p])
        # cross-section at the junction: tail px within 6 px of the torso/near-leg
        front = (self.L["torso"][..., 3] > 0) | (self.L["legNear"][..., 3] > 0)
        sec = (al >= 128) & ndi.binary_dilation(front, iterations=6)
        ys, xs = np.nonzero(sec)
        pts = np.c_[xs, ys].astype(float)
        pts = np.r_[pts, disk_pts(PIV["tail"], ROOT_R)]
        fn = hull_fn(pts)
        zone = self.hz(p) & any_cov(fn)
        known = lab >= 0
        Fd = F4.sdf_fields(lab, len(names))
        Fz = F4.harmonic(Fd, known, zone)
        a8, rgb8 = F4.render(zone, fn, F4.label_fn_from_fields(Fz, known | zone), cols)
        self.geom["tail_root"] = {"junction_px": int(sec.sum()), "root_disk": [list(PIV["tail"]), ROOT_R]}
        self.put(p, a8, rgb8, f"root behind the torso: convex hull of the tail's cross-section at the junction and a disk "
                              f"r={ROOT_R:.0f} about the tail pivot (rotation-invariant root); bands continued by harmonic SDF")

    # ------------------------------------------------------------ head under the ear
    def head(self):
        p = "head"
        L = self.L[p]
        names, lab, cols = F4.classify(L, PAL[p])
        bg = np.ones((H, W), bool)
        for q in STACK3:
            bg &= self.L[q][..., 3] == 0
        sil = (L[..., 3] > 0) & ndi.binary_dilation(bg, iterations=1)
        sy, sx = np.nonzero(sil)
        keep = ~(((sx > 405) & (sx < 535) & (sy < 255)) | ((sx > 495) & (sx < 700) & (sy < 200)) | (sx > 770) | (sy > 560))
        E = F4.fit_ellipse(sx[keep].astype(float), sy[keep].astype(float))
        shape = lambda X, Y: E(X, Y) <= 0
        zone = self.hz(p) & (self.L["earNear"][..., 3] == 255) & any_cov(shape)
        known = lab >= 0
        Fd = F4.sdf_fields(lab, len(names))
        Fz = F4.harmonic(Fd, known, zone)
        a8, rgb8 = F4.render(zone, shape, F4.label_fn_from_fields(Fz, known | zone), cols)
        self.put(p, a8, rgb8, "under the ear: inside the head ellipse (fitted to the head silhouette away from ear, "
                              "tuft and snout); back stripe / fur continued by harmonic SDF")

    # ------------------------------------------------------------ tuft
    def tuft(self):
        p = "tuft"
        T = self.new[p]
        hd = self.L["head"]
        reg = hd[170:300, 440:760].reshape(-1, 4)
        reg = reg[reg[:, 3] == 255][:, :3].astype(float)
        sel = reg[np.abs(reg - np.array(PAL["head"]["tan"])).max(1) < 12]
        flat = np.round(np.median(sel, 0)).astype(np.uint8)
        vis = T[..., 3] > 0
        n_rec = int(vis.sum())
        T[vis, :3] = flat
        T[~vis, :3] = 0
        self.seam[p] |= vis & (np.abs(self.L[p][..., :3].astype(int) - flat.astype(int)).max(-1) > 0)
        contact = vis & ndi.binary_dilation(hd[..., 3] > 0, iterations=2)
        ys, xs = np.nonzero(contact | (vis & (np.arange(H)[:, None] > np.nonzero(contact)[0].min() - 6)))
        pts = np.c_[xs, ys].astype(float)
        x0c, x1c = xs.min() - 2, xs.max() + 6
        pts = np.r_[pts, pts + [0, TUFT_SWEEP]]
        pts[:, 0] = np.clip(pts[:, 0], x0c, x1c)
        fn = hull_fn(pts)
        zone = self.hz(p) & any_cov(fn)
        a8, rgb8 = F4.render(zone, fn, lambda X, Y: np.zeros(X.shape, int), flat[None].astype(float))
        self.geom["tuft"] = {"flat_colour": [int(v) for v in flat], "contact_x": [int(x0c), int(x1c)], "sweep": TUFT_SWEEP}
        self.exceptions["tuft_flat"] = {"px_recoloured": n_rec, "colour": [int(v) for v in flat],
                                        "why": "front decision: tuft = one flat fur colour (alpha / silhouette kept)"}
        self.notes[p].append(f"flattened to the head's flat tan {list(map(int, flat))} (alpha kept) [{n_rec} px]")
        self.put(p, a8, rgb8, f"base extension under the head: contact rows swept {TUFT_SWEEP:.0f} px down (convex hull, "
                              "clipped to the contact span)")
        # left lobe continued behind the ear (the ear is in front of head and tuft), so a turning ear
        # uncovers tuft fur instead of a background notch
        vis = self.new[p][..., 3] > 0
        ys, xs = np.nonzero(vis & (np.arange(W)[None, :] < 545) & (np.arange(H)[:, None] < 175))
        pts = np.c_[xs, ys].astype(float)
        pts = np.r_[pts, pts + [-12.0, 4.0]]
        fn = hull_fn(pts)
        zone = self.hz(p) & (self.L["earNear"][..., 3] == 255) & any_cov(fn)
        a8, rgb8 = F4.render(zone, fn, lambda X, Y: np.zeros(X.shape, int), flat[None].astype(float))
        self.put(p, a8, rgb8, "left lobe continued 12 px behind the ear (convex hull), flat tan")

    # ------------------------------------------------------------ seams
    SEAM_PAIRS = [("torso", "head"), ("tail", "torso"), ("legNear", "torso"), ("legFar", "torso"),
                  ("armFar", "torso"), ("armFar", "pendant"),
                  ("legFar", "legNear"), ("torso", "pendant"), ("head", "earNear"), ("tail", "legNear")]
    SEAM_TOL = 12

    def seam_unmix(self):
        """front method (generic palette): back-part boundary px (alpha 255, 4-adjacent to an opaque front px)
        whose colour is > SEAM_TOL off the clean local colour of its own region -> clean colour; the FRONT layer
        gets a soft edge px there (alpha by projection, colour un-mixed) so front-over-back == original."""
        for back, front in self.SEAM_PAIRS:
            Bl, Fl = self.L[back], self.L[front]
            B, Fn = self.new[back], self.new[front]
            names, lab, cols = F4.classify(Bl, PAL[back])
            fa = Fl[..., 3] == 255
            adj = np.zeros((H, W), bool)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                adj |= np.roll(fa, (dy, dx), (0, 1))
            bnd = (Bl[..., 3] == 255) & adj & (Fl[..., 3] == 0)
            rgb = Bl[..., :3].astype(float)
            clean = (lab >= 0) & ~ndi.binary_dilation(bnd, iterations=1)
            n, res = 0, []
            for y, x in zip(*np.nonzero(bnd)):
                if any(self.seam[q][y, x] for q in self.seam if q != "tuft"):
                    continue
                y0, y1, x0, x1 = max(y - 3, 0), y + 4, max(x - 3, 0), x + 4
                cl = clean[y0:y1, x0:x1]
                if cl.sum() < 3:
                    continue
                k = np.bincount(lab[y0:y1, x0:x1][cl]).argmax()
                m = cl & (lab[y0:y1, x0:x1] == k)
                if m.sum() < 3:
                    continue
                Cc = np.round(np.median(rgb[y0:y1, x0:x1][m], 0))
                M = rgb[y, x]
                if np.abs(M - Cc).max() <= self.SEAM_TOL:
                    continue
                nb = [Fl[y + dy, x + dx, :3].astype(float) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= y + dy < H and 0 <= x + dx < W and Fl[y + dy, x + dx, 3] == 255]
                Hc = np.mean(nb, 0); d = Hc - Cc
                if np.dot(d, d) < 100:
                    continue
                a = float(np.clip(np.dot(M - Cc, d) / np.dot(d, d), 0, 1)); a8 = int(round(a * 255))
                if a8 < 8:
                    continue
                Fc = np.round(np.clip((M - (1 - a8 / 255) * Cc) / (a8 / 255), 0, 255))
                comp = a8 / 255 * Fc + (1 - a8 / 255) * Cc
                if np.abs(comp - M).max() > 1.0:
                    continue
                B[y, x, :3] = Cc; Fn[y, x, :3] = Fc; Fn[y, x, 3] = a8
                self.seam[back][y, x] = True; self.seam[front][y, x] = True
                n += 1; res.append(float(np.abs(comp - M).max()))
            if n:
                self.exceptions[f"{back}|{front}"] = {"back_visible_px_recoloured": n, "front_soft_edge_px_added": n,
                                                      "unmix_residual_max": round(max(res), 2)}
                self.notes[back].append(f"EXCEPTION seam un-mix vs {front}: {n} boundary px -> clean colour")
                self.notes[front].append(f"EXCEPTION seam un-mix over {back}: {n} soft edge px added")

    def unmix_torso_head(self):
        """front step-5 method: torso px within 2 px of the opaque head still carrying the head's light rim AA
        (head alpha jumps 255 -> 0 in step 3) -> clean collar colour (median of torso px 3-7 rows below);
        head gets a soft edge px (alpha by projection toward the nearest opaque head colour) so the rest
        composite is unchanged up to the reported residual (<= 10, mixes that are not exactly on the head/collar line)."""
        T, Hd = self.new["torso"], self.new["head"]
        front = np.zeros((H, W), bool)
        for k in ("pendant", "armNear", "collar_front", "tuft", "earNear"):
            front |= self.new[k][..., 3] > 0
        near = ndi.binary_dilation(self.L["head"][..., 3] == 255, iterations=3) & (self.L["torso"][..., 3] == 255) \
            & ~front & (self.L["head"][..., 3] < 255)
        far = ~ndi.binary_dilation(self.L["head"][..., 3] > 0, iterations=3) & (self.L["torso"][..., 3] == 255)
        _, (iy, ix) = ndi.distance_transform_edt(self.L["head"][..., 3] < 255, return_indices=True)
        n, res = 0, []
        for y, x in zip(*np.nonzero(near)):
            win = T[y + 4:y + 9, x - 1:x + 2]
            ok = far[y + 4:y + 9, x - 1:x + 2]
            if ok.sum() < 4:
                continue
            tc = np.median(win[ok][:, :3].astype(float), 0)
            C = self.L["torso"][y, x, :3].astype(float)
            if self.L["head"][y, x, 3] > 0:
                a0 = self.L["head"][y, x, 3] / 255
                C = a0 * self.L["head"][y, x, :3] + (1 - a0) * C
            if np.abs(C - tc).max() <= 5:
                continue
            hc = Hd[iy[y, x], ix[y, x], :3].astype(float)
            d = hc - tc
            b = float(np.clip(np.dot(C - tc, d) / max(np.dot(d, d), 1e-6), 0, 1))
            b8 = int(round(b * 255))
            if b8 == 0:
                continue
            bb = b8 / 255
            cc = np.clip(np.round((C - (1 - bb) * np.round(tc)) / bb), 0, 255)
            r = float(np.abs(bb * cc + (1 - bb) * np.round(tc) - C).max())
            if r > 10.0:
                continue
            T[y, x, :3] = np.round(tc).astype(np.uint8)
            Hd[y, x, :3] = cc.astype(np.uint8); Hd[y, x, 3] = b8
            self.seam["torso"][y, x] = True; self.seam["head"][y, x] = True
            n += 1; res.append(r)
        self.exceptions["torso|head (rim)"] = {"px": n, "max_rest_residual": round(max(res), 2) if res else 0,
                                               "why": " ".join(self.unmix_torso_head.__doc__.split())}

    def run(self):
        self.collar_front()
        self.torso()
        self.arm()
        self.seam_unmix()
        self.unmix_torso_head()
        self.leg("legNear"); self.leg("legFar")
        self.tail()
        self.head()
        self.tuft()


# ============================================================== motion test
def poses():
    B = (300, 40, 880, 1000)
    return [
        ("rest", {}, (120, 40, 860, 1000)),
        ("legs: near +25, far -25 (walk contact A)", {"legNear": 25, "legFar": -25}, (360, 780, 780, 1000)),
        ("legs: near -25, far +25 (walk contact B)", {"legNear": -25, "legFar": 25}, (360, 780, 780, 1000)),
        ("arm +30 (ccw: paw forward)", {"armNear": 30}, (480, 580, 800, 880)),
        ("arm -30 (cw: paw back)", {"armNear": -30}, (480, 580, 800, 880)),
        ("far arm +20 (ccw: paw up/forward)", {"armFar": 20}, (600, 600, 820, 820)),
        ("far arm -20 (cw: paw down/back)", {"armFar": -20}, (600, 600, 820, 820)),
        ("head +8", {"head": 8}, B),
        ("head -8", {"head": -8}, B),
        ("ear +15", {"earNear": 15}, (380, 80, 600, 300)),
        ("ear -15", {"earNear": -15}, (380, 80, 600, 300)),
        ("tuft +6", {"tuft": 6}, (460, 40, 720, 240)),
        ("tuft -6", {"tuft": -6}, (460, 40, 720, 240)),
        ("tail +10", {"tail": 10}, (100, 520, 520, 940)),
        ("tail -10", {"tail": -10}, (100, 520, 520, 940)),
        ("pendant +10", {"pendant": 10}, (620, 600, 780, 760)),
        ("pendant -10", {"pendant": -10}, (620, 600, 780, 760)),
    ]


HEAD_KIDS = ("earNear", "tuft")


def motion_sheet(layers, stack, path, title):
    from PIL import ImageFont
    pms = {p: F4.premul(layers[p]) for p in stack}
    _, acc0 = F4.compose(pms, stack, np.zeros(3))
    _, h0 = F4.holes(acc0)
    rows, stats = [], []
    TW = 520
    for name, rot, box in poses():
        cur = {}
        for p in stack:
            pm = pms[p]
            if p in rot:
                pm = F4.rotate_pm(pm, PIV[p], rot[p])
            if "head" in rot and p in HEAD_KIDS:
                pm = F4.rotate_pm(pm, PIV["head"], rot["head"])
            cur[p] = pm
        ims = []
        for bg in (GREEN, DARK):
            img, acc = F4.compose(cur, stack, np.array(bg, float))
            x0, y0, x1, y1 = box
            im = Image.fromarray(np.round(img[y0:y1, x0:x1]).clip(0, 255).astype(np.uint8))
            ims.append(im.resize((TW, int(im.height * TW / im.width)), Image.LANCZOS))
        _, hm = F4.holes(acc)
        nh = int((hm & ~h0).sum())
        stats.append({"pose": name, "new_enclosed_hole_px": nh})
        rows.append((f"{name}   [new enclosed holes vs rest: {nh} px]", ims))
    hgt = sum(r[1][0].height + 34 for r in rows) + 50
    sheet = Image.new("RGB", (2 * TW + 30, hgt), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), title + "   (angles + = counter-clockwise here; head carries ear + tuft)", fill=(0, 0, 0), font=F4.font(18))
    y = 50
    for label, ims in rows:
        d.text((10, y), label, fill=(0, 0, 0), font=F4.font(15))
        for j, im in enumerate(ims):
            sheet.paste(im, (10 + j * (TW + 10), y + 22))
        y += ims[0].height + 34
    sheet.save(path)
    return stats


def diag(layer, added, seam, path):
    bb = F4.bbox_of(layer[..., 3] > 0)
    x0, y0, x1, y1 = max(0, bb[0] - 10), max(0, bb[1] - 10), min(W, bb[2] + 10), min(H, bb[3] + 10)
    f = layer[y0:y1, x0:x1].astype(float); a = f[..., 3:4] / 255
    img = f[..., :3] * a + F4.checker(y1 - y0, x1 - x0, 8) * (1 - a)
    ad = added[y0:y1, x0:x1]
    img[ad] = img[ad] * 0.45 + np.array([255, 0, 255]) * 0.55
    img[seam[y0:y1, x0:x1]] = [0, 230, 255]
    im = Image.fromarray(np.round(img).astype(np.uint8))
    sc = max(2, int(900 / max(im.size)))
    im.resize((im.width * sc, im.height * sc), Image.NEAREST).save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = {p: np.array(Image.open(IN / f"{p}.png").convert("RGBA")) for p in STACK3}
    fill = Fill(L)
    fill.run()
    new = fill.new
    L = fill.L                                   # now includes the step-4 "collar_front" (copy, no new art)
    rep = {"source": "art/v2/tori-side/step3-layers", "draw_order_back_to_front": STACK,
           "rule": "pixels added only where the part is transparent and a layer in front of it is opaque (alpha 255)",
           "parts": {}}
    for p in STACK:
        Image.fromarray(new[p], "RGBA").save(OUT / f"{p}.png")
        diag(new[p], fill.added[p], fill.seam[p], OUT / f"filled-{p}.png")
        vis = L[p][..., 3] > 0
        ch = vis & (np.abs(new[p].astype(int) - L[p].astype(int)).max(-1) > 0)
        rep["parts"][p] = {"added_px": int(fill.added[p].sum()), "added_bbox": F4.bbox_of(fill.added[p]),
                           "method": fill.notes[p] or ["nothing hidden to fill -> unchanged"],
                           "visible_px_changed": int(ch.sum()),
                           "visible_px_changed_outside_listed_exceptions": int((ch & ~fill.seam[p]).sum())}
        print(f"{p:12s} added={int(fill.added[p].sum()):6d} visible_changed={int(ch.sum())} "
              f"outside_exc={int((ch & ~fill.seam[p]).sum())}")
    rep["exceptions"] = fill.exceptions
    rep["geometry"] = json.loads(json.dumps(fill.geom, default=float))
    rest = {}
    for name, bg in (("white", (255, 255, 255)), ("green", GREEN), ("dark", DARK)):
        a, _ = F4.compose({p: F4.premul(L[p]) for p in STACK3}, STACK3, np.array(bg, float))
        b, _ = F4.compose({p: F4.premul(new[p]) for p in STACK}, STACK, np.array(bg, float))
        d8 = np.abs(np.round(a) - np.round(b)).max(-1)
        tuft = new["tuft"][..., 3] > 0
        rest[name] = {"max_outside_tuft": int(d8[~tuft].max()), "px_diff_gt2_outside_tuft": int((d8[~tuft] > 2).sum()),
                      "px_diff_outside_tuft": int((d8[~tuft] > 0).sum()), "tuft_area_max (flattened)": int(d8[tuft].max())}
    rep["rest_pose_step4_vs_step3"] = rest
    print("rest", json.dumps(rest))
    L3 = {p: np.array(Image.open(IN / f"{p}.png").convert("RGBA")) for p in STACK3}
    before = motion_sheet(L3, STACK3, OUT / "motion-test-before.png", "STEP 3 layers (before fill)")
    after = motion_sheet(new, STACK, OUT / "motion-test.png", "STEP 4 layers (filled)")
    rep["provisional_pivots"] = {k: list(v) for k, v in PIV.items()}
    rep["motion_test"] = [{"pose": a["pose"], "holes_before": b["new_enclosed_hole_px"], "holes_after": a["new_enclosed_hole_px"]}
                          for a, b in zip(after, before)]
    (OUT / "fill-report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["motion_test"]))


if __name__ == "__main__":
    main()
