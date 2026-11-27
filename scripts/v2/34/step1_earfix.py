#!/usr/bin/env python3
"""
STEP 1-fix (3/4 view) - Tori v2: transplant cand4's FAR ear (viewer's right, drawn as an ear seen
from behind: one solid inner-ear-brown disc) onto the chosen base cand3.  Everything else stays
cand3 pixel-exact.

Inputs : art/v2/tori/views/34-original.png   (= cand3, cleaned; the base)
         art/v2/tori/views/34-cand4.png      (donor of the far ear)
Outputs: art/v2/tori-34/original.png              base with the transplanted far ear
         art/v2/tori-34/original-before-earfix.png copy of the base
         art/v2/tori-34/earfix-check.png          before / donor / after zoom (+ changed-px map)
         art/v2/tori-34/earfix.json               numbers (transform, curve fit, colour check)

Method
  1. Ear/head contact = the two concave notches where the ear silhouette meets the head silhouette
     (deepest points of the convexity defects of the foreground in the ear window), measured in
     both images.  Donor -> base similarity transform maps cand4's notch chord onto cand3's notch
     chord (the ear base sits exactly where cand3's ear base attached; scale ~1, a few degrees
     of rotation along the head arc).  A free similarity registration of cand4's head (stripes,
     right head side) onto cand3 is reported as a consistency check.
  2. cand3's head outline under its old ear (hidden behind the ear in cand3) = least-squares cubic
     y(x) through cand3's own visible evidence: the head-top edge left of the left notch, the
     straight cut of the old ear's inner oval (the head outline clips it, exactly as in the front
     step 2) and the head side right of the right notch.  cand4's (visible) arc, mapped through
     the same transform, is reported against it.
  3. Donor ear layer (cand4 coords): pixels above cand4's head arc; outer silhouette de-whited
     against the local ear colour (alpha = projection on white->ear, colour un-premultiplied); the
     part hidden behind cand4's head is filled with the nearest pure ear colour inside an ellipse
     fitted to the visible ear silhouette (so a 1-2 px curve difference never shows a gap).
     Colours: each donor ear tone is matched to cand3's palette (the dominant ear tone -> cand3's
     inner-ear colour, measured; the subtle crescent shade keeps its offset).
  4. Warp the premultiplied ear (bilinear) into cand3 coords and composite:
        out = a_h * H + (1 - a_h) * (ear over white)
     a_h = head coverage: 4x4 supersampled coverage of the fitted outline where the old ear touched
     the head; elsewhere the head's own white-blend alpha read from the cand3 pixel (so cand3's
     existing outline AA is kept); H = local head colour (nearest cand3 head pixel >= 3 px inside).
     Applied only in the zone covered by the old ear or the new ear (+2 px); all other pixels are
     byte-identical to cand3 (verified: changed-px bbox reported).
"""
import json, pathlib, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "art/v2/tori/views/34-original.png"
DONOR = ROOT / "art/v2/tori/views/34-cand4.png"
OUT = ROOT / "art/v2/tori-34"
WIN = (590, 100, 770, 300)            # x0, y0, x1, y1: far-ear window
H = W = 1024
SS = 4


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64)


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def notches(a):
    """deepest points of the two largest convexity defects in the ear window (ear/head notches)."""
    x0, y0, x1, y1 = WIN
    fg = ndi.binary_fill_holes(((255 - a.min(2)) > 40)[y0:y1, x0:x1])
    ys, xs = np.nonzero(fg)
    h = ConvexHull(np.c_[xs, ys])
    im = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in np.c_[xs, ys][h.vertices]], fill=1)
    hull = np.asarray(im) > 0
    defect = hull & ~fg
    dist = ndi.distance_transform_edt(hull)
    lab, n = ndi.label(defect)
    res = []
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() < 300:            # the two ear notches (>500 px); skip small pockets
            continue
        d = np.where(c, dist, 0)
        yy, xx = np.unravel_index(d.argmax(), d.shape)
        if yy + y0 > 260:            # pocket below the head side (cheek), not an ear notch
            continue
        res.append((float(xx + x0), float(yy + y0)))
    res.sort()
    assert len(res) == 2, res
    return np.array(res)


def similarity_from_pairs(src, dst):
    """complex-number similarity z -> s*e^{i t} z + b mapping 2 src points onto 2 dst points."""
    zs = src[:, 0] + 1j * src[:, 1]
    zd = dst[:, 0] + 1j * dst[:, 1]
    m = (zd[1] - zd[0]) / (zs[1] - zs[0])
    b = zd[0] - m * zs[0]
    return m, b


def register_head(A, B):
    """free similarity (scale, rot, shift about (660,250)) of donor head onto base, ears excluded."""
    yy, xx = np.mgrid[0:H, 0:W]
    reg = (xx > 560) & (xx < 760) & (yy > 150) & (yy < 300) & ~((xx > 605) & (xx < 745) & (yy < 238))
    ys, xs = np.nonzero(reg)
    C = np.array([660.0, 250.0])

    def err(p):
        s, dx, dy, th = p
        c, sn = np.cos(th), np.sin(th)
        X, Y = xs - C[0], ys - C[1]
        sx = C[0] + (c * X + sn * Y) / s - dx
        sy = C[1] + (-sn * X + c * Y) / s - dy
        v = np.stack([ndi.map_coordinates(B[..., k], [sy, sx], order=1) for k in range(3)], 1)
        return np.abs(v - A[ys, xs]).mean()
    best = min(((err((1, dx, dy, 0)), dx, dy) for dx in range(-8, 9, 2) for dy in range(-8, 9, 2)))
    x0 = [1, best[1], best[2], 0]
    simplex = [x0, [1.02, x0[1], x0[2], 0], [1, x0[1] + 1, x0[2], 0], [1, x0[1], x0[2] + 1, 0], [1, x0[1], x0[2], 0.02]]
    r = minimize(err, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, initial_simplex=simplex))
    s, dx, dy, th = r.x
    # as complex similarity: base = C + s e^{i th}(donor + d - C)
    m = s * np.exp(1j * th)
    b = (C[0] + 1j * C[1]) + m * ((dx + 1j * dy) - (C[0] + 1j * C[1]))
    return m, b, dict(scale=round(float(s), 4), rot_deg=round(float(np.rad2deg(th)), 3),
                      shift=[round(float(dx), 2), round(float(dy), 2)], mean_abs_err=round(float(r.fun), 2),
                      mean_abs_err_identity=round(float(err((1, 0, 0, 0))), 2))


def base_outline(A, nl, nr):
    """cubic y(x) for cand3's head outline under the old ear (see docstring 2)."""
    fg = (255 - A.min(2)) > 60
    e = fg & ~ndi.binary_erosion(fg)
    ys, xs = np.nonzero(e)
    left = (xs >= nl[0] - 20) & (xs <= nl[0] - 1) & (ys < nl[1] + 6) & (ys > nl[1] - 20)
    right = (xs >= nr[0] + 1) & (ys >= nr[1] + 2) & (ys <= nr[1] + 16) & (xs < nr[0] + 20)
    # inner-oval cut: per column, lowest pixel nearer the inner-ear colour than tan (+0.5 subpixel)
    T, E = np.array([234, 190, 129.0]), np.array([209, 155, 108.0])
    d = E - T
    cut = []
    for x in range(int(nl[0]) + 25, int(nr[0]) - 15):
        col = A[170:240, x]
        t = ((col - T) @ d) / (d @ d)
        k = np.nonzero(t > 0.5)[0]
        if len(k) == 0:
            continue
        y = 170 + k.max()
        # subpixel crossing between y and y+1
        t0, t1 = t[k.max()], t[min(k.max() + 1, len(t) - 1)]
        cut.append((x, y + (t0 - 0.5) / max(t0 - t1, 1e-6)))
    cut = np.array(cut)
    # keep the straight cut only (the oval's own round bottom curls back up near its right end)
    sl = np.diff(cut[:, 1])
    keep = np.r_[True, sl > 0.3]
    stop = np.argmax(~keep) if (~keep).any() else len(cut)
    cut = cut[:stop]
    cut = cut[cut[:, 0] < cut[:, 0].max() - 3]
    X = np.r_[xs[left], cut[:, 0], xs[right]].astype(float)
    Y = np.r_[ys[left], cut[:, 1], ys[right]].astype(float)
    wts = np.r_[np.ones(left.sum()), 2 * np.ones(len(cut)), np.ones(right.sum())]
    c = np.polyfit(X, Y, 3, w=wts)
    res = Y - np.polyval(c, X)
    return c, dict(n_left=int(left.sum()), n_cut=len(cut), n_right=int(right.sum()),
                   cut_x=[float(cut[0, 0]), float(cut[-1, 0])],
                   rms=round(float(res.std()), 2), max_abs=round(float(np.abs(res).max()), 2))


def donor_arc(B, nl, nr):
    """cand4's visible head arc under its far ear: tan/ear-colour crossing per column."""
    T, E = np.array([234, 190, 129.0]), np.array([208, 154, 107.0])
    d = T - E
    pts = []
    for x in range(int(nl[0]) + 18, int(nr[0]) - 2):
        col = B[120:245, x]
        t = ((col - E) @ d) / (d @ d)
        ear = np.abs(col - E).max(1) < 14
        ys = np.nonzero(ear)[0]
        if len(ys) == 0:
            continue
        y0 = ys.max()
        seg = t[y0:y0 + 6]
        k = np.nonzero(seg > 0.5)[0]
        if len(k) == 0 or k[0] == 0:
            continue
        k = k[0]
        pts.append((x, 120 + y0 + k - 1 + (0.5 - seg[k - 1]) / (seg[k] - seg[k - 1] + 1e-9)))
    pts = np.array(pts)
    # keep the monotonic (descending to the right) head arc only: drop the stripe/ear-edge outliers
    med = ndi.median_filter(pts[:, 1], size=7, mode="nearest")
    ok = np.abs(pts[:, 1] - med) < 2.5
    ok &= np.r_[True, np.diff(med) > -0.5]
    return pts[ok]


def fit_ellipse(x, y):
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


def donor_ear_layer(B, arc, palette_shift):
    """RGBA float (premultiplied) of cand4's far ear in cand4 coordinates."""
    x0, y0, x1, y1 = WIN
    yy, xx = np.mgrid[0:H, 0:W]
    cA = np.polyfit(arc[:, 0], arc[:, 1], 3)
    win = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < 245)
    # above the donor head: left of the arc fit range the head top is the stripe/top edge -> use
    # the ear colour class directly; right of it the head side (steep) -> same
    E = np.array([208, 154, 107.0])
    above = win & (yy < np.polyval(cA, np.clip(xx, arc[0, 0], arc[-1, 0])) - 1.5)
    # ear pixels = non-white pixels above the arc that are not head tan / stripe
    T = np.array([234, 190, 129.0])
    dT = np.abs(B - T).max(2)
    stripe = B.sum(2) < 400
    nonwhite = (255 - B.min(2)) > 6
    ear_vis = above & nonwhite & (dT > 12) & ~stripe
    ear_vis = ndi.binary_opening(ear_vis, iterations=1)
    lab, n = ndi.label(ear_vis)
    ear_vis = lab == (np.argmax(ndi.sum(ear_vis, lab, range(1, n + 1))) + 1)
    ear_vis = ndi.binary_fill_holes(ear_vis)
    # solid (pure) ear pixels and white-blend alpha of the silhouette
    pure = ndi.binary_erosion(ear_vis, iterations=3)
    _, (iy, ix) = ndi.distance_transform_edt(~pure, return_indices=True)
    S = B[iy, ix]                                         # local solid ear colour
    dC, dS = 255 - B, 255 - S
    a = np.clip((dC * dS).sum(2) / np.maximum((dS * dS).sum(2), 1e-6), 0, 1)
    a_min = dC.max(2) / 255
    a = np.maximum(a, a_min)
    a[pure] = 1.0
    sil = ear_vis & ~pure
    # silhouette ring: also include the outer AA ring (white-ish px next to the ear)
    ring = ndi.binary_dilation(ear_vis, iterations=2) & ~ear_vis & above & nonwhite & (dT > 12) & ~stripe
    vis = ear_vis | ring
    alpha = np.where(vis, a, 0.0)
    rgb = np.where(vis[..., None], S, 0.0)               # flat local colour, alpha carries the AA
    # hidden part under cand4's head: ellipse fitted to the visible outer silhouette
    sx_, sy_ = np.nonzero(((alpha > 0.15) & (alpha < 0.85)).T)
    keepp = sy_ < np.polyval(cA, np.clip(sx_, arc[0, 0], arc[-1, 0])) - 4
    F = fit_ellipse(sx_[keepp].astype(float), sy_[keepp].astype(float))
    inside = F(xx.astype(float), yy.astype(float)) < 0
    hid = inside & ~above & win
    rgb[hid] = S[hid]
    alpha[hid] = 1.0
    # palette correction (towards cand3's inner-ear colour)
    rgb = np.clip(rgb + palette_shift, 0, 255)
    pm = np.concatenate([rgb * alpha[..., None], alpha[..., None]], -1)
    return pm, dict(visible_px=int(vis.sum()), pure_px=int(pure.sum()), hidden_fill_px=int(hid.sum()))


def warp(pm, m, b):
    """sample donor premultiplied layer at the inverse similarity of every base pixel."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    z = (xx + 1j * yy - b) / m
    return np.stack([ndi.map_coordinates(pm[..., k], [z.imag, z.real], order=1, mode="constant", cval=0)
                     for k in range(4)], -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, B = load(BASE), load(DONOR)
    shutil.copy(BASE, OUT / "original-before-earfix.png")
    info = {"base": str(BASE.relative_to(ROOT)), "donor": str(DONOR.relative_to(ROOT))}

    nA, nB = notches(A), notches(B)
    m, b = similarity_from_pairs(nB, nA)
    info["notches_base_cand3"] = nA.round(1).tolist()
    info["notches_donor_cand4"] = nB.round(1).tolist()
    info["ear_transform_donor_to_base"] = {
        "rule": "maps cand4's ear/head notch chord onto cand3's ear/head notch chord",
        "scale": round(float(abs(m)), 4), "rot_deg": round(float(np.rad2deg(np.angle(m))), 3),
        "translation_at_ear_centre": None}
    mr, br, reginfo = register_head(A, B)
    info["head_registration_check"] = reginfo
    zc = complex(660, 175)
    info["ear_transform_donor_to_base"]["translation_at_ear_centre"] = [
        round(float((m * zc + b - zc).real), 2), round(float((m * zc + b - zc).imag), 2)]
    info["ear_transform_vs_head_registration_at_ear_centre_px"] = round(float(abs((m * zc + b) - (mr * zc + br))), 2)

    # --- base head outline under the old ear
    cH, finfo = base_outline(A, nA[0], nA[1])
    info["base_outline_cubic"] = {"coef": [float(v) for v in cH], **finfo}
    arc = donor_arc(B, nB[0], nB[1])
    za = (m * (arc[:, 0] + 1j * arc[:, 1]) + b)
    dev = za.imag - np.polyval(cH, za.real)
    info["donor_arc_mapped_vs_base_outline_px"] = {"mean": round(float(dev.mean()), 2),
                                                   "max_abs": round(float(np.abs(dev).max()), 2)}

    # --- palette check: donor ear tones vs base inner-ear colour
    yy, xx = np.mgrid[0:H, 0:W]
    win = (xx >= WIN[0]) & (xx < WIN[2]) & (yy >= WIN[1]) & (yy < 245)
    Einner3 = A[win & (np.abs(A - np.array([209, 155, 108.0])).max(2) < 8)]
    Einner3 = np.median(Einner3, 0)
    Ed = B[win & (np.abs(B - np.array([208, 155, 107.0])).max(2) < 6)]
    Ed = np.median(Ed, 0)
    shift = Einner3 - Ed
    info["palette"] = {"cand3_inner_ear": Einner3.tolist(), "cand4_far_ear_main": Ed.tolist(),
                       "shift_applied": shift.round(2).tolist(),
                       "cand3_tan": np.median(A[win & (np.abs(A - np.array([234, 190, 129.0])).max(2) < 8)], 0).tolist(),
                       "cand4_tan": np.median(B[win & (np.abs(B - np.array([234, 190, 129.0])).max(2) < 8)], 0).tolist()}

    pmB, einfo = donor_ear_layer(B, arc, shift)
    info["donor_ear_layer"] = einfo
    pm = warp(pmB, m, b)
    ea = pm[..., 3]
    ear_rgb = pm[..., :3] / np.maximum(ea, 1e-9)[..., None]
    ear_over_white = pm[..., :3] + 255 * (1 - ea[..., None])

    # --- old ear in the base: non-white px above the outline (in the window)
    def curve(X):
        return np.polyval(cH, X)
    xl, xr = nA[0, 0], nA[1, 0]
    above = win & (yy < curve(np.clip(xx, xl - 25, xr + 25)) - 0.5) & (xx > xl - 1)
    old = above & ((255 - A.min(2)) > 3)
    lab, n = ndi.label(old)
    old = lab == (np.argmax(ndi.sum(old, lab, range(1, n + 1))) + 1)
    old = ndi.binary_dilation(old, iterations=2) & win
    new = ea > 1e-3
    zone = (old | ndi.binary_dilation(new, iterations=2)) & win
    # head coverage
    x0, y0 = WIN[0], WIN[1]
    gy, gx = np.mgrid[WIN[1]:WIN[3], WIN[0]:WIN[2]].astype(float)
    cov = np.zeros((H, W))
    for sy in range(SS):
        for sx in range(SS):
            Xs = gx + (sx + 0.5) / SS - 0.5
            Ys = gy + (sy + 0.5) / SS - 0.5
            cov[WIN[1]:WIN[3], WIN[0]:WIN[2]] += Ys >= curve(Xs)
    cov /= SS * SS
    # local head colour: nearest base px >= 3 px below the outline and opaque-ish (not white)
    deep = win & (yy > curve(np.clip(xx, xl - 25, xr + 25)) + 3) & ((255 - A.min(2)) > 60)
    deep |= ~win & ((255 - A.min(2)) > 60)
    deep &= ~old
    _, (iy, ix) = ndi.distance_transform_edt(~deep, return_indices=True)
    Hc = A[iy, ix]
    # head alpha from the base pixel itself (white blend vs Hc) where the old ear did not touch the head
    dC, dS = 255 - A, 255 - Hc
    a_pix = np.clip((dC * dS).sum(2) / np.maximum((dS * dS).sum(2), 1e-6), 0, 1)
    geo = old & (np.abs(yy - curve(np.clip(xx, xl - 25, xr + 25))) < 3) & (xx >= xl - 2) & (xx <= xr + 2)
    a_h = np.where(geo, cov, np.where(old, (cov > 0.5) * 1.0 * (yy > curve(np.clip(xx, xl - 25, xr + 25))), a_pix))
    # inside the head (deep) the base pixel stays as it is
    head_px = np.where(a_h[..., None] >= 1.0, A, Hc)
    out = A.copy()
    res = a_h[..., None] * head_px + (1 - a_h[..., None]) * ear_over_white
    out[zone] = res[zone]
    out8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    Image.fromarray(out8).save(OUT / "original.png")
    ch = np.abs(out8.astype(int) - A.astype(int)).max(2) > 0
    ys, xs = np.nonzero(ch)
    info["changed_px"] = int(ch.sum())
    info["changed_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    info["changed_outside_window"] = int((ch & ~win).sum())
    # residual stripe/inner-ear colour below the outline (should be none)
    inner_left = (yy > curve(np.clip(xx, xl - 25, xr + 25)) + 1.5) & win & \
        (np.abs(out - np.array([209, 155, 108.0])).max(2) < 10)
    info["inner_ear_colour_px_left_inside_head"] = int(inner_left.sum())
    (OUT / "earfix.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    check_sheet(A, B, out8, ch, cH, nA, OUT / "earfix-check.png")


def check_sheet(A, B, out, ch, cH, nA, path):
    box = (560, 90, 790, 300)
    S = 3
    w, h = (box[2] - box[0]) * S, (box[3] - box[1]) * S
    tiles = []
    for lab, img in (("before (cand3)", A), ("donor (cand4)", B), ("after", out)):
        tiles.append((lab, Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8)).crop(box).resize((w, h), Image.NEAREST)))
    # after over dark (shows clean silhouette)
    o = out.astype(float)
    dark = np.where(((255 - o.min(2)) > 3)[..., None], o, 32)
    tiles.append(("after (bg px -> #202020)", Image.fromarray(dark.astype(np.uint8)).crop(box).resize((w, h), Image.NEAREST)))
    m = A * 0.35 + 255 * 0.65
    m[ch] = [255, 0, 0]
    im = Image.fromarray(m.astype(np.uint8)).crop(box).resize((w, h), Image.NEAREST)
    d = ImageDraw.Draw(im)
    xs = np.arange(nA[0, 0] - 25, nA[1, 0] + 25, 0.5)
    pts = [((x - box[0]) * S, (np.polyval(cH, x) - box[1]) * S) for x in xs]
    d.line(pts, fill=(0, 0, 255), width=2)
    tiles.append(("changed px (red) + fitted head outline (blue)", im))
    full = Image.new("RGB", (1024 // 2 * 2, 1024 // 2), (255, 255, 255))
    full.paste(Image.fromarray(A.astype(np.uint8)).resize((512, 512), Image.LANCZOS), (0, 0))
    full.paste(Image.fromarray(out).resize((512, 512), Image.LANCZOS), (512, 0))
    tiles.append(("full: before | after", full.resize((w, int(w / 2)))))
    cols = 3
    sheet = Image.new("RGB", (cols * (w + 10) + 10, 2 * (max(h, int(w / 2)) + 36) + 10), (50, 50, 50))
    dr = ImageDraw.Draw(sheet)
    for i, (lab, t) in enumerate(tiles):
        gx, gy = 10 + (i % cols) * (w + 10), 10 + (i // cols) * (h + 36)
        dr.text((gx, gy), lab, fill=(255, 255, 255), font=font(18))
        sheet.paste(t, (gx, gy + 26))
    sheet.save(path)


if __name__ == "__main__":
    main()
