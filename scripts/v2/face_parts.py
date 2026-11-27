#!/usr/bin/env python3
"""
FACE PARTS - Tori v2: blank head + swappable front-view face parts (eyes, brows, nose, mouths).

Input : art/v2/tori/original.png                  (1024x1024, the face is baked into the head)
        art/v2/tori/step5-rig/layers/head.png     (full-canvas step-5 head layer; == original on the face)
        public/assets/characters/tori-v2/         (step-5 rig: rig.json + cropped layers)
        art/v2/tori/face/work/ai-*.png            (image-model sheets; generated once if missing)
Output: public/assets/characters/tori-v2/head-blank.png         (same box as head.png)
        public/assets/characters/tori-v2/face/<part>.png + face.json
        art/v2/tori/face/  head-blank-full.png, recomposite-diff.png, expressions.png, parts.png,
                           measure.png, checks.json, work/ (AI sheets, refs, normalisation previews)

Usage   python3 scripts/v2/face_parts.py            (re-uses work/ai-*.png when present)
        python3 scripts/v2/face_parts.py --regen    (asks the image model again; costs money)

Method (short)
  * Segmentation: every face pixel's distance to the background colour segments cream-tan,
    cream-pink, tan-pink (sampled flat colours) > 12 -> feature cores; components = eyeL, eyeR,
    browL, browR, nose+mouth.  Hole = core dilated 3 px (AA fringe + the original's light halo).
  * Blank head: holes are filled with the surrounding flat colours by per-class Gaussian
    normalised convolution (cream / tan fields).  Under the nose the tan forehead wedge is continued
    from its measured sub-pixel edges (linear extrapolation of the last rows above the nose) and
    closed inside the nose silhouette (eroded 3 px), so it is always covered by the nose.  The
    cheeks are untouched (they do not meet any hole).  Alpha unchanged (head is opaque there).
  * Exact parts: alpha = projection of (pixel - fill) on (feature colour - fill); interior pixels
    keep the original values (pixel-exact), edge pixels the unmixed colour.  eye-open = the
    viewer's RIGHT eye (big white highlight upper-left + small white highlight lower-right, the
    rule); it is used for BOTH eyes (never mirrored).  brow-neutral = the viewer's LEFT brow; the
    right brow is the mirrored image.  nose = nose down to the philtrum; mouth-neutral = philtrum +
    w-smile.  Slot anchors of the reused images are snapped (<0.5 px) so that the rest pose is
    an integer shift (no resampling).
  * Variants: mouths from the image-model sheet (3x3 cells, every cell with the same nose):
    green key + decontamination, class per pixel (line / inside / tongue), per-cell scale from
    the AI nose width and the AI neutral mouth width (-> original mouth width), aligned at the
    nose bottom/philtrum; the lines are re-stroked from their skeleton at the original mouth-line
    thickness, the AI philtrum replaced by a straight one on the original philtrum axis, colours
    replaced by the original's sampled flat colours (line, eye dark brown, cheek pink).
    Eye lines (blink, happy) = skeleton of the AI strokes fitted with a polynomial, re-stroked at
    the mouth-line thickness, span = eye diameter, eye dark brown.  eye-surprised = the measured
    eye geometry (disc + 2 highlights) re-rendered 12 % larger.  eye-sad = eye-open + the AI
    tear-shine crescent (flat highlight white).  Brows = the exact brow rotated / raised.
"""
import json, pathlib, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from step5_rig import Rig, premul, norm_conv, to_img, font, GREEN  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIG = ROOT / "art/v2/tori/original.png"
HEAD_FULL = ROOT / "art/v2/tori/step5-rig/layers/head.png"
PUB = ROOT / "public/assets/characters/tori-v2"
PUBF = PUB / "face"
OUT = ROOT / "art/v2/tori/face"
WORK = OUT / "work"
H = W = 1024
SS = 8                      # supersampling for every rendered (non-exact) part
PAD = 2                     # transparent margin around each crop
FACE = (320, 280, 720, 560)  # x0,y0,x1,y1 window holding the whole face
EYE_SCALE_SURPRISED = 1.12

# ----------------------------------------------------------------------------- image model
AI_SHEETS = {
    "ai-mouths-1.png": dict(refs=["ref-nose-mouth.png", "ref-face.png"], prompt=(
        "Reference images: a cute flat-vector baby chipmunk face (lineless flat style, no outlines). Its nose is a "
        "soft rounded brown triangle, and under it a thin brown line goes down and splits into a small 'w'-shaped "
        "smile. Task: draw a 3x3 grid of 9 separate mouth expression variants for THIS character, for a 2D animation "
        "lip-sync and expression set. Every cell contains ONLY the same brown nose (identical size and shape in every "
        "cell, copied from the reference) and the short thin brown line under it, plus the mouth. Same scale in all "
        "cells, nose centered horizontally near the top of each cell, generous spacing between cells. Draw on a "
        "perfectly flat pure green #00FF00 background - no face, no fur, no cream skin, no shadows, no gradients, no "
        "text, no grid lines. Style: exactly the reference's flat lineless vector style, the same thin brown line "
        "weight as the reference smile, no black outlines. Open mouths are flat solid dark reddish-brown inside with "
        "a flat pink tongue, no teeth, no outline around them. Cells in reading order: 1) the original gentle "
        "w-shaped smile, unchanged; 2) big happy open smile (wide D shape, pink tongue); 3) talking, mouth slightly "
        "open (small); 4) talking, medium open; 5) talking, wide open; 6) sad, small downturned frown line; 7) grumpy "
        "pout, short wavy line; 8) surprised, small round o mouth; 9) wide flat 'ee' mouth, open thin.")),
    "ai-eyes-1.png": dict(refs=["ref-eye.png", "ref-face.png"], prompt=(
        "Reference images: a cute flat-vector baby chipmunk face (lineless flat style). Its eye is a big round solid "
        "dark brown circle with one big round white highlight at upper-left and one small round white highlight at "
        "lower-right. Task: draw a 2x3 grid of 6 separate eye variants of THIS exact eye for a 2D animation "
        "expression set, all at the same scale as each other (the open eye the same diameter as the reference "
        "circle), generous spacing. Perfectly flat pure green #00FF00 background - no face, no skin, no shadows, no "
        "gradients, no text, no grid lines. Same flat lineless vector style and the same dark brown color, no "
        "outlines. Cells in reading order: 1) the open eye exactly as the reference; 2) closed eye: one gentle "
        "downward-curving thick dark brown line (like a relaxed closed eye), as wide as the open eye, rounded ends; "
        "3) happy closed eye: an upward arc like a caret ^ shape, dark brown thick line, rounded ends, as wide as the "
        "open eye; 4) surprised: the open eye slightly bigger and rounder, same two white highlights; 5) teary sad "
        "eye: the open eye with the same highlights plus a thin glossy white shine along the bottom inside edge; "
        "6) sleepy half-closed eye: lower half of the dark brown circle with a flat top.")),
}


def ensure_ai(regen=False):
    WORK.mkdir(parents=True, exist_ok=True)
    o = Image.open(ORIG).convert("RGB")
    refs = {"ref-nose-mouth.png": ((432, 410, 592, 520), (640, 440)),
            "ref-face.png": ((320, 280, 704, 520), (960, 600)),
            "ref-eye.png": ((570, 354, 688, 472), (472, 472))}
    for n, (box, size) in refs.items():
        if not (WORK / n).exists():
            o.crop(box).resize(size, Image.LANCZOS).save(WORK / n)
    used = []
    for name, spec in AI_SHEETS.items():
        if (WORK / name).exists() and not regen:
            continue
        cmd = ["python3", str(ROOT / "scripts/gen_image.py"), "--prompt", spec["prompt"], "--out", str(WORK / name),
               "--aspect", "1:1", "--size", "1K"]
        for r in spec["refs"]:
            cmd += ["--ref", str(WORK / r)]
        subprocess.run(cmd, check=True)
        used.append(name)
    return used


# ----------------------------------------------------------------------------- helpers
def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1)


def proj_alpha(P, B, F):
    d = F - B
    num = ((P - B) * d).sum(-1)
    den = np.maximum((d * d).sum(-1), 1e-6)
    return np.clip(num / den, 0, 1)


def disc(r):
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return xx * xx + yy * yy <= r * r


def thin(mask):
    """Zhang-Suen thinning (bool -> bool skeleton)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = np.pad(mask[y0:y1, x0:x1].astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = m[:-2, 1:-1], m[:-2, 2:], m[1:-1, 2:]
            p5, p6, p7 = m[2:, 2:], m[2:, 1:-1], m[2:, :-2]
            p8, p9 = m[1:-1, :-2], m[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(s.astype(np.int16) for s in seq[:8])
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rem = (m[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rem.any():
                m[1:-1, 1:-1][rem] = 0
                changed = True
        if not changed:
            break
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = m[1:-1, 1:-1].astype(bool)
    return out


def ss_grid(box, ss=SS):
    x0, y0, x1, y1 = box
    xs = x0 + (np.arange((x1 - x0) * ss) + 0.5) / ss
    ys = y0 + (np.arange((y1 - y0) * ss) + 0.5) / ss
    return np.meshgrid(xs, ys)


def ss_to_canvas(box, paint, ss=SS):
    """paint = [(bool SS mask, rgb)] painted in order -> full-canvas premultiplied RGBA (float, a 0..1)."""
    x0, y0, x1, y1 = box
    hh, ww = (y1 - y0) * ss, (x1 - x0) * ss
    rgb = np.zeros((hh, ww, 3)); a = np.zeros((hh, ww))
    for m, col in paint:
        rgb[m] = col; a[m] = 1
    pm = np.concatenate([rgb * a[..., None], a[..., None]], -1)
    pm = pm.reshape(y1 - y0, ss, x1 - x0, ss, 4).mean((1, 3))
    out = np.zeros((H, W, 4)); out[y0:y1, x0:x1] = pm
    return out


def over(top, bot):
    return top + bot * (1 - top[..., 3:4])


def unpremul_u8(pm):
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 0)
    return np.clip(np.round(np.concatenate([rgb, a * 255], -1)), 0, 255).astype(np.uint8)


def stroke(box, pts, t, ss=SS):
    """SS bool mask of a round-capped stroke of width t along dense canvas points pts (N,2)."""
    X, Y = ss_grid(box, ss)
    hh, ww = X.shape
    seed = np.zeros((hh, ww), bool)
    ij = np.round((pts - [box[0], box[1]]) * ss - 0.5).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < ww) & (ij[:, 1] >= 0) & (ij[:, 1] < hh)
    seed[ij[ok, 1], ij[ok, 0]] = True
    return ndi.distance_transform_edt(~seed) <= t * ss / 2


def dense(pts, step=0.05):
    out = [pts[0]]
    for p, q in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(q - p) / step)))
        out += [p + (q - p) * k / n for k in range(1, n + 1)]
    return np.array(out)


def crop_part(pm, slot):
    """premultiplied full-canvas part -> (uint8 RGBA crop, x, y, anchor in crop px)."""
    a = pm[..., 3]
    ys, xs = np.nonzero(a > 1 / 512)
    x0, y0 = xs.min() - PAD, ys.min() - PAD
    x1, y1 = xs.max() + 1 + PAD, ys.max() + 1 + PAD
    im = unpremul_u8(pm[y0:y1, x0:x1])
    return im, int(x0), int(y0), [round(slot[0] - x0, 3), round(slot[1] - y0, 3)]


# ----------------------------------------------------------------------------- analysis of the original
def analyse(o):
    x0, y0, x1, y1 = FACE
    P = o[y0:y1, x0:x1]
    cream = np.median(o[330:345, 420:440].reshape(-1, 3), 0)
    tan = np.median(o[335:345, 505:520].reshape(-1, 3), 0)
    pink = np.median(o[465:478, 322:338].reshape(-1, 3), 0)
    d = np.minimum.reduce([seg_dist(P, cream, tan), seg_dist(P, cream, pink), seg_dist(P, tan, pink)])
    lab, n = ndi.label(d > 12)
    sizes = ndi.sum(np.ones_like(d), lab, range(1, n + 1))
    cores = {}
    for i in np.nonzero(sizes > 300)[0] + 1:
        ys, xs = np.nonzero(lab == i)
        cx, cy = xs.mean() + x0, ys.mean() + y0
        if sizes[i - 1] > 5000:
            name = "eyeL" if cx < 512 else "eyeR"
        elif cy < 340:
            name = "browL" if cx < 512 else "browR"
        elif abs(cx - 512) < 20 and cy > 415:
            name = "nosemouth"
        else:
            continue
        m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = lab == i
        # drop thin spurs (the slightly dark cream/tan AA edge touching the nose), keep the main blob
        m = ndi.binary_opening(m, iterations=1)
        ml, mn = ndi.label(m)
        m = ml == (np.argmax(ndi.sum(m, ml, range(1, mn + 1))) + 1)
        cores[name] = m
    assert sorted(cores) == ["browL", "browR", "eyeL", "eyeR", "nosemouth"], sorted(cores)
    return dict(cream=cream, tan=tan, pink=pink, cores=cores, dist=d)


def wedge_edges(o, cream, tan, rows):
    """sub-pixel left/right edge of the tan forehead wedge (x in pixel-edge coords) per row."""
    ab = tan - cream
    c = np.clip(((o - cream) @ ab) / (ab @ ab), 0, 1)
    xl = np.array([512 - c[y, 450:512].sum() for y in rows])
    xr = np.array([512 + c[y, 512:575].sum() for y in rows])
    return xl, xr


def blank_head(o, head, A):
    cream, tan = A["cream"], A["tan"]
    holes = {k: ndi.binary_dilation(m, iterations=3) for k, m in A["cores"].items()}
    hole = np.logical_or.reduce(list(holes.values()))
    x0, y0, x1, y1 = FACE
    win = (slice(y0 - 40, y1 + 40), slice(x0 - 40, x1 + 40))
    ab = tan - cream
    ctan = np.clip(((o - cream) @ ab) / (ab @ ab), 0, 1)
    dd = np.minimum(seg_dist(o, cream, tan), 999)
    known_c = ~hole & (ctan < 0.04) & (dd < 8)
    known_t = ~hole & (ctan > 0.96) & (dd < 8)
    sig = (2.0, 4.0, 8.0, 16.0, 32.0)
    fc = np.zeros_like(o); ft = np.zeros_like(o)
    fc[win] = norm_conv(o[win], known_c[win], sig)
    ft[win] = norm_conv(o[win], known_t[win], sig)
    # tan coverage inside the nose/mouth hole: wedge continuation, closed inside the nose
    nm = holes["nosemouth"]
    hy = np.nonzero(nm.any(1))[0]
    htop = hy.min()
    fit_rows = np.arange(htop - 10, htop)
    xl, xr = wedge_edges(o, cream, tan, fit_rows)
    pl = np.polyfit(fit_rows + 0.5, xl, 1); pr = np.polyfit(fit_rows + 0.5, xr, 1)
    core = A["cores"]["nosemouth"]
    # nose core = the part of the nose+mouth core above the philtrum split
    split = nose_split(core)
    nose_core = core.copy(); nose_core[split:] = False
    E = ndi.binary_erosion(nose_core, iterations=3)
    nw = nose_core.sum(1)
    nose_top = np.nonzero(nw)[0].min()
    widest = nose_top + int(np.argmax(nw[nose_top:nose_top + 30]))
    cov = np.zeros((H, W))
    ys, xs = np.nonzero(nm)
    sub = (np.arange(SS) + 0.5) / SS
    for y, x in zip(ys, xs):
        yy = y + sub[:, None] + 0 * sub[None, :]
        xx = x + sub[None, :] + 0 * sub[:, None]
        inw = (xx > np.polyval(pl, yy)) & (xx < np.polyval(pr, yy))
        keep = (yy < widest) | E[y, x]          # wedge down to the nose's widest row, then inside the nose
        cov[y, x] = (inw & keep).mean()
    fill = fc * (1 - cov[..., None]) + ft * cov[..., None]
    blank = head.astype(float).copy()
    blank[..., :3][hole] = fill[hole]
    info = dict(wedge_fit_rows=[int(fit_rows[0]), int(fit_rows[-1])],
                wedge_left=[round(v, 4) for v in pl], wedge_right=[round(v, 4) for v in pr],
                nose_erode_px=3, hole_dilate_px=3, nose_top=int(nose_top),
                nose_widest_row=int(widest))
    return blank, fill, holes, info


def nose_split(core):
    """first row below the widest nose row whose core width <= 10 px (nose taper -> philtrum)."""
    widths = core.sum(1)
    ys = np.nonzero(widths)[0]
    top = ys.min()
    widest = top + int(np.argmax(widths[top:top + 40]))
    for y in range(widest, ys.max()):
        if widths[y] <= 10:
            return y
    raise RuntimeError("no philtrum")


def extract(o, fill, hole, core):
    """exact feature layer: interior = original pixels, edge = unmixed against the fill."""
    inner = ndi.binary_erosion(ndi.binary_fill_holes(core), iterations=1)
    Ffield = norm_conv(o, inner, (1.0, 2.0, 4.0, 8.0))
    a = proj_alpha(o, fill, Ffield) * hole
    a[a < 0.03] = 0
    a[inner] = 1
    rgb_un = (o - (1 - a[..., None]) * fill) / np.maximum(a[..., None], 1e-6)
    rgb = np.where((a >= 0.6)[..., None], np.clip(rgb_un, 0, 255), Ffield)
    rgb[inner] = o[inner]
    return np.concatenate([rgb * a[..., None], a[..., None]], -1)


def centroid(a):
    ys, xs = np.mgrid[0:H, 0:W]
    s = a.sum()
    return [float((a * (xs + 0.5)).sum() / s), float((a * (ys + 0.5)).sum() / s)]


def snap(slot_measured, anchor_src_canvas):
    """slot near the measured one such that slot - anchor_src is integral (pure integer shift)."""
    d = np.round(np.array(slot_measured) - np.array(anchor_src_canvas))
    return [float(v) for v in np.array(anchor_src_canvas) + d]


# ----------------------------------------------------------------------------- AI sheet keying
def key_sheet(path, fg_metric):
    im = np.array(Image.open(path).convert("RGB")).astype(float)
    border = np.concatenate([im[:6].reshape(-1, 3), im[-6:].reshape(-1, 3), im[:, :6].reshape(-1, 3),
                             im[:, -6:].reshape(-1, 3)])
    bg = np.median(border, 0)
    g = im[..., 1] - np.maximum(im[..., 0], im[..., 2])
    gb = bg[1] - max(bg[0], bg[2])
    a = np.clip((gb - g) / (gb - fg_metric), 0, 1)
    F = np.where((a > 0.15)[..., None], (im - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 1e-3), 0)
    return im, a, np.clip(F, 0, 255), bg


def cells(a, n=3):
    lab, k = ndi.label(a > 0.5)
    out = {}
    if k == 0:
        return out
    com = ndi.center_of_mass(np.ones_like(a), lab, range(1, k + 1))
    sz = ndi.sum(np.ones_like(a), lab, range(1, k + 1))
    for i, (cy, cx) in enumerate(com):
        if sz[i] < 40:
            continue
        c = int(cy * n // a.shape[0]) * n + int(cx * n // a.shape[1])
        out.setdefault(c, np.zeros_like(a, bool))
        out[c] |= lab == i + 1
    return out


def row_widths(m):
    return m.sum(1)


# ----------------------------------------------------------------------------- mouths from the AI sheet
AI_MOUTH_CELLS = {0: "mouth-neutral-ai", 1: "mouth-smile-open", 2: "mouth-talk-s", 3: "mouth-talk-m",
                  4: "mouth-talk-l", 5: "mouth-sad", 6: "mouth-angry", 7: "mouth-o", 8: "mouth-ee"}


def ai_mouths(M, colors):
    im, a, F, bg = key_sheet(WORK / "ai-mouths-1.png", -40.0)
    centers = {"line": np.array([160, 121, 92.]), "inside": np.array([100, 40, 36.]),
               "tongue": np.array([232, 122, 130.])}
    names = list(centers)
    dist = np.stack([np.linalg.norm(F - centers[k], axis=-1) for k in names], -1)
    cls = np.argmin(dist, -1)
    cm = cells(a)
    t = M["line_thickness"]
    phil_x, split = M["philtrum_x"], M["split_y"]
    # per cell geometry in AI px
    geo = {}
    for c, m in cm.items():
        mm = m | (ndi.binary_dilation(m, iterations=2) & (a > 0.05))
        w = row_widths(m)
        ys = np.nonzero(w)[0]
        top = ys.min()
        widest = top + int(np.argmax(w[top:top + 60]))
        nose_w = w[widest]
        wv = w[top:top + 150]; wv = wv[wv > 0]
        lw = float(np.median(wv[wv <= wv.min() + 3]))
        nb = next(y for y in range(widest, ys.max()) if w[y] <= 1.5 * lw)
        xs = np.nonzero(m[nb + 2:nb + 8].any(0))[0]
        px = (xs.min() + xs.max() + 1) / 2
        below = m.copy(); below[:nb] = False
        bx = np.nonzero(below.any(0))[0]
        geo[c] = dict(mask=mm, top=int(top), nose_w=float(nose_w), lw=float(lw), nb=int(nb), px=float(px),
                      mouth_w=float(bx.max() + 1 - bx.min()))
    s1 = M["mouth_width"] / geo[0]["mouth_w"]
    results, info = {}, {}
    box = (int(phil_x) - 90, split - 10, int(phil_x) + 90, split + 120)
    X, Y = ss_grid(box)
    for c, name in AI_MOUTH_CELLS.items():
        g = geo[c]
        s = s1 * geo[0]["nose_w"] / g["nose_w"]
        xa = g["px"] + (X - phil_x) / s
        ya = g["nb"] + (Y - split) / s
        am = ndi.gaussian_filter(a * g["mask"], 0.7)
        alpha_s = ndi.map_coordinates(am, [ya - 0.5, xa - 0.5], order=1, cval=0)
        probs = []
        for k in range(3):
            pk = ndi.gaussian_filter(((cls == k) * a * g["mask"]).astype(float), 0.7)
            probs.append(ndi.map_coordinates(pk, [ya - 0.5, xa - 0.5], order=1, cval=0))
        lab_s = np.argmax(np.stack(probs, -1), -1)
        fg = (alpha_s > 0.5) & (Y >= split)
        L, D, K = fg & (lab_s == 0), fg & (lab_s == 1), fg & (lab_s == 2)
        shape = L | D | K
        wrow = shape.sum(1)
        thr = 2.5 * t * SS
        mt = int(np.nonzero(wrow > thr)[0].min())             # SS row where the mouth starts
        cols = np.nonzero(shape[mt:].any(0))[0]
        xc = (cols.min() + cols.max() + 1) / 2
        dx = int(round((phil_x - box[0]) * SS - xc))
        L, D, K = (np.roll(v, dx, 1) for v in (L, D, K))
        shape = L | D | K
        # line = only the brown band along the silhouette (outline / philtrum / line mouths); brownish AA
        # between the inside and the tongue is not a line -> goes back to the nearest fill class
        ext_d = ndi.distance_transform_edt(ndi.binary_fill_holes(shape))
        L = ndi.binary_closing(L, iterations=3) & shape & (ext_d <= 1.6 * g["lw"] * s * SS)
        D, K = D & ~L, K & ~L
        sk = thin(L)
        sk &= ext_d <= 0.8 * g["lw"] * s * SS          # drop inward spurs (true centreline ~0.5 lw deep)
        sk[:mt] = False
        pc = int(round((phil_x - box[0]) * SS - 0.5))
        hit = np.nonzero(shape[mt:, pc])[0]
        yhit = mt + (int(hit.min()) if len(hit) else 0)
        r = t * SS / 2
        sk[int((split - 6 - box[1]) * SS):int(yhit + 0.5 * r), pc] = True
        newL = ndi.distance_transform_edt(~sk) <= r
        inner = ndi.binary_fill_holes(newL | D | K) & ~newL
        DK = D | K
        if DK.any():
            _, (iy, ix) = ndi.distance_transform_edt(~DK, return_indices=True)
            Kn = K[iy, ix]
            Kin, Din = inner & Kn, inner & ~Kn
        else:
            Kin = Din = np.zeros_like(newL)
        paint = [(Din, colors["inside"]), (Kin, colors["tongue"]), (newL, colors["line"])]
        results[name] = ss_to_canvas(box, paint)
        info[name] = dict(ai_cell=c + 1, scale=round(float(s), 4), ai_nose_w=g["nose_w"], ai_line_w=g["lw"],
                          ai_line_w_scaled=round(float(g["lw"] * s), 2), x_shift_px=round(dx / SS, 3))
        # preview of the normalisation (AI cell crop | normalised)
    return results, info, geo


# ----------------------------------------------------------------------------- eyes
def eye_vector(cx, cy, k, G, colors):
    """measured eye geometry (disc + 2 highlights) re-rendered at scale k about the eye centre."""
    R = G["eye_d"] / 2 * k
    box = (int(cx - R) - 3, int(cy - R) - 3, int(cx + R) + 4, int(cy + R) + 4)
    X, Y = ss_grid(box)
    dsk = (X - cx) ** 2 + (Y - cy) ** 2 <= R * R
    paint = [(dsk, colors["eye"])]
    for hk in ("hl_big", "hl_small"):
        h = G[hk]
        hx, hy = cx + h["dx"] * k, cy + h["dy"] * k
        hr = h["d"] / 2 * k
        paint.append((((X - hx) ** 2 + (Y - hy) ** 2 <= hr * hr) & dsk, colors[hk]))
    return ss_to_canvas(box, paint)


def ai_eye_curves():
    im, a, F, bg = key_sheet(WORK / "ai-eyes-1.png", 0.0)
    cm = cells(a)
    out = {}
    # open eye (cell 1) -> AI eye diameter
    m0 = ndi.binary_fill_holes(cm[0])
    out["ai_eye_d"] = float(2 * np.sqrt(m0.sum() / np.pi))
    for c, nm in ((1, "blink"), (2, "happy")):
        m = ndi.binary_opening(ndi.gaussian_filter(cm[c].astype(float), 1.0) > 0.5, iterations=1)
        sk = thin(m)
        ys, xs = np.nonzero(sk)
        lw = m.sum() / max(len(xs), 1)
        xm = (xs.min() + xs.max()) / 2
        hw = (xs.max() - xs.min()) / 2
        yend = (ys[xs == xs.min()].mean() + ys[xs == xs.max()].mean()) / 2
        u = (xs - xm) / hw
        v = (ys - yend) / hw
        coef = np.polyfit(np.r_[u, -u], np.r_[v, v], 4)    # symmetric fit (mirrored samples)
        coef[[1, 3]] = 0
        out[nm] = dict(coef=coef.tolist(), ai_line_w=float(lw), ai_span=float(2 * hw + lw))
    # tear shine (cell 5)
    m4 = cm[4]
    dsk = ndi.binary_fill_holes(m4)
    ys, xs = np.nonzero(dsk)
    R = np.sqrt(dsk.sum() / np.pi)
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
    lum = F.mean(-1)
    white = dsk & (lum > 150)
    lab, n = ndi.label(white)
    best, bw = None, 0
    for i in range(1, n + 1):
        yy, xx = np.nonzero(lab == i)
        if yy.mean() > cy and xx.max() - xx.min() > bw:
            best, bw = i, xx.max() - xx.min()
    shine = ndi.gaussian_filter((lab == best).astype(float), 0.8)
    out["shine"] = dict(mask=shine, cx=cx, cy=cy, R=R)
    return out


def eye_line(cx, cy, curve, D, t, depth_max, mid_dy, color):
    coef = np.array(curve["coef"])
    u = np.linspace(-1, 1, 400)
    v = np.polyval(coef, u)
    v = v - v[0]
    depth = np.abs(v).max()
    k = min(1.0, depth_max / depth) if depth > 0 else 1.0
    hspan = (D - t) / 2
    X = cx + u * hspan
    Yc = v * k * hspan
    Y = cy + mid_dy + Yc - (Yc.min() + Yc.max()) / 2
    pts = np.stack([X, Y], -1)
    box = (int(cx - D / 2) - 4, int(Y.min() - t) - 4, int(cx + D / 2) + 5, int(Y.max() + t) + 5)
    return ss_to_canvas(box, [(stroke(box, dense(pts), t), color)]), dict(depth_ratio=round(float(depth * k / 2), 3))


# ----------------------------------------------------------------------------- brows
def transform_part(pm, center, deg, dx=0.0, dy=0.0, flip=False):
    """resample a premultiplied full-canvas part: rotate by deg (+ = clockwise) about center, then shift."""
    a = pm[..., 3]
    ys, xs = np.nonzero(a > 0)
    r = np.hypot(xs - center[0], ys - center[1]).max() + 4
    box = (int(center[0] - r + dx) - 2, int(center[1] - r + dy) - 2, int(center[0] + r + dx) + 3,
           int(center[1] + r + dy) + 3)
    jj, ii = np.mgrid[box[1]:box[3], box[0]:box[2]].astype(float)
    px, py = ii + 0.5 - dx - center[0], jj + 0.5 - dy - center[1]
    t = np.deg2rad(-deg)
    sx = np.cos(t) * px - np.sin(t) * py
    sy = np.sin(t) * px + np.cos(t) * py
    if flip:
        sx = -sx
    sx, sy = sx + center[0] - 0.5, sy + center[1] - 0.5
    out = np.zeros_like(pm)
    for k in range(4):
        out[box[1]:box[3], box[0]:box[2], k] = ndi.map_coordinates(pm[..., k], [sy, sx], order=1, cval=0)
    return out


def tilt_deg(pm):
    """(tilt of the long axis in deg, + = right end lower; length along it; thickness = area / length)."""
    a = pm[..., 3]
    ys, xs = np.nonzero(a > 0.5)
    w = a[ys, xs]
    mx, my = np.average(xs, weights=w), np.average(ys, weights=w)
    cov = np.cov(np.stack([xs - mx, ys - my]), aweights=w)
    ev, V = np.linalg.eigh(cov)
    v = V[:, -1]
    if v[0] < 0:
        v = -v
    pr = (xs - mx) * v[0] + (ys - my) * v[1]
    length = float(pr.max() - pr.min() + 1)
    return float(np.degrees(np.arctan2(v[1], v[0]))), length, float(a.sum() / length)


# ----------------------------------------------------------------------------- compositing
def place(img_u8, anchor, slot, mirror=False):
    """crop image -> premultiplied full canvas at slot (image top-left = slot - anchor)."""
    im = np.array(img_u8)
    ax, ay = anchor
    if mirror:
        im = im[:, ::-1]
        ax = im.shape[1] - ax
    pm = premul(im)
    ox, oy = slot[0] - ax, slot[1] - ay
    out = np.zeros((H, W, 4))
    if abs(ox - round(ox)) < 1e-6 and abs(oy - round(oy)) < 1e-6:
        ox, oy = int(round(ox)), int(round(oy))
        out[oy:oy + im.shape[0], ox:ox + im.shape[1]] = pm
        return out
    x0, y0 = int(np.floor(ox)) - 1, int(np.floor(oy)) - 1
    jj, ii = np.mgrid[y0:y0 + im.shape[0] + 3, x0:x0 + im.shape[1] + 3].astype(float)
    for k in range(4):
        out[y0:y0 + im.shape[0] + 3, x0:x0 + im.shape[1] + 3, k] = ndi.map_coordinates(
            pm[..., k], [jj - oy, ii - ox], order=1, cval=0)
    return out


ZORDER = ["eyeL", "eyeR", "browL", "browR", "mouth", "nose"]


def compose_face(blank_u8, manifest, parts_img, sel):
    acc = premul(blank_u8)
    for slot in ZORDER:
        pn = sel[slot]
        p = manifest["parts"][pn]
        s = manifest["slots"][slot]
        acc = over(place(parts_img[pn], p["anchor"], s["anchor"], s["mirror"]), acc)
    return acc


def expression_sel(manifest, name):
    e = manifest["expressions"][name]
    return {"eyeL": e["eyes"], "eyeR": e["eyes"], "browL": e["brows"], "browR": e["brows"],
            "nose": "nose", "mouth": e["mouth"]}


def render_with_head(R, head_pm_full, pose=None, bg=(255, 255, 255), box=(0, 0, W, H), scale=1.0):
    hp = R.rig["parts"]["head"]
    keep = R.pm["head"]
    R.pm["head"] = head_pm_full[hp["y"]:hp["y"] + hp["h"], hp["x"]:hp["x"] + hp["w"]]
    try:
        return R.render(pose or {}, bg, box=box, scale=scale)
    finally:
        R.pm["head"] = keep


def checker(h, w, s=8):
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // s + xx // s) % 2).astype(float)
    return np.stack([200 + 40 * c] * 3, -1)


# ----------------------------------------------------------------------------- main
def main(regen=False):
    PUBF.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    ai_used = ensure_ai(regen)
    o = np.array(Image.open(ORIG).convert("RGB")).astype(float)
    head = np.array(Image.open(HEAD_FULL).convert("RGBA")).astype(float)
    x0, y0, x1, y1 = FACE
    assert np.abs(head[y0:y1, x0:x1, :3] - o[y0:y1, x0:x1]).max() == 0, "head layer differs from original on face"
    assert head[y0:y1, x0:x1, 3].min() == 255
    A = analyse(o)
    blank, fill, holes, binfo = blank_head(o, head, A)
    blank_u8 = np.clip(np.round(blank), 0, 255).astype(np.uint8)

    # --- exact parts (full canvas premultiplied)
    ex = {k: extract(o, fill, holes[k], A["cores"][k]) for k in A["cores"]}
    core = A["cores"]["nosemouth"]
    split = nose_split(core)
    nose_pm = ex["nosemouth"].copy(); nose_pm[split:] = 0
    mouth_pm = ex["nosemouth"].copy(); mouth_pm[:split] = 0
    # philtrum: tuck 6 rows under the (opaque) nose by repeating the first philtrum row upward
    for y in range(split - 6, split):
        mouth_pm[y] = mouth_pm[split]
    # --- measurements
    eyeL_c = centroid(ex["eyeL"][..., 3])
    eyeR_c = centroid(ex["eyeR"][..., 3])
    eye_d = float(2 * np.sqrt((ex["eyeR"][..., 3].sum() + ex["eyeL"][..., 3].sum()) / 2 / np.pi))
    # highlights of the source (right) eye
    whiteness = np.clip((o.mean(-1) - 81) / (253 - 81), 0, 1) * ndi.binary_erosion(
        ex["eyeR"][..., 3] > 0.99, iterations=3)
    lab, n = ndi.label(whiteness > 0.3)
    hls = []
    for i in range(1, n + 1):
        m = ndi.binary_dilation(lab == i, iterations=2)
        w = whiteness * m
        if w.sum() < 50:
            continue
        c = centroid(w)
        hls.append(dict(dx=c[0] - eyeR_c[0], dy=c[1] - eyeR_c[1], d=float(2 * np.sqrt(w.sum() / np.pi)),
                        color=np.median(o[ndi.binary_erosion(lab == i, iterations=2)], 0)))
    hls.sort(key=lambda h: -h["d"])
    eye_inner = ndi.binary_erosion(ex["eyeR"][..., 3] > 0.99, iterations=3) & (whiteness < 0.05)
    eye_col = np.median(o[eye_inner], 0)
    # left eye's lower-right mark (brown dot in the original)
    dotm = ndi.binary_erosion(ex["eyeL"][..., 3] > 0.99, iterations=2) & (np.abs(o - [124, 89, 61]).max(-1) < 25)
    dot_c = centroid(dotm.astype(float))
    phil_rows = mouth_pm[split + 3:split + 16, :, 3]
    t_line = float(np.median(phil_rows.sum(1)))
    xs_ = np.arange(W) + 0.5
    phil_x = float((phil_rows * xs_).sum() / phil_rows.sum())
    ma = mouth_pm[..., 3].copy(); ma[:split] = 0
    mx = np.nonzero(ma.max(0) > 0.5)[0]; my = np.nonzero(ma.max(1) > 0.5)[0]
    mouth_w = float(mx.max() + 1 - mx.min())
    line_col = np.median(o[(ma > 0.99) & (np.arange(H)[:, None] >= split)], 0)
    na = nose_pm[..., 3]
    nx = np.nonzero(na.max(0) > 0.5)[0]; ny = np.nonzero(na.max(1) > 0.5)[0]
    nose_c = centroid(na)
    nose_col = np.median(o[ndi.binary_erosion(na > 0.99, iterations=2)], 0)
    browL_c = centroid(ex["browL"][..., 3]); browR_c = centroid(ex["browR"][..., 3])
    browL_tilt, browL_len, browL_th = tilt_deg(ex["browL"])
    browR_tilt, _, _ = tilt_deg(ex["browR"])
    brow_area = float(ex["browL"][..., 3].sum())
    brow_col = np.median(o[ndi.binary_erosion(ex["browL"][..., 3] > 0.99, iterations=2)], 0)
    colors = {"eye": eye_col, "hl_big": hls[0]["color"], "hl_small": hls[1]["color"], "line": line_col,
              "inside": eye_col, "tongue": A["pink"], "brow": brow_col, "nose": nose_col}
    G = dict(eye_d=eye_d, hl_big=hls[0], hl_small=hls[1])
    M = dict(line_thickness=t_line, philtrum_x=phil_x, split_y=int(split), mouth_width=mouth_w)

    # --- slots (anchors on the 1024 canvas)
    slots = {
        "eyeR": dict(anchor=eyeR_c, measured=eyeR_c, mirror=False),
        "eyeL": dict(anchor=snap(eyeL_c, eyeR_c), measured=eyeL_c, mirror=False),
        "browL": dict(anchor=browL_c, measured=browL_c, mirror=False),
        "nose": dict(anchor=nose_c, measured=nose_c, mirror=False),
        "mouth": dict(anchor=[phil_x, float(split)], measured=[phil_x, float(split)], mirror=False),
    }
    parts_pm = {}           # full-canvas premultiplied, drawn at the SOURCE slot
    src_slot = {}
    parts_pm["eye-open"] = ex["eyeR"]; src_slot["eye-open"] = "eyeR"
    parts_pm["brow-neutral"] = ex["browL"]; src_slot["brow-neutral"] = "browL"
    parts_pm["nose"] = nose_pm; src_slot["nose"] = "nose"
    parts_pm["mouth-neutral"] = mouth_pm; src_slot["mouth-neutral"] = "mouth"
    # --- eyes
    cx, cy = eyeR_c
    vec1 = eye_vector(cx, cy, 1.0, G, colors)
    parts_pm["eye-surprised"] = eye_vector(cx, cy, EYE_SCALE_SURPRISED, G, colors)
    ai = ai_eye_curves()
    parts_pm["eye-blink"], blink_i = eye_line(cx, cy, ai["blink"], eye_d, t_line, 0.30, 0.18 * eye_d / 2,
                                              colors["eye"])
    parts_pm["eye-happy"], happy_i = eye_line(cx, cy, ai["happy"], eye_d, t_line, 0.42, 0.0, colors["eye"])
    sh = ai["shine"]
    box = (int(cx - eye_d / 2) - 2, int(cy - eye_d / 2) - 2, int(cx + eye_d / 2) + 3, int(cy + eye_d / 2) + 3)
    X, Y = ss_grid(box)
    k = sh["R"] / (eye_d / 2)
    smp = ndi.map_coordinates(sh["mask"], [sh["cy"] + (Y - cy) * k - 0.5, sh["cx"] + (X - cx) * k - 0.5], order=1)
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= (eye_d / 2 - 2.5) ** 2
    # the AI crescent is bold at this size: thin it to ~45 % (erode at SS), keep its arc and tapering ends
    shine = (smp > 0.5) & inside
    th_ss = 2 * ndi.distance_transform_edt(shine).max()
    shine = ndi.binary_erosion(shine, iterations=int(th_ss * 0.28))
    shine_pm = ss_to_canvas(box, [(shine, colors["hl_big"])])
    parts_pm["eye-sad"] = over(shine_pm, ex["eyeR"])
    for n_ in ("eye-surprised", "eye-blink", "eye-happy", "eye-sad"):
        src_slot[n_] = "eyeR"
    # --- brows: exact brow rotated / shifted (viewer's LEFT brow; inner end = right end)
    brow_specs = {"brow-angry": dict(tilt=+16.0, dy=+2.0), "brow-sad": dict(tilt=-26.0, dy=-1.0),
                  "brow-worried": dict(tilt=-24.0, dy=-8.0), "brow-raised": dict(tilt=browL_tilt - 2, dy=-10.0)}
    for n_, sp in brow_specs.items():
        parts_pm[n_] = transform_part(ex["browL"], browL_c, sp["tilt"] - browL_tilt, 0, sp["dy"])
        src_slot[n_] = "browL"
    # --- mouths from the AI sheet
    mouths, minfo, geo = ai_mouths(M, colors)
    for n_, pm in mouths.items():
        parts_pm[n_] = pm; src_slot[n_] = "mouth"

    # --- crops + browR slot snapped for the mirrored brow
    parts_img, parts_meta = {}, {}
    for n_, pm in parts_pm.items():
        im, px, py, anc = crop_part(pm, slots[src_slot[n_]]["anchor"])
        parts_img[n_] = Image.fromarray(im, "RGBA")
        parts_meta[n_] = dict(file=f"{n_}.png", w=im.shape[1], h=im.shape[0], anchor=anc,
                              sourceBox=[px, py])
    bn = parts_meta["brow-neutral"]
    mir_anchor_canvas_x = bn["sourceBox"][0] + (bn["w"] - bn["anchor"][0])     # not used directly
    ax_m = bn["w"] - bn["anchor"][0]
    sx = round(browR_c[0] - ax_m) + ax_m
    sy = round(browR_c[1] - bn["anchor"][1]) + bn["anchor"][1]
    slots["browR"] = dict(anchor=[float(sx), float(sy)], measured=browR_c, mirror=True)
    _ = mir_anchor_canvas_x

    # --- outputs: blank head (cropped like head.png) + parts
    hp = json.loads((PUB / "rig.json").read_text())["parts"]["head"]
    Image.fromarray(blank_u8, "RGBA").save(OUT / "head-blank-full.png")
    Image.fromarray(blank_u8[hp["y"]:hp["y"] + hp["h"], hp["x"]:hp["x"] + hp["w"]], "RGBA").save(
        PUB / "head-blank.png")
    for n_, im in parts_img.items():
        if n_.endswith("-ai"):
            im.save(WORK / f"{n_}.png")
        else:
            im.save(PUBF / f"{n_}.png")

    rnd = lambda v, k=3: [round(float(x), k) for x in v]  # noqa: E731
    sources = {
        "eye-open": "EXACT crop of the original viewer's-right eye (big white highlight upper-left, small white "
                    "highlight lower-right); used for BOTH eyes, never mirrored",
        "eye-blink": "AI sheet ai-eyes-1 cell 2 centreline (symmetric 4th-order fit), depth capped (gentle), "
                     "re-stroked: span = eye diameter, width = mouth line, eye dark brown",
        "eye-happy": "AI sheet ai-eyes-1 cell 3 centreline (symmetric fit), re-stroked: span = eye diameter, "
                     "width = mouth line, eye dark brown",
        "eye-surprised": f"measured eye geometry (disc + both highlights, original colours) re-rendered x{EYE_SCALE_SURPRISED}",
        "eye-sad": "eye-open (exact) + tear-shine crescent from AI sheet ai-eyes-1 cell 5 (flat highlight white)",
        "brow-neutral": "EXACT crop of the original viewer's-left brow; right brow = mirrored",
        "nose": "EXACT crop of the original nose (down to the philtrum)",
        "mouth-neutral": "EXACT crop of the original philtrum + w smile",
    }
    for n_, sp in brow_specs.items():
        sources[n_] = (f"exact brow-neutral rotated to {sp['tilt']:+.0f} deg tilt (inner end "
                       f"{'down' if sp['tilt'] > 0 else 'up'}), shifted {sp['dy']:+.0f} px")
    for n_, mi in minfo.items():
        sources[n_] = (f"AI sheet ai-mouths-1 cell {mi['ai_cell']}: keyed, scaled x{mi['scale']} (AI nose/mouth width "
                       f"-> original), lines re-stroked at {t_line:.2f} px, straight philtrum, flat original colours")
    slots_out = {k: dict(anchor=rnd(v["anchor"]), measured=rnd(v["measured"]), mirror=v["mirror"])
                 for k, v in slots.items()}
    slot_of = {"eye": ["eyeL", "eyeR"], "brow": ["browL", "browR"], "nose": ["nose"], "mouth": ["mouth"]}
    parts_out = {}
    for n_, pm_ in parts_meta.items():
        if n_.endswith("-ai"):
            continue
        typ = n_.split("-")[0]
        parts_out[n_] = dict(file=pm_["file"], type=typ, slots=slot_of[typ], w=pm_["w"], h=pm_["h"],
                             anchor=pm_["anchor"], source=sources[n_])
    exprs = {
        "neutral": dict(eyes="eye-open", brows="brow-neutral", mouth="mouth-neutral"),
        "happy": dict(eyes="eye-happy", brows="brow-neutral", mouth="mouth-smile-open"),
        "angry": dict(eyes="eye-open", brows="brow-angry", mouth="mouth-angry"),
        "sad": dict(eyes="eye-sad", brows="brow-sad", mouth="mouth-sad"),
        "scared": dict(eyes="eye-surprised", brows="brow-worried", mouth="mouth-angry"),
        "surprised": dict(eyes="eye-surprised", brows="brow-raised", mouth="mouth-o"),
        "blink": dict(eyes="eye-blink", brows="brow-neutral", mouth="mouth-neutral"),
        "talk-s": dict(eyes="eye-open", brows="brow-neutral", mouth="mouth-talk-s"),
        "talk-m": dict(eyes="eye-open", brows="brow-neutral", mouth="mouth-talk-m"),
        "talk-l": dict(eyes="eye-open", brows="brow-neutral", mouth="mouth-talk-l"),
        "talk-ee": dict(eyes="eye-open", brows="brow-neutral", mouth="mouth-ee"),
    }
    hl_out = [dict(offset=rnd([h["dx"], h["dy"]]), diameter=round(h["d"], 2), color=rnd(h["color"], 0)) for h in hls]
    manifest = {
        "name": "tori-v2 face", "version": 1, "canvas": [W, H],
        "units": "px on the 1024x1024 source canvas (y down); continuous coords, pixel k covers [k, k+1)",
        "parent": "head",
        "placement": ("for slot s and part p: image top-left = s.anchor - p.anchor (if s.mirror: flip the image "
                      "horizontally and use anchor x = p.w - p.anchor[0]); the result is a child of 'head' at rest "
                      "canvas coords, i.e. apply the head's world matrix from rig.json (step5_rig.world_matrices) "
                      "exactly like head.png.  Slots of the same type share images: eyes are NEVER mirrored (the "
                      "highlights must be identical), brows ARE mirrored for browR."),
        "zOrder": ZORDER,
        "zOrderNote": "draw right after head-blank (all above the head), mouth below nose (philtrum tucks under it)",
        "headBlank": dict(file="../head-blank.png", x=hp["x"], y=hp["y"], w=hp["w"], h=hp["h"],
                          note="replaces head.png when the face parts are drawn (same box, same pivot)"),
        "slots": slots_out,
        "slotNotes": {
            "eyeL/eyeR": "eye centres (alpha centroid of the disc).  eyeR = measured; eyeL snapped by "
                         f"{slots['eyeL']['anchor'][0] - eyeL_c[0]:+.2f},{slots['eyeL']['anchor'][1] - eyeL_c[1]:+.2f} px "
                         "so the shared image lands on whole pixels",
            "browL/browR": "brow centres (alpha centroid).  browR uses the mirrored image, snapped to whole pixels",
            "nose": "nose alpha centroid",
            "mouth": "top-centre attach point under the nose = philtrum axis x at the nose/philtrum split row; "
                     "every mouth image's anchor is the top of its philtrum (the first 6 rows tuck under the nose)",
        },
        "parts": parts_out,
        "expressions": exprs,
        "metrics": {
            "eyeDiameter": round(eye_d, 2),
            "eyeCentersMeasured": {"L": rnd(eyeL_c), "R": rnd(eyeR_c)},
            "eyeHighlights": hl_out,
            "eyeHighlightNote": "offsets from the eye centre; identical in both eyes (one image). The original "
                                "left eye had a small BROWN dot (not white) at lower right "
                                f"({rnd([dot_c[0] - eyeL_c[0], dot_c[1] - eyeL_c[1]])} from its centre); "
                                "both eyes now use the right eye's white one (the rule)",
            "eyeSurprisedScale": EYE_SCALE_SURPRISED,
            "lineThickness": round(t_line, 2),
            "lineThicknessNote": "original mouth line (philtrum) width = sum of alpha across the line; used for "
                                 "every re-stroked mouth line and the blink/happy eye lines",
            "browThickness": round(browL_th, 2),
            "browLength": round(browL_len, 2),
            "browTiltDeg": {"L": round(browL_tilt, 2), "R": round(browR_tilt, 2),
                            "note": "angle of the outer->inner end axis, + = inner end lower (L brow)"},
            "browNote": "the original brows are thicker than the mouth line; all brow variants are the exact "
                        "original brow (rotated/raised) so they keep its thickness",
            "mouthWidth": round(mouth_w, 2),
            "mouthBox": [int(mx.min()), int(split), int(mx.max() + 1), int(my.max() + 1)],
            "noseSize": [int(nx.max() + 1 - nx.min()), int(ny.max() + 1 - ny.min())],
            "noseBox": [int(nx.min()), int(ny.min()), int(nx.max() + 1), int(ny.max() + 1)],
            "colors": {k: "#%02X%02X%02X" % tuple(int(round(c)) for c in v) for k, v in
                       {**colors, "cream": A["cream"], "tan": A["tan"], "cheek": A["pink"]}.items()},
        },
        "blankHead": dict(method="per-class Gaussian normalised convolution of the surrounding flat colours; tan "
                                 "forehead wedge continued under the nose", **binfo),
    }
    (PUBF / "face.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    # =========================================================== verification
    R = Rig(PUB / "rig.json")
    ref, _ = R.render({})
    neutral = compose_face(blank_u8, manifest, parts_img, expression_sel(manifest, "neutral"))
    rec, _ = render_with_head(R, neutral)
    fb = (slice(290, 515), slice(335, 690))
    d = np.abs(np.round(rec) - np.round(ref)).max(-1)
    dot_zone = np.zeros((H, W), bool)
    lr = (int(round(eyeL_c[0] + hls[1]["dx"])), int(round(eyeL_c[1] + hls[1]["dy"])))
    dot_zone[lr[1] - 13:lr[1] + 13, lr[0] - 13:lr[0] + 13] = True
    dz = d.copy(); dz[dot_zone] = 0
    checks = {"recomposite_vs_step5_rest": {
        "face_box_xywh": [fb[1].start, fb[0].start, fb[1].stop - fb[1].start, fb[0].stop - fb[0].start],
        "all": dict(max=int(d[fb].max()), mean=round(float(d[fb].mean()), 4),
                    px_gt8=int((d[fb] > 8).sum()), px_gt2=int((d[fb] > 2).sum())),
        "excl_left_eye_lower_right_mark": dict(max=int(dz[fb].max()), mean=round(float(dz[fb].mean()), 4),
                                               px_gt8=int((dz[fb] > 8).sum()), px_gt2=int((dz[fb] > 2).sum())),
        "whole_canvas": dict(max=int(d.max()), mean=round(float(d.mean()), 5)),
        "note": "the only intended change is the left eye's lower-right brown dot -> white highlight (rule)"}}
    # where the remaining differences are
    for nm_, m_ in [("eyes", holes["eyeL"] | holes["eyeR"]), ("brows", holes["browL"] | holes["browR"]),
                    ("nose+mouth", holes["nosemouth"])]:
        dm = dz[m_]
        checks["recomposite_vs_step5_rest"][f"in_{nm_}_holes"] = dict(max=int(dm.max()), mean=round(float(dm.mean()), 3))
    # diff picture
    vis = np.round(ref * 0.4 + 255 * 0.6)
    vis[d > 2] = [255, 160, 0]; vis[d > 8] = [255, 0, 0]
    pic = np.concatenate([np.round(ref[fb]), np.round(rec[fb]), vis[fb]], 1)
    to_img(pic).resize((pic.shape[1] * 2, pic.shape[0] * 2), Image.NEAREST).save(OUT / "recomposite-diff.png")
    # highlight identity in every open-eye variant
    ident = {}
    for ev in ("eye-open", "eye-surprised", "eye-sad"):
        sel = expression_sel(manifest, "neutral"); sel["eyeL"] = sel["eyeR"] = ev
        hpm = compose_face(blank_u8, manifest, parts_img, sel)
        pl_ = place(parts_img[ev], parts_meta[ev]["anchor"], slots["eyeL"]["anchor"])
        pr_ = place(parts_img[ev], parts_meta[ev]["anchor"], slots["eyeR"]["anchor"])
        dxx = int(round(slots["eyeR"]["anchor"][0] - slots["eyeL"]["anchor"][0]))
        dyy = int(round(slots["eyeR"]["anchor"][1] - slots["eyeL"]["anchor"][1]))
        shifted = np.roll(np.roll(pl_, dxx, 1), dyy, 0)
        opaque = pr_[..., 3] > 0.999
        a_ = hpm[..., :3]
        bl = np.roll(np.roll(a_, dxx, 1), dyy, 0)
        ident[ev] = dict(part_pixels_max_diff=float(np.abs(shifted - pr_).max()),
                         rendered_opaque_max_diff=float(np.abs(bl[opaque] - a_[opaque]).max()))
    checks["eye_highlights_identical"] = ident
    # vector eye validation (scale 1 re-render vs exact crop)
    va = vec1[..., 3]; ea = ex["eyeR"][..., 3]
    m_ = (va > 0) | (ea > 0)
    checks["eye_vector_x1_vs_exact"] = dict(alpha_mean_abs=round(float(np.abs(va - ea)[m_].mean()), 4),
                                            rgb_max=float(np.abs(over(vec1, premul(blank_u8))[..., :3] -
                                                                 over(ex["eyeR"], premul(blank_u8))[..., :3])[m_].max()))
    checks["mouth_normalisation"] = minfo
    checks["eye_lines"] = {"blink": blink_i, "happy": happy_i, "ai_eye_diameter": ai["ai_eye_d"],
                           "ai_line_w_blink": ai["blink"]["ai_line_w"], "ai_line_w_happy": ai["happy"]["ai_line_w"]}
    checks["ai_images_generated_this_run"] = ai_used
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks["recomposite_vs_step5_rest"], indent=1))
    print(json.dumps(ident, indent=1))

    sheets(R, blank_u8, manifest, parts_img, parts_meta, mouths, geo)
    return manifest, checks


# ----------------------------------------------------------------------------- review sheets
def sheets(R, blank_u8, manifest, parts_img, parts_meta, mouths, geo):
    f = font(22); fs = font(16)
    order = ["neutral", "happy", "angry", "sad", "scared", "surprised", "blink", "talk-s", "talk-m", "talk-l"]
    cw, full_s = 400, 0.30
    fbox = (150, 30, 980, 1000)
    zbox = (322, 276, 702, 566)
    cells_ = []
    for e in order:
        hpm = compose_face(blank_u8, manifest, parts_img, expression_sel(manifest, e))
        full, _ = render_with_head(R, hpm, bg=GREEN, box=fbox, scale=full_s)
        zoom, _ = render_with_head(R, hpm, bg=GREEN, box=zbox, scale=1.0)
        cell = Image.new("RGB", (cw, 34 + full.shape[0] + 8 + zoom.shape[0] + 50), (250, 250, 250))
        dr = ImageDraw.Draw(cell)
        ex = manifest["expressions"][e]
        dr.text((8, 6), e, fill=(30, 30, 30), font=f)
        cell.paste(to_img(full), ((cw - full.shape[1]) // 2, 34))
        cell.paste(to_img(zoom), ((cw - zoom.shape[1]) // 2, 34 + full.shape[0] + 8))
        dr.text((8, cell.height - 44), f"{ex['eyes']} / {ex['brows']}", fill=(90, 90, 90), font=fs)
        dr.text((8, cell.height - 22), ex["mouth"], fill=(90, 90, 90), font=fs)
        cells_.append(cell)
    ch = max(c.height for c in cells_) + 6
    sheet = Image.new("RGB", (cw * 5, ch * 2 + 10), (235, 235, 235))
    for i, c in enumerate(cells_):
        sheet.paste(c, ((i % 5) * cw, (i // 5) * ch))
    sheet.save(OUT / "expressions.png")

    # parts sheet on a checkerboard at 2x
    names = [n for n in manifest["parts"]]
    z = 2
    tiles = []
    for n in names:
        im = np.array(parts_img[n]).astype(float)
        pm = premul(im)
        bg = checker(im.shape[0], im.shape[1], 4)
        comp = pm[..., :3] + bg * (1 - pm[..., 3:4])
        t = to_img(comp).resize((im.shape[1] * z, im.shape[0] * z), Image.NEAREST)
        ax, ay = manifest["parts"][n]["anchor"]
        d = ImageDraw.Draw(t)
        d.line([(ax * z - 6, ay * z), (ax * z + 6, ay * z)], fill=(255, 0, 255), width=1)
        d.line([(ax * z, ay * z - 6), (ax * z, ay * z + 6)], fill=(255, 0, 255), width=1)
        tiles.append((n, t))
    colw = 260
    rows, row, x = [], [], 0
    for n, t in tiles:
        if x + max(colw, t.width + 10) > 1600 and row:
            rows.append(row); row, x = [], 0
        row.append((n, t)); x += max(colw, t.width + 10)
    rows.append(row)
    hh = sum(max(t.height for _, t in r) + 40 for r in rows) + 10
    ps = Image.new("RGB", (1600, hh), (255, 255, 255))
    dr = ImageDraw.Draw(ps)
    y = 5
    for r in rows:
        x = 5
        rh = max(t.height for _, t in r)
        for n, t in r:
            dr.text((x, y), f"{n} {manifest['parts'][n]['w']}x{manifest['parts'][n]['h']}", fill=(0, 0, 0), font=fs)
            ps.paste(t, (x, y + 22))
            x += max(colw, t.width + 10)
        y += rh + 40
    ps.save(OUT / "parts.png")

    # blank head preview + measure overlay
    hp = R.rig["parts"]["head"]
    bl, _ = render_with_head(R, premul(blank_u8), bg=GREEN, box=(300, 260, 724, 560), scale=2.0)
    to_img(bl).save(OUT / "head-blank-zoom.png")
    neutral = compose_face(blank_u8, manifest, parts_img, expression_sel(manifest, "neutral"))
    nr, _ = render_with_head(R, neutral, bg=(255, 255, 255), box=(300, 260, 724, 560), scale=2.0)
    im = to_img(nr); d = ImageDraw.Draw(im)
    for k, s in manifest["slots"].items():
        x, y = (s["anchor"][0] - 300) * 2, (s["anchor"][1] - 260) * 2
        d.line([(x - 10, y), (x + 10, y)], fill=(255, 0, 255), width=2)
        d.line([(x, y - 10), (x, y + 10)], fill=(255, 0, 255), width=2)
        d.text((x + 8, y + 4), k, fill=(200, 0, 200), font=fs)
    r_ = manifest["metrics"]["eyeDiameter"] / 2
    for k in ("eyeL", "eyeR"):
        x, y = manifest["slots"][k]["anchor"]
        d.ellipse([((x - r_) - 300) * 2, ((y - r_) - 260) * 2, ((x + r_) - 300) * 2, ((y + r_) - 260) * 2],
                  outline=(0, 160, 255), width=1)
    im.save(OUT / "measure.png")
    _ = hp
    # AI mouth sheet vs normalised (for review)
    tiles = []
    for nm in AI_MOUTH_CELLS.values():
        pm = mouths[nm]
        box = (420, 450, 604, 590)
        comp = pm[box[1]:box[3], box[0]:box[2], :3] + np.array([251, 243, 222.]) * (1 - pm[box[1]:box[3], box[0]:box[2], 3:4])
        tiles.append(to_img(comp).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS))
    ai = Image.open(WORK / "ai-mouths-1.png").convert("RGB").resize((768, 768))
    out = Image.new("RGB", (768 + 3 * 372, 768), (255, 255, 255))
    out.paste(ai, (0, 0))
    for i, t in enumerate(tiles):
        out.paste(t.resize((368, 280)), (768 + (i % 3) * 372, (i // 3) * 256))
    out.save(WORK / "mouths-normalised.png")


if __name__ == "__main__":
    main(regen="--regen" in sys.argv)
