#!/usr/bin/env python3
"""
STEP 2 (3/4 view) - Tori v2 part masks (masks only).  Mirrors scripts/v2/step2_masks.py (front).

Input : art/v2/tori-34/original.png  (cand3 + cand4 far ear, see step1_earfix.py)
Guides: art/v2/tori-34/step2-masks/work/ai_*.png  (Gemini edits of the original, used ONLY as guides;
        alignment verified by check_alignment; every mask pixel refers to the original)
  ai_no_tail   "Edit this image. Keep EVERYTHING exactly identical (...) and remove ONLY the big striped
                curled tail on the left side behind the body. Where the tail was, show white background,
                and complete the body's left side and the arm naturally where the tail was behind them..."
  ai_tail_only "Edit this image. Keep ONLY the big striped curled tail on the left exactly as it is (...).
                Paint every other pixel (...) pure flat green #00FF00."
  ai_no_ears   "Edit this image. Keep EVERYTHING exactly identical (...) and remove ONLY the two small round
                ears and the small puffy tuft on top of the head (...) continue the round outline of the head."
Output: art/v2/tori-34/step2-masks/<part>.png, labels.png, overlay.png, contact.png, parts.json

Naming: Near = closer to the viewer (viewer's LEFT side in this view: the character faces viewer-right),
Far = turned away (viewer's right).  Parts: head, tuft, earNear, earFar, torso, pendant, armNear,
armFar, legNear, legFar, tail.  The masks form a PARTITION of the visible foreground.

Rules (details next to the code)
  foreground : max(255-RGB) > 8, largest component, enclosed highlights filled (as front).
  tail       : changed by ai_no_tail AND not green in ai_tail_only.
  earNear    : outside the AI-continued head outline (ai_no_ears) + inner-ear coloured px connected to
               it (the head outline clips the inner oval, as front).
  earFar     : the transplanted cand4 ear is one solid inner-ear-brown disc behind the head: pixels in
               the far-ear window that are nearer to the ear brown than to the head tan / white
               (colour boundary = exact), + its white-blended rim.
  tuft       : middle component outside the AI outline + its soft lower shading over the head (as front).
  pendant    : acorn only, by colour in the pendant window, holes filled.  No glow on the belly in this
               view (checked in step 3).
  collar     : brown band -> torso.
  head       : component above the collar containing the face; its pinkish-beige bottom band is split
               from the tan arm tops by colour (blue channel).
  armNear    : the brown underside-shading wedge belongs to the arm (as front).  Torso/arm boundary =
               torso-side (right) edge of that wedge, measured per row, continued straight up to the
               collar (line fitted to the upper 50 rows).  Collar shadow on the shoulder -> arm.
  armFar     : the shading stripe on the far arm's inner side belongs to the arm; boundary = its
               torso-side (left) edge per row, continued straight up to the collar; below the hand the
               torso side (x = edge at the stripe bottom) runs straight down.
  legNear    : hip cut = straight line from the outer silhouette (y 880) to the top corner of the gap
               between the legs, then down the gap.
  legFar     : from the gap corner along the top edge of the inner-thigh shading crescent to its tip,
               then straight to the outer silhouette (y 880) (= front legR rule).
  leftovers  : nearest label; stray fragments re-labelled by neighbours (as front).
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
ORIG = ROOT / "art/v2/tori-34/original.png"
OUT = ROOT / "art/v2/tori-34/step2-masks"
WORK = OUT / "work"

PARTS = ["head", "tuft", "earNear", "earFar", "torso", "pendant", "armNear", "armFar", "legNear", "legFar", "tail"]
COLORS = {
    "head": (255, 99, 71), "tuft": (255, 0, 255), "earNear": (255, 215, 0), "earFar": (173, 255, 47),
    "torso": (30, 144, 255), "pendant": (0, 255, 200), "armNear": (255, 140, 0), "armFar": (148, 0, 211),
    "legNear": (0, 200, 0), "legFar": (220, 20, 60), "tail": (139, 90, 255),
}
IDX = {p: i + 1 for i, p in enumerate(PARTS)}


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int32)


def fg_of(a, t=8):
    return (255 - a.min(axis=2)) > t


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def poly_mask(pts, shape=(1024, 1024)):
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(im) > 0


def largest(m):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    s = ndi.sum(m, lab, range(1, n + 1))
    return lab == (np.argmax(s) + 1)


def check_alignment(name, ai, orig, fo, edited):
    keep = ~ndi.binary_dilation(edited, disk(6))
    fa = fg_of(ai)
    A, B = fo & keep, fa & keep
    iou = (A & B).sum() / max(1, (A | B).sum())
    md = np.abs(ai - orig).max(2)[keep].mean()
    ok = iou > 0.97 and md < 4
    print(f"  align {name:13s} untouched-IoU={iou:.4f} meandiff={md:.2f} -> {'OK' if ok else 'MISALIGNED'}")
    if not ok:
        sys.exit(f"{name} is misaligned; regenerate it")
    return dict(iou=round(float(iou), 4), meandiff=round(float(md), 2))


def main():
    O = load_rgb(ORIG)
    R, G, B = O[..., 0], O[..., 1], O[..., 2]
    H, W = R.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # ---------- foreground
    fg = largest(fg_of(O, 8))
    holes, nh = ndi.label(ndi.binary_fill_holes(fg) & ~fg)
    for i in range(1, nh + 1):
        h = holes == i
        ring = ndi.binary_dilation(h, disk(5)) & ~ndi.binary_dilation(h, disk(3))
        if np.median(O.min(2)[ring]) < 100 or h.sum() < 30:
            fg |= h
        else:
            print(f"  keeping enclosed background gap: {int(h.sum())} px around {np.argwhere(h).mean(0).astype(int)[::-1]}")
    print("foreground px:", int(fg.sum()))

    # ---------- AI guides
    print("AI guides:")
    nt = load_rgb(WORK / "ai_no_tail.png")
    to = load_rgb(WORK / "ai_tail_only.png")
    ne = load_rgb(WORK / "ai_no_ears.png")
    d_nt = np.abs(nt - O).max(2)
    d_ne = np.abs(ne - O).max(2)
    green = (to[..., 1] > 200) & (to[..., 0] < 90) & (to[..., 2] < 90)
    align = {}
    tail_core = fg & (d_nt > 40) & ~green & (xx < 420)
    align["ai_no_tail"] = check_alignment("ai_no_tail", nt, O, fg, fg & (d_nt > 40) & (xx < 420))
    keep = tail_core & ~ndi.binary_dilation(~tail_core, disk(3))
    md = np.abs(to - O).max(2)[keep].mean()
    print(f"  align ai_tail_only  inside-tail meandiff={md:.2f} -> {'OK' if md < 8 else 'MISALIGNED'}")
    if md >= 8:
        sys.exit("ai_tail_only misaligned")
    agree = (fg & (d_nt > 40) & (xx < 420))
    agree_frac = float((tail_core.sum()) / max(1, agree.sum()))
    align["ai_tail_only"] = dict(inside_tail_meandiff=round(float(md), 2),
                                 frac_of_no_tail_change_that_is_non_green=round(agree_frac, 4))
    ne_bg = ~fg_of(ne, 30)
    align["ai_no_ears"] = check_alignment("ai_no_ears", ne, O, fg, fg & ne_bg)

    L = np.zeros((H, W), np.uint8)

    def put(part, m, override=False):
        m = m & fg
        if not override:
            m = m & (L == 0)
        L[m] = IDX[part]

    # ---------- tail
    tail = ndi.binary_opening(tail_core, disk(1))
    lab, n = ndi.label(tail)
    sizes = ndi.sum(tail, lab, range(1, n + 1))
    tail = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 60])
    put("tail", tail)

    # ---------- far ear (solid inner-ear-brown disc): colour boundary against head tan / white
    E = np.array([209, 155, 108]); T = np.array([234, 190, 129])
    fwin = (xx > 600) & (xx < 760) & (yy > 100) & (yy < 250)
    d = (T - E).astype(float)
    t_ht = ((O - E) @ d) / (d @ d)                       # 0 = ear brown, 1 = head tan
    ear_core = fwin & (np.abs(O - E).max(2) < 16)
    ear_core = largest(ndi.binary_opening(ear_core, disk(2)))
    ear_core = ndi.binary_fill_holes(ear_core)
    # rim: px next to the ear core that are ear-brown blended with WHITE (not with head tan)
    near_e = ndi.binary_dilation(ear_core, disk(3)) & ~ear_core & fwin & fg
    # head-side boundary px: mixed ear/tan -> ear if nearer ear (t < 0.5)
    wb = (O.min(2) > 0) & ((O[..., 2] - E[2]) > (O[..., 0] - E[0]) * 1.2)   # bluer than brown->tan line = white blend
    rim = near_e & ((t_ht < 0.5) | wb) & ~(np.abs(O - T).max(2) < 8)
    put("earFar", ear_core | rim)

    # ---------- near ear + tuft (outside the AI-continued head outline)
    ext = fg & ne_bg & (yy < 260) & (L == 0)
    ext = ndi.binary_opening(ext, disk(2))
    lab, n = ndi.label(ext)
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() < 80:
            continue
        cx = xx[c].mean()
        if cx < 420:
            put("earNear", c)
        elif cx < 600:
            put("tuft", c)
    inner = (R < 226) & (G < 180) & (G > 140) & (B < 124) & (yy > 100) & (yy < 250) & fg & (xx < 420)
    rim = ndi.binary_dilation(inner, disk(1)) & (G < 186) & fg & (xx < 420)
    lab, n = ndi.label(inner | rim)
    ids = np.unique(lab[(L == IDX["earNear"]) & (lab > 0)])
    put("earNear", np.isin(lab, ids[ids > 0]))
    tuft_core = L == IDX["tuft"]
    near = ndi.binary_dilation(tuft_core, disk(8))
    stripe = (R < 140) & (G < 110)
    tx = xx[tuft_core]
    tuft_sh = fg & near & (d_ne > 12) & (yy < 185) & ~stripe & (xx > tx.min() - 5) & (xx < tx.max() + 5)
    tuft_sh = ndi.binary_opening(tuft_sh, disk(1))
    put("tuft", tuft_sh)

    # ---------- pendant (acorn only)
    win = (xx > 550) & (xx < 665) & (yy > 610) & (yy < 785)
    olive = (np.abs(R - G) < 16) & (B < 100) & (R < 175) & (G > 100)
    nut = (R < 200) & (G > 200) & (B < 225) & (G - R > 25)
    pend = win & (olive | nut)
    pend = ndi.binary_opening(pend, disk(1))
    pend = largest(ndi.binary_closing(pend, disk(2)))
    pend = ndi.binary_fill_holes(pend)
    put("pendant", pend)

    # ---------- collar (torso): brown band
    brown = (R < 185) & (R - G > 28) & (G < 140) & (yy > 595) & (yy < 700) & (xx > 340) & (xx < 700)
    collar = largest(ndi.binary_opening(brown & (L == 0), disk(1)))
    put("torso", collar)
    rim = ndi.binary_dilation(collar, disk(2)) & fg & (L == 0) & (B > 140) & ((xx < 400) | (xx > 640)) & (yy > 610)
    rim &= ~((B > 160) & (R > 225))            # not the head's beige band
    put("torso", rim)

    # ---------- head: component above the collar containing the face
    beige_band = (B > 150) & (yy > 540)                  # head bottom band is pinkish beige (B ~ 175)
    rest = fg & (L == 0) & (yy < 640) & ~((yy > 600) & ~beige_band & ((xx < 360) | (xx > 640)))
    core = ndi.binary_erosion(rest, disk(2))
    lab, n = ndi.label(core)
    head_core = lab == lab[400, 512]
    head = ndi.binary_dilation(head_core, disk(3)) & rest
    put("head", head)

    # ---------- arms
    shade = (R < 222) & (G < 170) & (G > 120)
    cuts = {}

    def edge_rows(x0, x1, y0, y1, side):
        ys, xs = [], []
        for y in range(y0, y1):
            c = np.where(shade[y, x0:x1] & (L[y, x0:x1] == 0))[0]
            if len(c):
                ys.append(y); xs.append(x0 + (c.max() if side == "N" else c.min()))
        return np.array(ys), np.array(xs)

    # near arm: right edge of the underside wedge, rows 705..822, x 320..445
    ys, xs = edge_rows(330, 446, 700, 826, "N")
    top = ys < ys.min() + 50
    coef = np.polyfit(ys[top], xs[top], 1)
    bound = np.full(H, np.nan)
    for y in range(595, int(ys.min())):
        bound[y] = np.polyval(coef, y)
    for y, x in zip(ys, xs):
        bound[y] = x + 0.5
    ymaxN = int(ys.max())
    regionN = (xx <= np.nan_to_num(bound, nan=-1)[:, None]) & (yy >= 595) & (yy <= ymaxN) & (xx > 230)
    cuts["armNear"] = [(float(bound[y]), y) for y in range(595, ymaxN + 1) if not np.isnan(bound[y])]
    arm = regionN & fg & (L == 0)
    arm = largest(ndi.binary_opening(arm, disk(2)))
    put("armNear", arm)

    # far arm: left edge of the inner shading stripe, rows 676..792, x 660..720
    ys, xs = edge_rows(660, 722, 674, 800, "F")
    # stop where the stripe turns along the hand's bottom contour (edge moves right > 3 px over 3 rows);
    # below that the torso side runs straight down
    ok = np.r_[[True] * 3, (xs[3:] - xs[:-3]) <= 3]
    stop = np.argmax(~ok) if (~ok).any() else len(xs)
    ys, xs = ys[:stop], xs[:stop]
    top = ys < ys.min() + 50
    coef = np.polyfit(ys[top], xs[top], 1)
    bound = np.full(H, np.nan)
    for y in range(595, int(ys.min())):
        bound[y] = np.polyval(coef, y)
    for y, x in zip(ys, xs):
        bound[y] = x - 0.5
    ybot = int(ys.max())
    for y in range(ybot + 1, 815):
        bound[y] = bound[ybot]
    regionF = (xx >= np.nan_to_num(bound, nan=9999)[:, None]) & (yy >= 595) & (yy < 815) & (xx < 800)
    cuts["armFar"] = [(float(bound[y]), y) for y in range(595, 815) if not np.isnan(bound[y])]
    arm = regionF & fg & (L == 0)
    arm = largest(ndi.binary_opening(arm, disk(2)))
    put("armFar", arm)

    # ---------- legs
    # gap between the legs: background columns below the torso; its top corners
    gap = ~fg & (yy > 900) & (yy < 960) & (xx > 495) & (xx < 570)
    gap = largest(gap)
    gy, gx = np.nonzero(gap)
    gtop = gy.min()
    gl = gx[gy <= gtop + 2].min(); gr = gx[gy <= gtop + 2].max()
    yo = 880
    rowo = np.nonzero(fg[yo])[0]
    xoL, xoR = rowo[rowo > 300].min(), rowo[rowo < 760].max()
    legN_poly = [(300, yo), (int(xoL), yo), (int(gl), int(gtop)), (int(gl) + 14, int(gtop) + 14), (int(gl) + 14, 1010), (300, 1010)]
    put("legNear", poly_mask(legN_poly))
    # far leg: top edge of the inner-thigh shading crescent (x from gap corner to the tip)
    pts = []
    for x in range(int(gr) + 2, 660):
        c = np.where(shade[880:930, x])[0]
        if len(c):
            pts.append((x, 880 + c.min()))
    pts = np.array(pts)
    tip = pts[np.argmax(pts[:, 0])]
    edge = [(int(x), int(y)) for x, y in pts[::4]] + [(int(tip[0]), int(tip[1]))]
    legF_poly = [(int(gr) - 14, 1010), (int(gr) - 14, int(gtop) + 14), (int(gr), int(gtop))] + edge + \
                [(int(xoR), yo), (760, yo), (760, 1010)]
    put("legFar", poly_mask(legF_poly))

    # ---------- torso = everything else in the body
    body_rest = fg & (L == 0) & (yy > 595)
    lab, n = ndi.label(ndi.binary_opening(body_rest, disk(1)))
    put("torso", lab == lab[800, 560])

    # ---------- leftovers
    def fill_nearest():
        un = fg & (L == 0)
        if un.any():
            _, (iy, ix) = ndi.distance_transform_edt(L == 0, return_indices=True)
            L[un] = L[iy[un], ix[un]]

    fill_nearest()
    allowed_extra = {"tail": 60}
    for it in range(3):
        changed = 0
        for p in PARTS:
            m = L == IDX[p]
            lab, n = ndi.label(m)
            if n <= 1:
                continue
            s = ndi.sum(m, lab, range(1, n + 1))
            main_i = np.argmax(s) + 1
            for i in range(1, n + 1):
                if i == main_i or s[i - 1] >= allowed_extra.get(p, 10**9):
                    continue
                c = lab == i
                ring = ndi.binary_dilation(c, disk(1)) & ~c & fg
                nb = L[ring]
                nb = nb[(nb != 0) & (nb != IDX[p])]
                L[c] = np.bincount(nb).argmax() if len(nb) else 0
                changed += 1
        fill_nearest()
        if not changed:
            break
    L[~fg] = 0
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray((fg * 255).astype(np.uint8)).save(WORK / "fg.png")
    Image.fromarray(L).save(WORK / "label_map.png")
    masks = {p: L == IDX[p] for p in PARTS}
    for p, m in masks.items():
        Image.fromarray((m * 255).astype(np.uint8), "L").save(OUT / f"{p}.png")

    stack = np.stack([masks[p] for p in PARTS]).astype(np.int32)
    overlap = int((stack.sum(0) > 1).sum())
    unassigned = int((fg & (stack.sum(0) == 0)).sum())
    in_bg = int((~fg & (stack.sum(0) > 0)).sum())
    print(f"partition: overlap={overlap} unassigned_fg={unassigned} mask_px_in_bg={in_bg}")

    notes = {
        "head": "Face/head incl. stripes, cheeks, pinkish-beige bottom band; bottom = colour edge vs collar / tail / arm tops / background; near-ear + tuft junctions = AI-continued outline (ai_no_ears); far-ear junction = colour edge (ear brown vs head tan).",
        "tuft": "Cowlick blobs above the AI-continued head outline + their soft lower shading over the head.",
        "earNear": "Viewer-left (near) ear: original px outside the ai_no_ears head outline + inner-oval px clipped by it.",
        "earFar": "Viewer-right (far) ear (cand4 transplant, step 1): solid ear-brown disc by colour (exact edge vs head tan) + white-blended rim.",
        "torso": "Body incl. collar band, belly, crotch; bounded by arm shading edges, hip cuts, tail mask, head band.",
        "pendant": "Acorn only (olive cap+stem, green nut, highlights) by colour in the pendant window, holes filled.",
        "armNear": "Torso-side (right) edge of the arm's underside shading wedge per row, continued straight up to the collar; collar shadow on the shoulder included; outer side bounded by the AI tail mask.",
        "armFar": "Torso-side (left) edge of the far arm's inner shading stripe per row, continued straight up to the collar; below the hand a vertical line (torso side).",
        "legNear": "Hip cut: straight line from the outer silhouette (y=880) to the top-left corner of the gap between the legs, then down the gap.",
        "legFar": "Gap top-right corner -> along the top edge of the inner-thigh shading crescent to its tip -> straight to the outer silhouette at y=880.",
        "tail": "Pixels changed by ai_no_tail AND non-green in ai_tail_only (incl. the dark stripe sliver visible between the near hand and the torso).",
    }
    info = {}
    for p in PARTS:
        m = masks[p]
        ys, xs = np.where(m)
        lab, n = ndi.label(m)
        info[p] = {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   "area": int(m.sum()), "components": int(n), "note": notes[p]}
        print(f"  {p:8s} bbox={info[p]['bbox']} area={info[p]['area']} comps={n}")
    meta = {
        "source": "art/v2/tori-34/original.png", "canvas": [W, H], "foreground_px": int(fg.sum()),
        "naming": "Near = closer to the viewer (viewer-left: near ear, near arm in front of the tail, near leg); Far = turned away (viewer-right)",
        "partition_check": {"overlap_px": overlap, "unassigned_fg_px": unassigned, "mask_px_in_background": in_bg},
        "ai_guides": align,
        "arm_cuts_xy": {k: [[round(x, 1), y] for x, y in v[::10]] for k, v in cuts.items()},
        "leg_polys": {"legNear": legN_poly, "legFar": legF_poly},
        "parts": info,
    }
    (OUT / "parts.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    render_previews(O, fg, L, masks, info)


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def render_previews(O, fg, L, masks, info):
    H, W = L.shape
    lab_img = np.full((H, W, 3), 30, np.uint8)
    for p, m in masks.items():
        lab_img[m] = COLORS[p]
    im = Image.fromarray(lab_img)
    d = ImageDraw.Draw(im)
    f = font(16)
    for p, m in masks.items():
        dist = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        tw = d.textlength(p, font=f)
        d.rectangle([x - tw / 2 - 3, y - 10, x + tw / 2 + 3, y + 10], fill=(0, 0, 0))
        d.text((x - tw / 2, y - 9), p, fill=(255, 255, 255), font=f)
    im.save(OUT / "labels.png")

    S = 2
    base = Image.fromarray(O.astype(np.uint8)).resize((W * S, H * S), Image.LANCZOS)
    Lb = np.asarray(Image.fromarray(L).resize((W * S, H * S), Image.NEAREST))
    col = np.zeros((H * S, W * S, 3), np.uint8)
    for p in PARTS:
        col[Lb == IDX[p]] = COLORS[p]
    b = np.asarray(base).astype(np.float32)
    a = (Lb > 0)[..., None] * 0.35
    out = (b * (1 - a) + col * a).astype(np.uint8)
    edge = (Lb != np.roll(Lb, 1, 0)) | (Lb != np.roll(Lb, 1, 1))
    edge &= (Lb > 0) | (np.roll(Lb, 1, 0) > 0) | (np.roll(Lb, 1, 1) > 0)
    out[edge] = (0, 0, 0)
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    f = font(26)
    for p in PARTS:
        m = masks[p]
        dist = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        d.text((x * S - 30, y * S - 13), p, fill=COLORS[p], font=f, stroke_width=3, stroke_fill=(0, 0, 0))
    im.save(OUT / "overlay.png")

    cell, pad, cols = 300, 8, 4
    rows = (len(PARTS) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + 40 + pad) + pad), (40, 40, 40))
    yy, xx = np.mgrid[0:H, 0:W]
    checker = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 200, 235).astype(np.uint8).repeat(3, 2)
    d = ImageDraw.Draw(sheet)
    for k, p in enumerate(PARTS):
        m = masks[p]
        x0, y0, x1, y1 = info[p]["bbox"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = max(x1 - x0, y1 - y0) + 20
        bx0, by0 = max(0, cx - side // 2), max(0, cy - side // 2)
        bx1, by1 = min(W, bx0 + side), min(H, by0 + side)
        img = np.where(m[..., None], O.astype(np.uint8), checker)[by0:by1, bx0:bx1]
        tile = Image.fromarray(img)
        tile.thumbnail((cell, cell), Image.LANCZOS)
        gx, gy = pad + (k % cols) * (cell + pad), pad + (k // cols) * (cell + 40 + pad)
        sheet.paste(tile, (gx + (cell - tile.width) // 2, gy + 36 + (cell - tile.height) // 2))
        d.text((gx + 4, gy + 2), f"{p}  {info[p]['area']} px", fill=COLORS[p], font=font(18))
        d.text((gx + 4, gy + 21), f"bbox {info[p]['bbox']}", fill=(200, 200, 200), font=font(13))
    sheet.save(OUT / "contact.png")


if __name__ == "__main__":
    main()
