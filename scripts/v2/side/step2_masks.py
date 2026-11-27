#!/usr/bin/env python3
"""
STEP 2 - Tori v2 SIDE view: part masks (partition of the visible foreground; no layers, no fill).

Input : art/v2/tori-side/original.png (step-1 fixed), art/v2/tori-side/step1-fix/work/arm_layer.png
Output: art/v2/tori-side/step2-masks/<part>.png (L, 255 = part), labels.png, overlay.png (2x),
        contact.png, parts.json, work/label_map.png, work/fg.png
No AI images: every boundary is a colour edge of the original or a curve fitted to its silhouette.

Parts: head, tuft, earNear, torso (incl. collar), pendant (acorn only), armNear, armFar (v2), legNear, legFar,
  tail.  (earFar: not visible -> no mask.)
Rules
  foreground : max(255-C) > 8, largest component, enclosed holes filled.
  armNear    : step-1 arm layer alpha >= 128 (the arm was composited by step 1, so its coverage is exact).
  collar     : brown band (R<180, R-G>25, G<140) under the head -> torso.
  pendant    : olive cap/stem + mint nut (+ highlights) in the acorn window, holes filled.
  head       : with the collar removed the head (+ear+tuft) is its own connected component; its pixels
               minus ear and tuft.  AA pixels between head and collar -> nearest.
  tuft       : head-component pixels OUTSIDE the head outline (ellipse fitted to the head silhouette
               away from ear, tuft and snout) in the tuft window.
  earNear    : the ear is drawn IN FRONT of the head (it covers the head's back stripe): ear = head-
               component pixels inside the ear outline = ellipse fitted to the ear's visible silhouette
               (against background and against the dark head stripe), incl. its inner oval and the dark
               lower rim crescent that lies inside the ellipse.
  tail       : body-component pixels left of the torso's back edge: per row, the torso's back edge is the
               first non-fur pixel walking left from inside the torso (the tail touches the torso only with
               its shade / dark stripe / cream bands, never with fur tan).
  legFar     : the darker (shade) far leg below the belly: shade-coloured component at the bottom right.
  legNear    : below the hip cut (the belly-bottom contour continued from the gap between the legs to the
               back silhouette, polyline LEG_CUT); the near leg's fur is the torso's colour, so the cut is
               the continuation of the torso's own bottom contour.
  torso      : the rest of the body component.
  leftovers  : AA pixels -> label of the nearest labelled pixel; fragments < 40 px -> neighbours.
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
ORIG = ROOT / "art/v2/tori-side/original.png"
ARM = ROOT / "art/v2/tori-side/step1-fix/work/arm_layer.png"
FARARM = ROOT / "art/v2/tori-side/step1-fix/work/farArm_layer.png"
NOFAR = ROOT / "art/v2/tori-side/step1-fix/work/original_no_farArm.png"
OUT = ROOT / "art/v2/tori-side/step2-masks"
WORK = OUT / "work"
PARTS = ["head", "tuft", "earNear", "torso", "pendant", "armNear", "armFar", "legNear", "legFar", "tail"]
COLORS = {"head": (255, 99, 71), "tuft": (255, 0, 255), "earNear": (255, 215, 0), "torso": (30, 144, 255),
          "pendant": (0, 255, 200), "armNear": (255, 140, 0), "armFar": (120, 60, 0), "legNear": (0, 200, 0), "legFar": (220, 20, 60),
          "tail": (139, 90, 255)}
IDX = {p: i + 1 for i, p in enumerate(PARTS)}
LEG_CUT = [(400, 872), (436, 880), (470, 894), (505, 902), (546, 905), (548, 905)]
TAN = np.array([235, 191, 130])


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def largest(m):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    return lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)


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


def boundary(m, other):
    return m & ndi.binary_dilation(other, iterations=1)


def main():
    O = np.asarray(Image.open(ORIG).convert("RGB")).astype(np.int32)
    R, G, B = O[..., 0], O[..., 1], O[..., 2]
    H, W = R.shape
    yy, xx = np.mgrid[0:H, 0:W]
    fg = largest((255 - O.min(2)) > 8)
    fg = ndi.binary_fill_holes(fg)
    bg = ~fg
    L = np.zeros((H, W), np.uint8)
    geom = {}

    def put(p, m, override=False):
        m = m & fg
        if not override:
            m &= L == 0
        L[m] = IDX[p]

    armA = np.asarray(Image.open(ARM))[..., 3]
    put("armNear", armA >= 128)
    # far arm (v2): drawn behind the body -> visible where the step-1 image WITHOUT the far arm is background
    farA = np.asarray(Image.open(FARARM))[..., 3]
    nofar = np.asarray(Image.open(NOFAR).convert("RGB")).astype(np.int32)
    put("armFar", (farA >= 128) & ((255 - nofar.min(2)) <= 8))

    win = (xx > 640) & (xx < 745) & (yy > 625) & (yy < 745)
    olive = (np.abs(R - G) < 22) & (B < 100) & (R < 170) & (G > 90)
    nut = (G > 190) & (R < 200) & (G - R > 25)
    pend = ndi.binary_opening(win & (olive | nut), disk(1))
    pend = ndi.binary_fill_holes(largest(ndi.binary_closing(pend, disk(2))))
    put("pendant", pend)

    collar = (R < 180) & (R - G > 25) & (G < 140) & (yy > 600) & (yy < 720) & (xx > 430) & (xx < 720)
    collar = largest(ndi.binary_opening(collar & (L == 0), disk(1)))
    put("torso", collar)

    # head component (head + ear + tuft)
    rest = fg & ~ndi.binary_dilation(collar | (L == IDX["pendant"]), iterations=2)
    lab, n = ndi.label(rest)
    headc = lab == lab[400, 600]
    # head outline ellipse
    sil = boundary(headc, bg)
    sy, sx = np.nonzero(sil)
    keep = ~(((sx > 405) & (sx < 535) & (sy < 255)) | ((sx > 495) & (sx < 700) & (sy < 200)) |
             (sx > 770) | (sy > 560))
    headE = fit_ellipse(sx[keep].astype(float), sy[keep].astype(float))
    inside_head = headE(xx.astype(float), yy.astype(float)) <= 0
    # ear outline ellipse: visible ear silhouette vs background (left/top) + bottom edge of the dark lower
    # rim crescent + the inner-ear oval's right/bottom boundary pushed out by an 8 px rim (the ear's
    # right side has no visible edge: ear fur and head fur are the same tan)
    stripe = (R < 150) & (G < 115)
    tanm = np.abs(O - TAN).max(2) < 12
    shd = (np.abs(R - 208) < 16) & (np.abs(G - 153) < 16)
    zone = (xx > 405) & (xx < 540) & (yy > 100) & (yy < 255)
    lab_i, _ = ndi.label(shd & (xx > 420) & (xx < 540) & (yy > 130) & (yy < 240))
    inner = ndi.binary_fill_holes(lab_i == lab_i[185, 470])
    s1 = headc & zone & ~stripe & ndi.binary_dilation(bg, iterations=1) & (xx < 495) & (yy < 200)
    s3 = shd & zone & (yy > 215) & (xx < 470) & np.roll(tanm, -2, 0)
    ib = inner & ~ndi.binary_erosion(inner)
    iy, ix = np.nonzero(ib)
    c = np.array([ix.mean(), iy.mean()])
    sel = (ix > c[0] - 5) | (iy > c[1] + 20)
    v = np.c_[ix - c[0], iy - c[1]].astype(float); v /= np.linalg.norm(v, axis=1)[:, None]
    off = np.c_[ix, iy][sel] + 8 * v[sel]
    ey, ex = np.nonzero(s1 | s3)
    earE = fit_ellipse(np.r_[ex, off[:, 0]].astype(float), np.r_[ey, off[:, 1]].astype(float))
    in_ear = earE(xx.astype(float), yy.astype(float)) <= 0
    # ear = inside its ellipse (+ its visible silhouette rim up to 8 px outside the fit, x < 508); the fur bridge
    # between the ear's top-right and the tuft's left lobe further out goes to the tuft (it is behind the
    # head like the tuft), so the ear keeps a round outline when it turns
    ear = headc & zone & ~stripe & (in_ear | (~inside_head & ndi.binary_dilation(in_ear, iterations=8) & (xx < 508)))
    ear = largest(ndi.binary_opening(ear, disk(1)))
    cres = shd & zone & (yy > 205) & (xx < 480) & ndi.binary_dilation(ear, iterations=3)
    ear |= cres
    put("earNear", ear)
    geom["ear_inner_oval_centre"] = [round(float(c[0]), 1), round(float(c[1]), 1)]
    # tuft: outside the head outline, in the tuft window
    tuft = headc & ~inside_head & (xx > 470) & (xx < 700) & (yy < 205) & (L == 0)
    tuft = largest(ndi.binary_opening(tuft, disk(2)))
    put("tuft", tuft)
    put("head", headc)
    geom["head_ellipse_points_used"] = int(keep.sum())
    geom["ear_points_used"] = int(len(ex))

    # tail: left of the torso back edge
    body = fg & (L == 0) & ~headc
    fur = (np.abs(O - TAN).max(2) < 26)
    back = np.full(H, -1)
    for y in range(640, 905):
        x = 470
        if not fur[y, x]:
            continue
        while x > 300 and (fur[y, x - 1] or (fur[y, x - 2] and fur[y, x - 3])):
            x -= 1
        back[y] = x
    tail = np.zeros((H, W), bool)
    for y in range(540, 878):            # tail bottom (cream underside) ends at y~872
        if back[y] >= 0:
            tail[y, :back[y]] = True
        elif y < 700:
            tail[y, :445] = True
    tail &= body
    tail = largest(ndi.binary_opening(tail, disk(1)))
    put("tail", tail)
    geom["torso_back_edge_xy"] = [[int(back[y]), y] for y in range(640, 905, 10) if back[y] >= 0]

    # far leg: shade component at the bottom right
    shade = (np.abs(R - 205) < 16) & (np.abs(G - 148) < 16) & (np.abs(B - 99) < 18)
    lf = shade & (yy > 845) & (xx > 540) & (L == 0)
    lf = largest(ndi.binary_opening(lf, disk(1)))
    lf = ndi.binary_closing(lf, disk(2)) & fg & (L == 0)
    # AA rim of the far leg: pixels on the shade->white blend line (not tan / cream body pixels)
    cand = ndi.binary_dilation(lf, iterations=3) & fg & (L == 0) & ~lf
    Sv = np.array([205.0, 148.0, 99.0]); dS = 255.0 - Sv
    dC = 255.0 - O.astype(float)
    a = (dC @ dS) / (dS @ dS)
    res = np.linalg.norm(dC - a[..., None] * dS, axis=-1)
    lf |= cand & (res < 12) & (a < 0.98)
    put("legFar", lf)
    # near leg: below the hip cut
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(LEG_CUT + [(548, 1023), (400, 1023)], fill=255)
    put("legNear", (np.asarray(im) > 0))
    geom["leg_cut"] = LEG_CUT

    # torso = rest of body
    bodyrest = fg & (L == 0) & ~headc
    lab, n = ndi.label(ndi.binary_opening(bodyrest, disk(1)))
    # v2: the near arm now crosses the belly, so the torso visible area is split in pieces (main body +
    # the cream / collar-shadow wedge between the arm, the collar and the acorn): every piece >= 40 px
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    put("torso", np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz >= 40]))

    def fill_nearest():
        un = fg & (L == 0)
        if un.any():
            _, (iy, ix) = ndi.distance_transform_edt(L == 0, return_indices=True)
            L[un] = L[iy[un], ix[un]]
    fill_nearest()
    for it in range(3):
        changed = 0
        for p in PARTS:
            m = L == IDX[p]
            lab, n = ndi.label(m)
            if n <= 1:
                continue
            s = ndi.sum(m, lab, range(1, n + 1))
            mi = np.argmax(s) + 1
            for i in range(1, n + 1):
                if i == mi:
                    continue
                c = lab == i
                ring = ndi.binary_dilation(c, disk(1)) & ~c & fg
                nb = L[ring]; nb = nb[(nb != 0) & (nb != IDX[p])]
                L[c] = np.bincount(nb).argmax() if len(nb) else 0
                changed += 1
        fill_nearest()
        if not changed:
            break
    L[~fg] = 0

    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    Image.fromarray((fg * 255).astype(np.uint8)).save(WORK / "fg.png")
    Image.fromarray(L).save(WORK / "label_map.png")
    masks = {p: L == IDX[p] for p in PARTS}
    for p, m in masks.items():
        Image.fromarray((m * 255).astype(np.uint8), "L").save(OUT / f"{p}.png")
    stack = np.stack([masks[p] for p in PARTS]).astype(int)
    part = {"overlap_px": int((stack.sum(0) > 1).sum()), "unassigned_fg_px": int((fg & (stack.sum(0) == 0)).sum()),
            "mask_px_in_background": int((~fg & (stack.sum(0) > 0)).sum())}
    print("partition", part)
    notes = {
        "head": "head disc incl. stripes, face, cheek, bottom band; ear (in front) and tuft (outside the fitted head ellipse) removed; collar boundary = brown colour edge.",
        "tuft": "head-component pixels outside the head ellipse in the tuft window (x478-700, y<205), ear excluded.",
        "earNear": "inside the ellipse fitted to the ear's visible silhouette (vs background / dark head stripe); ear is in front of the head stripe; incl. inner oval and lower-rim crescent.",
        "torso": "body incl. collar band, collar-shadow band, belly; bounded by arm (step-1 layer), tail (back edge), hip cut and far leg colour edge.",
        "pendant": "acorn only (olive cap/stem + mint nut + highlights).",
        "armNear": "step-1 arm layer, alpha >= 128 (v2: forward, over the cream belly).",
        "armFar": "step-1 far-arm layer alpha >= 128 where the step-1 image without the far arm is background (the far arm is behind the body and the acorn).",
        "legNear": "below the hip cut polyline LEG_CUT (belly-bottom contour continued to the back silhouette).",
        "legFar": "shade-coloured leg below the belly (colour edge vs torso tan/cream).",
        "tail": "left of the torso back edge (first non-fur pixel walking left, per row).",
    }
    info = {}
    for p in PARTS:
        m = masks[p]
        ys, xs = np.nonzero(m)
        info[p] = {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   "area": int(m.sum()), "components": int(ndi.label(m)[1]), "note": notes[p]}
        print(f"  {p:8s} {info[p]['bbox']} area={info[p]['area']} comps={info[p]['components']}")
    meta = {"source": "art/v2/tori-side/original.png", "canvas": [W, H], "foreground_px": int(fg.sum()),
            "partition_check": part, "ai_guides": "none", "geometry": geom, "parts": info,
            "not_visible": {"earFar": "hidden behind the head in cand3"}}
    (OUT / "parts.json").write_text(json.dumps(meta, indent=1))
    previews(O, L, masks, info)


def font(sz):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(f, sz) if pathlib.Path(f).exists() else ImageFont.load_default()


def previews(O, L, masks, info):
    H, W = L.shape
    lab_img = np.full((H, W, 3), 30, np.uint8)
    for p, m in masks.items():
        lab_img[m] = COLORS[p]
    im = Image.fromarray(lab_img); d = ImageDraw.Draw(im); f = font(18)
    for p, m in masks.items():
        dist = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        tw = d.textlength(p, font=f)
        d.rectangle([x - tw / 2 - 3, y - 11, x + tw / 2 + 3, y + 11], fill=(0, 0, 0))
        d.text((x - tw / 2, y - 10), p, fill=(255, 255, 255), font=f)
    im.save(OUT / "labels.png")
    S = 2
    base = np.asarray(Image.fromarray(O.astype(np.uint8)).resize((W * S, H * S), Image.LANCZOS)).astype(float)
    Lb = np.asarray(Image.fromarray(L).resize((W * S, H * S), Image.NEAREST))
    col = np.zeros((H * S, W * S, 3))
    for p in PARTS:
        col[Lb == IDX[p]] = COLORS[p]
    a = (Lb > 0)[..., None] * 0.35
    out = (base * (1 - a) + col * a).astype(np.uint8)
    edge = (Lb != np.roll(Lb, 1, 0)) | (Lb != np.roll(Lb, 1, 1))
    out[edge] = 0
    im = Image.fromarray(out); d = ImageDraw.Draw(im)
    for p in PARTS:
        dist = ndi.distance_transform_edt(masks[p])
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        d.text((x * S - 20, y * S - 13), p, fill=COLORS[p], font=font(26), stroke_width=3, stroke_fill=(0, 0, 0))
    im.save(OUT / "overlay.png")
    cell, pad, cols = 300, 8, 3
    rows = (len(PARTS) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + 40 + pad) + pad), (40, 40, 40))
    yy, xx = np.mgrid[0:H, 0:W]
    ck = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 200, 235).astype(np.uint8).repeat(3, 2)
    d = ImageDraw.Draw(sheet)
    for k, p in enumerate(PARTS):
        x0, y0, x1, y1 = info[p]["bbox"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = max(x1 - x0, y1 - y0) + 20
        bx0, by0 = max(0, cx - side // 2), max(0, cy - side // 2)
        img = np.where(masks[p][..., None], O.astype(np.uint8), ck)[by0:by0 + side, bx0:bx0 + side]
        t = Image.fromarray(img); t.thumbnail((cell, cell), Image.LANCZOS)
        gx, gy = pad + (k % cols) * (cell + pad), pad + (k // cols) * (cell + 40 + pad)
        sheet.paste(t, (gx + (cell - t.width) // 2, gy + 36 + (cell - t.height) // 2))
        d.text((gx + 4, gy + 2), f"{p}  {info[p]['area']} px", fill=COLORS[p], font=font(18))
        d.text((gx + 4, gy + 21), f"bbox {info[p]['bbox']}", fill=(200, 200, 200), font=font(13))
    sheet.save(OUT / "contact.png")


if __name__ == "__main__":
    main()
