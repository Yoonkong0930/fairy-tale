#!/usr/bin/env python3
"""
STEP 2 – Tori v2 part-mask extraction (masks only; no layer PNGs, no occlusion fill).

Input : art/v2/tori/original.png (1024x1024, lineless flat art, white background)
Guides: art/v2/tori/step2-masks/work/ai_*.png  (Gemini edits of the original; only
        used as *guides*, every final mask pixel refers to the ORIGINAL image)
Output: art/v2/tori/step2-masks/
          <part>.png  (mode L, 255 = visible pixel of that part)
          labels.png, overlay.png (2x), contact.png, parts.json
          work/label_map.png (palette index map), work/fg.png

AI guide images (Gemini, generated once with scripts/gen_image.py --ref original.png
--aspect 1:1 --size 1K; all three verified pixel-aligned by check_alignment below):
  work/ai_no_tail.png   "Edit this image. Keep EVERYTHING exactly identical (same position, same
                         size, same colors, same flat lineless style, same white background) and
                         remove ONLY the big striped tail on the right side behind the body. Where
                         the tail was, show white background. Keep both arms, the body, legs and
                         head completely unchanged. Do not move, resize or redraw anything else."
  work/ai_tail_only.png "Edit this image. Keep ONLY the big striped tail exactly as it is (same
                         position, same size, same colors). Paint every other pixel - the head,
                         ears, body, arms, legs, collar and background - pure flat green #00FF00.
                         The visible part of the tail must stay pixel-identical."
  work/ai_no_ears.png   "Edit this image. Keep EVERYTHING exactly identical (...same as above...)
                         and remove ONLY the two small round ears and the small puffy tuft on top
                         of the head. Where they were, show white background, and continue the
                         round outline of the head smoothly. Do not move, resize or redraw
                         anything else."
  work/ai_no_arms.png   (also generated, NOT used: the model re-drew slimmer arms hugging the
                         body instead of removing them, so it says nothing about the torso edge.)

The masks form a PARTITION of the visible foreground of the original:
every foreground pixel belongs to exactly one part, background pixels to none.

How each boundary is decided (details next to the code):
  foreground : max(255-R,255-G,255-B) > 8, largest component, holes filled
               (eye highlights are pure white but enclosed -> head). The faint ghost of
               the removed 2nd tail (diff <= 5) is excluded.
  tail       : pixels changed by the "remove the tail" edit (ai_no_tail) AND not green in
               the "tail only on green" edit (ai_tail_only). The two guides agree on 99.9 %.
               Arm-underside shading below the right hand is given to the arm (no_tail
               keeps it, mirrors the left arm's underside band).
  earL/earR  : foreground pixels that are background in the "remove ears + tuft" edit
               (ai_no_ears), i.e. outside the AI-continued head outline.
  tuft       : same as ears (middle component) + the soft shading of the tuft's lower
               blobs that overlaps the head (pixels the edit changed, within 8 px of the
               protruding tuft, above y=172).
  pendant    : acorn only = olive cap/stem (|R-G| small, low B) + green nut (R < 181,
               G > 200) inside the pendant window, holes filled (highlights). The soft
               green glow around it stays with the torso.
  collar     : brown band pixels (kept with the torso).
  head       : the component above the collar (head-bottom band is a pinkish beige that
               meets the collar brown / white background / tail -> pure colour boundary).
  armL/armR  : the dark "inner-arm" shading wedge between each arm and the belly is the
               arm's own underside shading (on the left it joins the underside band of
               the hand). Torso/arm boundary = torso-side edge of that wedge, measured
               per row from the original; above the wedge tip it is continued to the
               collar by a quadratic fitted to the upper wedge edge. Collar brown stays
               torso; the translucent collar shadow lying on the shoulders goes to the arm.
  legL/legR  : hip cut = straight line from the outer silhouette (y=880) to the tip of
               the inner-thigh shading, then along the shading's top edge to the gap
               between the legs, then straight down the gap edge. The central crotch
               shading band between the legs stays with the torso.
  leftovers  : anti-aliased pixels not claimed by any rule get the label of the nearest
               labelled pixel; stray fragments (< 40 px, not the part's main component)
               are re-labelled by their neighbours.
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIG = ROOT / "art/v2/tori/original.png"
OUT = ROOT / "art/v2/tori/step2-masks"
WORK = OUT / "work"

PARTS = ["head", "tuft", "earL", "earR", "torso", "pendant", "armL", "armR", "legL", "legR", "tail"]
COLORS = {
    "head": (255, 99, 71), "tuft": (255, 0, 255), "earL": (255, 215, 0), "earR": (173, 255, 47),
    "torso": (30, 144, 255), "pendant": (0, 255, 200), "armL": (255, 140, 0), "armR": (148, 0, 211),
    "legL": (0, 200, 0), "legR": (220, 20, 60), "tail": (139, 90, 255),
}
IDX = {p: i + 1 for i, p in enumerate(PARTS)}

# ---------------------------------------------------------------- helpers
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
    """Compare AI edit with original outside the (dilated) edited zone."""
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


# ---------------------------------------------------------------- main
def main():
    O = load_rgb(ORIG)
    R, G, B = O[..., 0], O[..., 1], O[..., 2]
    H, W = R.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # ---------- foreground
    fg = largest(fg_of(O, 8))
    # fill enclosed holes only when they are highlights inside a dark shape (eye catch-lights);
    # the real background gap enclosed by head/collar/right arm/tail must stay background.
    holes, nh = ndi.label(ndi.binary_fill_holes(fg) & ~fg)
    for i in range(1, nh + 1):
        h = holes == i
        ring = ndi.binary_dilation(h, disk(5)) & ~ndi.binary_dilation(h, disk(3))
        if np.median(O.min(2)[ring]) < 100:
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
    tail_core = fg & (d_nt > 40) & ~green
    align["ai_no_tail"] = check_alignment("ai_no_tail", nt, O, fg, fg & (d_nt > 40))
    # tail_only: compare non-green pixels with original inside the tail (must be identical art)
    keep = tail_core & ~ndi.binary_dilation(~tail_core, disk(3))
    md = np.abs(to - O).max(2)[keep].mean()
    print(f"  align ai_tail_only  inside-tail meandiff={md:.2f} -> {'OK' if md < 8 else 'MISALIGNED'}")
    if md >= 8:
        sys.exit("ai_tail_only misaligned")
    align["ai_tail_only"] = dict(inside_tail_meandiff=round(float(md), 2))
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
    tail = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 60])  # main + stripe triangle
    put("tail", tail)

    # ---------- ears + tuft (outside the AI-continued head outline)
    ext = fg & ne_bg & (yy < 260)
    ext = ndi.binary_opening(ext, disk(2))  # drop 1-2px anti-alias slivers along the head edge
    lab, n = ndi.label(ext)
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() < 80:
            continue
        cx = xx[c].mean()
        part = "earL" if cx < 420 else ("earR" if cx > 600 else "tuft")
        put(part, c)
    # The AI-continued head arc clips the bottom of each ear's darker inner oval, but those
    # pixels are visibly inner-ear colour, so they belong to the ear: add inner-ear coloured
    # pixels (+1px anti-alias rim that is darker than head tan) connected to the ear's oval.
    inner = (R < 226) & (G < 180) & (G > 140) & (B < 124) & (yy > 130) & (yy < 240) & fg
    rim = ndi.binary_dilation(inner, disk(1)) & (G < 186) & fg
    for part in ("earL", "earR"):
        lab, n = ndi.label(inner | rim)
        ids = np.unique(lab[(L == IDX[part]) & (lab > 0)])
        put(part, np.isin(lab, ids[ids > 0]))
    # tuft lower-blob shading overlapping the head (edit changed it slightly)
    tuft_core = L == IDX["tuft"]
    near = ndi.binary_dilation(tuft_core, disk(8))
    stripe = (R < 140) & (G < 110)  # head stripes stay with head
    tuft_sh = fg & near & (d_ne > 12) & (yy < 172) & ~stripe & (xx > 425) & (xx < 595)
    tuft_sh = ndi.binary_opening(tuft_sh, disk(1))
    put("tuft", tuft_sh)

    # ---------- pendant (acorn only)
    win = (xx > 455) & (xx < 570) & (yy > 650) & (yy < 785)
    olive = (np.abs(R - G) < 16) & (B < 93) & (R < 175) & (G > 100)
    nut = (R < 181) & (G > 200) & (B < 215)
    pend = win & (olive | nut)
    pend = ndi.binary_opening(pend, disk(1))
    pend = largest(ndi.binary_closing(pend, disk(2)))
    pend = ndi.binary_fill_holes(pend)
    put("pendant", pend)

    # ---------- collar (torso) : brown band
    brown = (R < 185) & (R - G > 28) & (G < 140) & (yy > 595) & (yy < 700) & (xx > 320) & (xx < 710)
    collar = largest(ndi.binary_opening(brown & (L == 0), disk(1)))
    put("torso", collar)
    # anti-aliased rim of the collar's two rounded ends (brown blended with white background):
    # keep it with the collar so no sliver ends up on the shoulders.
    rim = ndi.binary_dilation(collar, disk(2)) & fg & (L == 0) & (B > 140) & ((xx < 365) | (xx > 660)) & (yy > 610)
    put("torso", rim)

    # ---------- head : component of the rest that contains the face, above the collar
    rest = fg & (L == 0) & (yy < 640)
    core = ndi.binary_erosion(rest, disk(2))
    lab, n = ndi.label(core)
    head_core = lab == lab[400, 512]
    head = ndi.binary_dilation(head_core, disk(3)) & rest
    put("head", head)

    # ---------- arms
    shade = (R < 222) & (G < 170) & (G > 120)

    def wedge_edge(side):
        ys, xs = [], []
        for y in range(686, 790):
            if side == "L":
                c = np.where(shade[y, 280:385])[0]
                if len(c):
                    ys.append(y); xs.append(280 + c.max())
            else:
                c = np.where(shade[y, 640:700])[0]
                if len(c):
                    ys.append(y); xs.append(640 + c.min())
        return np.array(ys), np.array(xs)

    cuts = {}
    for side in ("L", "R"):
        ys, xs = wedge_edge(side)
        top = ys < ys.min() + 50
        coef = np.polyfit(ys[top], xs[top], 1)  # straight continuation is the most neutral
        bound = np.full(H, np.nan)
        for y in range(600, int(ys.min())):
            bound[y] = np.polyval(coef, y)
        for y, x in zip(ys, xs):
            bound[y] = x + (0.5 if side == "L" else -0.5)
        if side == "L":
            # below the wedge: arm corner -> background gap between hand and torso (y 796)
            for y in range(int(ys.max()) + 1, 812):
                bound[y] = np.interp(y, [ys.max(), 796], [xs[-1], 324])
            region = (xx <= np.nan_to_num(bound, nan=-1)[:, None]) & (yy >= 600) & (yy < 812) & (xx > 220)
        else:
            for y in range(int(ys.max()) + 1, 812):
                bound[y] = 692
            region = (xx >= np.nan_to_num(bound, nan=9999)[:, None]) & (yy >= 600) & (yy < 812) & (xx < 800)
        cuts[side] = [(float(bound[y]), y) for y in range(600, 812) if not np.isnan(bound[y])]
        arm = region & fg & (L == 0)
        arm = largest(ndi.binary_opening(arm, disk(2)))  # drops the 2-3px tail-edge strip at the right shoulder gap
        put("arm" + side, arm)

    # ---------- legs
    legL_poly = [(300, 880), (421, 887), (447, 894), (470, 900), (469, 918), (462, 1010), (300, 1010)]
    legR_poly = [(724, 880), (604, 887), (578, 894), (556, 900), (556, 918), (566, 1010), (724, 1010)]
    put("legL", poly_mask(legL_poly))
    put("legR", poly_mask(legR_poly))

    # ---------- torso = everything else in the body (below head)
    body_rest = fg & (L == 0) & (yy > 600)
    # opening first, so 1-px anti-alias strips along the tail/arm edges are not pulled into
    # the torso; they get their label from the nearest part below.
    lab, n = ndi.label(ndi.binary_opening(body_rest, disk(1)))
    # main torso component (contains belly centre)
    put("torso", lab == lab[800, 512])

    # ---------- leftovers: anti-aliased / unclaimed pixels -> nearest label
    def fill_nearest():
        un = fg & (L == 0)
        if un.any():
            _, (iy, ix) = ndi.distance_transform_edt(L == 0, return_indices=True)
            L[un] = L[iy[un], ix[un]]

    fill_nearest()

    # ---------- fragment cleanup: each part keeps its main component (+ allowed extras)
    allowed_extra = {"tail": 60}  # tail stripe triangle peeking between right arm and torso
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

    # ---------- save masks
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    Image.fromarray((fg * 255).astype(np.uint8)).save(WORK / "fg.png")
    Image.fromarray(L).save(WORK / "label_map.png")
    masks = {p: L == IDX[p] for p in PARTS}
    for p, m in masks.items():
        Image.fromarray((m * 255).astype(np.uint8), "L").save(OUT / f"{p}.png")

    # ---------- checks
    stack = np.stack([masks[p] for p in PARTS]).astype(np.int32)
    overlap = int((stack.sum(0) > 1).sum())
    unassigned = int((fg & (stack.sum(0) == 0)).sum())
    in_bg = int((~fg & (stack.sum(0) > 0)).sum())
    print(f"partition: overlap={overlap} unassigned_fg={unassigned} mask_px_in_bg={in_bg}")

    notes = {
        "head": "Face/head disc incl. stripes, cheeks and pinkish bottom band; bottom = colour edge of the band vs collar brown / background / tail; top/ear junctions = AI-continued head outline (ai_no_ears).",
        "tuft": "Cowlick blobs protruding above the AI-continued head outline + their soft lower-blob shading over the head; head stripes stay head. Merge into head if a separate tuft is not wanted.",
        "earL": "Viewer-left ear: original pixels outside the head outline continued by ai_no_ears (cut runs notch-to-notch along the head arc).",
        "earR": "Viewer-right ear: same rule as earL.",
        "torso": "Body incl. collar band, collar shadow over chest, belly, pendant glow, crotch shading band; bounded by arm wedge edges, hip cuts, tail mask.",
        "pendant": "Acorn only (olive cap+stem, green nut, highlights) by colour in pendant window, holes filled; glow stays torso.",
        "armL": "Torso-side edge of the dark inner-arm shading wedge (measured per row), continued straight up to the collar; collar brown excluded, shoulder collar-shadow included.",
        "armR": "Mirror of armL; separated from the tail by the AI tail mask (arm keeps its underside shading band, as in ai_no_tail).",
        "legL": "Hip cut: line from outer silhouette (y=880) to inner-thigh shading tip (421,887), along shading top edge to the leg gap (470,900), then down the gap edge.",
        "legR": "Mirror of legL: (604,887) -> (556,900) -> gap edge; right outer side bounded by the tail mask.",
        "tail": "Pixels changed by ai_no_tail AND non-green in ai_tail_only; includes the small dark stripe triangle visible between right arm and torso (joined to the main tail by its shading).",
    }
    info = {}
    for p in PARTS:
        m = masks[p]
        ys, xs = np.where(m)
        lab, n = ndi.label(m)
        info[p] = {
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "area": int(m.sum()),
            "components": int(n),
            "note": notes[p],
        }
        print(f"  {p:8s} bbox={info[p]['bbox']} area={info[p]['area']} comps={n}")
    meta = {
        "source": "art/v2/tori/original.png",
        "canvas": [W, H],
        "foreground_px": int(fg.sum()),
        "partition_check": {"overlap_px": overlap, "unassigned_fg_px": unassigned, "mask_px_in_background": in_bg},
        "ai_guides": align,
        "arm_cuts_xy": {k: [[round(x, 1), y] for x, y in v[::10]] for k, v in cuts.items()},
        "leg_polys": {"legL": legL_poly, "legR": legR_poly},
        "parts": info,
    }
    (OUT / "parts.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    render_previews(O, fg, L, masks, info)


# ---------------------------------------------------------------- previews
def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def render_previews(O, fg, L, masks, info):
    H, W = L.shape
    # labels.png
    lab_img = np.full((H, W, 3), 30, np.uint8)
    for p, m in masks.items():
        lab_img[m] = COLORS[p]
    im = Image.fromarray(lab_img)
    d = ImageDraw.Draw(im)
    f = font(18)
    for p, m in masks.items():
        dist = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        tw = d.textlength(p, font=f)
        d.rectangle([x - tw / 2 - 3, y - 11, x + tw / 2 + 3, y + 11], fill=(0, 0, 0))
        d.text((x - tw / 2, y - 10), p, fill=(255, 255, 255), font=f)
    im.save(OUT / "labels.png")

    # overlay.png (2x)
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
        x0, y0, x1, y1 = info[p]["bbox"]
        m = masks[p]
        dist = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        d.text((x * S - 20, y * S - 13), p, fill=COLORS[p], font=f, stroke_width=3, stroke_fill=(0, 0, 0))
    im.save(OUT / "overlay.png")

    # contact.png
    cell, pad = 300, 8
    cols = 4
    rows = (len(PARTS) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + 40 + pad) + pad), (40, 40, 40))
    yy, xx = np.mgrid[0:H, 0:W]
    checker = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 200, 235).astype(np.uint8).repeat(3, 2)
    f = font(18)
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
        d.text((gx + 4, gy + 2), f"{p}  {info[p]['area']} px", fill=COLORS[p], font=f)
        d.text((gx + 4, gy + 21), f"bbox {info[p]['bbox']}", fill=(200, 200, 200), font=font(13))
    sheet.save(OUT / "contact.png")


if __name__ == "__main__":
    main()
