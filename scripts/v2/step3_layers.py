#!/usr/bin/env python3
"""
STEP 3 - Tori v2 transparent layer separation (visible pixels only; no occlusion fill).

Input : art/v2/tori/original.png            (1024x1024 RGB, white background)
        art/v2/tori/step2-masks/<part>.png  (mode L, 255 = part; exact partition of the fg)
Output: art/v2/tori/step3-layers/
          <part>.png              RGBA 1024x1024, same canvas/coordinates as the original
          stack-order.json        back-to-front draw order + per-layer bbox + stats
          recomposite-white.png   all layers over white  (== original except removed glow)
          recomposite-dark.png    all layers over #202020
          recomposite-green.png   all layers over #6BBF59
          contact.png             each layer on a checkerboard, labelled
          glow-before-after.png   zoomed belly: original / torso layer after glow removal
          edge-check.png          zoomed silhouette crops over dark + green (halo review)

Layer construction
  RGB   = original pixels, alpha = 255 on the part's mask, 0 elsewhere.
  Outer silhouette (part pixels within EDGE_LEVELS steps of the background, walking only
  through pixels of the SAME part) is "de-whited": the pixel C is modelled as
      C = a*F + (1-a)*white
  with F ~ S, the part's local solid colour (masked 3x3 mean of the nearest "core" pixel
  of the same part, core = part pixels deeper than the band).  a = projection of (white-C)
  onto (white-S), clamped to [a_min, 1] where a_min = max_c(255-C_c)/255 keeps F inside the
  RGB gamut; then F = (C-(1-a)*255)/a (un-premultiply).  So compositing over white gives back
  C (up to 8-bit rounding), and over a dark/coloured background there is no white fringe.
  Pixels darker than S (the slightly darker rim the art has on its silhouette) get a = 1.
  Pixels the S-blend does not explain are re-fitted against the nearest solid pixel, else
  treated as opaque art when clearly coloured (see soft_alpha); narrow notches whose
  white-blended pixels lie deeper than the band are added (see dewhite).
  Internal boundaries between two parts are never in the band -> alpha stays 255.

Pendant glow removal (torso layer only, user-requested change; see remove_glow)
  1. Glow model: normal blend toward a glow colour M, C = (1-t)*B + t*M, B = clean colour of
     the torso region underneath.  Clean region colours (belly cream, pink-beige collar-shadow
     band, collar light band, collar dark band, tan) = medians of that region's pixels in a
     ring just outside the glow (60..110 px from the acorn).  M is least-squares fitted from
     the glowing pixels; per-pixel t from the best-fitting region line, smoothed/filled with a
     normalised Gaussian (sigma 3) and limited to the glow zone (full to 50 px from the acorn,
     0 at 70 px; measured glow < 0.01 beyond ~55 px).
  2. Region map inside the zone from the blue channel (the glow mostly lowers R and barely
     moves B): <70 dark collar, <130 light collar, <190 beige band, else cream; 5x5 majority
     vote; anti-aliased region-colour image (Gaussian 0.7 on the label masks).
  3. Result = lerp(original, region colour, smoothstep(t, 0.005, 0.045)) -> uniform clean
     colours where the glow was, imperceptible hand-over at the glow's faint outer edge (no
     ring).  Torso pixels within 3 px of the acorn (acorn AA mixed into the torso) take the
     clean colour of the nearest region; torso pixels touching head/arms are only un-blended
     ((C - t*M)/(1-t)) to keep their seam.  Region boundaries (collar edges) keep their
     original positions.  The pendant layer and the torso's pendant hole are untouched.
"""
import json, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIG = ROOT / "art/v2/tori/original.png"
MASKS = ROOT / "art/v2/tori/step2-masks"
OUT = ROOT / "art/v2/tori/step3-layers"

PARTS = ["head", "tuft", "earL", "earR", "torso", "pendant", "armL", "armR", "legL", "legR", "tail"]
# back-to-front (see STACK_REASONS)
STACK = ["tail", "legL", "legR", "torso", "pendant", "armL", "armR", "earL", "earR", "head", "tuft"]
STACK_REASONS = {
    "tail": "behind everything: its left edge disappears behind the torso and the right arm's "
            "silhouette is drawn over it (only a small stripe wedge shows between arm and body).",
    "legL": "behind torso: the crotch/hip shading band that caps the legs belongs to the torso, "
            "so the torso's lower edge covers the hip joint when the legs rotate.",
    "legR": "same as legL.",
    "torso": "body incl. collar; drawn over tail and legs, under arms, pendant and head.",
    "pendant": "hangs on the collar/belly: in front of the torso (sits in the torso's hole).",
    "armL": "the inner-arm shading wedge lies on top of the belly edge -> arms in front of torso.",
    "armR": "in front of torso and of the tail (arm silhouette overlaps the tail).",
    "earL": "behind head: the head's round outline passes over the ear bases.",
    "earR": "same as earL.",
    "head": "head's lower band overlaps the collar and the arm tops -> in front of torso/arms.",
    "tuft": "in front of head: the tuft blobs' rounded bottoms and their soft shading are drawn "
            "over the head's top silhouette and over the head stripes.",
}
COLORS = {
    "head": (255, 99, 71), "tuft": (255, 0, 255), "earL": (255, 215, 0), "earR": (173, 255, 47),
    "torso": (30, 144, 255), "pendant": (0, 255, 200), "armL": (255, 140, 0), "armR": (148, 0, 211),
    "legL": (0, 200, 0), "legR": (220, 20, 60), "tail": (138, 100, 255),
}
DARK = (0x20, 0x20, 0x20)
GREEN = (0x6B, 0xBF, 0x59)
EDGE_LEVELS = 3          # outer-silhouette band depth (px, 8-connected, inside the part)
EXT_LEVELS = 8           # max depth of the notch extension
EXT_MAX_ALPHA = 0.8      # notch pixels must read as clearly white-blended
K8 = np.ones((3, 3), bool)

# glow-removal parameters
GLOW_FULL, GLOW_ZERO = 50.0, 70.0   # px from the acorn: envelope 1 .. 0
REF_RING = (60.0, 110.0)            # px from the acorn: where clean region colours are sampled
FIT_RES = 5.0                       # max distance (RGB) to a region's blend line to count as fit
SEEDS = {                           # approximate clean colours (refined from the ring)
    "cream": (251, 243, 220), "beige": (234, 203, 174), "collar_light": (158, 113, 80),
    "collar_dark": (140, 91, 60), "tan": (236, 192, 131),
}
M_INIT = np.array([151.0, 243.0, 190.0])


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64)


def bbox_of(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# ---------------------------------------------------------------- edge / alpha
def outer_band(part, bg, levels=EDGE_LEVELS):
    """level[y,x] = 1..levels for part pixels reachable from bg within the part, else 0."""
    level = np.zeros(part.shape, np.uint8)
    front = bg.copy()
    for lv in range(1, levels + 1):
        new = ndi.binary_dilation(front, K8) & part & (level == 0)
        level[new] = lv
        front = new
    return level


def solid_color(O, core, ys, xs):
    """Local solid colour for pixels (ys, xs): 3x3 mean of core pixels at the nearest core pixel."""
    cnt = ndi.uniform_filter(core.astype(np.float64), 3)
    S_img = np.stack([ndi.uniform_filter(O[..., c] * core, 3) for c in range(3)], -1) / np.maximum(cnt, 1e-9)[..., None]
    _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)
    return S_img[iy[ys, xs], ix[ys, xs]]


def white_alpha(C, S):
    """C = a*F + (1-a)*white with F ~ S: projected alpha, clamped to [a_min, 1]."""
    d_s, d_c = 255.0 - S, 255.0 - C
    denom = (d_s * d_s).sum(1)
    a = np.where(denom > 1.0, (d_c * d_s).sum(1) / np.maximum(denom, 1e-9), 1.0)
    a_min = d_c.max(1) / 255.0
    return np.clip(np.maximum(a, a_min), 0.0, 1.0), a_min


def soft_alpha(O, part, core, by, bx, C, S0):
    """Alpha of silhouette pixels C (positions by, bx), S0 = nearest core colour.

    Pass 1: white blend of S0 (projection).  Pixels NOT explained by that blend (perpendicular
    residual > 10 + 0.1*|white-C|) get pass 2: blend against the nearest "solid" pixel (core,
    or an unexplained band pixel with |white-C| >= 120).  If that fails too, a clearly coloured
    pixel (|white-C| >= 100) is opaque art (e.g. the thin tan sliver on top of the tail whose
    nearest core is the dark stripe) -> alpha 1; a light one takes the better of both fits."""
    H, W = part.shape
    dC = 255.0 - C
    nC = np.linalg.norm(dC, axis=1)

    def evaluate(S):
        dS = 255.0 - S
        a = (dC * dS).sum(1) / np.maximum((dS * dS).sum(1), 1e-9)
        return a, np.linalg.norm(dC - a[:, None] * dS, axis=1)

    a, r = evaluate(S0)
    tol = 10.0 + 0.1 * nC
    bad = r > tol
    if bad.any():
        # pass 2: blend against the nearest solid pixel (core, or an unexplained strongly
        # coloured band pixel such as the tail-top tan sliver)
        opaque = core.copy()
        solid = bad & (nC >= 120.0)
        opaque[by[solid], bx[solid]] = True
        a1, r1 = evaluate(solid_color(O, opaque, by, bx))
        use1 = bad & (r1 <= tol)
        a = np.where(use1, a1, a)
        rest = bad & ~use1
        a = np.where(rest & (nC >= 100.0), 1.0, a)                     # opaque art
        a = np.where(rest & (nC < 100.0) & (r1 < r), a1, a)           # light: best of both
    a_min = dC.max(1) / 255.0
    return np.clip(np.maximum(a, a_min), 0.0, 1.0)


def dewhite(O, part, bg):
    """Return (rgb float, alpha float 0..1, band level) for one part.

    Band = EDGE_LEVELS px inside the silhouette.  Narrow notches (e.g. the V between two tuft
    blobs) carry white-blended pixels deeper than that; they are added as an extension when
    they are connected to the band, lie at most EXT_LEVELS deep and still read as clearly
    blended with white (alpha < EXT_MAX_ALPHA against the deep-core colour)."""
    level = outer_band(part, bg, EXT_LEVELS)
    band = (level > 0) & (level <= EDGE_LEVELS)
    alpha = part.astype(np.float64)
    rgb = O.copy()
    if not band.any():
        return rgb, alpha, level
    # --- notch extension
    deep = part & (level == 0)
    ext = np.zeros_like(part)
    if deep.sum() >= 20:
        cand = part & (level > EDGE_LEVELS)
        cy, cx = np.nonzero(cand)
        a_c, _ = white_alpha(O[cy, cx], solid_color(O, deep, cy, cx))
        ok = np.zeros_like(part); ok[cy, cx] = a_c < EXT_MAX_ALPHA
        front = band
        for _ in range(EXT_LEVELS):
            new = ndi.binary_dilation(front, K8) & ok & ~ext
            if not new.any():
                break
            ext |= new
            front = new
    level = np.where(band, level, np.where(ext, EDGE_LEVELS + 1, 0)).astype(np.uint8)
    soft_zone = band | ext
    core = part & ~soft_zone
    if core.sum() < 20:                       # tiny part: fall back to its darkest pixels
        core = part & (O.sum(2) <= np.percentile(O.sum(2)[part], 30))
    by, bx = np.nonzero(band | ext)
    S0 = solid_color(O, core, by, bx)
    C = O[by, bx]
    a = soft_alpha(O, part, core, by, bx, C, S0)
    a_min = (255.0 - C).max(1) / 255.0
    a8 = np.round(a * 255.0)
    a8 = np.maximum(a8, np.ceil(a_min * 255.0 - 1e-9))   # keep F in gamut after quantisation
    a8 = np.clip(a8, 1, 255)
    aq = a8 / 255.0
    F = (C - (1.0 - aq)[:, None] * 255.0) / aq[:, None]
    rgb[by, bx] = np.clip(F, 0, 255)
    alpha[by, bx] = aq
    return rgb, alpha, level


# ---------------------------------------------------------------- glow removal
def remove_glow(O, torso, pendant):
    H, W = torso.shape
    dp = ndi.distance_transform_edt(~pendant)
    names = list(SEEDS)
    # clean region colours from the ring just outside the glow ("nearest clean pixels")
    refs = {}
    for k in names:
        s = np.array(SEEDS[k], float)
        m = torso & (dp >= REF_RING[0]) & (dp <= REF_RING[1]) & (np.abs(O - s).max(2) < 12)
        refs[k] = np.median(O[m], 0)
    Bs = np.array([refs[k] for k in names])

    zone = torso & (dp < GLOW_ZERO)
    zy, zx = np.nonzero(zone)
    C = O[zy, zx]

    # --- 1. glow colour M and per-pixel glow strength t (normal-blend model)
    def fit(M):
        res, ts = [], []
        for B in Bs:
            D = M - B
            t = np.clip(((C - B) @ D) / (D @ D), 0.0, 0.95)
            res.append(np.linalg.norm(C - (B + t[:, None] * D), axis=1))
            ts.append(t)
        res, ts = np.array(res), np.array(ts)
        k = res.argmin(0)
        idx = np.arange(len(k))
        return k, res[k, idx], ts[k, idx]

    M = M_INIT.copy()
    for _ in range(8):
        k, r, t = fit(M)
        g = (r < FIT_RES) & (t > 0.05)
        B = Bs[k[g]]; T = t[g]; Cg = C[g]
        M = (T[:, None] * (Cg - B + T[:, None] * B)).sum(0) / (T ** 2).sum()
    k, r, t = fit(M)
    good = r < FIT_RES
    # smooth glow-strength field (fitted pixels -> normalised Gaussian average), with envelope
    tf = np.zeros((H, W)); wf = np.zeros((H, W))
    tf[zy[good], zx[good]] = t[good]; wf[zy[good], zx[good]] = 1.0
    t_s = ndi.gaussian_filter(tf, 3.0) / np.maximum(ndi.gaussian_filter(wf, 3.0), 1e-6)
    env = np.clip((GLOW_ZERO - dp) / (GLOW_ZERO - GLOW_FULL), 0.0, 1.0)
    t_s = np.clip(t_s * env, 0.0, 0.95) * zone

    # --- 2. spatial region map. The blue channel separates the regions under the glow
    #        (glow lowers R a lot, changes B little): dark collar < 70 < light collar < 130
    #        < beige shadow band < 190 < belly cream.  5x5 majority vote removes noise.
    ridx = {"collar_dark": 0, "collar_light": 1, "beige": 2, "cream": 3}
    order = ["collar_dark", "collar_light", "beige", "cream"]
    Bo = np.array([refs[n] for n in order])
    lab0 = np.digitize(O[..., 2], [70, 130, 190])
    ring = ndi.binary_dilation(pendant, K8, iterations=3) & zone        # acorn AA mixed into torso
    vote_ok = zone & ~ring
    cnt = np.stack([ndi.uniform_filter(((lab0 == i) & vote_ok).astype(float), 5) for i in range(4)])
    lab = cnt.argmax(0)
    known = vote_ok
    _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
    lab = np.where(known, lab, lab[iy, ix])                             # ring: nearest region
    # anti-aliased region-colour image (soft label boundaries, normalised inside the torso)
    soft = np.stack([ndi.gaussian_filter(((lab == i) & torso).astype(float), 0.7) for i in range(4)])
    target = np.einsum("khw,kc->hwc", soft, Bo) / np.maximum(soft.sum(0), 1e-6)[..., None]

    # --- 3. replace: smooth weight from glow strength -> no ring at the glow's outer edge
    x = np.clip((t_s - 0.005) / 0.04, 0, 1)
    w = x * x * (3 - 2 * x)
    w[ring] = 1.0
    res = O.copy()
    res[zone] = O[zone] + w[zone][:, None] * (target[zone] - O[zone])
    res[ring] = Bo[lab[ring]]                    # acorn-adjacent torso pixels: clean region colour
    # torso pixels on the boundary with head/arms (not the acorn): un-blend only (keeps the AA
    # with the neighbouring part; the glow there is weak)
    other = ndi.binary_dilation(~torso & ~pendant, K8) & zone & ~ring
    oy, ox = np.nonzero(other)
    tt = t_s[oy, ox][:, None]
    res[oy, ox] = (O[oy, ox] - tt * M) / (1 - tt)
    res = np.round(np.clip(res, 0, 255))
    changed = torso & (np.abs(res - O).max(2) > 0)
    info = {
        "glow_color_M": [round(float(v), 1) for v in M],
        "region_clean_colors": {n: [round(float(v), 1) for v in refs[n]] for n in names},
        "fitted_px": int(good.sum()), "zone_px": int(zone.sum()),
        "t_max": round(float(t_s.max()), 3),
        "changed_px": int(changed.sum()),
        "changed_px_diff_gt_2": int((torso & (np.abs(res - O).max(2) > 2)).sum()),
        "acorn_ring_px": int(ring.sum()),
    }
    return res, changed, t_s, info


# ---------------------------------------------------------------- compositing / previews
def composite(layers, bg_rgb, order):
    out = np.empty((1024, 1024, 3)); out[:] = bg_rgb
    for p in order:
        rgba = layers[p].astype(np.float64)
        a = rgba[..., 3:4] / 255.0
        out = rgba[..., :3] * a + out * (1 - a)
    return out


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def checker(h, w, s=16):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // s + xx // s) % 2)[..., None] == 0, 200, 235).astype(np.float64).repeat(3, 2)


def contact_sheet(layers, info, path):
    cell, pad, cols = 300, 8, 4
    rows = (len(STACK) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + 40 + pad) + pad), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    ck = checker(1024, 1024)
    for i, p in enumerate(STACK):
        rgba = layers[p].astype(np.float64)
        a = rgba[..., 3:4] / 255.0
        img = rgba[..., :3] * a + ck * (1 - a)
        x0, y0, x1, y1 = info[p]["bbox"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = max(x1 - x0, y1 - y0) + 20
        bx0, by0 = max(0, cx - side // 2), max(0, cy - side // 2)
        tile = Image.fromarray(np.round(img[by0:by0 + side, bx0:bx0 + side]).astype(np.uint8))
        tile.thumbnail((cell, cell), Image.LANCZOS)
        gx, gy = pad + (i % cols) * (cell + pad), pad + (i // cols) * (cell + 40 + pad)
        sheet.paste(tile, (gx + (cell - tile.width) // 2, gy + 36 + (cell - tile.height) // 2))
        d.text((gx + 4, gy + 2), f"{i}: {p}", fill=COLORS[p], font=font(18))
        d.text((gx + 4, gy + 21), f"bbox {info[p]['bbox']}", fill=(200, 200, 200), font=font(13))
    sheet.save(path)


def glow_sheet(O, torso_before, torso_after, tmap, path):
    box = (390, 600, 640, 840)   # x0,y0,x1,y1
    S = 3
    ck = checker(1024, 1024, 8)

    def on_ck(rgba):
        a = rgba[..., 3:4] / 255.0
        return rgba[..., :3] * a + ck * (1 - a)

    tiles = [("original", O),
             ("torso layer BEFORE (raw)", on_ck(torso_before)),
             ("torso layer AFTER glow removal", on_ck(torso_after)),
             ("removed glow strength t (x3)", np.repeat(np.clip(tmap * 3 * 255, 0, 255)[..., None], 3, 2))]
    w, h = (box[2] - box[0]) * S, (box[3] - box[1]) * S
    sheet = Image.new("RGB", (2 * w + 30, 2 * (h + 34) + 20), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for i, (lab, img) in enumerate(tiles):
        crop = Image.fromarray(np.round(np.clip(img, 0, 255)).astype(np.uint8)).crop(box).resize((w, h), Image.NEAREST)
        gx, gy = 10 + (i % 2) * (w + 10), 10 + (i // 2) * (h + 34)
        d.text((gx, gy), lab, fill=(255, 255, 255), font=font(20))
        sheet.paste(crop, (gx, gy + 28))
    sheet.save(path)


def edge_sheet(comps, path):
    # ear, cheek, left arm, tuft notch, tail outer edge, right leg, head/collar/tail gap
    boxes = [(280, 115, 345, 180), (205, 430, 270, 495), (250, 690, 315, 755),
             (440, 80, 505, 145), (890, 660, 955, 725), (540, 900, 605, 965), (620, 590, 685, 655)]
    S = 5
    w = 65 * S
    sheet = Image.new("RGB", (len(boxes) * (w + 6) + 6, len(comps) * (w + 30) + 6), (60, 60, 60))
    d = ImageDraw.Draw(sheet)
    for r, (lab, img) in enumerate(comps):
        im = Image.fromarray(np.round(np.clip(img, 0, 255)).astype(np.uint8))
        for c, b in enumerate(boxes):
            gx, gy = 6 + c * (w + 6), 6 + r * (w + 30)
            if c == 0:
                d.text((gx, gy), lab, fill=(255, 255, 255), font=font(16))
            sheet.paste(im.crop(b).resize((w, w), Image.NEAREST), (gx, gy + 22))
    sheet.save(path)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    O = load_rgb(ORIG)
    H, W, _ = O.shape
    masks = {p: np.asarray(Image.open(MASKS / f"{p}.png").convert("L")) > 127 for p in PARTS}
    fg = np.zeros((H, W), bool)
    for m in masks.values():
        assert not (fg & m).any(), "masks overlap"
        fg |= m
    bg = ~fg

    # glow removal works on the original pixels of the torso
    O_torso, glow_changed, tmap, glow_info = remove_glow(O, masks["torso"], masks["pendant"])
    print("glow:", json.dumps(glow_info))

    layers, info = {}, {}
    raw_torso = None
    for p in PARTS:
        src = O_torso if p == "torso" else O
        rgb, alpha, level = dewhite(src, masks[p], bg)
        a8 = np.round(alpha * 255).astype(np.uint8)
        a8[~masks[p]] = 0
        rgb8 = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        rgb8[a8 == 0] = 0
        layers[p] = np.dstack([rgb8, a8])
        Image.fromarray(layers[p], "RGBA").save(OUT / f"{p}.png")
        if p == "torso":
            r2, al2, _ = dewhite(O, masks[p], bg)
            b8 = np.round(al2 * 255).astype(np.uint8); b8[~masks[p]] = 0
            raw_torso = np.dstack([np.round(r2), b8]).astype(np.float64)
        edge = masks[p] & (level > 0)
        info[p] = {
            "bbox": bbox_of(a8 > 0), "area_px": int(masks[p].sum()),
            "outer_edge_px": int(edge.sum()),
            "soft_alpha_px": int(((a8 > 0) & (a8 < 255)).sum()),
            "internal_boundary_alpha_min": int(a8[masks[p] & ~(level > 0)].min()),
        }
        print(f"  {p:8s} bbox={info[p]['bbox']} area={info[p]['area_px']} soft_alpha={info[p]['soft_alpha_px']}")

    # ---- recomposites
    white = composite(layers, (255, 255, 255), STACK)
    dark = composite(layers, DARK, STACK)
    green = composite(layers, GREEN, STACK)
    for name, img in [("white", white), ("dark", dark), ("green", green)]:
        Image.fromarray(np.round(img).astype(np.uint8)).save(OUT / f"recomposite-{name}.png")

    diff = np.abs(np.round(white) - O).max(2)
    keep = ~glow_changed
    soft = np.zeros((H, W), bool)
    for p in PARTS:
        soft |= (layers[p][..., 3] > 0) & (layers[p][..., 3] < 255)
    near_edge = ndi.binary_dilation(bg, K8, iterations=EDGE_LEVELS) & ~ndi.binary_erosion(bg, K8, iterations=1) | soft
    stats = {
        "all_px": {"max": int(diff.max()), "mean": round(float(np.abs(np.round(white) - O).mean()), 4)},
        "excluding_glow_change": {
            "max": int(diff[keep].max()),
            "mean": round(float(np.abs(np.round(white) - O)[keep].mean()), 4),
            "px_diff_gt_0": int((diff[keep] > 0).sum()),
            "px_diff_gt_2": int((diff[keep] > 2).sum()),
            "max_inside_fg_away_from_edges": int(diff[keep & fg & ~near_edge].max()),
            "max_in_background": int(diff[keep & bg].max()),
            "max_on_soft_edge": int(diff[keep & soft].max()) if soft.any() else 0,
        },
        "glow_changed_px": int(glow_changed.sum()),
        "glow_area_mean_diff": round(float(np.abs(np.round(white) - O)[glow_changed].mean()), 2),
        "glow_area_max_diff": int(diff[glow_changed].max()),
    }
    print("recomposite vs original:", json.dumps(stats, indent=1))

    order = {
        "canvas": [W, H],
        "source": "art/v2/tori/original.png",
        "masks": "art/v2/tori/step2-masks",
        "draw_order_back_to_front": STACK,
        "layers": {p: {"z": STACK.index(p), "file": f"{p}.png", **info[p], "why": STACK_REASONS[p]} for p in STACK},
        "edge_alpha": {
            "method": "outer silhouette band (<= %d px from background through the same part) de-whited: "
                      "alpha = projection of (white-C) on (white-S), S = local core colour, clamped to "
                      "[a_min,1]; colour un-premultiplied. Internal part/part boundaries alpha 255." % EDGE_LEVELS,
        },
        "glow_removal": {**glow_info, "envelope_px_from_acorn": [GLOW_FULL, GLOW_ZERO]},
        "recomposite_white_vs_original": stats,
    }
    (OUT / "stack-order.json").write_text(json.dumps(order, indent=1))

    # ---- previews
    contact_sheet(layers, info, OUT / "contact.png")
    glow_sheet(O, raw_torso, layers["torso"].astype(np.float64), tmap, OUT / "glow-before-after.png")
    raw_dark = O * fg[..., None] + np.array(DARK, float) * bg[..., None]
    edge_sheet([("hard mask cut-out (no de-white) over #202020", raw_dark),
                ("layers over #202020", dark), ("layers over #6BBF59", green),
                ("layers over white", white), ("original", O)], OUT / "edge-check.png")


if __name__ == "__main__":
    main()
