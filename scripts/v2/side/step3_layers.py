#!/usr/bin/env python3
"""
STEP 3 - Tori v2 SIDE view: transparent layers (visible pixels only; no occlusion fill).

Input : art/v2/tori-side/original.png, art/v2/tori-side/step2-masks/<part>.png,
        art/v2/tori-side/step1-fix/work/{arm_layer,body_under_arm}.png
Output: art/v2/tori-side/step3-layers/<part>.png (RGBA 1024, same coords), stack-order.json,
        recomposite-{white,dark,green}.png, contact.png, edge-check.png

Same method as the front (scripts/v2/step3_layers.py, imported): RGB = original, alpha 255 on the mask;
outer silhouette band (<=3 px from background through the same part) de-whited (projection on the
white->local-core-colour line, un-premultiplied).  Internal part/part boundaries alpha 255.
Exception - armNear: the arm was composited by step 1, so its exact layer is known: armNear layer =
step-1 arm layer (coverage AA kept), and the torso layer gets the step-1 body pixels (the image before the
arm) under the arm's soft-edge pixels, so arm-over-torso reproduces the original exactly and no
arm-AA seam is baked into the torso (front step-4 lesson applied up front).  These torso pixels are
"hidden" (fully under/inside the arm's coverage), the visible result is unchanged.
Pendant glow: checked - the belly around the acorn has no glow in cand3 (torso px within 4..14 px of the
acorn are within 6 levels of the flat cream / collar / band colours), so nothing is removed.
"""
import importlib.util, json, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("front3", ROOT / "scripts/v2/step3_layers.py")
F3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(F3)

ORIG = ROOT / "art/v2/tori-side/original.png"
MASKS = ROOT / "art/v2/tori-side/step2-masks"
S1 = ROOT / "art/v2/tori-side/step1-fix/work"
OUT = ROOT / "art/v2/tori-side/step3-layers"
PARTS = ["head", "tuft", "earNear", "torso", "pendant", "armNear", "legNear", "legFar", "tail"]
STACK = ["tail", "legFar", "legNear", "torso", "pendant", "armNear", "tuft", "head", "earNear"]
REASONS = {
    "tail": "behind everything: it touches only the torso's back edge (its shade / stripe / cream bands go behind the body).",
    "legFar": "far leg: behind the body and behind the near leg (darker, it disappears under the belly).",
    "legNear": "behind the torso: the near leg's fur continues the torso's fur, the hip end will be hidden under the belly bottom.",
    "torso": "body incl. collar; over the legs and tail, under the arm, pendant and head.",
    "pendant": "hangs on the collar front: in front of the torso.",
    "armNear": "near arm lies on the body side: in front of the torso (its top is tucked under the collar -> collar_front in step 5).",
    "tuft": "behind the head (front decision: flat tuft behind the head, base extended under it in step 5).",
    "head": "head's lower band overlaps the collar -> in front of the torso.",
    "earNear": "in front of the head: the ear covers the head's back stripe.",
}
COLORS = {"head": (255, 99, 71), "tuft": (255, 0, 255), "earNear": (255, 215, 0), "torso": (30, 144, 255),
          "pendant": (0, 255, 200), "armNear": (255, 140, 0), "legNear": (0, 200, 0), "legFar": (220, 20, 60),
          "tail": (139, 90, 255)}
DARK, GREEN = (0x20, 0x20, 0x20), (0x6B, 0xBF, 0x59)


def font(sz):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(f, sz) if pathlib.Path(f).exists() else ImageFont.load_default()


def composite(layers, bg, order):
    out = np.empty((1024, 1024, 3)); out[:] = bg
    for p in order:
        r = layers[p].astype(float); a = r[..., 3:4] / 255
        out = r[..., :3] * a + out * (1 - a)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    O = np.asarray(Image.open(ORIG).convert("RGB")).astype(float)
    masks = {p: np.asarray(Image.open(MASKS / f"{p}.png")) > 127 for p in PARTS}
    fg = np.zeros(O.shape[:2], bool)
    for m in masks.values():
        assert not (fg & m).any(); fg |= m
    bg = ~fg
    arm = np.asarray(Image.open(S1 / "arm_layer.png").convert("RGBA")).copy()
    body = np.asarray(Image.open(S1 / "body_under_arm.png").convert("RGB")).astype(float)

    # glow check around the acorn
    dp = ndi.distance_transform_edt(~masks["pendant"])
    ring = masks["torso"] & (dp >= 4) & (dp <= 14)
    flats = np.array([[249, 241, 220], [234, 203, 174], [235, 191, 130], [205, 148, 100], [154, 113, 81], [137, 92, 60]], float)
    dev = np.abs(O[ring][:, None, :] - flats[None]).max(2).min(1)
    glow = {"ring_px": int(ring.sum()), "px_dev_gt6_from_flat_colours": int((dev > 6).sum()),
            "px_greenish(G>R)": int((O[ring][:, 1] > O[ring][:, 0]).sum())}
    print("glow check", glow)

    layers, info = {}, {}
    for p in PARTS:
        if p == "armNear":
            L = arm.copy()
            level = np.zeros(O.shape[:2], np.uint8)
        else:
            rgb, alpha, level = F3.dewhite(O, masks[p], bg)
            a8 = np.round(alpha * 255).astype(np.uint8); a8[~masks[p]] = 0
            rgb8 = np.round(np.clip(rgb, 0, 255)).astype(np.uint8); rgb8[a8 == 0] = 0
            L = np.dstack([rgb8, a8])
        layers[p] = L
    # torso under the arm's soft edge: body pixels (exact un-mix of the step-1 composite)
    soft = ((arm[..., 3] > 0) | masks["armNear"]) & (arm[..., 3] < 255)
    T = layers["torso"]
    T[soft, :3] = np.round(body[soft]).astype(np.uint8); T[soft, 3] = 255
    for p in PARTS:
        Image.fromarray(layers[p], "RGBA").save(OUT / f"{p}.png")
        a8 = layers[p][..., 3]
        ys, xs = np.nonzero(a8)
        lv = F3.outer_band(masks[p], bg) if p != "armNear" else np.zeros_like(a8)
        info[p] = {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   "area_px": int(masks[p].sum()), "outer_edge_px": int((masks[p] & (lv > 0)).sum()),
                   "soft_alpha_px": int(((a8 > 0) & (a8 < 255)).sum())}
    white = composite(layers, (255, 255, 255), STACK)
    Image.fromarray(np.round(white).astype(np.uint8)).save(OUT / "recomposite-white.png")
    Image.fromarray(np.round(composite(layers, DARK, STACK)).astype(np.uint8)).save(OUT / "recomposite-dark.png")
    Image.fromarray(np.round(composite(layers, GREEN, STACK)).astype(np.uint8)).save(OUT / "recomposite-green.png")
    diff = np.abs(np.round(white) - O).max(2)
    soft_all = np.zeros_like(fg)
    for p in PARTS:
        soft_all |= (layers[p][..., 3] > 0) & (layers[p][..., 3] < 255)
    stats = {"max": int(diff.max()), "mean": round(float(diff.mean()), 4), "px_diff_gt2": int((diff > 2).sum()),
             "max_inside_fg_off_soft_edges": int(diff[fg & ~ndi.binary_dilation(soft_all | bg, iterations=1)].max()),
             "max_in_background": int(diff[bg].max()),
             "max_under_arm_edge": int(diff[soft].max())}
    print("recomposite vs original", stats)
    order = {"canvas": [1024, 1024], "source": "art/v2/tori-side/original.png", "masks": "art/v2/tori-side/step2-masks",
             "draw_order_back_to_front": STACK,
             "layers": {p: {"z": STACK.index(p), "file": f"{p}.png", **info[p], "why": REASONS[p]} for p in STACK},
             "edge_alpha": "front method (scripts/v2/step3_layers.py dewhite); armNear = exact step-1 arm layer",
             "glow_check": glow, "recomposite_white_vs_original": stats}
    (OUT / "stack-order.json").write_text(json.dumps(order, indent=1))
    # contact
    cell, pad, cols = 300, 8, 3
    rows = (len(STACK) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + 40 + pad) + pad), (40, 40, 40))
    d = ImageDraw.Draw(sheet); ck = F3.checker(1024, 1024)
    for i, p in enumerate(STACK):
        r = layers[p].astype(float); a = r[..., 3:4] / 255
        img = r[..., :3] * a + ck * (1 - a)
        x0, y0, x1, y1 = info[p]["bbox"]; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        side = max(x1 - x0, y1 - y0) + 20
        bx0, by0 = max(0, cx - side // 2), max(0, cy - side // 2)
        t = Image.fromarray(np.round(img[by0:by0 + side, bx0:bx0 + side]).astype(np.uint8)); t.thumbnail((cell, cell))
        gx, gy = pad + (i % cols) * (cell + pad), pad + (i // cols) * (cell + 40 + pad)
        sheet.paste(t, (gx + (cell - t.width) // 2, gy + 36 + (cell - t.height) // 2))
        d.text((gx + 4, gy + 2), f"{i}: {p}", fill=COLORS[p], font=font(18))
        d.text((gx + 4, gy + 21), f"bbox {info[p]['bbox']}", fill=(200, 200, 200), font=font(13))
    sheet.save(OUT / "contact.png")
    # edge check: ear, tuft, tail edge, legs, pendant, head-collar
    boxes = [(410, 105, 475, 170), (600, 70, 665, 135), (130, 640, 195, 705), (440, 900, 505, 965),
             (660, 900, 725, 965), (680, 660, 745, 725), (420, 590, 485, 655)]
    S = 5; w = 65 * S
    comps = [("hard cut over #202020", O * fg[..., None] + np.array(DARK) * bg[..., None]),
             ("layers over #202020", composite(layers, DARK, STACK)), ("layers over #6BBF59", composite(layers, GREEN, STACK)),
             ("original", O)]
    sh = Image.new("RGB", (len(boxes) * (w + 6) + 6, len(comps) * (w + 30) + 6), (60, 60, 60)); d = ImageDraw.Draw(sh)
    for r_, (lab, img) in enumerate(comps):
        im = Image.fromarray(np.round(np.clip(img, 0, 255)).astype(np.uint8))
        for c, b in enumerate(boxes):
            gx, gy = 6 + c * (w + 6), 6 + r_ * (w + 30)
            if c == 0:
                d.text((gx, gy), lab, fill=(255, 255, 255), font=font(16))
            sh.paste(im.crop(b).resize((w, w), Image.NEAREST), (gx, gy + 22))
    sh.save(OUT / "edge-check.png")


if __name__ == "__main__":
    main()
