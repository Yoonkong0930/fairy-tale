#!/usr/bin/env python3
"""
STEP 3 (3/4 view) - Tori v2 transparent layers (visible pixels only, no occlusion fill).

Same method as the front (scripts/v2/step3_layers.py; its edge functions are imported and reused):
  RGB = original pixels, alpha 255 on the part's mask; the OUTER silhouette band (<= 3 px from the
  background through the same part, + narrow white notches) is de-whited (alpha = projection of
  (white-C) on (white-S), S = local core colour, colour un-premultiplied) -> no white fringe over dark /
  coloured backgrounds; internal part/part boundaries stay alpha 255.
Pendant glow: none in this view (belly cream median is identical 3..110 px from the acorn:
  (249-250, 242-243, 218-219), std <= 2) -> nothing removed; the torso layer is the raw original.

Input : art/v2/tori-34/original.png, art/v2/tori-34/step2-masks/<part>.png
Output: art/v2/tori-34/step3-layers/<part>.png (RGBA 1024x1024), stack-order.json,
        recomposite-{white,dark,green}.png, contact.png, edge-check.png
"""
import importlib.util, json, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("front3", ROOT / "scripts/v2/step3_layers.py")
F3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F3)

ORIG = ROOT / "art/v2/tori-34/original.png"
MASKS = ROOT / "art/v2/tori-34/step2-masks"
OUT = ROOT / "art/v2/tori-34/step3-layers"

PARTS = ["head", "tuft", "earNear", "earFar", "torso", "pendant", "armNear", "armFar", "legNear", "legFar", "tail"]
STACK = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "earFar", "earNear", "head", "tuft"]
STACK_REASONS = {
    "tail": "behind everything: the near arm's silhouette is drawn over it, the torso's left side and the "
            "head's bottom band cover its root (only the dark stripe sliver shows between hand and torso).",
    "legFar": "behind torso (hip cut under the torso's lower edge, as front); far leg first.",
    "legNear": "behind torso (the torso's side continues into the leg; the hip joint hides under the torso).",
    "armFar": "FAR arm behind the torso: the torso's right side edge lies over the arm's inner shading "
              "stripe, and the collar end / head band are drawn over its shoulder (turned-away limb).",
    "torso": "body incl. collar; over tail, legs and far arm; under pendant, near arm and head.",
    "pendant": "hangs on the collar/belly: in front of the torso.",
    "armNear": "NEAR arm in front of torso and tail: its underside shading wedge lies on the body, its "
               "silhouette overlaps the tail.",
    "earFar": "behind head: the head outline passes over the ear base (step-1 transplant was composited so).",
    "earNear": "behind head: the head's round outline clips the inner-ear oval.",
    "head": "head's bottom band overlaps the collar, both arm tops and the tail top.",
    "tuft": "in front of head at step 3 (its soft lower shading overlaps the head top); step 5 flattens it "
            "and moves it behind the head, as decided for the front.",
}
COLORS = {
    "head": (255, 99, 71), "tuft": (255, 0, 255), "earNear": (255, 215, 0), "earFar": (173, 255, 47),
    "torso": (30, 144, 255), "pendant": (0, 255, 200), "armNear": (255, 140, 0), "armFar": (148, 0, 211),
    "legNear": (0, 200, 0), "legFar": (220, 20, 60), "tail": (138, 100, 255),
}
DARK, GREEN = F3.DARK, F3.GREEN


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    O = F3.load_rgb(ORIG)
    H, W, _ = O.shape
    masks = {p: np.asarray(Image.open(MASKS / f"{p}.png").convert("L")) > 127 for p in PARTS}
    fg = np.zeros((H, W), bool)
    for m in masks.values():
        assert not (fg & m).any(), "masks overlap"
        fg |= m
    bg = ~fg

    layers, info = {}, {}
    for p in PARTS:
        rgb, alpha, level = F3.dewhite(O, masks[p], bg)
        a8 = np.round(alpha * 255).astype(np.uint8)
        a8[~masks[p]] = 0
        rgb8 = np.round(np.clip(rgb, 0, 255)).astype(np.uint8)
        rgb8[a8 == 0] = 0
        layers[p] = np.dstack([rgb8, a8])
        Image.fromarray(layers[p], "RGBA").save(OUT / f"{p}.png")
        edge = masks[p] & (level > 0)
        info[p] = {"bbox": F3.bbox_of(a8 > 0), "area_px": int(masks[p].sum()),
                   "outer_edge_px": int(edge.sum()),
                   "soft_alpha_px": int(((a8 > 0) & (a8 < 255)).sum()),
                   "internal_boundary_alpha_min": int(a8[masks[p] & ~(level > 0)].min())}
        print(f"  {p:8s} bbox={info[p]['bbox']} area={info[p]['area_px']} soft_alpha={info[p]['soft_alpha_px']}")

    white = F3.composite(layers, (255, 255, 255), STACK)
    dark = F3.composite(layers, DARK, STACK)
    green = F3.composite(layers, GREEN, STACK)
    for name, img in [("white", white), ("dark", dark), ("green", green)]:
        Image.fromarray(np.round(img).astype(np.uint8)).save(OUT / f"recomposite-{name}.png")
    diff = np.abs(np.round(white) - O).max(2)
    soft = np.zeros((H, W), bool)
    for p in PARTS:
        soft |= (layers[p][..., 3] > 0) & (layers[p][..., 3] < 255)
    near_edge = ndi.binary_dilation(bg, F3.K8, iterations=F3.EDGE_LEVELS) & ~ndi.binary_erosion(bg, F3.K8, iterations=1) | soft
    stats = {"max": int(diff.max()), "mean": round(float(np.abs(np.round(white) - O).mean()), 4),
             "px_diff_gt_0": int((diff > 0).sum()), "px_diff_gt_2": int((diff > 2).sum()),
             "max_inside_fg_away_from_edges": int(diff[fg & ~near_edge].max()),
             "max_in_background": int(diff[bg].max()),
             "max_on_soft_edge": int(diff[soft].max()) if soft.any() else 0}
    print("recomposite vs original:", json.dumps(stats))
    order = {"canvas": [W, H], "source": "art/v2/tori-34/original.png", "masks": "art/v2/tori-34/step2-masks",
             "draw_order_back_to_front": STACK,
             "layers": {p: {"z": STACK.index(p), "file": f"{p}.png", **info[p], "why": STACK_REASONS[p]} for p in STACK},
             "edge_alpha": {"method": "as front step 3 (functions imported from scripts/v2/step3_layers.py): outer "
                                      "silhouette band <= %d px de-whited; internal boundaries alpha 255" % F3.EDGE_LEVELS},
             "glow_removal": "none needed: no pendant glow on the belly in this view",
             "recomposite_white_vs_original": stats}
    (OUT / "stack-order.json").write_text(json.dumps(order, indent=1))

    # previews (contact sheet uses the front helper with this view's stack/colours)
    F3.STACK, F3.COLORS = STACK, COLORS
    F3.contact_sheet(layers, info, OUT / "contact.png")
    raw_dark = O * fg[..., None] + np.array(DARK, float) * bg[..., None]
    # near ear, tuft notch, far ear, near hand/tail, far hand, legs, head/collar/arm
    boxes = [(280, 120, 345, 185), (455, 90, 520, 155), (690, 150, 755, 215), (280, 750, 345, 815),
             (700, 740, 765, 805), (480, 900, 545, 965), (640, 590, 705, 655)]
    _edge_sheet(raw_dark, dark, green, white, O, boxes, OUT / "edge-check.png")


def _edge_sheet(raw_dark, dark, green, white, O, boxes, path):
    from PIL import ImageDraw
    comps = [("hard mask cut-out over #202020", raw_dark), ("layers over #202020", dark),
             ("layers over #6BBF59", green), ("layers over white", white), ("original", O)]
    S, w = 5, 65 * 5
    sheet = Image.new("RGB", (len(boxes) * (w + 6) + 6, len(comps) * (w + 30) + 6), (60, 60, 60))
    d = ImageDraw.Draw(sheet)
    for r, (lab, img) in enumerate(comps):
        im = Image.fromarray(np.round(np.clip(img, 0, 255)).astype(np.uint8))
        for c, b in enumerate(boxes):
            gx, gy = 6 + c * (w + 6), 6 + r * (w + 30)
            if c == 0:
                d.text((gx, gy), lab, fill=(255, 255, 255), font=F3.font(16))
            sheet.paste(im.crop(b).resize((w, w), Image.NEAREST), (gx, gy + 22))
    sheet.save(path)


if __name__ == "__main__":
    main()
