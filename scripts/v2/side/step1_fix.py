#!/usr/bin/env python3
"""STEP 1 - Tori v2 SIDE view: the requested fixes on the chosen candidate (cand3).   [v2: two arms]

Input : art/v2/tori-side/original-before-fix.png  (= art/v2/tori/views/side-cand3.png, unchanged copy)
        art/v2/tori-side/step1-fix/work/ai_arm_green_e.png  (Gemini: one arm alone on #00FF00, drawn
              with the front arm + front-ref as style refs) - silhouette source for BOTH arms
Output: art/v2/tori-side/original.png             fixed original (eye + near arm + far arm)
        art/v2/tori-side/fix-check.png            before/after zooms (cand3 | v1 | v2) of eye and arms
        art/v2/tori-side/step1-fix/fix-report.json
        art/v2/tori-side/step1-fix/work/arm_layer.png       near arm, exact RGBA as composited (over everything)
        art/v2/tori-side/step1-fix/work/farArm_layer.png    far arm, exact full RGBA (drawn BEHIND the body)
        art/v2/tori-side/step1-fix/work/body_under_arm.png  eye fixed + line removed, no arms (= body pixels)
        art/v2/tori-side/step1-fix/work/original_no_farArm.png  body + near arm only (reference for the
              de-whitening of the silhouettes in step 3, which must see the white background, not the far arm)
v1 (single near arm on the body side) is kept in art/v2/tori-side/v1/.

1. Eye highlights (step1_base.eye_fix): unchanged from v1 (big highlight upper-left, small lower-right).
2. Old U-line painted out with the body's tan (step1_base.remove_line): unchanged from v1.
3. NEAR arm (v2, user option 2): same AI silhouette, re-rendered flat (arm_render2.render_arm2), now hung
   further FORWARD - shoulder under the collar at the front of the side, the arm lying over the cream belly
   and its paw reaching the belly front - so it reads by its tan-against-cream silhouette.  No band along an
   edge: one soft flat shade area under the paw only (offset crescent in the arm's own frame).  Shoulder top
   clipped at the collar's bottom edge (collar_front in step 5), collar-shadow band continued on its top.
4. FAR arm (v2, user sketch): the same silhouette, smaller, in the shade tan of the far leg with a darker
   underside, composited BEHIND the body (body = base de-whitened as one part, so the belly's AA edge lies
   over the arm) and behind the acorn; only its forearm + paw show, emerging from the belly front just below
   the collar / acorn and pointing forward-down.
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from step1_base import eye_fix, remove_line, ROOT, SRC, WORK  # noqa: E402
from arm_render import render_arm  # noqa: E402,F401  (v1)
from arm_render2 import render_arm2  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("front3", ROOT / "scripts/v2/step3_layers.py")
F3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F3)

OUT = ROOT / "art/v2/tori-side"
ARM = dict(s=0.33, rot=-8.0, src=(405, 290), dst=(608, 662), shade_ai=(24, 70))          # near arm (v2)
FAR = dict(s=0.14, rot=-30.0, src=(405, 290), dst=(692, 702), shade_ai=(36, 105))       # far arm (v2)
FAR_COLS = {"fill": (205, 148, 100), "shade": (188, 132, 88), "band": (0, 0, 0)}
V1 = ROOT / "art/v2/tori-side/v1/original.png"
FRONT_ARM = {"tan": [235, 190, 131], "shade": [207, 147, 97]}


def font(sz):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(f, sz) if pathlib.Path(f).exists() else ImageFont.load_default()


def collar_curves(B):
    tan = np.array([235, 191, 130.0])
    xs = np.arange(458, 600); cb, bb = [], []
    for x in xs:
        col = B[600:720, x]
        yb = np.nonzero(col[:, 0] < 175)[0].max()
        t = np.nonzero((np.abs(col - tan).max(1) < 10) & (np.arange(120) > yb))[0].min()
        cb.append(600 + yb + 0.5); bb.append(600 + t)
    pc, pb = np.polyfit(xs, cb, 2), np.polyfit(xs, bb, 2)
    return (lambda X: np.polyval(pc, X)), (lambda X: np.polyval(pb, X)), pc, pb


def main():
    O = np.asarray(Image.open(SRC).convert("RGB")).astype(float)
    E, einfo, em = eye_fix(O)
    B, linfo, line = remove_line(E)
    CB, BB, pc, pb = collar_curves(B)
    # body palette (cand3's own flat colours)
    tanm = (np.abs(B - [235, 191, 130]).max(2) < 8); tanm[:, :] &= np.arange(1024)[:, None] > 680
    tan = np.median(B[700:840, 470:580][tanm[700:840, 470:580]], 0)
    bandm = np.zeros(B.shape[:2], bool)
    for x in range(470, 590):
        bandm[int(CB(x)) + 3:int(BB(x)) - 2, x] = True
    band = np.median(B[bandm], 0)
    shade = np.array([207.0, 147.0, 97.0])   # front arm shade (cand3's collar band on tan is (205,148,100))
    clip = lambda X: np.polyval(pc, np.clip(X, 458, 600))
    bandf = lambda X: np.polyval(pb, np.clip(X, 458, 600))
    # far arm behind the body: body = the whole foreground of B de-whitened as one part
    fg = (255 - B.min(2)) > 8
    lab, n = ndi.label(fg)
    fg = ndi.binary_fill_holes(lab == np.argmax(ndi.sum(fg, lab, range(1, n + 1))) + 1)
    brgb, balpha, _ = F3.dewhite(B, fg, ~fg)
    Af, Cf, fsh = render_arm2(FAR, lambda X: np.full(X.shape, -1e9), lambda X: np.full(X.shape, -1e9), FAR_COLS,
                              (560, 600, 860, 880))
    bg = np.full(B.shape, 255.0)
    out = Cf * Af[..., None] + bg * (1 - Af[..., None])
    out = brgb * balpha[..., None] + out * (1 - balpha[..., None])
    Bf = np.where((Af > 0)[..., None], out, B)                    # far arm only where it is drawn
    A, C, shcov = render_arm2(ARM, clip, bandf, {"fill": tan, "shade": shade, "band": band}, (450, 600, 800, 900))
    F = Bf * (1 - A[..., None]) + C * A[..., None]
    F8 = np.round(F).astype(np.uint8)
    Image.fromarray(F8).save(OUT / "original.png")
    rgba = np.dstack([np.round(C).clip(0, 255).astype(np.uint8), np.round(A * 255).astype(np.uint8)])
    rgba[rgba[..., 3] == 0, :3] = 0
    Image.fromarray(rgba, "RGBA").save(WORK / "arm_layer.png")
    frgba = np.dstack([np.round(Cf).clip(0, 255).astype(np.uint8), np.round(Af * 255).astype(np.uint8)])
    frgba[frgba[..., 3] == 0, :3] = 0
    Image.fromarray(frgba, "RGBA").save(WORK / "farArm_layer.png")
    Image.fromarray(np.round(B).astype(np.uint8)).save(WORK / "body_under_arm.png")
    Image.fromarray(np.round(B * (1 - A[..., None]) + C * A[..., None]).astype(np.uint8)).save(WORK / "original_no_farArm.png")
    Image.fromarray(np.round(E).astype(np.uint8)).save(WORK / "eye_fixed_only.png")
    far_vis = Af * (1 - balpha) * (1 - A)

    # checks
    cream = (B.min(2) > 205) & (B[..., 0] > 240)
    arm = A > 0.5
    ys, xs = np.nonzero(A > 0)
    d = np.abs(F8.astype(int) - O.astype(int)).max(2) > 0
    eye_box = em["eye"]
    rep = {"eye": einfo, "line": linfo,
           "farArm": {"ai_source": "work/ai_arm_green_e.png", "transform": FAR, "colours": FAR_COLS,
                      "drawn_px": int((Af > 0).sum()), "visible_px(cov>=0.5)": int((far_vis >= 0.5).sum()),
                      "bbox_drawn_x0y0x1y1": [int(v) for v in np.r_[np.nonzero((Af > 0).any(0))[0][[0]], np.nonzero((Af > 0).any(1))[0][[0]], np.nonzero((Af > 0).any(0))[0][[-1]] + 1, np.nonzero((Af > 0).any(1))[0][[-1]] + 1]],
                      "note": "behind the body and the acorn; only forearm + paw visible in front of the belly line"},
           "arm": {"ai_source": "work/ai_arm_green_e.png", "transform": ARM,
                   "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   "opaque_px": int((A >= 0.999).sum()), "coverage_px": int((A > 0).sum()),
                   "shade_band_px": int((shcov > 0.5).sum()),
                   "colours": {"tan": [int(v) for v in tan], "shade/band": [int(v) for v in shade],
                               "front_arm_ref": FRONT_ARM},
                   "collar_bottom_poly": [float(v) for v in pc], "band_bottom_poly": [float(v) for v in pb],
                   "paw_over_cream_px": int((arm & cream).sum())},
           "changed_px_total": int(d.sum()),
           "changed_px_outside(eye|line|arm|farArm)": int((d & ~eye_box & ~line & ~(A > 0) & ~(Af > 0)).sum())}
    (OUT / "step1-fix").mkdir(exist_ok=True)
    (OUT / "step1-fix/fix-report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

    # fix-check sheet: cand3 | v1 (previous fix) | v2 (this fix)
    ims = [("cand3 (before)", Image.fromarray(O.astype(np.uint8))),
           ("v1 (previous: near arm on the side)", Image.open(V1).convert("RGB")),
           ("v2 (now: near arm forward + far arm)", Image.fromarray(F8))]
    rows = [("full", (100, 40, 900, 1000), 0.5), ("eye x4", (650, 355, 746, 467), 4),
            ("arms x2", (440, 600, 800, 880), 2)]
    cw = max(int((b_[2] - b_[0]) * sc) for _, b_, sc in rows)
    sheet = Image.new("RGB", (3 * (cw + 10) + 10, 40 + sum(int((b_[3] - b_[1]) * sc) + 30 for _, b_, sc in rows)),
                      (240, 240, 240))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), "Tori side - step 1 fix check (eye highlights + arms): " + " | ".join(t for t, _ in ims),
            fill=(0, 0, 0), font=font(18))
    y = 40
    for lab, box, sc in rows:
        w, h = int((box[2] - box[0]) * sc), int((box[3] - box[1]) * sc)
        dr.text((10, y), lab, fill=(0, 0, 0), font=font(14))
        for k, (_, im) in enumerate(ims):
            sheet.paste(im.crop(box).resize((w, h), Image.LANCZOS if sc < 1 else Image.NEAREST), (10 + k * (cw + 10), y + 20))
        y += h + 30
    sheet.save(OUT / "fix-check.png")


if __name__ == "__main__":
    main()
