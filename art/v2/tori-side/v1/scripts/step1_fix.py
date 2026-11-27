#!/usr/bin/env python3
"""STEP 1 - Tori v2 SIDE view: the two requested fixes on the chosen candidate (cand3).

Input : art/v2/tori-side/original-before-fix.png  (= art/v2/tori/views/side-cand3.png, unchanged copy)
        art/v2/tori-side/step1-fix/work/ai_arm_green_e.png  (Gemini: one arm alone on #00FF00, drawn
              with the front arm (art/v2/tori/step5-rig/layers/armR.png on green) + front-ref as style refs)
Output: art/v2/tori-side/original.png             fixed original (eye + arm)
        art/v2/tori-side/fix-check.png            before/after zooms of the eye and the arm
        art/v2/tori-side/step1-fix/fix-report.json
        art/v2/tori-side/step1-fix/work/arm_layer.png       the new arm as an exact RGBA layer
              (its coverage/colours as composited; used by steps 2/3 instead of guessing the cut)
        art/v2/tori-side/step1-fix/work/body_under_arm.png  the image before the arm was composited
              (eye fixed, line removed) = the torso's own pixels under the arm (used by step 4)

1. Eye highlights (step1_base.eye_fix): big highlight moved to UPPER-LEFT, small to LOWER-RIGHT by
   mirroring cand3's own highlight pixels about the eye's vertical axis; vacated pixels painted with the
   eye's own brown.  Only pixels >= 3 px inside the eye change.
2. Near arm: the brown U-line is painted out with the body's own tan (step1_base.remove_line); the new
   arm (arm_render.py) = the AI arm silhouette, re-rendered flat in the body's tan + shade with 4x4
   supersampled AA; shading band = the AI's own back-edge band united with an offset crescent (silhouette
   minus itself shifted toward the front/top), so the band is a filled wedge that wraps under the paw
   like the front arms' underside band; shoulder top clipped at the collar's bottom edge (tucked under
   the collar); arm pixels inside the collar-shadow band take the band colour (as on the front arms).
   Placement: the paw tip just overlaps the cream belly edge - a tan paw over tan fur has no visible
   front edge in a lineless style, so the paw needs the cream behind it to read as a paw.
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from step1_base import eye_fix, remove_line, ROOT, SRC, WORK  # noqa: E402
from arm_render import render_arm  # noqa: E402

OUT = ROOT / "art/v2/tori-side"
ARM = dict(s=0.29, rot=-2.0, src=(405, 290), dst=(530, 661), offset=(50, -60))
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
    shade = band.copy()           # cand3's collar-shadow band on tan == its shade colour
    A, C, shcov = render_arm(ARM, CB, BB, tan, shade, band)
    F = B * (1 - A[..., None]) + C * A[..., None]
    F8 = np.round(F).astype(np.uint8)
    Image.fromarray(F8).save(OUT / "original.png")
    rgba = np.dstack([np.round(C).clip(0, 255).astype(np.uint8), np.round(A * 255).astype(np.uint8)])
    rgba[rgba[..., 3] == 0, :3] = 0
    Image.fromarray(rgba, "RGBA").save(WORK / "arm_layer.png")
    Image.fromarray(np.round(B).astype(np.uint8)).save(WORK / "body_under_arm.png")
    Image.fromarray(np.round(E).astype(np.uint8)).save(WORK / "eye_fixed_only.png")

    # checks
    cream = (B.min(2) > 205) & (B[..., 0] > 240)
    arm = A > 0.5
    ys, xs = np.nonzero(A > 0)
    d = np.abs(F8.astype(int) - O.astype(int)).max(2) > 0
    eye_box = em["eye"]
    rep = {"eye": einfo, "line": linfo,
           "arm": {"ai_source": "work/ai_arm_green_e.png", "transform": ARM,
                   "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   "opaque_px": int((A >= 0.999).sum()), "coverage_px": int((A > 0).sum()),
                   "shade_band_px": int((shcov > 0.5).sum()),
                   "colours": {"tan": [int(v) for v in tan], "shade/band": [int(v) for v in shade],
                               "front_arm_ref": FRONT_ARM},
                   "collar_bottom_poly": [float(v) for v in pc], "band_bottom_poly": [float(v) for v in pb],
                   "paw_over_cream_px": int((arm & cream).sum())},
           "changed_px_total": int(d.sum()),
           "changed_px_outside(eye|line|arm)": int((d & ~eye_box & ~line & ~(A > 0)).sum())}
    (OUT / "step1-fix").mkdir(exist_ok=True)
    (OUT / "step1-fix/fix-report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

    # fix-check sheet
    O8 = Image.fromarray(O.astype(np.uint8)); N8 = Image.fromarray(F8)
    rows = [("eye  (x5)", (650, 355, 746, 467), 5), ("arm  (x2.5)", (420, 600, 680, 880), 2.5)]
    tiles = []
    for lab, box, sc in rows:
        w, h = int((box[2] - box[0]) * sc), int((box[3] - box[1]) * sc)
        tiles.append((lab, O8.crop(box).resize((w, h), Image.NEAREST), N8.crop(box).resize((w, h), Image.NEAREST)))
    full_o = O8.resize((400, 400), Image.LANCZOS); full_n = N8.resize((400, 400), Image.LANCZOS)
    Wd = max(t[1].width for t in tiles) * 2 + 30
    Ht = sum(t[1].height + 40 for t in tiles) + 450
    sheet = Image.new("RGB", (max(Wd, 830), Ht), (240, 240, 240))
    dr = ImageDraw.Draw(sheet)
    y = 10
    dr.text((10, y), "Tori side - step 1 fix check: BEFORE (cand3) | AFTER", fill=(0, 0, 0), font=font(20)); y += 34
    sheet.paste(full_o, (10, y)); sheet.paste(full_n, (420, y)); y += 410
    for lab, a, b in tiles:
        dr.text((10, y), lab + "  before | after", fill=(0, 0, 0), font=font(16))
        sheet.paste(a, (10, y + 22)); sheet.paste(b, (20 + a.width, y + 22)); y += a.height + 40
    sheet.save(OUT / "fix-check.png")


if __name__ == "__main__":
    main()
