#!/usr/bin/env python3
"""STEP 1 (part A) - side view: eye-highlight fix + removal of the U-shaped arm line.

Input : art/v2/tori-side/original-before-fix.png (= art/v2/tori/views/side-cand3.png)
Output: art/v2/tori-side/step1-fix/work/base_eye_noline.png  (eye fixed, line removed, no arm yet)
        art/v2/tori-side/step1-fix/work/line_mask.png, eye_masks.png
Imported by step1_fix.py (functions eye_fix / remove_line).

Eye: the rule (same light as the front view) is big white highlight UPPER-LEFT, small one LOWER-RIGHT;
cand3 has them mirrored.  Only pixels strictly inside the eye (>= 3 px from the eye's outer edge) are
touched: highlight pixels (+2 px AA ring) are first painted with the eye's own dark brown (normalised
convolution of the surrounding eye pixels), then the original highlight pixels are copied to the
position mirrored about the eye's vertical axis (x' = 2*cx - x, cx = centre of the eye ellipse
measured per row), so shapes, sizes, colours and AA are cand3's own.
Line: the thin brown U (one connected component of non-tan pixels inside the body, |C-tan|>10,
separate from the collar shadow band) + 2 px AA ring (pixels within 3 of tan excluded) is painted with
the body's own tan (normalised convolution of the surrounding body pixels; the tan there is flat).
"""
import pathlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "art/v2/tori-side/original-before-fix.png"
WORK = ROOT / "art/v2/tori-side/step1-fix/work"
TAN = np.array([235, 191, 130.0])


def norm_fill(img, known, target, sigmas=(1.5, 3, 6, 12)):
    out = img.copy()
    done = np.zeros(known.shape, bool)
    k = known.astype(float)
    for s in sigmas:
        ws = ndi.gaussian_filter(k, s)
        ok = target & ~done & (ws > 1e-3)
        for c in range(3):
            num = ndi.gaussian_filter(img[..., c] * k, s)
            out[..., c][ok] = num[ok] / ws[ok]
        done |= ok
    return out


def eye_fix(O):
    """returns new image, info, masks dict"""
    win = np.zeros(O.shape[:2], bool); win[340:480, 640:760] = True
    dark = win & (O.sum(2) < 300)
    eye = ndi.binary_fill_holes(ndi.binary_closing(dark, iterations=3))
    eye = eye & win
    lab, n = ndi.label(eye)
    eye = lab == np.argmax(ndi.sum(eye, lab, range(1, n + 1))) + 1
    ys, xs = np.nonzero(eye)
    mids = [(xs[ys == y].min() + xs[ys == y].max()) / 2 for y in np.unique(ys)]
    cx = float(np.median(mids))
    X2 = int(round(2 * cx))                      # mirror: x' = X2 - x
    interior = ndi.binary_erosion(eye, iterations=3)
    hl = interior & (O.sum(2) > 330)             # highlights (white big, greyish small)
    hlr = ndi.binary_dilation(hl, iterations=2) & interior
    O2 = O.copy()
    known = interior & ~hlr
    O2 = norm_fill(O2, known, hlr)
    hy, hx = np.nonzero(hlr)
    mx = X2 - hx
    ok = interior[hy, mx]
    O2[hy[ok], mx[ok]] = O[hy[ok], hx[ok]]
    bb = lambda m: [int(np.nonzero(m)[1].min()), int(np.nonzero(m)[0].min()), int(np.nonzero(m)[1].max()) + 1, int(np.nonzero(m)[0].max()) + 1]
    lab, n = ndi.label(hl)
    comps = sorted([(int((lab == i).sum()), bb(lab == i)) for i in range(1, n + 1)], reverse=True)
    info = {"eye_bbox": bb(eye), "eye_axis_x": round(cx, 2), "mirror_X2": X2,
            "highlights_before(px,bbox)": comps,
            "highlight_px_mirrored": int(ok.sum()), "highlight_px_outside_interior_skipped": int((~ok).sum()),
            "changed_px": int((np.abs(O2 - O).max(2) > 0).sum()),
            "changed_outside_eye": int(((np.abs(O2 - O).max(2) > 0) & ~eye).sum())}
    return O2, info, {"eye": eye, "hl": hl}


def remove_line(O):
    d = np.abs(O - TAN).max(2)
    win = np.zeros(O.shape[:2], bool); win[600:870, 430:640] = True
    lab, n = ndi.label((d > 10) & win)
    best = None
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        if ys.min() > 670 and 2000 < m.sum() < 8000:     # the U: starts below the collar band
            best = m
    assert best is not None
    line = best | (ndi.binary_dilation(best, iterations=2) & (d > 3))
    body = ~ndi.binary_dilation(line, iterations=1) & (np.abs(O - TAN).max(2) < 8)
    ring = ndi.binary_dilation(line, iterations=8) & body
    O2 = norm_fill(O, ring, line)
    ys, xs = np.nonzero(line)
    info = {"line_core_px": int(best.sum()), "line_px_painted": int(line.sum()),
            "line_bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "fill_colour_range": [[int(v) for v in O2[line].min(0)], [int(v) for v in O2[line].max(0)]]}
    return O2, info, line


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    O = np.asarray(Image.open(SRC).convert("RGB")).astype(float)
    E, einfo, em = eye_fix(O)
    B, linfo, line = remove_line(E)
    print(einfo); print(linfo)
    Image.fromarray(np.round(B).astype(np.uint8)).save(WORK / "base_eye_noline.png")
    Image.fromarray((line * 255).astype(np.uint8)).save(WORK / "line_mask.png")


if __name__ == "__main__":
    main()
