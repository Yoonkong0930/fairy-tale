"""Near-arm rendering for the side view (used by step1_fix.py).

Shape source: art/v2/tori-side/step1-fix/work/ai_arm_green_e.png (Gemini, arm alone on #00FF00,
front arm as style reference).  The arm is keyed (green-ness G-max(R,B) < 120, holes filled) and each
arm pixel is labelled by its nearest AI colour (tan / shade band / top cap); the cap (hidden under the
collar) counts as tan.  The arm is then re-rendered on the 1024 canvas with a similarity transform
(scale, rotation, anchor) and 4x4 supersampling: coverage = fraction of sub-samples inside the keyed
silhouette, colour = mean of the flat palette colours of the covered sub-samples, where the palette is
the BODY's own tan / shade (identical to the front arm's colours within 3 levels).  Sub-samples above
the collar's bottom edge are dropped (the shoulder top is tucked under the collar) and sub-samples in
the body's collar-shadow band take the band colour (the collar shadow falls across the arm top, as on
the front view's arms).
"""
import pathlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
AI_ARM = ROOT / "art/v2/tori-side/step1-fix/work/ai_arm_green_e.png"
AI_PAL = np.array([[235, 188, 125], [203, 143, 93], [240, 205, 185], [212, 160, 120]], float)
SS = 4


def load_ai_arm(offset=None):
    e = np.asarray(Image.open(AI_ARM).convert("RGB")).astype(float)
    g = e[..., 1] - np.maximum(e[..., 0], e[..., 2])
    m = g < 120
    lab, n = ndi.label(m)
    m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    m = ndi.binary_fill_holes(m)
    cl = ((e[..., None, :] - AI_PAL) ** 2).sum(-1).argmin(-1)
    shade = (cl == 1) & m
    # clean the shade class: 1-px AA speckles between tan and green are not band
    shade = ndi.binary_opening(shade, iterations=1)
    if offset is not None:
        # wider crescent band: arm pixels not covered by the arm silhouette shifted by `offset`
        # (dx, dy in AI px, toward the lit front/top) = flat-vector offset-shadow construction;
        # union with the AI's own band so the AI band shape is kept where it is wider.
        dx, dy = offset
        sh2 = np.zeros_like(m)
        H_, W_ = m.shape
        ys, xs = np.nonzero(m)
        ys2, xs2 = ys + dy, xs + dx
        ok = (ys2 >= 0) & (ys2 < H_) & (xs2 >= 0) & (xs2 < W_)
        moved = np.zeros_like(m); moved[ys2[ok], xs2[ok]] = True
        shade = (shade | (m & ~moved)) & m
    return m, shade


def render_arm(params, collar_bottom, band_bottom, tan, shade_col, band_col, box=(380, 600, 680, 900)):
    """params: s (scale), rot (deg, + = clockwise on screen), src (E anchor x,y), dst (canvas x,y).
    collar_bottom(X), band_bottom(X): vectorised curves (canvas y) of the collar's bottom edge and of
    the collar-shadow band's bottom edge.  Returns full-canvas alpha (float 0..1) and rgb (float),
    and the label coverage maps (for diagnostics)."""
    m, sh = load_ai_arm(params.get("offset"))
    s, rot = params["s"], np.deg2rad(params["rot"])
    sx, sy = params["src"]; dx, dy = params["dst"]
    x0, y0, x1, y1 = box
    o = (np.arange(SS) + 0.5) / SS - 0.5
    X = (np.arange(x0, x1)[:, None] + o[None, :]).ravel()
    Y = (np.arange(y0, y1)[:, None] + o[None, :]).ravel()
    X, Y = np.meshgrid(X, Y)
    # inverse similarity: E = src + R(-rot) (P - dst) / s
    c, si = np.cos(rot), np.sin(rot)
    u, v = (X - dx) / s, (Y - dy) / s
    ex = sx + c * u + si * v
    ey = sy - si * u + c * v
    xi = np.clip(np.round(ex).astype(int), 0, m.shape[1] - 1)
    yi = np.clip(np.round(ey).astype(int), 0, m.shape[0] - 1)
    inb = (ex >= 0) & (ex < m.shape[1]) & (ey >= 0) & (ey < m.shape[0])
    inside = m[yi, xi] & inb & (Y >= collar_bottom(X))
    is_sh = sh[yi, xi]
    in_band = Y < band_bottom(X)
    col = np.where(in_band[..., None], band_col, np.where(is_sh[..., None], shade_col, tan))
    h, w = y1 - y0, x1 - x0
    pool = lambda a: a.reshape(h, SS, w, SS, *a.shape[2:]).mean((1, 3))
    cov = pool(inside.astype(float))
    rgb = pool(col * inside[..., None]) / np.maximum(cov, 1e-9)[..., None]
    A = np.zeros((1024, 1024)); C = np.zeros((1024, 1024, 3))
    A[y0:y1, x0:x1] = cov; C[y0:y1, x0:x1] = rgb
    shade_cov = np.zeros((1024, 1024)); shade_cov[y0:y1, x0:x1] = pool((inside & is_sh & ~in_band).astype(float))
    return A, C, shade_cov
