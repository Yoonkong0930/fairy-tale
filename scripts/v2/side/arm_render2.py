"""Side view v2 arms: generic flat re-rendering of the keyed AI arm silhouette (ai_arm_green_e.png).

render_arm2(params, clip_top, band_bottom, cols, box) -> (alpha, rgb, shade_cov, band_cov) on the 1024 canvas.
  params: s (scale), rot (deg, + = clockwise on screen), src (AI anchor), dst (canvas anchor),
          shade_k (px): underside shade = the part of the silhouette whose point k px ABOVE is outside the
          silhouette (canvas-space offset: a flat lit-from-above underside area, widest under the paw and
          tapering to nothing along the steep upper arm - no stroke-like band along an edge),
          shade_ai ([dx, dy] AI px, optional): instead of shade_k, offset in the arm's own frame: (0, +k)
          shades the paw end and the lower curve only, tapering to nothing along the straight arm sides,
          shade_ymin (px, optional): no shade above this canvas row (keeps the steep upper arm edge clean),
          flip (bool): mirror the AI silhouette left-right before the transform.
  cols: {"fill", "shade", "band"}; band = colour inside the collar-shadow band (y < band_bottom(X)).
  4x4 supersampled coverage; colour = mean of the flat colours of the covered sub-samples.
"""
import numpy as np
from arm_render import load_ai_arm, SS


def render_arm2(params, clip_top, band_bottom, cols, box):
    m, _ = load_ai_arm()
    if params.get("flip"):
        m = m[:, ::-1]
    s, rot = params["s"], np.deg2rad(params["rot"])
    sx, sy = params["src"]; dx, dy = params["dst"]
    x0, y0, x1, y1 = box
    o = (np.arange(SS) + 0.5) / SS - 0.5
    X = (np.arange(x0, x1)[:, None] + o[None, :]).ravel()
    Y = (np.arange(y0, y1)[:, None] + o[None, :]).ravel()
    X, Y = np.meshgrid(X, Y)
    c, si = np.cos(rot), np.sin(rot)

    def inside(Xq, Yq, off=(0.0, 0.0)):
        u, v = (Xq - dx) / s, (Yq - dy) / s
        ex = sx + c * u + si * v + off[0]
        ey = sy - si * u + c * v + off[1]
        xi = np.clip(np.round(ex).astype(int), 0, m.shape[1] - 1)
        yi = np.clip(np.round(ey).astype(int), 0, m.shape[0] - 1)
        return m[yi, xi] & (ex >= 0) & (ex < m.shape[1]) & (ey >= 0) & (ey < m.shape[0])
    ins = inside(X, Y)
    k = params.get("shade_k", 0)
    sh = ins & ~inside(X, Y - k) if k > 0 else np.zeros_like(ins)
    if "shade_ai" in params:                      # offset in the AI arm's own frame (along its axis)
        sh = ins & ~inside(X, Y, tuple(params["shade_ai"]))
    if "shade_ymin" in params:                   # shade only on the forearm / paw underside
        sh &= Y >= params["shade_ymin"]
    ins &= Y >= clip_top(X)
    band = Y < band_bottom(X)
    col = np.where(band[..., None], np.array(cols["band"], float),
                   np.where(sh[..., None], np.array(cols["shade"], float), np.array(cols["fill"], float)))
    h, w = y1 - y0, x1 - x0
    pool = lambda a: a.reshape(h, SS, w, SS, *a.shape[2:]).mean((1, 3))
    cov = pool(ins.astype(float))
    rgb = pool(col * ins[..., None]) / np.maximum(cov, 1e-9)[..., None]
    A = np.zeros((1024, 1024)); C = np.zeros((1024, 1024, 3))
    A[y0:y1, x0:x1] = cov; C[y0:y1, x0:x1] = rgb
    S_ = np.zeros((1024, 1024)); S_[y0:y1, x0:x1] = pool((ins & sh & ~band).astype(float))
    return A, C, S_
