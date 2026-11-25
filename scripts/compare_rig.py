#!/usr/bin/env python3
"""원본 설정화 정면 vs 리그 비교 이미지 (out/compare-<who>.png)
먼저: npx remotion still RigCompare out/cmp-<who>.png --frame=0 --props={"who":"<who>"}
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def load_crop(path, box=None):
    im = Image.open(path).convert("RGB")
    if box: im = im.crop(box)
    a = np.array(im).astype(int)
    fg = (a.min(axis=2) < 235)
    ys, xs = np.where(fg)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

def measure(im):
    a = np.array(im).astype(int); H, W, _ = a.shape
    fg = a.min(axis=2) < 235
    # 눈동자: 아주 어두운 덩어리 중 위쪽 65%에 있는 큰 것 2개
    dark = (a.max(axis=2) < 110) & (a[..., 0] - a[..., 2] < 70)
    lab, n = ndimage.label(dark)
    blobs = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i; area = m.sum()
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        cy = (sl[0].start + sl[0].stop) / 2
        if area > (H * 0.02) ** 2 and cy < H * 0.65 and 0.6 < h / max(w, 1) < 1.6:
            blobs.append((area, (sl[1].start + sl[1].stop) / 2, cy, w))
    blobs = sorted(blobs, reverse=True)[:2]
    widths = fg.sum(axis=1)
    return dict(H=H, W=W, eyes=sorted([(round(b[1]), round(b[2]), round(b[3])) for b in blobs]), widths=widths)

def report(name, m):
    H = m["H"]; e = m["eyes"]
    out = {"키": H}
    if len(e) == 2:
        out["눈 높이(위에서, 키 대비)"] = round(np.mean([e[0][1], e[1][1]]) / H, 3)
        out["눈 사이 거리(키 대비)"] = round((e[1][0] - e[0][0]) / H, 3)
        out["눈 지름(키 대비)"] = round(np.mean([e[0][2], e[1][2]]) / H, 3)
    w = m["widths"]
    for f in (0.3, 0.45, 0.55, 0.7, 0.85):
        out[f"폭@{int(f*100)}%"] = round(w[int(H * f)] / H, 3)
    print(name, out)

def side_by_side(orig, rig, out_path, target_h=700):
    o = orig.resize((round(orig.width * target_h / orig.height), target_h))
    r = rig.resize((round(rig.width * target_h / rig.height), target_h))
    pad = 40
    g = Image.new("RGB", (o.width + r.width + o.width + pad * 4, target_h + 60), "white")
    g.paste(o, (pad, 30)); g.paste(r, (o.width + pad * 2, 30))
    # 겹쳐보기: 원본(빨강) vs 리그(파랑) 외곽
    ov = Image.new("RGB", (max(o.width, r.width), target_h), "white")
    oa = np.array(o.convert("L")) < 200; ra = np.array(r.convert("L")) < 200
    canvas = np.full((target_h, ov.width, 3), 255, np.uint8)
    ox = (ov.width - o.width) // 2; rx = (ov.width - r.width) // 2
    canvas[:, ox:ox + o.width][oa] = [230, 60, 60]
    sub = canvas[:, rx:rx + r.width]; sub[ra & (sub[..., 1] == 255)] = [60, 90, 230]; sub[ra & (sub[..., 0] == 230)] = [120, 40, 160]
    g.paste(Image.fromarray(canvas), (o.width + r.width + pad * 3, 30))
    d = ImageDraw.Draw(g)
    for f in (0.25, 0.5, 0.75):
        y = 30 + int(target_h * f); d.line((0, y, g.width, y), fill=(0, 170, 0), width=1)
    d.text((pad, 5), "ORIGINAL", fill="black"); d.text((o.width + pad * 2, 5), "RIG", fill="black"); d.text((o.width + r.width + pad * 3, 5), "OVERLAY red=orig blue=rig", fill="black")
    g.save(out_path)

for who, sheet in (("tori", "art/final/tori-model-sheet.png"), ("momo", "art/final/momo-model-sheet.png")):
    S = Image.open(sheet); w3 = S.width // 3
    orig = load_crop(sheet, (0, 0, w3, S.height))
    rig = load_crop(f"out/cmp-{who}.png")
    report(who + " 원본", measure(orig)); report(who + " 리그", measure(rig))
    side_by_side(orig, rig, f"out/compare-{who}.png")
