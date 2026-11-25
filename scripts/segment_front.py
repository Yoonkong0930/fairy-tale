#!/usr/bin/env python3
"""
정면 원본(초록 배경, 얼굴 지운 그림)을 외곽선 기준 영역으로 나눠서 목록 출력
사용: python3 scripts/segment_front.py <이미지> [미리보기.png]
- 외곽선(진갈색)으로 둘러싸인 영역마다 번호·위치·크기·평균색을 출력
- 미리보기 이미지에 번호를 표시 → 어떤 번호가 머리/귀/몸통인지 고름
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

src = sys.argv[1]
im = np.array(Image.open(src).convert("RGB")).astype(int)
r, g, b = im[..., 0], im[..., 1], im[..., 2]
fg = (g - np.maximum(r, b)) < 60
line = fg & (im.max(axis=2) < 115) & (r > b)
# 외곽선 틈 메우기: 선을 살짝 두껍게 해서 영역이 새지 않게
thick = ndimage.binary_dilation(line, iterations=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
inner = fg & ~thick
lab, n = ndimage.label(inner)
rows = []
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    m = lab[sl] == i
    a = int(m.sum())
    if a < 1500:
        continue
    col = im[sl][m].mean(axis=0).astype(int)
    rows.append((i, sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, a, tuple(col)))
for row in sorted(rows, key=lambda t: -t[5]):
    print(row)
if len(sys.argv) > 2:
    rng = np.random.default_rng(1)
    vis = np.zeros_like(im)
    for row in rows:
        vis[lab == row[0]] = rng.integers(60, 255, 3)
    vis[line] = 0
    out = Image.fromarray(vis.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for row in rows:
        cx, cy = (row[1] + row[3]) // 2, (row[2] + row[4]) // 2
        d.text((cx, cy), str(row[0]), fill="white")
    out.resize((out.width // 2, out.height // 2)).save(sys.argv[2])
np.save(src.replace(".png", "-labels.npy"), lab)
