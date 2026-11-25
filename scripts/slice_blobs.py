#!/usr/bin/env python3
"""
초록 배경 파츠 시트 → 덩어리(연결 요소)별 투명 PNG
칸 크기가 제각각이거나 AI가 글자를 넣어도 동작하도록:
- 초록 배경 제거 후, 붙어 있는 덩어리 단위로 파츠를 찾음
- 검정 글자(라벨)는 제거
- 위→아래, 왼쪽→오른쪽 순서로 이름을 붙임

사용: python3 scripts/slice_blobs.py <시트.png> <출력폴더> <행별 개수, 예: 4,4,4> 이름1 이름2 ...
필요: pip install scipy
"""
import sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

OUTLINE = np.array([0x4E, 0x23, 0x13])


def main():
    src, outdir, per_row, *names = sys.argv[1:]
    per_row = [int(x) for x in per_row.split(",")]
    im = np.array(Image.open(src).convert("RGB")).astype(float)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    greenness = g - np.maximum(r, b)
    alpha = np.clip(1 - (greenness - 40) / 120, 0, 1)
    im[..., 1] = np.where(alpha < 1, np.minimum(g, np.maximum(r, b)), g)

    # 라벨 글자 = 순수 검정에 가까운 픽셀 (도토 외곽선은 갈색이라 R이 더 높음)
    text = (im.max(axis=2) < 45) & (np.abs(r - b) < 20)
    solid = alpha > 0.05
    lab, n = ndimage.label(solid)
    blobs = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        size = m.sum()
        if size < 400:
            continue
        if text[sl][m].mean() > 0.5:   # 대부분 검정 → 글자
            continue
        cy = (sl[0].start + sl[0].stop) / 2
        cx = (sl[1].start + sl[1].stop) / 2
        blobs.append((cy, cx, i, sl))

    # 행 묶기: y 중심으로 정렬 후 행별 개수만큼 자름
    blobs.sort()
    rows, k = [], 0
    for cnt in per_row:
        rows.append(sorted(blobs[k:k + cnt], key=lambda t: t[1])); k += cnt
    ordered = [b for row in rows for b in row]
    if len(ordered) != len(names) or k != len(blobs):
        print(f"warning: found {len(blobs)} blobs, expected {len(names)}")

    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    for (cy, cx, i, sl), name in zip(ordered, names):
        pad = 6
        y0, y1 = max(sl[0].start - pad, 0), sl[0].stop + pad
        x0, x1 = max(sl[1].start - pad, 0), sl[1].stop + pad
        rgb = im[y0:y1, x0:x1].copy()
        a = alpha[y0:y1, x0:x1] * (lab[y0:y1, x0:x1] == i)
        dark = rgb.max(axis=2) < 60
        rgb[dark] = OUTLINE
        Image.fromarray(np.dstack([rgb, a * 255]).clip(0, 255).astype(np.uint8)).save(out / f"{name}.png")
        print(f"{name}.png {x1-x0}x{y1-y0} at ({int(cx)},{int(cy)})")


if __name__ == "__main__":
    main()
