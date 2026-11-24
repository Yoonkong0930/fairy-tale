#!/usr/bin/env python3
"""
초록 배경 파츠 시트 → 파츠별 투명 PNG

사용: python3 scripts/slice_parts.py <시트.png> <출력폴더> <열> <행> 이름1 이름2 ...
- 초록(#00FF00) 배경을 투명하게 만들고, 가장자리 초록 번짐을 제거
- 거의 검정인 선은 도토 외곽선 색(진갈색)으로 맞춤
- 칸마다 여백을 잘라 저장
"""
import sys, pathlib
import numpy as np
from PIL import Image

OUTLINE = np.array([0x4E, 0x23, 0x13])  # 표정 시트에서 샘플링한 외곽선 갈색


def key_green(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(float) for i in range(3)]
    greenness = g - np.maximum(r, b)          # 초록이 다른 채널보다 얼마나 센지
    alpha = np.clip(1 - (greenness - 40) / 120, 0, 1)
    return alpha


def main():
    src, outdir, cols, rows, *names = sys.argv[1:]
    cols, rows = int(cols), int(rows)
    im = np.array(Image.open(src).convert("RGB")).astype(float)
    H, W, _ = im.shape
    alpha = key_green(im)

    # 초록 번짐 제거: 반투명 가장자리의 초록 채널을 낮춤
    g_limit = np.maximum(im[..., 0], im[..., 2])
    im[..., 1] = np.where(alpha < 1, np.minimum(im[..., 1], g_limit), im[..., 1])

    # 검정 선 → 외곽선 갈색
    dark = im.max(axis=2) < 60
    im[dark] = OUTLINE

    rgba = np.dstack([im, alpha * 255]).astype(np.uint8)
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    ch, cw = H // rows, W // cols
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        cell = rgba[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
        ys, xs = np.where(cell[..., 3] > 10)
        if len(xs) == 0:
            print("empty:", name); continue
        pad = 6
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, ch)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, cw)
        Image.fromarray(cell[y0:y1, x0:x1]).save(out / f"{name}.png")
        print(f"{name}.png {x1-x0}x{y1-y0}")


if __name__ == "__main__":
    main()
