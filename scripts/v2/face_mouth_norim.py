#!/usr/bin/env python3
"""벌린 입의 갈색 테두리 제거 (face_parts.py 다음에 실행)

테두리를 투명하게 지우면 입이 작아지므로, 테두리 픽셀을 입 안쪽 색(#512313)으로 바꿔
같은 크기의 '테두리 없는' 입 모양으로 만든다. 코-입을 잇는 세로 선(입 위쪽)은 남긴다.
원본(테두리 있는) 이미지는 art/v2/tori/face/rim-backup/ 에 보관.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

D = "public/assets/characters/tori-v2/face/"
BK = "art/v2/tori/face/rim-backup/"
INSIDE = np.array([0x51, 0x23, 0x13], float)
LINE = np.array([0xA1, 0x7F, 0x62], float)
NAMES = ["mouth-smile-open", "mouth-talk-s", "mouth-talk-m", "mouth-talk-l", "mouth-o", "mouth-ee"]

for n in NAMES:
    a = np.array(Image.open(BK + n + ".png").convert("RGBA")).astype(float)
    rgb, al = a[..., :3], a[..., 3]
    solid = al > 10
    H, W = solid.shape
    yy, xx = np.mgrid[:H, :W]
    # 줄기: 이미지 맨 위에서 내려오는 가는 세로 선. 폭이 좁은 줄(행)이 끝나는 곳 = 입 윗선
    widths = solid.sum(1)
    stem_w = np.median(widths[6:16])
    mouth_top = int(np.argmax(widths > stem_w * 2.5))
    xs_top = np.where(solid[2])[0]
    cx = (xs_top.min() + xs_top.max()) / 2
    # 가운데(줄기 바로 옆)에서 입이 시작하는 높이: 웃는 입처럼 입꼬리가 올라간 경우 대비
    side = int(round(cx + stem_w * 1.5))
    center_top = int(np.argmax(solid[:, side])) if solid[:, side].any() else mouth_top
    center_top = max(center_top, mouth_top)
    stem_col = np.abs(xx - cx) <= stem_w / 2 + 2
    stem = solid & stem_col & (yy < center_top)
    # 입 모양 = 줄기를 뺀 불투명 영역 (입 윗선 아래, 구멍 채움)
    shape = ndimage.binary_fill_holes(solid & ~stem & (yy >= mouth_top))
    # 혀 + 혀 가장자리 1px 은 원래 색 유지, 나머지는 전부 입 안 색으로
    tongue = shape & (rgb[..., 0] > 200) & (rgb[..., 1] < 215) & (rgb[..., 2] < 215)
    lab, k = ndimage.label(tongue)
    if k:
        big = max(range(1, k + 1), key=lambda i: (lab == i).sum())
        tongue = ndimage.binary_dilation(lab == big, iterations=1) & shape
    out = a.copy()
    fill = shape & ~tongue
    out[fill, :3] = INSIDE
    # 줄기가 입 윗선과 만나는 1~2줄은 입 쪽 알파를 따라 부드럽게
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(D + n + ".png")
    print(n, "mouth_top", mouth_top, "recolored px", int(fill.sum()))
