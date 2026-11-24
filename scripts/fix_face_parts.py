#!/usr/bin/env python3
"""
얼굴 파츠 후처리
1) eye-open: 안쪽 그라데이션 제거 → 단색 눈동자 + 흰 반짝이만 남김
2) 입 파츠에서 코를 떼어내고, 공용 nose.png 하나로 통일
   - 각 입의 '코가 있던 자리'(기준점)와 코 크기를 parts.json에 기록
   - Remotion에서 코를 고정 위치에 두고 입을 기준점에 맞춰 붙임 (코 크기 기준으로 배율 보정)
"""
import json, pathlib
import numpy as np
from PIL import Image

D = pathlib.Path("public/assets/characters/doto/face")
PUPIL = np.array([78, 35, 19])


def load(n):
    return np.array(Image.open(D / f"{n}.png").convert("RGBA")).astype(int)


def save(n, a):
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(D / f"{n}.png")


# 1) 눈 단색화
eye = load("eye-open")
rgb = eye[..., :3]
not_white = rgb.min(axis=2) < 200
eye[not_white & (eye[..., 3] > 0), :3] = PUPIL
save("eye-open", eye)


def nose_mask(a):
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    return (al > 0) & (r > 100) & (r < 180) & (g > 62) & (b < 120)


meta = {}
from PIL import ImageFilter


def components(mask):
    """4방향 연결 요소 라벨링 (작은 잔여 점 제거용)"""
    H, W = mask.shape
    lab = np.zeros((H, W), int); n = 0; sizes = {}
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not lab[y, x]:
                n += 1; stack = [(y, x)]; lab[y, x] = n; c = 0
                while stack:
                    cy, cx = stack.pop(); c += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = n; stack.append((ny, nx))
                sizes[n] = c
    return lab, sizes


ref_w = None
for n in ["mouth-neutral", "mouth-happy", "mouth-talk-s", "mouth-talk-m",
          "mouth-talk-l", "mouth-sad", "mouth-angry", "mouth-o"]:
    a = load(n)
    fill = nose_mask(a)
    fill[int(a.shape[0] * 0.45):, :] = False
    # 코 색 영역 중 가장 큰 덩어리 = 코
    lab, sizes = components(fill)
    big = max(sizes, key=sizes.get)
    fill = lab == big
    ys, xs = np.where(fill)
    # 코 외곽선까지: 코 영역을 7px 넓힌 범위 안의 불투명 픽셀
    ring = np.array(Image.fromarray((fill * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(15))) > 0
    nose_full = ring & (a[..., 3] > 0)
    # 줄기(코 아래 중앙)는 지우지 않음: 코 색 영역 아래쪽보다 낮은 픽셀은 보존
    nose_full[ys.max() + 1:, :] &= False
    nys, nxs = np.where(nose_full)
    x0, x1, y0, y1 = nxs.min(), nxs.max() + 1, nys.min(), nys.max() + 1
    if n == "mouth-neutral":
        nose = a.copy(); nose[~nose_full, 3] = 0
        Image.fromarray(nose[y0:y1, x0:x1].astype(np.uint8)).save(D / "nose.png")
        ref_w = x1 - x0
    nose_w = x1 - x0
    a[nose_full, 3] = 0
    # 작은 잔여 점 제거
    lab, sizes = components(a[..., 3] > 10)
    for k, c in sizes.items():
        if c < 80:
            a[lab == k, 3] = 0
    ys2, xs2 = np.where(a[..., 3] > 10)
    cx0, cy0 = xs2.min(), ys2.min()
    a = a[cy0:ys2.max() + 1, cx0:xs2.max() + 1]
    save(n, a)
    meta[n] = {
        "w": int(a.shape[1]), "h": int(a.shape[0]),
        "anchorX": float((x0 + x1) / 2 - cx0),
        "anchorY": float(ys.max() - cy0),   # 코 색 영역 아래 끝 = 줄기 시작
        "scale": round(float(ref_w / nose_w), 4),
    }
    print(n, meta[n])

nose = Image.open(D / "nose.png")
meta["nose"] = {"w": nose.width, "h": nose.height}
(D / "parts.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
