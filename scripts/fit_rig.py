#!/usr/bin/env python3
"""
원본 정면 설정화에 맞춘 리그 파츠 만들기

입력 (art/fit/, AI 편집으로 만든 '같은 위치' 그림들. 모두 초록 배경, 2K)
  <who>-noface.png     얼굴만 지운 원본
  <who>-headonly.png   + 귀·머리장식 지움 → 머리 모양
  <who>-torsoonly.png  + 팔·다리·꼬리 지움 → 몸통 모양
  <who>-only-arms.png  팔만 남김
  <who>-only-legs.png  다리(+꼬리)만 남김
  <who>-only-top.png   귀(+애교머리/꽃핀)만 남김
  <who>-front-pad.png  얼굴 있는 원본 (눈·코·입 위치 측정용, 흰 배경)

출력
  public/assets/characters/<who>/fit/*.png   파츠 이미지
  src/characters/<who>-fit.json              위치표 (리그 좌표: 발바닥 가운데 = 0,0)
"""
import json, sys, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage

who = sys.argv[1]
F = pathlib.Path("art/fit")
OUT = pathlib.Path(f"public/assets/characters/{who}/fit"); OUT.mkdir(parents=True, exist_ok=True)
OUTLINE = np.array([0x4E, 0x23, 0x13])


def load(name):
    a = np.array(Image.open(F / f"{who}-{name}.png").convert("RGB")).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    greenness = g - np.maximum(r, b)
    alpha = np.clip(1 - (greenness - 40) / 120, 0, 1)
    a[..., 1] = np.where(alpha < 1, np.minimum(g, np.maximum(r, b)), g)  # 초록 번짐 제거
    return a, alpha


def line_mask(a, alpha):
    return (alpha > 0.5) & (a.max(axis=2) < 115) & (a[..., 0] > a[..., 2])


def region_at(a, alpha, pt, thick=6):
    """외곽선으로 둘러싸인 영역 중 pt가 속한 영역 + 그 외곽선"""
    line = ndimage.binary_dilation(line_mask(a, alpha), iterations=thick)
    inner = (alpha > 0.5) & ~line
    lab, _ = ndimage.label(inner)
    k = lab[pt[1], pt[0]]
    assert k, f"no region at {pt}"
    m = ndimage.binary_fill_holes(lab == k)
    m = ndimage.binary_dilation(m, iterations=thick + 12) & (alpha > 0.05)
    return m


def save_part(name, a, alpha, mask, origin):
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgb = a[y0:y1, x0:x1].copy()
    al = (alpha * mask)[y0:y1, x0:x1]
    rgb[rgb.max(axis=2) < 60] = OUTLINE
    Image.fromarray(np.dstack([rgb, al * 255]).clip(0, 255).astype(np.uint8)).save(OUT / f"{name}.png")
    ox, oy = origin
    return {"file": f"fit/{name}", "x": int(x0 - ox), "y": int(y0 - oy), "w": int(x1 - x0), "h": int(y1 - y0)}


def blobs(alpha, min_area=3000):
    lab, n = ndimage.label(alpha > 0.5)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab == i
        if m.sum() >= min_area:
            out.append(m)
    return sorted(out, key=lambda m: np.where(m)[1].mean())  # 왼쪽 → 오른쪽


def bbox(m):
    ys, xs = np.where(m); return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def narrow_neck(mask, neck_y, extra=10):
    """머리 뒤에 숨는 몸통 윗부분을 목 폭으로 좁힘 → 머리가 기울어도 몸통 윗선이 삐져나오지 않음"""
    ref = int(neck_y + 45)
    xs = np.where(mask[ref])[0]
    if len(xs) == 0:
        return mask
    l, r = xs.min() + 25, xs.max() - 25
    band = np.zeros_like(mask); band[: int(neck_y + 15)] = True
    keep = np.zeros_like(mask); keep[:, max(l - extra, 0): r + extra] = True
    return mask & ~(band & ~keep)


# ---------- 기준 좌표 ----------
full, full_al = load("noface")
fb = bbox(full_al > 0.5)
origin = ((fb[0] + fb[2]) / 2, fb[3])  # 발바닥 가운데
layout = {"origin_px": [float(origin[0]), float(origin[1])], "parts": {}}
P = layout["parts"]

head_a, head_al = load("headonly")
torso_a, torso_al = load("torsoonly")
hb = bbox(head_al > 0.5)

# ---------- 머리 ----------
if who == "tori":
    head_pt = (int((hb[0] + hb[2]) / 2), int(hb[1] + (hb[3] - hb[1]) * 0.25))
    region = region_at(head_a, head_al, head_pt)
    # 줄무늬가 외곽선에 닿아 구멍으로 빠지지 않게: 열마다 '머리 영역의 맨 아래'까지 전부 머리로
    bottom = np.full(region.shape[1], -1)
    cols = np.where(region.any(axis=0))[0]
    for x in cols:
        bottom[x] = np.where(region[:, x])[0].max()
    yy = np.arange(region.shape[0])[:, None]
    head_mask = (head_al > 0.05) & (yy <= bottom[None, :]) & (bottom[None, :] >= 0)
    neck_y = bbox(region)[3]
else:
    # 모모: 머리와 몸이 이어져 있음 → 목의 잘록한 곳을 찾아 둥근 선으로 자름
    m = head_al > 0.5
    rows = range(int(hb[1] + (hb[3] - hb[1]) * 0.35), int(hb[1] + (hb[3] - hb[1]) * 0.75))
    widths = [(m[y].sum(), y) for y in rows]
    # 머리에서 가장 넓은 줄 아래쪽에서 폭이 가장 좁은 곳
    widest = max(widths)[1]
    neck_y = min((w, y) for w, y in widths if y > widest)[1]
    xs = np.arange(m.shape[1]); cx = (hb[0] + hb[2]) / 2; hw = (hb[2] - hb[0]) / 2
    cut = neck_y + 0.06 * (hb[3] - hb[1]) * np.clip(1 - ((xs - cx) / (hw * 0.55)) ** 2, 0, 1)
    yy = np.arange(m.shape[0])[:, None]
    head_mask = (head_al > 0.05) & (yy < cut[None, :])
P["head"] = save_part("head", head_a, head_al, head_mask, origin)
P["head"]["pivot"] = [float((hb[0] + hb[2]) / 2 - origin[0]), float(neck_y - origin[1] - 10)]

# ---------- 몸통 (+ 초커) ----------
tb = bbox(torso_al > 0.5)
if who == "tori":
    # 초커·펜던트까지 지운 몸통 (펜던트는 코드로 그림)
    t2_a, t2_al = load("torsoonly2")
    t2b = bbox(t2_al > 0.5)
    belly_pt = (int((t2b[0] + t2b[2]) / 2), int(t2b[3] - (t2b[3] - neck_y) * 0.3))
    torso_mask = region_at(t2_a, t2_al, belly_pt)
    # 목에 선이 없어 머리까지 이어짐 → 머리 아래 끝보다 조금 위에서 자름 (겹치는 부분은 머리 뒤에 숨음)
    yy = np.arange(torso_mask.shape[0])[:, None]
    torso_mask &= yy >= neck_y - 90
    # 잘린 띠 안의 머리 외곽선·볼 색은 몸통 털색으로 덮어 머리가 기울어도 안 보이게
    band = torso_mask & (yy < neck_y + 10)
    fur = np.median(t2_a[int(neck_y) + 60:int(neck_y) + 100, belly_pt[0] - 250:belly_pt[0] - 150].reshape(-1, 3), axis=0)
    cream = np.median(t2_a[int(neck_y) + 60:int(neck_y) + 100, belly_pt[0] - 20:belly_pt[0] + 20].reshape(-1, 3), axis=0)
    edge = ndimage.binary_erosion(t2_al > 0.5, iterations=16)
    xx = np.arange(torso_mask.shape[1])[None, :]
    bw = 140  # 배 무늬 반폭
    t2_a[band & edge & (np.abs(xx - belly_pt[0]) > bw)] = fur
    t2_a[band & edge & (np.abs(xx - belly_pt[0]) <= bw)] = cream
    # 몸통 아래쪽을 원본 실루엣에 맞춰 자르고, 잘린 가장자리에 외곽선을 그림
    lower = yy > (neck_y + (origin[1] - neck_y) * 0.55)
    removed = torso_mask & lower & ~(full_al > 0.5)
    torso_mask &= ~removed
    near = ndimage.binary_dilation(removed, iterations=12) & torso_mask
    t2_a[near] = OUTLINE
    # 원본 펜던트 위치·크기 (민트색 덩어리)
    mint = (full_al > 0.5) & (full[..., 1] > full[..., 0] + 15) & (full[..., 1] > 150)
    lab_m, _ = ndimage.label(mint)
    if lab_m.max():
        big = max(range(1, lab_m.max() + 1), key=lambda i: (lab_m == i).sum())
        pb_ = bbox(lab_m == big)
        P["pendant_box"] = [float(pb_[0] - origin[0]), float(pb_[1] - origin[1]), float(pb_[2] - pb_[0]), float(pb_[3] - pb_[1])]
    torso_mask = narrow_neck(torso_mask, neck_y)
    P["torso"] = save_part("torso", t2_a, t2_al, torso_mask, origin)
    c_a, c_al = load("only-collar")
    cm = blobs(c_al, 1500)
    cm = max(cm, key=lambda m: m.sum())
    P["collar"] = save_part("collar", c_a, c_al, cm, origin)
    cb = bbox(cm)
    P["pendant_at"] = [float((cb[0] + cb[2]) / 2 - origin[0]), float(cb[3] - origin[1] - 8)]
else:
    xs = np.arange(torso_al.shape[1]); cx = (tb[0] + tb[2]) / 2; hw = (hb[2] - hb[0]) / 2
    cut = (neck_y - 60) + 0.06 * (hb[3] - hb[1]) * np.clip(1 - ((xs - cx) / (hw * 0.55)) ** 2, 0, 1)
    yy = np.arange(torso_al.shape[0])[:, None]
    torso_mask = (torso_al > 0.05) & (yy >= cut[None, :])
    # 목 위쪽에 남은 볼 아래 외곽선 조각은 털색으로 덮음 (머리 뒤에 숨는 부분)
    fur = np.median(torso_a[int(neck_y) + 40:int(neck_y) + 80, int(cx) - 30:int(cx) + 30].reshape(-1, 3), axis=0)
    top_zone = torso_mask & (yy < neck_y + 25)
    inner_top = ndimage.binary_erosion(torso_al > 0.5, iterations=14) & top_zone
    torso_a[inner_top] = fur
    lower = yy > (neck_y + (origin[1] - neck_y) * 0.55)
    removed = torso_mask & lower & ~(full_al > 0.5)
    torso_mask &= ~removed
    near = ndimage.binary_dilation(removed, iterations=12) & torso_mask
    torso_a[near] = OUTLINE
    torso_mask = narrow_neck(torso_mask, neck_y)
    P["torso"] = save_part("torso", torso_a, torso_al, torso_mask, origin)

# ---------- 팔 ----------
arm_a, arm_al = load("only-arms")
arms = blobs(arm_al)[:2]
for side, m in zip(("armL", "armR"), arms):
    P[side] = save_part(side, arm_a, arm_al, m, origin)
    x0, y0, x1, y1 = bbox(m)
    # 어깨 = 위쪽, 몸 쪽 가장자리
    inner_x = x1 - (x1 - x0) * 0.35 if side == "armL" else x0 + (x1 - x0) * 0.35
    P[side]["pivot"] = [float(inner_x - origin[0]), float(y0 + (y1 - y0) * 0.1 - origin[1])]

# ---------- 다리·꼬리 ----------
leg_a, leg_al = load("only-legs")
lb = blobs(leg_al)
if who == "momo":
    legs = sorted(lb, key=lambda m: -m.sum())[:2]
    legs = sorted(legs, key=lambda m: np.where(m)[1].mean())
    for side, m in zip(("legL", "legR"), legs):
        P[side] = save_part(side, leg_a, leg_al, m, origin)
else:
    # 토리: 다리 편집이 실패 → 원본에서 몸통 밖으로 나온 다리 영역에 기존 다리 파츠를 맞춤
    diff = (full_al > 0.5) & ~(torso_al > 0.5)
    diff[: int(origin[1] - (origin[1] - neck_y) * 0.35)] = False
    dl = [m for m in blobs(diff.astype(float), 2000)]
    dl = sorted(sorted(dl, key=lambda m: -m.sum())[:2], key=lambda m: np.where(m)[1].mean())
    for side, m, src in zip(("legL", "legR"), dl, ("leg-l", "leg-r")):
        x0, y0, x1, y1 = bbox(m)
        part = Image.open(f"public/assets/characters/tori/body/{src}.png")
        w = int((x1 - x0) * 1.05); h = int(part.height * w / part.width)
        part.resize((w, h), Image.LANCZOS).save(OUT / f"{side}.png")
        P[side] = {"file": f"fit/{side}", "x": int((x0 + x1) / 2 - w / 2 - origin[0]), "y": int(y1 - h - origin[1]), "w": w, "h": h}
    # 꼬리: 다리 편집 그림에서 '원본 꼬리 자리'와 가장 많이 겹치는 덩어리
    tail_zone = (full_al > 0.5) & ~(torso_al > 0.5) & ~(head_al > 0.5)
    best = max(lb, key=lambda m: (m & tail_zone).sum())
    P["tail"] = save_part("tail", leg_a, leg_al, best, origin)
    # 편집 그림의 꼬리 위치가 원본과 어긋나 있음 → 원본에서 보이는 꼬리 영역에 맞춰 옮기고 크기 조정
    tz = (full_al > 0.5) & ~(torso_al > 0.5); tz[:, int(origin[0]):] = False  # 화면 왼쪽
    tz[int(origin[1] - (origin[1] - neck_y) * 0.35):] = False  # 다리 높이 제외
    tl = [m for m in blobs(tz.astype(float), 3000)]
    if tl:
        zone = max(tl, key=lambda m: m.sum()); zx0, zy0, zx1, zy1 = bbox(zone)
        t = P["tail"]; scale = (zy1 - zy0) * 1.08 / t["h"]
        img = Image.open(OUT / "tail.png"); w, h = int(t["w"] * scale), int(t["h"] * scale)
        img.resize((w, h), Image.LANCZOS).save(OUT / "tail.png")
        P["tail"] = {"file": "fit/tail", "x": int(zx0 - origin[0] - 6), "y": int(zy0 - origin[1] - 10), "w": w, "h": h}
for side in ("legL", "legR"):
    p = P[side]; P[side]["pivot"] = [p["x"] + p["w"] / 2, p["y"] + p["h"] * 0.15]

# ---------- 귀·머리장식 ----------
top_a, top_al = load("only-top")
tops = blobs(top_al, 1500)
names = ("earL", "tuft", "earR") if who == "tori" else ("earL", "earR")
tops = sorted(sorted(tops, key=lambda m: -m.sum())[: len(names)], key=lambda m: np.where(m)[1].mean())
for nm, m in zip(names, tops):
    if who == "momo" and nm == "earR":
        # 꽃핀은 머리 앞에 보여야 함 → 귀에서 분리 (분홍·노랑 꽃잎 + 그 외곽선)
        r_, g_, b_ = top_a[..., 0], top_a[..., 1], top_a[..., 2]
        petal = m & (((r_ > 200) & (g_ < 175) & (b_ > 130) & (r_ - g_ > 50)) | ((r_ > 200) & (g_ > 180) & (b_ < 140)))
        lab_f, _ = ndimage.label(ndimage.binary_closing(petal, iterations=4))
        if lab_f.max():
            big = max(range(1, lab_f.max() + 1), key=lambda i: (lab_f == i).sum())
            flower = ndimage.binary_fill_holes(lab_f == big)
            flower = ndimage.binary_dilation(flower, iterations=14) & m
            P["flower"] = save_part("flower", top_a, top_al, flower, origin)
    P[nm] = save_part(nm, top_a, top_al, m, origin)
    x0, y0, x1, y1 = bbox(m)
    P[nm]["pivot"] = [float((x0 + x1) / 2 - origin[0]), float(y1 - (y1 - y0) * 0.12 - origin[1])]

# ---------- 얼굴 위치 (얼굴 있는 원본에서 측정) ----------
fp = np.array(Image.open(F / f"{who}-front-pad.png").convert("RGB")).astype(int)
ffg = fp.min(axis=2) < 235
pb = bbox(ffg)
sx = (fb[2] - fb[0]) / (pb[2] - pb[0]); sy = (fb[3] - fb[1]) / (pb[3] - pb[1])
to_rig = lambda x, y: (float((x - pb[0]) * sx + fb[0] - origin[0]), float((y - pb[1]) * sy + fb[1] - origin[1]))
dark = (fp.max(axis=2) < 110)
core = ndimage.binary_erosion(dark, iterations=3)  # 얇은 선 제거 → 눈동자만
lab, _ = ndimage.label(core)
cand = []
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    m = lab[sl] == i; a = m.sum()
    h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
    if a > 60 and 0.6 < h / max(w, 1) < 1.7:
        cand.append((a, (sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2, w + 6, h + 6))
eyes = sorted(sorted(cand, reverse=True)[:2], key=lambda c: c[1])
face = {}
for nm, e in zip(("eyeL", "eyeR"), eyes):
    x, y = to_rig(e[1], e[2]); face[nm] = {"x": x, "y": y, "d": float(e[3] * sx), "h": float(e[4] * sy)}
# 코·입·눈썹: 확대 이미지에서 직접 잰 값 (front-pad 픽셀 기준)
#  - 자동 검출은 토리 이마 줄무늬·볼 색과 섞여서 부정확했음
MANUAL = {
    "tori": {"nose": (181, 210, 297, 311), "mouth": (163, 229, 314, 336), "brow": (112, 137, 228)},
    "momo": {"nose": (204, 231, 344, 360), "mouth": (191, 244, 366, 382), "brow": (133, 158, 273)},
}[who]
nx0, nx1, ny0, ny1 = MANUAL["nose"]
x, y = to_rig((nx0 + nx1) / 2, (ny0 + ny1) / 2)
face["nose"] = {"x": x, "y": y, "w": float((nx1 - nx0) * sx), "bottom": to_rig(0, ny1)[1]}
mx0, mx1, my0, my1 = MANUAL["mouth"]
face["mouth"] = {"w": float((mx1 - mx0) * sx), "top": to_rig(0, my0)[1], "bottom": to_rig(0, my1)[1]}
bx0, bx1, by = MANUAL["brow"]
face["brow"] = {"y": to_rig(0, by)[1], "w": float((bx1 - bx0) * sx)}
layout["face"] = face
layout["height"] = float(origin[1] - fb[1])
pathlib.Path(f"src/characters/{who}-fit.json").write_text(json.dumps(layout, indent=1, ensure_ascii=False))
print(json.dumps({k: (v if k != "parts" else {n: (p.get("w"), p.get("h")) if isinstance(p, dict) else p for n, p in v.items()}) for k, v in layout.items()}, ensure_ascii=False, indent=1)[:2500])
