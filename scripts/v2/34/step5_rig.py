#!/usr/bin/env python3
"""
STEP 5 (3/4 view) - Tori v2: final layers, pivots, hierarchy, limits (rig.json, SAME schema as the front
art/v2/tori/step5-rig/rig.json so one animation engine drives both views).  The front's rig math,
compositor, gap metrics and sheets are imported from scripts/v2/step5_rig.py and reused unchanged.

Input : art/v2/tori-34/step4-filled/<part>.png (incl. collar_front.png)
Output: art/v2/tori-34/step5-rig/ layers/<part>.png, rig.json, pivots.png, range-test.png, poses.png,
        tuft-check.png, rest-check.json, rest-diff-tuft.png
        public/assets/characters/tori-v2-34/<part>.png (cropped) + rig.json

Usage
  python3 scripts/v2/34/step5_rig.py                       build everything
  python3 scripts/v2/34/step5_rig.py render pose.json out.png [--bg 6BBF59] [--scale 1]

Changes vs step 4 (same user decisions as the front)
  * tuft: one flat fur colour (head's flat tan), drawn BEHIND the head, base extended (rounded, hidden
    under the opaque head); the strip of head top that was under the tuft is re-coloured to continue
    the head pixels below it (front make_tuft_and_head, reused).
  * armNear: shoulder AA un-mixed into collar_front (front unmix_arm_collar, here for the near arm only;
    the far arm is behind the torso so its shoulder needs no collar_front).
  * torso/head: head's bottom-edge AA un-mixed from the torso into the head (front unmix_torso_head).
Names: Near = closer to the viewer (viewer-left), Far = turned away (viewer-right).  Angles + = clockwise.
"""
import importlib.util, json, pathlib, shutil, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("front5", ROOT / "scripts/v2/step5_rig.py")
F5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F5)

IN = ROOT / "art/v2/tori-34/step4-filled"
OUT = ROOT / "art/v2/tori-34/step5-rig"
PUB = ROOT / "public/assets/characters/tori-v2-34"
H = W = 1024
GREEN, DARK = F5.GREEN, F5.DARK
PAD = F5.PAD

STEP4_ORDER = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "collar_front",
               "earFar", "earNear", "head", "tuft"]
ORDER = ["tail", "legFar", "legNear", "armFar", "torso", "pendant", "armNear", "collar_front",
         "earFar", "earNear", "tuft", "head"]
PARENT = {"torso": "root", "legNear": "root", "legFar": "root",
          "tail": "torso", "pendant": "torso", "armNear": "torso", "armFar": "torso", "collar_front": "torso",
          "head": "torso", "earNear": "head", "earFar": "head", "tuft": "head"}

PIVOTS = {
    "torso": [532.5, 887.0],
    "legNear": [444.5, 893.0],
    "legFar": [620.5, 881.0],
    "tail": [385.0, 835.0],
    "pendant": [599.0, 660.0],
    "armNear": [409.0, 656.0],
    "armFar": [672.0, 660.0],
    "collar_front": [532.5, 887.0],
    "head": [526.0, 630.0],
    "earNear": [363.0, 220.4],
    "earFar": [662.3, 212.3],
    "tuft": [498.0, 167.0],
}
# + = clockwise on screen.  armNear: + = raise outward (hand to the viewer's left), like front armL.
# armFar: - = raise outward (hand to the viewer's right), like front armR.
LIMITS = {
    "torso": [-5, 5],
    "legNear": [-15, 15],
    "legFar": [-15, 15],
    "tail": [-8, 8],
    "pendant": [-10, 10],
    "armNear": [-15, 75],
    "armFar": [-20, 10],
    "collar_front": [0, 0],
    "head": [-8, 8],
    "earNear": [-15, 15],
    "earFar": [-15, 15],
    "tuft": [-8, 8],
}
PIVOT_NOTES = {
    "torso": "hip centre between the two hip-cap centres (lean / sway); legs are root children so the feet stay planted",
    "legNear": "centre of the near leg's hip-cap half-ellipse filled under the torso in step 4",
    "legFar": "centre of the far leg's hip-cap half-ellipse filled under the torso in step 4",
    "tail": "tail root behind the torso's near side (step-4 root fill), so the curl swings while the root stays hidden",
    "pendant": "top of the acorn stem where it hangs from the collar",
    "armNear": "centre of the rounded shoulder end (step-4 cap), hidden under collar_front: ball joint",
    "armFar": "far shoulder behind the collar's right end / torso (the far arm is drawn behind the torso)",
    "collar_front": "rigid part of the torso (same pivot, never rotated on its own; limits 0)",
    "head": "neck: under the head's bottom edge, inside the step-4 neck/collar fill",
    "earNear": "ear base where the near ear enters the head outline",
    "earFar": "base of the far (turned-away) ear where it enters the head outline",
    "tuft": "tuft root ~14 px below the head outline at the centre of the contact span (base hidden behind the head)",
}
LIMIT_NOTES = {
    "armNear": "+ = raise outward (wave: +45..+75); - = in towards the belly, kept to 15.",
    "armFar": "- = raise outward. The far arm is BEHIND the torso: lifting it shows the torso's side edge, which is "
              "the hard step-2 cut along the arm's inner shading -> kept to -25 (subtle swing); + (in) 10.",
    "head": "+-8 (step-4 neck/collar fill under the head).",
    "earNear": "+-15.", "earFar": "+-15.",
    "tuft": "+-8 as front (flat tuft behind the head).",
    "legNear": "+-15: beyond, the torso's hard hip-cut corner at the gap between the legs starts to show.",
    "legFar": "+-15 (same reason).",
    "tail": "+-8.", "pendant": "+-10.", "torso": "+-5 lean about the hips, feet planted.",
}

BOXES = {
    "torso": (60, 40, 830, 1000), "legNear": (320, 800, 740, 1000), "legFar": (320, 800, 740, 1000),
    "tail": (60, 520, 480, 920), "pendant": (520, 600, 690, 800), "armNear": (180, 520, 520, 880),
    "armFar": (560, 560, 840, 860), "head": (150, 30, 880, 780), "earNear": (240, 90, 460, 300),
    "earFar": (560, 90, 780, 300), "tuft": (390, 40, 640, 240),
}
FULL = (60, 30, 840, 1000)


def combined_poses(rig):
    lim = {n: rig["parts"][n]["limits"] for n in rig["parts"]}
    return [
        ("wave: armNear max raise (+%d), head -4, earNear +8, tuft -5" % lim["armNear"]["maxDeg"],
         {"armNear": lim["armNear"]["maxDeg"], "head": -4, "earNear": 8, "tuft": -5}),
        ("wave mid: armNear +50, armFar -10, head +3", {"armNear": 50, "armFar": -10, "head": 3}),
        ("head tilt min (%d) + ears +12 + tuft max" % lim["head"]["minDeg"],
         {"head": lim["head"]["minDeg"], "earNear": 12, "earFar": 12, "tuft": lim["tuft"]["maxDeg"]}),
        ("head tilt max (%d) + ears -12 + tuft min" % lim["head"]["maxDeg"],
         {"head": lim["head"]["maxDeg"], "earNear": -12, "earFar": -12, "tuft": lim["tuft"]["minDeg"]}),
        ("walk A: legNear +12 legFar -12, armNear -12 armFar -12 (opposite), torso +2, tail +6",
         {"legNear": 12, "legFar": -12, "armNear": -12, "armFar": -12, "torso": 2, "tail": 6, "pendant": -5}),
        ("walk B: legNear -12 legFar +12, armNear +15 armFar +8, torso -2, tail -6",
         {"legNear": -12, "legFar": 12, "armNear": 15, "armFar": 8, "torso": -2, "tail": -6, "pendant": 5}),
        ("breath in: torso sy 1.03 (anchor breath = feet line), head -3, tuft +5",
         {"torso": {"sy": 1.03, "sx": 1.0, "anchor": "breath"}, "head": -3, "tuft": 5}),
        ("hop squash: torso sy 0.97 sx 1.02, legs +-8, arms out (armNear +30, armFar -20)",
         {"torso": {"sy": 0.97, "sx": 1.02, "anchor": "breath"}, "legNear": 8, "legFar": -8, "armNear": 30, "armFar": -20}),
        ("lean: torso -5, head +6, arms (armNear +20, armFar -15), tail +8",
         {"torso": -5, "head": 6, "armNear": 20, "armFar": -15, "tail": 8}),
        ("lean: torso +5, head -6, tail -8", {"torso": 5, "head": -6, "tail": -8}),
    ]


def load_step4():
    return {p: np.array(Image.open(IN / f"{p}.png").convert("RGBA")) for p in STEP4_ORDER}


def unmix_arm_collar(final):
    """front unmix_arm_collar, for armNear (the only arm under collar_front)."""
    info = {}
    below = ["tail", "legFar", "legNear", "armFar", "torso", "pendant"]
    cf = final["collar_front"].copy()
    arm = "armNear"
    A = final[arm].copy()
    a = A[..., 3]
    full = a == 255
    closed = ndi.binary_closing(full, structure=np.ones((3, 3)), iterations=1)
    P = closed & (a > 0) & (a < 255) & (final["torso"][..., 3] == 255) & (cf[..., 3] == 0)
    Bc, _ = F5.compose_full({k: final[k] for k in below}, below, np.zeros(3))
    res = []
    for y, x in zip(*np.nonzero(P)):
        al = a[y, x] / 255.0
        C = al * A[y, x, :3].astype(float) + (1 - al) * Bc[y, x]
        win = A[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        ok = win[..., 3] == 255
        ac = np.median(win[ok][:, :3].astype(float), 0) if ok.sum() >= 3 else A[y, x, :3].astype(float)
        b8 = 255 - int(a[y, x]); b = b8 / 255.0
        c = np.clip(np.round((C - (1 - b) * ac) / b), 0, 255)
        A[y, x, :3] = np.round(ac).astype(np.uint8); A[y, x, 3] = 255
        cf[y, x, :3] = c.astype(np.uint8); cf[y, x, 3] = b8
        res.append(float(np.abs(b * c + (1 - b) * np.round(ac) - C).max()))
    near = ndi.binary_dilation(cf[..., 3] > 0, iterations=2) & (cf[..., 3] < 255) & (A[..., 3] == 255) \
        & (final["torso"][..., 3] == 255)
    _, (iy, ix) = ndi.distance_transform_edt(cf[..., 3] < 255, return_indices=True)
    n2, res2 = 0, []
    for y, x in zip(*np.nonzero(near)):
        win = A[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        ok = win[..., 3] == 255
        ac = np.median(win[ok][:, :3].astype(float), 0)
        al0 = cf[y, x, 3] / 255.0
        C = al0 * cf[y, x, :3].astype(float) + (1 - al0) * A[y, x, :3].astype(float)
        if np.abs(A[y, x, :3].astype(float) - ac).max() <= 6:
            continue
        c = cf[iy[y, x], ix[y, x], :3].astype(float)
        d = c - ac
        b = float(np.clip(np.dot(C - ac, d) / max(np.dot(d, d), 1e-6), 0, 1))
        b8 = int(round(b * 255)); b = b8 / 255.0
        cc = np.clip(np.round((C - (1 - b) * ac) / b), 0, 255) if b8 > 0 else c
        r = float(np.abs(b * cc + (1 - b) * np.round(ac) - C).max())
        if r > 2.0:
            continue
        A[y, x, :3] = np.round(ac).astype(np.uint8)
        cf[y, x, :3] = cc.astype(np.uint8); cf[y, x, 3] = b8
        n2 += 1; res2.append(r)
    final[arm] = A
    final["collar_front"] = cf
    info[arm] = {"interior_alpha_px": int(P.sum()), "max_rest_residual": round(max(res), 2) if res else 0,
                 "dark_edge_px": n2, "dark_edge_max_rest_residual": round(max(res2), 2) if res2 else 0}
    return info


def unmix_torso_head(final):
    """front unmix_torso_head with this view's part names."""
    T, Hd = final["torso"].copy(), final["head"].copy()
    front = np.zeros((H, W), bool)
    for k in ("pendant", "armNear", "collar_front", "earNear", "earFar", "tuft", "head"):
        front |= final[k][..., 3] > 0
    near = ndi.binary_dilation(Hd[..., 3] == 255, iterations=2) & (T[..., 3] == 255) & ~front
    far = ~ndi.binary_dilation(Hd[..., 3] > 0, iterations=2) & (T[..., 3] == 255)
    _, (iy, ix) = ndi.distance_transform_edt(Hd[..., 3] < 255, return_indices=True)
    n, res = 0, []
    for y, x in zip(*np.nonzero(near)):
        win = T[y + 3:y + 8, x - 1:x + 2]
        ok = far[y + 3:y + 8, x - 1:x + 2]
        if ok.sum() < 4:
            continue
        tc = np.median(win[ok][:, :3].astype(float), 0)
        C = T[y, x, :3].astype(float)
        if np.abs(C - tc).max() <= 5:
            continue
        hc = Hd[iy[y, x], ix[y, x], :3].astype(float)
        d = hc - tc
        b = float(np.clip(np.dot(C - tc, d) / max(np.dot(d, d), 1e-6), 0, 1))
        b8 = int(round(b * 255))
        if b8 == 0:
            cc, b = hc, 0.0
        else:
            b = b8 / 255.0
            cc = np.clip(np.round((C - (1 - b) * np.round(tc)) / b), 0, 255)
        r = float(np.abs(b * cc + (1 - b) * np.round(tc) - C).max())
        if r > 3.0:
            continue
        T[y, x, :3] = np.round(tc).astype(np.uint8)
        if b8 > 0:
            Hd[y, x, :3] = cc.astype(np.uint8); Hd[y, x, 3] = b8
        n += 1; res.append(r)
    final["torso"], final["head"] = T, Hd
    return {"px": n, "max_rest_residual": round(max(res), 2) if res else 0}


def feet_line(layers):
    a = np.maximum(layers["legNear"][..., 3], layers["legFar"][..., 3])
    return int(np.nonzero(a.any(1))[0].max()) + 1


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layers").mkdir(exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    L4 = load_step4()
    head2, tuft2, tinfo, Z, tuft_area = F5.make_tuft_and_head(L4["head"], L4["tuft"])
    final = dict(L4)
    final["head"], final["tuft"] = head2, tuft2
    arm_fix = unmix_arm_collar(final)
    arm_fix["torso_head_unmix"] = unmix_torso_head(final)
    for p in ORDER:
        Image.fromarray(final[p], "RGBA").save(OUT / "layers" / f"{p}.png")
    parts = {}
    for z, p in enumerate(ORDER):
        bb = F5.bbox_of(final[p][..., 3] > 0)
        x0, y0 = max(0, bb[0] - PAD), max(0, bb[1] - PAD)
        x1, y1 = min(W, bb[2] + PAD), min(H, bb[3] + PAD)
        fn = f"{p}.png"
        Image.fromarray(final[p][y0:y1, x0:x1], "RGBA").save(PUB / fn, optimize=True)
        parts[p] = {"file": fn, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                    "pivot": [round(float(v), 1) for v in PIVOTS[p]], "parent": PARENT[p], "z": z,
                    "limits": {"minDeg": LIMITS[p][0], "maxDeg": LIMITS[p][1]}}
        if p in PIVOT_NOTES:
            parts[p]["pivotNote"] = PIVOT_NOTES[p]
        if p in LIMIT_NOTES:
            parts[p]["limitsNote"] = LIMIT_NOTES[p]
    fy = feet_line(final)
    allm = np.zeros((H, W), bool)
    for p in ORDER:
        allm |= final[p][..., 3] > 0
    bb = F5.bbox_of(allm)
    top = F5.bbox_of(np.any([final[p][..., 3] >= 128 for p in ORDER], 0))[1]
    front_rig = json.loads((ROOT / "art/v2/tori/step5-rig/rig.json").read_text())
    rig = {
        "name": "tori-v2-34",
        "version": 1,
        "view": "three-quarter (3/4), facing the viewer's right; Near = viewer-left side, Far = viewer-right side",
        "canvas": [W, H],
        "units": front_rig["units"],
        "angleConvention": "degrees, + = clockwise on screen (same sign as CSS rotate()); Near/Far = closer to / "
                           "turned away from the viewer (Near on the viewer's left in this view)",
        "transformModel": front_rig["transformModel"].replace("scripts/v2/step5_rig.py", "scripts/v2/step5_rig.py "
                                                              "(3/4 build: scripts/v2/34/step5_rig.py)").replace(
            "e.g. tail behind legs", "e.g. tail and far arm behind the torso"),
        "partNameMap": {"note": "front-view names -> 3/4-view names (same roles)",
                        "earL": "earNear", "earR": "earFar", "armL": "armNear", "armR": "armFar",
                        "legL": "legNear", "legR": "legFar"},
        "root": {"origin": [float(PIVOTS["torso"][0]), float(fy)],
                 "originNote": "bottom-centre between the feet (x = torso/hip centre, y = first row below the soles); "
                               "place this point on the scene ground line",
                 "height": int(fy - top), "heightNote": "feet line to the top of the tuft (alpha>=128)",
                 "bbox": bb},
        "drawOrder": ORDER,
        "parts": parts,
        "anchors": {
            "breath": {"point": [float(PIVOTS["torso"][0]), float(fy)], "target": "torso",
                       "use": "torso scaleY (breathing / squash) anchored at the feet line; children (head, arms, "
                              "tail, pendant, collar_front) follow; legs are root children so they stay planted",
                       "range": {"sy": [0.97, 1.03], "sx": [0.99, 1.02]}},
            "lean": {"point": list(PIVOTS["torso"]), "target": "torso",
                     "use": "body lean / sway = torso rotation about the hip centre (feet stay planted)"},
            "headBob": {"point": list(PIVOTS["head"]), "target": "head",
                        "use": "head squash (scaleY ~0.97-1.03) anchored at the neck"},
        },
    }
    rig["_tuft"] = tinfo
    rig["_armFix"] = arm_fix
    (OUT / "rig.json").write_text(json.dumps(rig, indent=1, ensure_ascii=False))
    shutil.copy(OUT / "rig.json", PUB / "rig.json")
    return L4, final, Z, tuft_area, tinfo, arm_fix


def rest_checks(R, L4, final, tuft_area):
    res = {}
    for name, bg in (("white", (255, 255, 255)), ("green", GREEN), ("dark", DARK)):
        bgv = np.array(bg, float)
        rig_o, _ = R.render({}, bg)
        fin_o, _ = F5.compose_full(final, ORDER, bgv)
        s4_o, _ = F5.compose_full(L4, STEP4_ORDER, bgv)
        r8, f8, s8 = (np.round(x) for x in (rig_o, fin_o, s4_o))
        d_rig = np.abs(r8 - f8).max(-1)
        d4 = np.abs(r8 - s8).max(-1)
        out = ~tuft_area
        res[name] = {
            "rig_render_from_crops_vs_final_full_layers": {"max": int(d_rig.max()), "px_diff": int((d_rig > 0).sum())},
            "vs_step4_rest": {
                "outside_tuft_area": {"max": int(d4[out].max()), "px_diff": int((d4[out] > 0).sum()),
                                      "px_diff_gt2": int((d4[out] > 2).sum())},
                "inside_tuft_area": {"px": int(tuft_area.sum()), "px_diff": int((d4[tuft_area] > 0).sum()),
                                     "max": int(d4[tuft_area].max())}},
        }
        if name == "white":
            res["_diff_img"] = d4
    return res


def poses_sheet(R, path):
    rows = []
    TW = 380
    for name, pose in combined_poses(R.rig):
        g, _ = F5.thumb(R, pose, FULL, TW, GREEN)
        d, _ = F5.thumb(R, pose, FULL, TW, DARK)
        rows.append((name, [g, d]))
    cols = 2
    cw = 2 * TW + 15
    ch = rows[0][1][0].height + 30
    nrow = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw + 10, nrow * ch + 40), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), "Tori v2 3/4 rig - combined poses (green | dark)", fill=(0, 0, 0), font=F5.font(18))
    for i, (name, ims) in enumerate(rows):
        gx, gy = 10 + (i % cols) * cw, 36 + (i // cols) * ch
        dr.text((gx, gy), name[:90], fill=(0, 0, 0), font=F5.font(12))
        sheet.paste(ims[0], (gx, gy + 18)); sheet.paste(ims[1], (gx + TW + 5, gy + 18))
    sheet.save(path)


def main():
    L4, final, Z, tuft_area, tinfo, arm_fix = build()
    # the front's sheet / metric helpers read these module globals
    F5.ORDER, F5.BOXES, F5.FULL, F5.combined_poses = ORDER, BOXES, FULL, combined_poses
    F5.STEP4_OPT_ORDER = STEP4_ORDER
    R = F5.Rig(PUB / "rig.json")
    checks = rest_checks(R, L4, final, tuft_area)
    d4 = checks.pop("_diff_img")
    base, _ = R.render({})
    im = (base * 0.35 + 255 * 0.65)
    im[d4 > 0] = [255, 0, 0]
    edge = tuft_area & ~ndi.binary_erosion(tuft_area)
    im[edge] = [0, 0, 255]
    F5.to_img(im[40:260, 380:650]).resize((810, 660), Image.NEAREST).save(OUT / "rest-diff-tuft.png")
    F5.pivots_sheet(R, OUT / "pivots.png")
    F5.tuft_sheet(R, L4, OUT / "tuft-check.png")
    tuft_sway = {}
    for deg in (-10, -8, -5, 5, 8, 10):
        s, _ = F5.joint_gap(R, {"tuft": deg}, "tuft", 90)
        tuft_sway[str(deg)] = s
    stats, combo = F5.range_sheet(R, OUT / "range-test.png")
    poses_sheet(R, OUT / "poses.png")
    rig = json.loads((OUT / "rig.json").read_text())
    rig["checks"] = {"rest": checks, "tuft": tinfo, "tuft_sway": tuft_sway, "arm_collar_unmix": arm_fix,
                     "range_test": stats, "combined_poses": combo}
    rig.pop("_tuft", None); rig.pop("_armFix", None)
    (OUT / "rig.json").write_text(json.dumps(rig, indent=1, ensure_ascii=False))
    shutil.copy(OUT / "rig.json", PUB / "rig.json")
    (OUT / "rest-check.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    print(json.dumps(stats, indent=1))
    print(json.dumps(combo, indent=1))


def cli_render(argv):
    pose = json.loads(pathlib.Path(argv[0]).read_text())
    bg = (255, 255, 255)
    scale = 1.0
    if "--bg" in argv:
        h = argv[argv.index("--bg") + 1]; bg = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if "--scale" in argv:
        scale = float(argv[argv.index("--scale") + 1])
    o, _ = F5.Rig(PUB / "rig.json").render(pose, bg, scale=scale)
    F5.to_img(o).save(argv[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        cli_render(sys.argv[2:])
    else:
        main()
