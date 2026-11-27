#!/usr/bin/env python3
"""
STEP 5 - Tori v2 SIDE view: final layers, part positions, pivots, hierarchy, limits (rig.json) + tests.

Input : art/v2/tori-side/step4-filled/<part>.png (incl. collar_front.png; tuft already flat + behind head)
Output: art/v2/tori-side/step5-rig/
          layers/<part>.png   final full-canvas layers (1024, same coords as the original)
          rig.json            SAME schema as the front rig (art/v2/tori/step5-rig/rig.json); copied to
                              public/assets/characters/tori-v2-side/rig.json
          pivots.png          rest pose x2, pivots + parent links
          range-test.png      every joint at min / mid / max of its limits (#6BBF59 | #202020) + combined poses
          walk-cycle.png      8-frame walk test strip (legs opposite, arm opposite to the near leg, root bob,
                              tail counter-sway) + its numbers in rig.json -> checks.walk
          rest-check.json     rig render (from the cropped PNGs) vs step-4 rest composite
        public/assets/characters/tori-v2-side/<part>.png  cropped layers (placed back at x,y)
Transform model / compositor: identical to the front (scripts/v2/step5_rig.py world_matrices / Rig.render,
imported).  Angles + = clockwise on screen.  The character faces screen-RIGHT, so for legs and arm
NEGATIVE (counter-clockwise) swings the foot / paw FORWARD (to the right).
Usage: python3 scripts/v2/side/step5_rig.py            build + tests
       python3 scripts/v2/side/step5_rig.py render pose.json out.png [--bg 6BBF59]
"""
import importlib.util, json, pathlib, shutil, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("front5", ROOT / "scripts/v2/step5_rig.py")
F5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(F5)

IN = ROOT / "art/v2/tori-side/step4-filled"
OUT = ROOT / "art/v2/tori-side/step5-rig"
PUB = ROOT / "public/assets/characters/tori-v2-side"
H = W = 1024
GREEN, DARK = (0x6B, 0xBF, 0x59), (0x20, 0x20, 0x20)
PAD = 2
ORDER = ["tail", "legFar", "legNear", "torso", "pendant", "armNear", "collar_front", "tuft", "head", "earNear"]
PARENT = {"torso": "root", "legNear": "root", "legFar": "root",
          "tail": "torso", "pendant": "torso", "armNear": "torso", "collar_front": "torso", "head": "torso",
          "earNear": "head", "tuft": "head"}
PIVOTS = {
    "torso": [559.0, 895.0], "legNear": [490.0, 896.0], "legFar": [628.0, 893.0], "tail": [458.0, 823.0],
    "pendant": [686.0, 641.0], "armNear": [534.0, 662.0], "collar_front": [559.0, 895.0],
    "head": [565.0, 615.0], "earNear": [482.0, 228.0], "tuft": [581.0, 182.0],
}
LIMITS = {
    "torso": [-5, 5], "legNear": [-25, 25], "legFar": [-25, 25], "tail": [-10, 10], "pendant": [-10, 10],
    "armNear": [-30, 30], "collar_front": [0, 0], "head": [-8, 8], "earNear": [-15, 15], "tuft": [-5, 5],
}
PIVOT_NOTES = {
    "torso": "hip centre between the two hip pivots (lean / sway); legs are root children so the feet stay planted",
    "legNear": "centre of the step-4 hip disk (r 54) under the belly: the leg's top is round about this point",
    "legFar": "centre of the step-4 hip disk (r 48, + a 38 px disk toward the belly front) under the belly",
    "tail": "centre of the step-4 tail-root disk (r 50) behind the torso's back edge",
    "pendant": "top of the acorn stem where it hangs from the collar",
    "armNear": "centre of the rounded shoulder end (step-4), under collar_front: ball-joint-like turn",
    "collar_front": "rigid part of the torso (same pivot, never rotated on its own; limits 0)",
    "head": "neck: above the collar, inside the step-4 neck fill, under the head's bottom edge",
    "earNear": "ear base where the ear's lower-right meets the head (the ear is drawn in front of the head)",
    "tuft": "tuft root below the head outline at the centre of the contact span (base hidden behind the head)",
}
LIMIT_NOTES = {
    "legNear": "+-25 (walk). - = foot forward (right). The rounded hip shows as a small round bump at the back "
               "silhouette at -25.",
    "legFar": "+-25 (walk). - = foot forward (right). Behind torso and near leg.",
    "armNear": "+-30 (walk swing). - = paw forward (right). The collar-shadow band on the arm top turns with the arm.",
    "tail": "+-10: small (2-4 px) corner of the root fill / cream notch at the junction at the extremes; +-8 cleaner.",
    "head": "+-8: collar fill above the visible collar shows; a faint 1-px original collar-top edge row is visible.",
    "earNear": "+-15: turning reveals the head's back stripe continued under the ear (step-4 fill).",
    "tuft": "+-5: beyond, a 1-2 px sliver of background opens under the right lobe (tuft +) / the hidden lobe "
            "behind the ear peeks out (tuft -).",
    "pendant": "+-10 clean (belly front silhouette continued under the acorn in step 4).",
    "torso": "+-5 lean about the hips with the legs planted; children follow.",
}
BOXES = {
    "torso": (100, 40, 880, 1000), "legNear": (360, 780, 760, 1000), "legFar": (360, 780, 760, 1000),
    "tail": (100, 520, 560, 940), "pendant": (600, 590, 780, 770), "armNear": (400, 580, 700, 880),
    "head": (250, 40, 880, 720), "earNear": (380, 80, 600, 300), "tuft": (440, 40, 720, 260),
}
FULL = (100, 40, 880, 1000)


def load_final():
    return {p: np.array(Image.open(IN / f"{p}.png").convert("RGBA")) for p in ORDER}


def build():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "layers").mkdir(exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)
    final = load_final()
    parts = {}
    for z, p in enumerate(ORDER):
        Image.fromarray(final[p], "RGBA").save(OUT / "layers" / f"{p}.png")
        bb = F5.bbox_of(final[p][..., 3] > 0)
        x0, y0 = max(0, bb[0] - PAD), max(0, bb[1] - PAD)
        x1, y1 = min(W, bb[2] + PAD), min(H, bb[3] + PAD)
        Image.fromarray(final[p][y0:y1, x0:x1], "RGBA").save(PUB / f"{p}.png", optimize=True)
        parts[p] = {"file": f"{p}.png", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                    "pivot": [round(float(v), 1) for v in PIVOTS[p]], "parent": PARENT[p], "z": z,
                    "limits": {"minDeg": LIMITS[p][0], "maxDeg": LIMITS[p][1]}}
        if p in PIVOT_NOTES:
            parts[p]["pivotNote"] = PIVOT_NOTES[p]
        if p in LIMIT_NOTES:
            parts[p]["limitsNote"] = LIMIT_NOTES[p]
    a = np.maximum(final["legNear"][..., 3], final["legFar"][..., 3])
    fy = int(np.nonzero(a.any(1))[0].max()) + 1
    allm = np.zeros((H, W), bool)
    for p in ORDER:
        allm |= final[p][..., 3] > 0
    top = F5.bbox_of(np.any([final[p][..., 3] >= 128 for p in ORDER], 0))[1]
    front = json.loads((ROOT / "art/v2/tori/step5-rig/rig.json").read_text())
    rig = {
        "name": "tori-v2-side",
        "version": 1,
        "view": "side (profile), facing screen-right; cand3 + step-1 fixes (eye highlights, near arm)",
        "canvas": [W, H],
        "units": front["units"],
        "angleConvention": front["angleConvention"] + "; facing right: negative leg / arm angles swing the foot / "
                                                       "paw forward (to the right)",
        "transformModel": front["transformModel"].replace("scripts/v2/step5_rig.py", "scripts/v2/step5_rig.py "
                                                          "(side: scripts/v2/side/step5_rig.py)"),
        "root": {"origin": [float(PIVOTS["torso"][0]), float(fy)],
                 "originNote": "bottom point under the hips (x = hip centre between the two hip pivots, y = first row "
                               "below the soles); place this point on the scene ground line",
                 "height": int(fy - top), "heightNote": "feet line to the top of the tuft (alpha>=128)",
                 "bbox": F5.bbox_of(allm)},
        "drawOrder": ORDER,
        "parts": parts,
        "anchors": {
            "breath": {"point": [float(PIVOTS["torso"][0]), float(fy)], "target": "torso",
                       "use": "torso scaleY (breathing / squash) anchored at the feet line; children follow; legs "
                              "are root children so they stay planted", "range": {"sy": [0.97, 1.03], "sx": [0.99, 1.02]}},
            "lean": {"point": list(PIVOTS["torso"]), "target": "torso",
                     "use": "body lean / sway = torso rotation about the hip centre (feet stay planted)"},
            "headBob": {"point": list(PIVOTS["head"]), "target": "head",
                        "use": "head squash (scaleY ~0.97-1.03) anchored at the neck"},
        },
    }
    (OUT / "rig.json").write_text(json.dumps(rig, indent=1, ensure_ascii=False))
    shutil.copy(OUT / "rig.json", PUB / "rig.json")
    return final


def thumb(R, pose, box, tw, bg):
    o, a = R.render(pose, bg, box, scale=tw / (box[2] - box[0]))
    return F5.to_img(o), a


def combined_poses():
    return [
        ("walk contact A: legNear -25 legFar +25, arm +30, tail +6, root dy 0",
         {"legNear": -25, "legFar": 25, "armNear": 30, "tail": 6}),
        ("walk contact B: legNear +25 legFar -25, arm -30, tail -6",
         {"legNear": 25, "legFar": -25, "armNear": -30, "tail": -6}),
        ("head -8 + ear +15 + tuft +5", {"head": -8, "earNear": 15, "tuft": 5}),
        ("head +8 + ear -15 + tuft -5", {"head": 8, "earNear": -15, "tuft": -5}),
        ("lean: torso -5, head +5, arm -20, tail +8", {"torso": -5, "head": 5, "armNear": -20, "tail": 8}),
        ("squash: torso sy 0.97 sx 1.02 (breath), legs +-10", {"torso": {"sy": 0.97, "sx": 1.02, "anchor": "breath"},
                                                               "legNear": -10, "legFar": 10}),
    ]


def range_sheet(R, path):
    rig = R.rig
    TW = 250
    rows, stats = [], {}
    joints = [n for n in ORDER if rig["parts"][n]["limits"]["minDeg"] != rig["parts"][n]["limits"]["maxDeg"]]
    for j in joints:
        lo, hi = rig["parts"][j]["limits"]["minDeg"], rig["parts"][j]["limits"]["maxDeg"]
        mid = round((lo + hi) / 2, 1)
        ims, st = [], {}
        for deg in (lo, mid, hi):
            pose = {j: deg}
            ims += [thumb(R, pose, BOXES[j], TW, GREEN)[0], thumb(R, pose, BOXES[j], TW, DARK)[0]]
            if deg != 0:
                s, _ = F5.joint_gap(R, pose, j, 90 if j not in ("head", "torso") else 320)
                st[str(deg)] = s
        stats[j] = st
        txt = "  ".join("%s: crack+%d hole+%d" % (k, v["new_crack_px"], v["new_hole_px"]) for k, v in st.items())
        rows.append((f"{j}  pivot {rig['parts'][j]['pivot']}  parent {rig['parts'][j]['parent']}  "
                     f"min {lo} / mid {mid} / max {hi}    [{txt}]", ims))
    combo = []
    _, a0 = R.render({})
    for name, pose in combined_poses():
        g, _ = thumb(R, pose, FULL, 2 * TW + 5, GREEN)
        d, _ = thumb(R, pose, FULL, 2 * TW + 5, DARK)
        _, a = R.render(pose)
        Ms = F5.world_matrices(rig, pose)
        hole, pocket = F5.new_holes(a0, a, [Ms[k] for k in pose if k != "root"])
        combo.append({"pose": name, "new_hole_px": int(hole.sum()), "closed_off_background_pocket_px": pocket})
        rows.append((f"{name}   [new holes {int(hole.sum())} px]", [g, d]))
    W6 = 6 * (TW + 5) + 10
    hgt = 60 + sum(max(im.height for im in r[1]) + 30 for r in rows)
    sheet = Image.new("RGB", (W6, hgt), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 10), "Tori v2 SIDE step 5 - range test: each joint at min / mid / max (children follow parents); "
                      "pairs = #6BBF59 | #202020; + = clockwise", fill=(0, 0, 0), font=F5.font(15))
    dr.text((10, 32), "crack = new thin background crack within the joint radius vs rest; hole = new enclosed "
                      "background px where the body was opaque at rest", fill=(60, 60, 60), font=F5.font(13))
    y = 60
    for label, ims in rows:
        dr.text((10, y), label, fill=(0, 0, 0), font=F5.font(13))
        x = 5
        for im in ims:
            sheet.paste(im, (x, y + 20)); x += im.width + 5
        y += max(im.height for im in ims) + 30
    sheet.save(path)
    return stats, combo


def pivots_sheet(R, path):
    rig = R.rig
    o, _ = R.render({}, (255, 255, 255), (0, 0, W, H), scale=2)
    im = F5.to_img(o).convert("RGB")
    d = ImageDraw.Draw(im)
    pts = {n: rig["parts"][n]["pivot"] for n in rig["parts"]}
    pts["root"] = rig["root"]["origin"]
    for n, p in rig["parts"].items():
        q = pts[p["parent"]]
        d.line((p["pivot"][0] * 2, p["pivot"][1] * 2, q[0] * 2, q[1] * 2), fill=(255, 0, 200), width=3)
    f = F5.font(22)
    cols = {"root": (0, 0, 0), "torso": (200, 0, 0), "head": (0, 90, 220)}
    for n, p in sorted(pts.items()):
        if n == "collar_front":
            continue
        x, y = p[0] * 2, p[1] * 2
        c = cols.get(n, (230, 0, 120))
        d.line((x - 14, y, x + 14, y), fill=c, width=4); d.line((x, y - 14, x, y + 14), fill=c, width=4)
        lab = f"{n} ({p[0]:.0f},{p[1]:.0f})"
        tx, ty = x + 16, y - 26 + (30 if n == "torso" else 0)
        d.rectangle((tx - 2, ty, tx + d.textlength(lab, font=f) + 2, ty + 24), fill=(255, 255, 255))
        d.text((tx, ty), lab, fill=c, font=f)
    a = rig["anchors"]["breath"]["point"]
    d.line((a[0] * 2 - 400, a[1] * 2, a[0] * 2 + 400, a[1] * 2), fill=(0, 160, 0), width=2)
    d.text((a[0] * 2 + 410, a[1] * 2 - 12), "feet line / root", fill=(0, 120, 0), font=F5.font(20))
    d.text((20, 20), "Tori v2 SIDE rig - rest pose x2: pivots (cross) + parent links (magenta)", fill=(0, 0, 0), font=F5.font(28))
    im.save(path)


def walk_frames(n=8):
    """phase t = 2*pi*k/n.  Facing right: negative angle = forward swing.  The drawn rest pose is already a
    stride (near leg back ~15 deg, far leg forward ~30 deg) = contact pose A at t=0; contact B at t=pi uses the
    full limit (near -25 = forward, far +25 = back); passing poses at t=pi/2, 3pi/2."""
    fr = []
    for k in range(n):
        t = 2 * np.pi * k / n
        c = np.cos(t)
        pose = {
            "legNear": round(-12.5 + 12.5 * c, 2), "legFar": round(12.5 - 12.5 * c, 2),
            "armNear": round(-20 * c, 2),                                # opposite to the near leg
            "root": {"dy": round(-6 * (1 - np.cos(2 * t)) / 2, 2)},      # low at contact, up 6 px at passing
            "tail": round(5 * np.cos(2 * t), 2),                         # counter-sway against the bob
            "head": round(-1.5 * np.cos(2 * t), 2), "pendant": round(4 * np.sin(t), 2),
        }
        fr.append(pose)
    return fr


def walk_sheet(R, path):
    box = (100, 40, 880, 1000)
    TW = 260
    frames = walk_frames()
    ims, stats = [], []
    _, a0 = R.render({})
    for k, pose in enumerate(frames):
        g, a = thumb(R, pose, box, TW, GREEN)
        d = ImageDraw.Draw(g)
        gy = (R.rig["root"]["origin"][1] - box[1]) * TW / (box[2] - box[0])
        d.line((0, gy, TW, gy), fill=(40, 90, 30), width=1)
        ims.append((k, pose, g))
        Ms = F5.world_matrices(R.rig, pose)
        _, af = R.render(pose)
        _, ar = R.render({"root": pose["root"]})          # rest with the same root bob (cracks compared in place)
        hole, pocket = F5.new_holes(ar, af, [Ms[q] for q in pose if q != "root"])
        cr = F5.crack_mask(af) & ~F5.binary_dilation(F5.crack_mask(ar)) if hasattr(F5, "binary_dilation") else \
            F5.crack_mask(af) & ~F5.ndi.binary_dilation(F5.crack_mask(ar), iterations=2)
        stats.append({"frame": k, "pose": pose, "new_hole_px": int(hole.sum()), "new_crack_px": int(cr.sum())})
    # second row: dark bg
    sheet = Image.new("RGB", (len(ims) * (TW + 6) + 6, 2 * (ims[0][2].height + 46) + 50), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    dr.text((8, 8), "Tori v2 SIDE - walk cycle test (8 frames, legs opposite over their 25 deg range, arm +-20 opposite to the near leg, "
                    "root bob 6 px, tail counter-sway +-5, head +-1.5, pendant +-4; f0 = drawn rest = contact A, f4 = contact B)", fill=(0, 0, 0), font=F5.font(15))
    for k, pose, g in ims:
        x = 6 + k * (TW + 6)
        dr.text((x, 30), f"f{k}: near {pose['legNear']:+.0f} far {pose['legFar']:+.0f} arm {pose['armNear']:+.0f} "
                         f"dy {pose['root']['dy']:+.1f}", fill=(0, 0, 0), font=F5.font(11))
        sheet.paste(g, (x, 46))
        dk, _ = thumb(R, pose, box, TW, DARK)
        sheet.paste(dk, (x, 46 + g.height + 8))
    sheet.save(path)
    return stats


def rest_checks(R, final):
    s4 = {p: np.array(Image.open(IN / f"{p}.png").convert("RGBA")) for p in ORDER}
    res = {}
    for name, bg in (("white", (255, 255, 255)), ("green", GREEN), ("dark", DARK)):
        ro, _ = R.render({}, bg)
        fo, _ = F5.compose_full(final, ORDER, np.array(bg, float))
        so, _ = F5.compose_full(s4, ORDER, np.array(bg, float))
        d1 = np.abs(np.round(ro) - np.round(fo)).max(-1)
        d2 = np.abs(np.round(ro) - np.round(so)).max(-1)
        res[name] = {"rig_render_from_crops_vs_final_full_layers": {"max": int(d1.max()), "px_diff": int((d1 > 0).sum())},
                     "rig_render_vs_step4_rest": {"max": int(d2.max()), "px_diff": int((d2 > 0).sum())}}
    return res


def main():
    final = build()
    R = F5.Rig(PUB / "rig.json")
    checks = rest_checks(R, final)
    print(json.dumps(checks))
    pivots_sheet(R, OUT / "pivots.png")
    stats, combo = range_sheet(R, OUT / "range-test.png")
    walk = walk_sheet(R, OUT / "walk-cycle.png")
    rig = json.loads((OUT / "rig.json").read_text())
    rig["checks"] = {"rest": checks, "range_test": stats, "combined_poses": combo,
                     "walk": {"frames": walk, "formula": "t=2*pi*k/8: legNear=-12.5+12.5 cos t, legFar=12.5-12.5 cos t, armNear=-20 cos t, "
                                                         "root.dy=-6(1-cos 2t)/2, tail=5 cos 2t, head=-1.5 cos 2t, pendant=4 sin t; "
                                                         "f0 = drawn rest stride = contact A, f4 = contact B"}}
    (OUT / "rig.json").write_text(json.dumps(rig, indent=1, ensure_ascii=False))
    shutil.copy(OUT / "rig.json", PUB / "rig.json")
    (OUT / "rest-check.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps({j: {k: (v["new_crack_px"], v["new_hole_px"]) for k, v in s.items()} for j, s in stats.items()}))
    print(json.dumps(combo))
    print(json.dumps([(w["frame"], w["new_hole_px"], w["new_crack_px"]) for w in walk]))


def cli_render(argv):
    pose = json.loads(pathlib.Path(argv[0]).read_text())
    bg = (255, 255, 255)
    if "--bg" in argv:
        h = argv[argv.index("--bg") + 1]; bg = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    o, _ = F5.Rig(PUB / "rig.json").render(pose, bg)
    F5.to_img(o).save(argv[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        cli_render(sys.argv[2:])
    else:
        main()
