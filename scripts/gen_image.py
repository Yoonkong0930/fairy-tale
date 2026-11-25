#!/usr/bin/env python3
"""
Gemini(Nano Banana) 이미지 생성 스크립트

사용 예:
  python3 scripts/gen_image.py --prompt-file prompts/characters/tori-model-sheet.md \
      --out public/assets/characters/tori/model-sheet.png --aspect 16:9
  python3 scripts/gen_image.py --prompt "같은 캐릭터, 눈을 감은 얼굴만" \
      --ref public/assets/characters/tori/model-sheet.png --out ...

API 키는 작업 환경의 네트워크 비밀값(x-goog-api-key 헤더)으로 자동 주입됩니다.
로컬에서 직접 돌릴 때는 GEMINI_API_KEY 환경변수를 설정하세요.
"""
import argparse, base64, json, mimetypes, os, pathlib, re, sys, urllib.request, urllib.error

MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def read_prompt(path: str) -> str:
    """마크다운 파일에서 ```prompt 블록만 꺼냄 (없으면 파일 전체)."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    m = re.search(r"```prompt\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--ref", action="append", default=[], help="기준 이미지 (여러 번 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aspect", default="1:1")
    ap.add_argument("--size", default="2K", help="1K / 2K / 4K")
    a = ap.parse_args()

    prompt = a.prompt or (read_prompt(a.prompt_file) if a.prompt_file else None)
    if not prompt:
        sys.exit("--prompt 또는 --prompt-file 이 필요합니다")

    parts = []
    for ref in a.ref:
        mime = mimetypes.guess_type(ref)[0] or "image/png"
        data = base64.b64encode(pathlib.Path(ref).read_bytes()).decode()
        parts.append({"inline_data": {"mime_type": mime, "data": data}})
    parts.append({"text": prompt})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": a.aspect, "imageSize": a.size},
        },
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            # 클라우드 환경에서는 프록시가 실제 키로 덮어씁니다
            "x-goog-api-key": os.environ.get("GEMINI_API_KEY", "proxy-injected"),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"API 오류 {e.code}: {e.read().decode()[:800]}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for cand in res.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            blob = p.get("inline_data") or p.get("inlineData")
            if blob:
                path = out if n == 0 else out.with_name(f"{out.stem}-{n}{out.suffix}")
                path.write_bytes(base64.b64decode(blob["data"]))
                print("saved", path)
                n += 1
            elif p.get("text"):
                print("text:", p["text"][:300])
    if n == 0:
        sys.exit("이미지가 생성되지 않았습니다: " + json.dumps(res)[:800])


if __name__ == "__main__":
    main()
