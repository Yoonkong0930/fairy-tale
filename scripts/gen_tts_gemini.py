#!/usr/bin/env python3
"""
(테스트용) Gemini TTS로 대사 음성 생성 → wav
실제 제작 음성은 타입캐스트를 쓸 예정. 입 모양 동기화 테스트용.

사용: python3 scripts/gen_tts_gemini.py "대사" public/audio/test.wav [--style "밝고 신나게, 어린아이 목소리로"] [--voice Puck]
"""
import argparse, base64, json, os, urllib.request, wave

ap = argparse.ArgumentParser()
ap.add_argument("text"); ap.add_argument("out")
ap.add_argument("--style", default="밝고 귀여운 어린아이 목소리로, 신나게 말해줘")
ap.add_argument("--voice", default="Leda")
ap.add_argument("--model", default="gemini-3.8-flash-tts")
a = ap.parse_args()

body = {
    "contents": [{"parts": [{"text": f"{a.style}: {a.text}"}]}],
    "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": a.voice}}},
    },
}
req = urllib.request.Request(
    f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "x-goog-api-key": os.environ.get("GEMINI_API_KEY", "proxy-injected")},
)
res = json.load(urllib.request.urlopen(req, timeout=120))
part = res["candidates"][0]["content"]["parts"][0]
blob = part.get("inlineData") or part.get("inline_data")
pcm = base64.b64decode(blob["data"])
rate = 24000
mt = blob.get("mimeType", "")
if "rate=" in mt:
    rate = int(mt.split("rate=")[1].split(";")[0])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
if pcm[:4] == b"RIFF":  # 이미 wav 형식이면 그대로 저장
    open(a.out, "wb").write(pcm)
    with wave.open(a.out) as w:
        print("saved", a.out, f"{w.getnframes()/w.getframerate():.2f}s")
    raise SystemExit
with wave.open(a.out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
print("saved", a.out, f"{len(pcm)/2/rate:.2f}s", mt)
