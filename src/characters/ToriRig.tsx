import { staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { blink, wave } from "../lib/anim";
import { FitPart, LEGACY_HEIGHT, Part, useMouthLevel, Voice, walkCycle } from "../lib/rig";
import FIT from "./tori-fit.json";

/**
 * 토리 컷아웃 리그 (원본 설정화에 맞춘 파츠)
 * - 몸 파츠·위치: scripts/fit_rig.py → src/characters/tori-fit.json, public/assets/characters/tori/fit/
 * - 얼굴 파츠(눈·눈썹·코·입): public/assets/characters/tori/face/ (위치·크기는 원본에서 잰 값)
 * - 마음 도토리 펜던트는 SVG로 그림 (감정별 색)
 */

export type ToriEmotion = "neutral" | "happy" | "angry" | "sad" | "scared" | "surprised";

export const HEART_COLORS: Record<ToriEmotion, string> = {
  neutral: "#8FE3C0",
  happy: "#FFD84D",
  angry: "#FF5A4E",
  sad: "#6FA8FF",
  scared: "#9B6BFF",
  surprised: "#FF9A3D",
};

const BASE = "assets/characters/tori";
const S = (p: string) => staticFile(`${BASE}/${p}.png`);
const PARTS = FIT.parts;
const FACE = FIT.face;

// 얼굴 파츠 원본 크기 (face/parts.json 참고)
const EYE_PNG: Record<string, [number, number]> = {
  "face/eye-open": [295, 292],
  "face/eye-surprised": [295, 292],
  "face/eye-happy": [298, 159],
  "face/eye-blink": [317, 121],
};
const EYE_VISIBLE = 283; // eye-open 그림 안의 실제 눈 지름
const MOUTH: Record<string, { w: number; h: number; ax: number; ay: number }> = {
  "mouth-neutral": { w: 224, h: 86, ax: 109.5, ay: -1 },
  "mouth-happy": { w: 257, h: 219, ax: 125.5, ay: -1 },
  "mouth-talk-s": { w: 108, h: 130, ax: 53, ay: -1 },
  "mouth-talk-m": { w: 179, h: 160, ax: 88, ay: -1 },
  "mouth-talk-l": { w: 194, h: 239, ax: 94.5, ay: -1 },
  "mouth-sad": { w: 142, h: 101, ax: 69, ay: -1 },
  "mouth-angry": { w: 234, h: 111, ax: 115.5, ay: -25 },
  "mouth-o": { w: 122, h: 132, ax: 60, ay: -43 },
};

type Props = {
  x: number;
  y: number;
  /** 화면 배율 (1 = 키 약 1050px) */
  size?: number;
  emotion?: ToriEmotion;
  talking?: boolean;
  waving?: boolean;
  seed?: number;
  walking?: boolean;
  facing?: 1 | -1;
  voice?: Voice;
};

export const ToriRig: React.FC<Props> = ({
  x, y, size = 0.6, emotion = "neutral", talking = false, waving = false, seed = 0, walking = false, facing = 1, voice,
}) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const mouthLevel = useMouthLevel(voice, f, fps);
  const walk = walkCycle(f, walking);
  const k = 2; // 리그 좌표가 예전보다 약 2배 → 움직임 크기 보정

  const breathe = wave(f, 60, seed) * 0.012;
  const headTilt = wave(f, 90, seed) * 2 + (emotion === "sad" ? 5 : 0);
  const tailSwing = wave(f, 50, seed) * 5 + (emotion === "happy" ? wave(f, 8) * 5 : 0);
  const earTwitch = (n: number) => (wave(f + n * 7, 140, seed) > 0.96 ? -8 : 0);
  const shiver = emotion === "scared" ? wave(f, 3) * 3 * k : 0;
  const hop = emotion === "happy" ? -Math.abs(wave(f, 18)) * 30 * k : 0;
  const closed = blink(f, 95, seed * 13 + 30);
  const earDrop = emotion === "sad" || emotion === "scared" ? 18 : 0;

  const armL = walking ? walk.stepWave * 12 : emotion === "happy" ? 25 + wave(f, 18) * 8 : 2 + wave(f, 60, seed) * 2;
  const armR = walking ? -walk.stepWave * 12 : emotion === "happy" ? -25 - wave(f, 18) * 8 : -2 - wave(f, 60, seed) * 2;

  // 눈
  const eyeSrc =
    emotion === "happy" ? "face/eye-happy" : emotion === "surprised" || emotion === "scared" ? "face/eye-surprised" : closed > 0.5 ? "face/eye-blink" : "face/eye-open";
  const [ew, eh] = EYE_PNG[eyeSrc];
  const eyeScale = FACE.eyeL.d / EYE_VISIBLE;

  // 눈썹
  const brow: [string, number, number] =
    emotion === "angry" ? ["face/brow-angry", 278, 125] : emotion === "sad" ? ["face/brow-sad", 278, 125] : emotion === "scared" ? ["face/brow-worried", 316, 80] : ["face/brow-neutral", 301, 71];
  const browScale = (FACE.brow.w * 1.15) / 301;

  // 입
  const talkFrames = ["mouth-talk-s", "mouth-talk-m", "mouth-talk-l", "mouth-talk-m"];
  const emotionMouth = ({ neutral: "mouth-neutral", happy: "mouth-happy", angry: "mouth-angry", sad: "mouth-sad", scared: "mouth-angry", surprised: "mouth-o" } as const)[emotion];
  const voiceMouths = ["mouth-neutral", "mouth-talk-s", "mouth-talk-m", "mouth-talk-l"];
  let mouthName: string = emotionMouth;
  if (mouthLevel > 0) mouthName = voiceMouths[mouthLevel];
  else if (mouthLevel < 0 && talking) mouthName = talkFrames[Math.floor(f / 4) % talkFrames.length];
  const m = MOUTH[mouthName];
  const mouthScale = FACE.mouth.w / 224;
  const noseScale = FACE.nose.w / 101;
  const noseH = 59 * noseScale;

  const heart = HEART_COLORS[emotion];
  const unit = (size * LEGACY_HEIGHT) / FIT.height;
  const head = PARTS.head;
  const shoulderR = PARTS.armR.pivot;
  // 원본 펜던트 영역(빛 포함)에 맞춰 크기·위치
  const [pbx, pby, pbw, pbh] = PARTS.pendant_box;
  const pendScale = (pbw * 0.66) / 46;

  return (
    <div style={{ position: "absolute", left: x + shiver * unit, top: y + (hop + walk.bob * k) * unit, width: 0, height: 0, transform: `scale(${unit * facing}, ${unit})`, transformOrigin: "0 0" }}>
      {/* 꼬리: 몸 쪽 아래(오른쪽 아래)가 뿌리 */}
      <FitPart base={BASE} p={PARTS.tail} pivot={[PARTS.tail.x + PARTS.tail.w * 0.9, PARTS.tail.y + PARTS.tail.h * 0.92]} rotate={-tailSwing} z={1} />
      <FitPart base={BASE} p={PARTS.legL} dy={-walk.liftL * k} rotate={-walk.liftL * 0.25} z={2} />
      <FitPart base={BASE} p={PARTS.legR} dy={-walk.liftR * k} rotate={walk.liftR * 0.25} z={2} />
      <FitPart base={BASE} p={PARTS.torso} pivot={[0, 0]} scaleY={1 + breathe} z={3} />
      <FitPart base={BASE} p={PARTS.collar} z={4} />
      <FitPart base={BASE} p={PARTS.armL} rotate={armL} z={5} />
      {waving ? (
        <Part src={S("body/arm-r-wave")} w={445} h={630} x={shoulderR[0] - 20} y={shoulderR[1] + 60} px={0.22} py={0.9} rotate={wave(f, 10) * 10 + 15} scale={(PARTS.armR.h / 402) * 1.2} z={9} />
      ) : (
        <FitPart base={BASE} p={PARTS.armR} rotate={armR} z={5} />
      )}

      {/* 마음 도토리 */}
      <svg width={70} height={80} viewBox="-35 -40 70 80" style={{ position: "absolute", left: pbx + pbw / 2 - 35, top: pby + pbh * 0.36 - 40, transform: `scale(${pendScale})`, transformOrigin: "50% 50%", zIndex: 4, overflow: "visible" }}>
        <defs>
          <radialGradient id={`glow-${seed}`}>
            <stop offset="0%" stopColor={heart} stopOpacity={0.6} />
            <stop offset="100%" stopColor={heart} stopOpacity={0} />
          </radialGradient>
        </defs>
        <circle cx={0} cy={6} r={42 + wave(f, 30) * 4} fill={`url(#glow-${seed})`} />
        <path d="M-20 -6 C -22 16, -10 30, 0 32 C 10 30, 22 16, 20 -6 Z" fill={heart} stroke="#4E2313" strokeWidth={4} strokeLinejoin="round" />
        <path d="M-23 -3 C -23 -20, 23 -20, 23 -3 Z" fill="#8A5A35" stroke="#4E2313" strokeWidth={4} strokeLinejoin="round" />
        <path d="M0 -17 L 0 -26" stroke="#4E2313" strokeWidth={5} strokeLinecap="round" />
        <circle cx={-6} cy={6} r={4} fill="white" opacity={0.9} />
      </svg>

      {/* 머리 묶음: 목이 회전 중심 */}
      <div style={{ position: "absolute", left: 0, top: 0, width: 0, height: 0, transformOrigin: `${head.pivot[0]}px ${head.pivot[1]}px`, transform: `rotate(${headTilt}deg)`, zIndex: 6 }}>
        <FitPart base={BASE} p={PARTS.earL} rotate={earTwitch(1) - earDrop} z={0} />
        <FitPart base={BASE} p={PARTS.earR} rotate={-earTwitch(2) + earDrop} z={0} />
        <FitPart base={BASE} p={PARTS.tuft} rotate={wave(f, 70, seed) * 3} scaleY={emotion === "surprised" ? 1.15 : 1} z={0} />
        <FitPart base={BASE} p={head} z={1} />
        {[FACE.eyeL, FACE.eyeR].map((e, i) => (
          <Part key={i} src={S(eyeSrc)} w={ew} h={eh} x={e.x} y={e.y} scale={eyeScale} flipX={facing} z={2} />
        ))}
        {[FACE.eyeL, FACE.eyeR].map((e, i) => (
          <Part key={`b${i}`} src={S(brow[0])} w={brow[1]} h={brow[2]} x={e.x} y={FACE.brow.y} scale={browScale} flipX={i ? -1 : 1} z={2} />
        ))}
        <Part src={S("face/nose")} w={107} h={59} x={FACE.nose.x} y={FACE.nose.y} scale={noseScale} z={4} />
        <Part src={S(`face/${mouthName}`)} w={m.w} h={m.h} x={FACE.nose.x} y={FACE.nose.y + noseH / 2 - 4 * noseScale} px={m.ax / m.w} py={m.ay / m.h} scale={mouthScale} z={3} />
      </div>
    </div>
  );
};
