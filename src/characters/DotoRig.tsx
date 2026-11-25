import { Img, staticFile, useCurrentFrame } from "remotion";
import { blink, wave } from "../lib/anim";

/**
 * 도토 컷아웃 리그 (실제 파츠 이미지 사용)
 *
 * 좌표계: "리그 단위" = 몸 파츠 시트의 픽셀. 원점(0,0)은 두 발 사이 바닥.
 * 각 파츠는 pivot(회전 중심)을 기준으로 배치·회전합니다.
 * 위치 숫자는 DOTO_LAYOUT 한 곳에서 조정합니다.
 */

export type DotoEmotion = "neutral" | "happy" | "angry" | "sad" | "scared" | "surprised";

/** 마음 도토리 색 (docs/02-세계관-캐릭터.md 규칙) */
export const HEART_COLORS: Record<DotoEmotion, string> = {
  neutral: "#8FE3C0",
  happy: "#FFD84D",
  angry: "#FF5A4E",
  sad: "#6FA8FF",
  scared: "#9B6BFF",
  surprised: "#FF9A3D",
};

const P = (p: string) => staticFile(`assets/characters/doto/${p}.png`);

type PartProps = {
  src: string;
  w: number;
  h: number;
  /** 파츠를 붙일 위치 (리그 좌표) */
  x: number;
  y: number;
  /** 이미지 안에서 pivot 위치 (0~1 비율) */
  px?: number;
  py?: number;
  rotate?: number;
  scale?: number;
  scaleY?: number;
  z?: number;
};

const Part: React.FC<PartProps> = ({ src, w, h, x, y, px = 0.5, py = 0.5, rotate = 0, scale = 1, scaleY = 1, z }) => (
  <Img
    src={src}
    style={{
      position: "absolute",
      width: w,
      height: h,
      left: x - w * px,
      top: y - h * py,
      transformOrigin: `${px * 100}% ${py * 100}%`,
      transform: `rotate(${rotate}deg) scale(${scale}, ${scale * scaleY})`,
      zIndex: z,
    }}
  />
);

/** 파츠 위치표 (리그 단위). 숫자를 바꿔가며 맞춤 */
export const DOTO_LAYOUT = {
  legL: { x: -78, y: -6 },
  legR: { x: 78, y: -6 },
  torso: { x: 0, y: -70 }, // 몸통 아래 끝
  head: { x: 0, y: -440 }, // 목(머리 회전 중심)
  shoulderL: { x: -165, y: -395 },
  shoulderR: { x: 165, y: -395 },
  tail: { x: 120, y: -150 }, // 꼬리 뿌리
  // 머리 이미지(650x539) 안의 얼굴 위치
  face: {
    eyeL: { x: 200, y: 232 },
    eyeR: { x: 450, y: 232 },
    browY: 150,
    nose: { x: 325, y: 300 },
  },
};

const FACE = { eye: 0.36, brow: 0.22, nose: 0.46, mouth: 0.44 };

type Props = {
  x: number;
  y: number;
  /** 화면 배율 (1 = 리그 단위 그대로, 키 약 1050px) */
  size?: number;
  emotion?: DotoEmotion;
  talking?: boolean;
  waving?: boolean;
  /** 깜빡임 타이밍을 캐릭터마다 다르게 */
  seed?: number;
};

export const DotoRig: React.FC<Props> = ({ x, y, size = 0.6, emotion = "neutral", talking = false, waving = false, seed = 0 }) => {
  const f = useCurrentFrame();
  const L = DOTO_LAYOUT;

  // 생명감
  const breathe = wave(f, 60, seed) * 0.012;
  const headTilt = wave(f, 90, seed) * 2.5 + (emotion === "sad" ? 6 : 0);
  const tailSwing = wave(f, 50, seed) * 7 + (emotion === "happy" ? wave(f, 8) * 6 : 0);
  const earTwitch = (k: number) => (wave(f + k * 7, 140, seed) > 0.96 ? -8 : 0);
  const shiver = emotion === "scared" ? wave(f, 3) * 3 : 0;
  const hop = emotion === "happy" ? -Math.abs(wave(f, 18)) * 30 : 0;
  const closed = blink(f, 95, seed * 13 + 30);

  // 팔
  const armL = emotion === "happy" ? -25 + wave(f, 18) * 8 : 4 + wave(f, 60, seed) * 2;
  const armR = waving ? -138 + wave(f, 10) * 14 : emotion === "happy" ? 25 - wave(f, 18) * 8 : -4 - wave(f, 60, seed) * 2;

  // 눈
  const eyeSrc =
    emotion === "happy" ? "face/eye-happy" : emotion === "surprised" || emotion === "scared" ? "face/eye-surprised" : closed > 0.5 ? "face/eye-blink" : "face/eye-open";
  const eyeSize: Record<string, [number, number]> = {
    "face/eye-open": [295, 292],
    "face/eye-surprised": [295, 292],
    "face/eye-happy": [298, 159],
    "face/eye-blink": [317, 121],
  };
  const [ew, eh] = eyeSize[eyeSrc];

  // 눈썹
  const brow =
    emotion === "angry" ? ["face/brow-angry", 278, 125] : emotion === "sad" ? ["face/brow-sad", 278, 125] : emotion === "scared" ? ["face/brow-worried", 316, 80] : ["face/brow-neutral", 301, 71];

  // 입 (talking이면 크기 3단계를 번갈아)
  const talkFrames = ["mouth-talk-s", "mouth-talk-m", "mouth-talk-l", "mouth-talk-m"];
  const mouthName = talking
    ? talkFrames[Math.floor(f / 4) % talkFrames.length]
    : ({ neutral: "mouth-neutral", happy: "mouth-happy", angry: "mouth-angry", sad: "mouth-sad", scared: "mouth-angry", surprised: "mouth-o" } as const)[emotion];
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
  const m = MOUTH[mouthName];
  const noseW = 107 * FACE.nose;
  const noseH = 59 * FACE.nose;
  const noseBottom = L.face.nose.y + noseH / 2 - 4;

  const heart = HEART_COLORS[emotion];

  return (
    <div
      style={{
        position: "absolute",
        left: x + shiver,
        top: y + hop,
        width: 0,
        height: 0,
        transform: `scale(${size})`,
        transformOrigin: "0 0",
      }}
    >
      {/* 꼬리 (가장 뒤) */}
      <Part src={P("body/tail")} w={467} h={435} x={L.tail.x} y={L.tail.y} px={0.12} py={0.88} rotate={tailSwing} z={1} />

      {/* 다리 */}
      <Part src={P("body/leg-l")} w={256} h={365} x={L.legL.x} y={L.legL.y} px={0.5} py={1} z={2} />
      <Part src={P("body/leg-r")} w={256} h={365} x={L.legR.x} y={L.legR.y} px={0.5} py={1} z={2} />

      {/* 몸통: 아래 끝 기준으로 숨쉬기 */}
      <Part src={P("body/torso")} w={451} h={451} x={L.torso.x} y={L.torso.y} px={0.5} py={1} scaleY={1 + breathe} z={3} />

      {/* 팔: 어깨가 pivot */}
      <Part src={P("body/arm-l")} w={207} h={402} x={L.shoulderL.x} y={L.shoulderL.y} px={0.6} py={0.1} rotate={armL} scale={0.72} z={5} />
      <Part src={P("body/arm-r")} w={206} h={402} x={L.shoulderR.x} y={L.shoulderR.y} px={0.4} py={0.1} rotate={armR} scale={0.72} z={waving ? 9 : 5} />

      {/* 머리 묶음: 목이 pivot */}
      <div
        style={{
          position: "absolute",
          left: L.head.x - 325,
          top: L.head.y - 520,
          width: 650,
          height: 539,
          transformOrigin: "50% 96%",
          transform: `rotate(${headTilt}deg)`,
          zIndex: 6,
        }}
      >
        {/* 귀·애교머리는 머리 뒤 */}
        <Part src={P("body/ear-l")} w={230} h={247} x={150} y={110} px={0.6} py={0.85} rotate={earTwitch(1) + (emotion === "sad" || emotion === "scared" ? -25 : 0)} scale={0.9} z={0} />
        <Part src={P("body/ear-r")} w={230} h={247} x={500} y={110} px={0.4} py={0.85} rotate={-earTwitch(2) + (emotion === "sad" || emotion === "scared" ? 25 : 0)} scale={0.9} z={0} />
        <Part src={P("body/tuft")} w={336} h={247} x={325} y={75} px={0.5} py={0.9} rotate={wave(f, 70, seed) * 3} scale={0.8} scaleY={emotion === "surprised" ? 1.2 : 1} z={0} />
        <Part src={P("body/head")} w={650} h={539} x={325} y={269.5} z={1} />

        {/* 눈: 같은 그림을 양쪽에 (좌우 반전 없음 → 반짝이 위치 동일) */}
        {[L.face.eyeL, L.face.eyeR].map((p, i) => (
          <Part key={i} src={P(eyeSrc)} w={ew} h={eh} x={p.x} y={p.y} scale={FACE.eye} z={2} />
        ))}
        {/* 눈썹: 오른쪽은 좌우 반전 (선이라 반짝이 문제 없음) */}
        {[L.face.eyeL, L.face.eyeR].map((p, i) => (
          <div key={i} style={{ position: "absolute", left: 0, top: 0, width: 650, height: 539, transform: i ? "scaleX(-1)" : undefined, zIndex: 2 }}>
            <Part src={P(brow[0] as string)} w={brow[1] as number} h={brow[2] as number} x={i ? 650 - p.x : p.x} y={L.face.browY} scale={FACE.brow} />
          </div>
        ))}
        {/* 코 + 입 (입은 코 아래 기준점에 붙임) */}
        <Part src={P("face/nose")} w={107} h={59} x={L.face.nose.x} y={L.face.nose.y} scale={FACE.nose} z={4} />
        <Part
          src={P(`face/${mouthName}`)}
          w={m.w}
          h={m.h}
          x={L.face.nose.x}
          y={noseBottom}
          px={m.ax / m.w}
          py={m.ay / m.h}
          scale={FACE.mouth}
          z={3}
        />
      </div>

      {/* 초커 + 마음 도토리 (코드로 그려서 색 변경) */}
      <Part src={P("body/collar")} w={426} h={134} x={0} y={L.head.y - 10} scale={0.62} z={7} />
      <svg width={70} height={80} viewBox="-35 -40 70 80" style={{ position: "absolute", left: -35, top: L.head.y - 2, transform: "scale(1.7)", transformOrigin: "50% 0%", zIndex: 8, overflow: "visible" }}>
        <defs>
          <radialGradient id={`glow-${seed}`}>
            <stop offset="0%" stopColor={heart} stopOpacity={0.6} />
            <stop offset="100%" stopColor={heart} stopOpacity={0} />
          </radialGradient>
        </defs>
        <circle cx={0} cy={6} r={42 + wave(f, 30) * 4} fill={`url(#glow-${seed})`} />
        {/* 도토리 몸 */}
        <path d="M-17 -6 C -19 14, -8 28, 0 32 C 8 28, 19 14, 17 -6 Z" fill={heart} stroke="#4E2313" strokeWidth={4} strokeLinejoin="round" />
        {/* 깍지 */}
        <path d="M-21 -4 C -21 -20, 21 -20, 21 -4 Z" fill="#8A5A35" stroke="#4E2313" strokeWidth={4} strokeLinejoin="round" />
        <path d="M0 -17 L 0 -26" stroke="#4E2313" strokeWidth={5} strokeLinecap="round" />
        <circle cx={-6} cy={6} r={4} fill="white" opacity={0.9} />
      </svg>
    </div>
  );
};
