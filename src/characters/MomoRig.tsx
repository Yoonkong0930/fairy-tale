import { staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { blink, wave } from "../lib/anim";
import { Part, useMouthLevel, Voice, walkCycle } from "../lib/rig";

/**
 * 모모 (아기토끼) 컷아웃 리그
 * - 귀가 감정 표현의 핵심: 평소(하나 쫑긋·하나 접힘) / 슬픔(축 처짐) / 무서움(뒤로 납작) / 놀람(둘 다 쫑긋)
 * - 눈은 한 그림을 양쪽에 (좌우 반전 금지 → 반짝이 위치 동일)
 */

export type MomoEmotion = "neutral" | "happy" | "worried" | "sad" | "scared" | "surprised";

const P = (p: string) => staticFile(`assets/characters/momo/${p}.png`);

export const MOMO_LAYOUT = {
  legL: { x: -72, y: -6 },
  legR: { x: 72, y: -6 },
  torso: { x: 0, y: -60 },
  head: { x: 0, y: -430 }, // 목(머리 회전 중심)
  shoulderL: { x: -158, y: -385 },
  shoulderR: { x: 158, y: -385 },
  tail: { x: 150, y: -120 },
  // 머리 이미지(584x495) 안의 위치
  face: {
    eyeL: { x: 185, y: 262 },
    eyeR: { x: 399, y: 262 },
    browY: 175,
    nose: { x: 292, y: 312 },
    earLeftBase: { x: 165, y: 70 }, // 화면 왼쪽 귀 (접힌 귀)
    earRightBase: { x: 420, y: 70 }, // 화면 오른쪽 귀 (쫑긋 귀)
  },
};

const FACE = { eye: 0.35, brow: 0.22, nose: 0.22, mouth: 0.27 };

const MOUTH: Record<string, [number, number]> = {
  "mouth-neutral": [293, 95],
  "mouth-happy": [279, 185],
  "mouth-talk-s": [129, 136],
  "mouth-talk-m": [278, 192],
  "mouth-talk-l": [309, 285],
  "mouth-sad": [205, 108],
  "mouth-worried": [301, 69],
  "mouth-o": [151, 164],
};

type Props = {
  x: number;
  y: number;
  size?: number;
  emotion?: MomoEmotion;
  talking?: boolean;
  waving?: boolean;
  walking?: boolean;
  facing?: 1 | -1;
  seed?: number;
  voice?: Voice;
};

export const MomoRig: React.FC<Props> = ({
  x, y, size = 0.6, emotion = "neutral", talking = false, waving = false, walking = false, facing = 1, seed = 7, voice,
}) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const L = MOMO_LAYOUT;
  const mouthLevel = useMouthLevel(voice, f, fps);
  const walk = walkCycle(f, walking);

  const breathe = wave(f, 64, seed) * 0.012;
  const headTilt = wave(f, 96, seed) * 2.5 + (emotion === "sad" ? -6 : emotion === "worried" ? -3 : 0);
  const shiver = emotion === "scared" ? wave(f, 3) * 3 : 0;
  const hop = emotion === "happy" ? -Math.abs(wave(f, 18)) * 26 : 0;
  const closed = blink(f, 105, seed * 11 + 50);

  // 귀 (화면 기준 왼쪽 = 접힌 귀, 오른쪽 = 쫑긋 귀)
  const earSway = wave(f, 80, seed) * 3;
  let leftEarSrc = "body/ear-r-fold";
  let leftEarRot = earSway;
  let rightEarRot = -earSway + (wave(f + 40, 150, seed) > 0.95 ? 10 : 0); // 가끔 쫑긋
  if (emotion === "surprised") { leftEarSrc = "body/ear-r-up"; leftEarRot = -8; rightEarRot = 8; }
  if (emotion === "happy") { rightEarRot = 6 + wave(f, 9) * 6; }
  if (emotion === "worried") { leftEarRot = -25; rightEarRot = 22; }
  if (emotion === "sad") { leftEarRot = -75; rightEarRot = 125; }
  if (emotion === "scared") { leftEarRot = -60 + wave(f, 3) * 3; rightEarRot = 100 + wave(f, 3) * 3; }
  const leftEar = leftEarSrc === "body/ear-r-fold" ? { w: 453, h: 344, px: 0.72, py: 0.88 } : { w: 268, h: 450, px: 0.62, py: 0.92 };

  // 팔
  const armL = walking ? walk.stepWave * 14 : emotion === "happy" ? -22 + wave(f, 18) * 8 : emotion === "scared" || emotion === "worried" ? -18 : 3 + wave(f, 60, seed) * 2;
  const armR = walking ? -walk.stepWave * 14 : emotion === "happy" ? 22 - wave(f, 18) * 8 : emotion === "scared" || emotion === "worried" ? 18 : -3 - wave(f, 60, seed) * 2;

  // 눈·눈썹
  const eyeSrc =
    emotion === "happy" ? "face/eye-happy" : emotion === "surprised" || emotion === "scared" ? "face/eye-surprised" : closed > 0.5 ? "face/eye-blink" : "face/eye-open";
  const EYE: Record<string, [number, number]> = {
    "face/eye-open": [312, 311], "face/eye-surprised": [271, 270], "face/eye-happy": [284, 133], "face/eye-blink": [286, 97],
  };
  const [ew, eh] = EYE[eyeSrc];
  const brow: [string, number, number] =
    emotion === "worried" || emotion === "sad" || emotion === "scared" ? ["face/brow-worried", 221, 112] : ["face/brow-neutral", 278, 65];

  // 입
  const emotionMouth = ({ neutral: "mouth-neutral", happy: "mouth-happy", worried: "mouth-worried", sad: "mouth-sad", scared: "mouth-worried", surprised: "mouth-o" } as const)[emotion];
  const talkFrames = ["mouth-talk-s", "mouth-talk-m", "mouth-talk-l", "mouth-talk-m"];
  const voiceMouths = ["mouth-neutral", "mouth-talk-s", "mouth-talk-m", "mouth-talk-l"];
  let mouthName: string = emotionMouth;
  if (mouthLevel > 0) mouthName = voiceMouths[mouthLevel];
  else if (mouthLevel < 0 && talking) mouthName = talkFrames[Math.floor(f / 4) % talkFrames.length];
  const [mw, mh] = MOUTH[mouthName];
  const noseBottom = L.face.nose.y + (106 * FACE.nose) / 2;

  return (
    <div
      style={{
        position: "absolute",
        left: x + shiver,
        top: y + hop + walk.bob,
        width: 0,
        height: 0,
        transform: `scale(${size * facing}, ${size})`,
        transformOrigin: "0 0",
      }}
    >
      <Part src={P("body/tail")} w={282} h={279} x={L.tail.x} y={L.tail.y} rotate={wave(f, 40, seed) * 6} scale={0.6} z={1} />
      <Part src={P("body/leg-l")} w={238} h={344} x={L.legL.x} y={L.legL.y - walk.liftL} px={0.5} py={1} rotate={-walk.liftL * 0.25} z={2} />
      <Part src={P("body/leg-r")} w={238} h={344} x={L.legR.x} y={L.legR.y - walk.liftR} px={0.5} py={1} rotate={walk.liftR * 0.25} z={2} />
      <Part src={P("body/torso")} w={481} h={475} x={L.torso.x} y={L.torso.y} px={0.5} py={1} scaleY={1 + breathe} z={3} />
      <Part src={P("body/arm-l")} w={259} h={444} x={L.shoulderL.x} y={L.shoulderL.y} px={0.6} py={0.1} rotate={armL} scale={0.56} z={5} />
      {waving ? (
        <Part src={P("body/arm-r-wave")} w={401} h={394} x={L.shoulderR.x + 10} y={L.shoulderR.y + 30} px={0.15} py={0.85} rotate={wave(f, 10) * 10 - 10} scale={0.66} z={9} />
      ) : (
        <Part src={P("body/arm-r")} w={259} h={444} x={L.shoulderR.x} y={L.shoulderR.y} px={0.4} py={0.1} rotate={armR} scale={0.56} z={5} />
      )}

      {/* 머리 묶음 */}
      <div
        style={{
          position: "absolute",
          left: L.head.x - 292,
          top: L.head.y - 480,
          width: 584,
          height: 495,
          transformOrigin: "50% 96%",
          transform: `rotate(${headTilt}deg) scale(1.18)`,
          zIndex: 6,
        }}
      >
        <Part src={P(leftEarSrc)} w={leftEar.w} h={leftEar.h} x={L.face.earLeftBase.x} y={L.face.earLeftBase.y} px={leftEar.px} py={leftEar.py} rotate={leftEarRot} scale={0.62} z={0} />
        {/* 쫑긋 귀: 왼쪽 귀 그림을 좌우 반전해서 바깥으로 기울게 */}
        <Part src={P("body/ear-l")} w={264} h={430} x={L.face.earRightBase.x} y={L.face.earRightBase.y} px={0.38} py={0.92} rotate={rightEarRot} flipX={-1} scale={0.62} z={0} />
        <Part src={P("body/head")} w={584} h={495} x={292} y={247.5} z={1} />
        <Part src={P("body/flower")} w={260} h={253} x={L.face.earRightBase.x + 10} y={L.face.earRightBase.y + 30} rotate={rightEarRot * 0.3} scale={0.34} z={3} />

        {[L.face.eyeL, L.face.eyeR].map((p, i) => (
          <Part key={i} src={P(eyeSrc)} w={ew} h={eh} x={p.x} y={p.y} scale={FACE.eye} flipX={facing} z={2} />
        ))}
        {[L.face.eyeL, L.face.eyeR].map((p, i) => (
          <Part key={`b${i}`} src={P(brow[0])} w={brow[1]} h={brow[2]} x={p.x} y={L.face.browY} scale={FACE.brow} flipX={i ? -1 : 1} z={2} />
        ))}
        <Part src={P("face/nose")} w={162} h={106} x={L.face.nose.x} y={L.face.nose.y} scale={FACE.nose} z={4} />
        <Part src={P(`face/${mouthName}`)} w={mw} h={mh} x={L.face.nose.x} y={noseBottom + 3} px={0.5} py={0} scale={FACE.mouth} z={3} />
      </div>
    </div>
  );
};
