import { staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { blink, wave } from "../lib/anim";
import { FitPart, LEGACY_HEIGHT, Part, useMouthLevel, Voice, walkCycle } from "../lib/rig";
import FIT from "./momo-fit.json";

/**
 * 모모 (아기토끼) 컷아웃 리그 (원본 설정화에 맞춘 파츠)
 * - 몸 파츠·위치: scripts/fit_rig.py → src/characters/momo-fit.json, public/assets/characters/momo/fit/
 * - 귀가 감정 표현의 핵심: 평소(하나 접힘·하나 쫑긋) / 걱정·슬픔·무서움(처짐) / 놀람(둘 다 쫑긋)
 */

export type MomoEmotion = "neutral" | "happy" | "worried" | "sad" | "scared" | "surprised";

const BASE = "assets/characters/momo";
const S = (p: string) => staticFile(`${BASE}/${p}.png`);
const PARTS = FIT.parts;
const FACE = FIT.face;

const EYE_PNG: Record<string, [number, number]> = {
  "face/eye-open": [312, 311], "face/eye-surprised": [271, 270], "face/eye-happy": [284, 133], "face/eye-blink": [286, 97],
};
const EYE_VISIBLE = 300;
const MOUTH: Record<string, [number, number]> = {
  "mouth-neutral": [293, 95], "mouth-happy": [279, 185], "mouth-talk-s": [129, 136], "mouth-talk-m": [278, 192],
  "mouth-talk-l": [309, 285], "mouth-sad": [205, 108], "mouth-worried": [301, 69], "mouth-o": [151, 164],
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
  const mouthLevel = useMouthLevel(voice, f, fps);
  const walk = walkCycle(f, walking);
  const k = 2;

  const breathe = wave(f, 64, seed) * 0.012;
  const headTilt = wave(f, 96, seed) * 2 + (emotion === "sad" ? -5 : emotion === "worried" ? -2 : 0);
  const shiver = emotion === "scared" ? wave(f, 3) * 3 * k : 0;
  const hop = emotion === "happy" ? -Math.abs(wave(f, 18)) * 26 * k : 0;
  const closed = blink(f, 105, seed * 11 + 50);

  // 귀: 왼쪽(화면) = 접힌 귀, 오른쪽 = 꽃핀 달린 쫑긋 귀. 각도는 바깥으로 처지는 방향
  const sway = wave(f, 80, seed) * 3;
  let earLRot = sway;
  let earRRot = -sway + (wave(f + 40, 150, seed) > 0.95 ? 6 : 0);
  let earLUp = false;
  if (emotion === "happy") earRRot = 4 + wave(f, 9) * 5;
  if (emotion === "worried") { earLRot = -12; earRRot = 18; }
  if (emotion === "sad") { earLRot = -35; earRRot = 55; }
  if (emotion === "scared") { earLRot = -28 + wave(f, 3) * 2; earRRot = 45 + wave(f, 3) * 2; }
  if (emotion === "surprised") { earLUp = true; earLRot = 0; earRRot = -4; }

  // 팔
  const armL = walking ? walk.stepWave * 12 : emotion === "happy" ? 22 + wave(f, 18) * 8 : emotion === "scared" || emotion === "worried" ? -14 : 2 + wave(f, 60, seed) * 2;
  const armR = walking ? -walk.stepWave * 12 : emotion === "happy" ? -22 - wave(f, 18) * 8 : emotion === "scared" || emotion === "worried" ? 14 : -2 - wave(f, 60, seed) * 2;

  // 눈·눈썹
  const eyeSrc =
    emotion === "happy" ? "face/eye-happy" : emotion === "surprised" || emotion === "scared" ? "face/eye-surprised" : closed > 0.5 ? "face/eye-blink" : "face/eye-open";
  const [ew, eh] = EYE_PNG[eyeSrc];
  const eyeScale = FACE.eyeL.d / EYE_VISIBLE;
  const brow: [string, number, number] =
    emotion === "worried" || emotion === "sad" || emotion === "scared" ? ["face/brow-worried", 221, 112] : ["face/brow-neutral", 278, 65];
  const browScale = (FACE.brow.w * 1.1) / 278;

  // 입
  const emotionMouth = ({ neutral: "mouth-neutral", happy: "mouth-happy", worried: "mouth-worried", sad: "mouth-sad", scared: "mouth-worried", surprised: "mouth-o" } as const)[emotion];
  const talkFrames = ["mouth-talk-s", "mouth-talk-m", "mouth-talk-l", "mouth-talk-m"];
  const voiceMouths = ["mouth-neutral", "mouth-talk-s", "mouth-talk-m", "mouth-talk-l"];
  let mouthName: string = emotionMouth;
  if (mouthLevel > 0) mouthName = voiceMouths[mouthLevel];
  else if (mouthLevel < 0 && talking) mouthName = talkFrames[Math.floor(f / 4) % talkFrames.length];
  const [mw, mh] = MOUTH[mouthName];
  const mouthScale = FACE.mouth.w / 293;
  const noseScale = FACE.nose.w / 150;

  const unit = (size * LEGACY_HEIGHT) / FIT.height;
  const head = PARTS.head;
  const eL = PARTS.earL;
  const shoulderR = PARTS.armR.pivot;
  // 놀람: 접힌 귀 대신 '쫑긋 선 귀' 그림 (원본 쫑긋 귀 크기에 맞춤)
  const upH = PARTS.earR.h * 0.95;
  const upW = (268 / 450) * upH;

  return (
    <div style={{ position: "absolute", left: x + shiver * unit, top: y + (hop + walk.bob * k) * unit, width: 0, height: 0, transform: `scale(${unit * facing}, ${unit})`, transformOrigin: "0 0" }}>
      <FitPart base={BASE} p={PARTS.legL} dy={-walk.liftL * k} rotate={-walk.liftL * 0.25} z={2} />
      <FitPart base={BASE} p={PARTS.legR} dy={-walk.liftR * k} rotate={walk.liftR * 0.25} z={2} />
      <FitPart base={BASE} p={PARTS.torso} pivot={[0, 0]} scaleY={1 + breathe} z={3} />
      <FitPart base={BASE} p={PARTS.armL} rotate={armL} z={5} />
      {waving ? (
        <Part src={S("body/arm-r-wave")} w={401} h={394} x={shoulderR[0] - 10} y={shoulderR[1] + 40} px={0.15} py={0.85} rotate={wave(f, 10) * 10 + 8} scale={(PARTS.armR.h / 444) * 1.4} z={9} />
      ) : (
        <FitPart base={BASE} p={PARTS.armR} rotate={armR} z={5} />
      )}

      <div style={{ position: "absolute", left: 0, top: 0, width: 0, height: 0, transformOrigin: `${head.pivot[0]}px ${head.pivot[1]}px`, transform: `rotate(${headTilt}deg)`, zIndex: 6 }}>
        {earLUp ? (
          <Part src={S("body/ear-r-up")} w={upW} h={upH} x={eL.pivot[0] + 30} y={eL.pivot[1] + 40} px={0.62} py={0.95} rotate={-6} z={0} />
        ) : (
          <FitPart base={BASE} p={eL} pivot={[eL.x + eL.w * 0.78, eL.y + eL.h * 0.92]} rotate={earLRot} z={0} />
        )}
        <FitPart base={BASE} p={PARTS.earR} rotate={earRRot} z={0} />
        <FitPart base={BASE} p={head} z={1} />
        {/* 꽃핀: 머리 앞에 보이고, 쫑긋 귀와 같이 움직임 */}
        <FitPart base={BASE} p={PARTS.flower} pivot={PARTS.earR.pivot} rotate={earRRot * 0.6} z={5} />
        {[FACE.eyeL, FACE.eyeR].map((e, i) => (
          <Part key={i} src={S(eyeSrc)} w={ew} h={eh} x={e.x} y={e.y} scale={eyeScale} flipX={facing} z={2} />
        ))}
        {[FACE.eyeL, FACE.eyeR].map((e, i) => (
          <Part key={`b${i}`} src={S(brow[0])} w={brow[1]} h={brow[2]} x={e.x} y={FACE.brow.y} scale={browScale} flipX={i ? -1 : 1} z={2} />
        ))}
        <Part src={S("face/nose")} w={162} h={106} x={FACE.nose.x} y={FACE.nose.y} scale={noseScale} z={4} />
        <Part src={S(`face/${mouthName}`)} w={mw} h={mh} x={FACE.nose.x} y={FACE.mouth.top} px={0.5} py={0} scale={mouthScale} z={3} />
      </div>
    </div>
  );
};
