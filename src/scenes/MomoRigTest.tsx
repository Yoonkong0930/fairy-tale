import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { MomoEmotion, MomoRig } from "../characters/MomoRig";
import { ToriRig } from "../characters/ToriRig";

/** 모모 리그 확인: 감정 6종(각 2초) → 말하기 → 손 흔들기. 옆에 토리를 세워 크기 비교 */
export const MomoRigTest: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const order: MomoEmotion[] = ["neutral", "happy", "worried", "sad", "scared", "surprised"];
  const seg = Math.floor(f / (fps * 2));
  const emotion = seg < order.length ? order[seg] : "neutral";
  const talking = seg >= order.length && seg < order.length + 2;
  const waving = seg >= order.length + 2;
  const label: Record<MomoEmotion, string> = { neutral: "평소", happy: "기쁨", worried: "걱정", sad: "슬픔", scared: "무서움", surprised: "놀람" };
  return (
    <AbsoluteFill style={{ background: "linear-gradient(#CDEBFF 0%, #EEF8FF 62%, #B5DE96 62%, #9CD07E 100%)" }}>
      <ToriRig x={620} y={1000} size={0.72} />
      <MomoRig x={1300} y={1000} size={0.72} emotion={emotion} talking={talking} waving={waving} />
      <div style={{ position: "absolute", top: 40, left: 60, fontSize: 56, fontWeight: 800, fontFamily: "sans-serif", color: "#4E2313" }}>
        모모 · {talking ? "말하기" : waving ? "안녕~" : label[emotion]}
      </div>
    </AbsoluteFill>
  );
};
