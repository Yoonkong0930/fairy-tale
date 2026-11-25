import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { ToriEmotion, ToriRig } from "../characters/ToriRig";

/** 리그 확인용: 감정 6종을 2초씩 + 말하기 + 손 흔들기 */
export const RigTest: React.FC<{ still?: ToriEmotion }> = ({ still }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const order: ToriEmotion[] = ["neutral", "happy", "angry", "sad", "scared", "surprised"];
  const seg = Math.floor(f / (fps * 2));
  const emotion = still ?? (seg < order.length ? order[seg] : "neutral");
  const talking = !still && seg >= order.length && seg < order.length + 2;
  const waving = !still && seg >= order.length + 2;
  const label: Record<ToriEmotion, string> = { neutral: "평소", happy: "기쁨", angry: "화남", sad: "슬픔", scared: "무서움", surprised: "놀람" };

  return (
    <AbsoluteFill style={{ background: "linear-gradient(#CDEBFF 0%, #EEF8FF 62%, #B5DE96 62%, #9CD07E 100%)" }}>
      <ToriRig x={960} y={1000} size={0.82} emotion={emotion} talking={talking} waving={waving} />
      <div style={{ position: "absolute", top: 40, left: 60, fontSize: 56, fontWeight: 800, fontFamily: "sans-serif", color: "#4E2313" }}>
        {talking ? "말하기" : waving ? "안녕~" : label[emotion]}
      </div>
    </AbsoluteFill>
  );
};
