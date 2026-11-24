import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Doto, Emotion } from "../characters/Doto";
import { popIn } from "../lib/anim";

/**
 * 파이프라인 테스트 장면
 * 0~2초: 도토가 통통 튀어나와 손 흔들며 인사
 * 2~4초: 말하기 (자막)
 * 4~6초: 감정 표정 순서대로 보여주기
 */
export const TestScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const enter = popIn(frame, fps, 5);
  const emotions: Emotion[] = ["happy", "surprised", "sad", "angry", "scared"];
  const emotion: Emotion =
    frame < fps * 4 ? "happy" : emotions[Math.floor((frame - fps * 4) / 12) % emotions.length];

  const subtitle =
    frame < fps * 2 ? "" : frame < fps * 4 ? "안녕! 나는 몽글숲의 아기다람쥐 도토야!" : "기쁠 때, 놀랄 때, 슬플 때…";

  return (
    <AbsoluteFill style={{ background: "linear-gradient(#BFE6FF 0%, #E9F7FF 60%, #A8D98A 60%, #8CC870 100%)" }}>
      {/* 임시 배경: 언덕 + 나무 */}
      <div style={{ position: "absolute", left: 180, top: 260, width: 220, height: 220, borderRadius: "50%", background: "#6DB65B" }} />
      <div style={{ position: "absolute", left: 270, top: 460, width: 40, height: 200, background: "#8A5A35", borderRadius: 12 }} />
      <div style={{ position: "absolute", right: 220, top: 300, width: 180, height: 180, borderRadius: "50%", background: "#79C065" }} />
      <div style={{ position: "absolute", right: 290, top: 470, width: 34, height: 170, background: "#8A5A35", borderRadius: 12 }} />

      <div style={{ transform: `scale(${enter})`, transformOrigin: "960px 800px", position: "absolute", inset: 0 }}>
        <Doto
          x={960}
          y={800}
          scale={1.25}
          emotion={emotion}
          waving={frame < fps * 2}
          talking={frame >= fps * 2 && frame < fps * 4}
        />
      </div>

      {subtitle && (
        <div
          style={{
            position: "absolute",
            bottom: 60,
            width: "100%",
            textAlign: "center",
            fontSize: 64,
            fontWeight: 800,
            fontFamily: "sans-serif",
            color: "#5A3418",
            textShadow: "0 0 12px white, 0 0 12px white",
            opacity: interpolate(frame % (fps * 2), [0, 8], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          {subtitle}
        </div>
      )}
    </AbsoluteFill>
  );
};
