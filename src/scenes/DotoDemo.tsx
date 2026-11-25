import { AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { DotoRig } from "../characters/DotoRig";

/**
 * 도토 데모: 걸어 들어와서 → 손 흔들고 → 음성에 맞춰 말하고 → 걸어 나감
 */
export const DotoDemo: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const S = (sec: number) => Math.round(sec * fps);

  const walkInEnd = S(3);
  const waveEnd = S(5);
  const voiceFrom = S(5.3);
  const voiceEnd = voiceFrom + S(10.1);
  const walkOutStart = voiceEnd + S(0.5);

  const x = interpolate(f, [0, walkInEnd, walkOutStart, walkOutStart + S(2.5)], [-250, 960, 960, -300], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const walking = f < walkInEnd || f >= walkOutStart;
  const facing = f >= walkOutStart ? -1 : 1;
  const waving = f >= walkInEnd && f < waveEnd;
  const speaking = f >= voiceFrom && f < voiceEnd;

  return (
    <AbsoluteFill style={{ background: "linear-gradient(#CDEBFF 0%, #EEF8FF 62%, #B5DE96 62%, #9CD07E 100%)" }}>
      <Sequence from={voiceFrom}>
        <Audio src={staticFile("audio/test-doto-hello.wav")} />
      </Sequence>
      <DotoRig
        x={x}
        y={1000}
        size={0.82}
        emotion={speaking || waving ? "happy" : "neutral"}
        walking={walking}
        facing={facing}
        waving={waving}
        voice={{ src: staticFile("audio/test-doto-hello.wav"), from: voiceFrom }}
      />
      {speaking && (
        <div style={{ position: "absolute", bottom: 50, width: "100%", textAlign: "center", fontSize: 60, fontWeight: 800, fontFamily: "sans-serif", color: "#4E2313", textShadow: "0 0 12px white, 0 0 12px white" }}>
          안녕! 나는 몽글숲의 아기다람쥐 도토야! 만나서 정말 반가워!
        </div>
      )}
    </AbsoluteFill>
  );
};
