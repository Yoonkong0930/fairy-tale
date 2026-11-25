import { useMemo } from "react";
import { useAudioData } from "@remotion/media-utils";
import { Img, staticFile } from "remotion";

/** 캐릭터 파츠 하나: pivot(px,py 비율)을 기준으로 위치·회전·배율 */
export type PartProps = {
  src: string;
  w: number;
  h: number;
  x: number;
  y: number;
  px?: number;
  py?: number;
  rotate?: number;
  scale?: number;
  scaleY?: number;
  /** 좌우 반전 (-1) */
  flipX?: number;
  z?: number;
};

export const Part: React.FC<PartProps> = ({ src, w, h, x, y, px = 0.5, py = 0.5, rotate = 0, scale = 1, scaleY = 1, flipX = 1, z }) => (
  <Img
    src={src}
    style={{
      position: "absolute",
      width: w,
      height: h,
      left: x - w * px,
      top: y - h * py,
      transformOrigin: `${px * 100}% ${py * 100}%`,
      transform: `rotate(${rotate}deg) scale(${scale * flipX}, ${scale * scaleY})`,
      zIndex: z,
    }}
  />
);

export type Voice = { src: string; from: number };

/** 음성 크기 → 입 크기 단계 (-1 음성 없음, 0 닫힘, 1 작게, 2 중간, 3 크게) */
export const useMouthLevel = (voice: Voice | undefined, frame: number, fps: number) => {
  const data = useAudioData(voice ? voice.src : staticFile("audio/silence.wav"));
  const rmsTrack = useMemo(() => {
    if (!voice || !data) return null;
    const x = data.channelWaveforms[0];
    const win = Math.floor(data.sampleRate / fps);
    const out: number[] = [];
    for (let i = 0; i + win <= x.length; i += win) {
      let sum = 0;
      for (let j = i; j < i + win; j++) sum += x[j] * x[j];
      out.push(Math.sqrt(sum / win));
    }
    const peak = Math.max(...out, 1e-6);
    return out.map((v) => v / peak);
  }, [data, voice, fps]);
  if (!voice || !rmsTrack) return -1;
  const i = frame - voice.from;
  if (i < 0 || i >= rmsTrack.length) return -1;
  const v = (rmsTrack[i] * 2 + (rmsTrack[i - 1] ?? 0) + (rmsTrack[i + 1] ?? 0)) / 4;
  return v < 0.08 ? 0 : v < 0.3 ? 1 : v < 0.6 ? 2 : 3;
};

/** 걷기 공통 값: 한 걸음 = 12프레임 */
export const walkCycle = (f: number, walking: boolean) => {
  const step = walking ? (f % 24) / 24 : 0;
  const stepWave = walking ? Math.sin(step * Math.PI * 2) : 0;
  return {
    stepWave,
    bob: walking ? -Math.abs(stepWave) * 18 : 0,
    liftL: walking ? Math.max(0, stepWave) * 34 : 0,
    liftR: walking ? Math.max(0, -stepWave) * 34 : 0,
  };
};
