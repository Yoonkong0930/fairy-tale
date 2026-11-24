import { interpolate, spring } from "remotion";

/** 숨쉬기/둥실거림 같은 반복 움직임 (-1 ~ 1) */
export const wave = (frame: number, periodFrames: number, phase = 0) =>
  Math.sin(((frame + phase) / periodFrames) * Math.PI * 2);

/** 일정 간격으로 눈 깜빡임. 0 = 뜬 눈, 1 = 감은 눈 */
export const blink = (frame: number, every = 90, offset = 0) => {
  const t = (frame + offset) % every;
  return interpolate(t, [0, 3, 6], [0, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
};

/** 통통 튀며 등장 (0 → 1) */
export const popIn = (frame: number, fps: number, delay = 0) =>
  spring({ frame: frame - delay, fps, config: { damping: 9, stiffness: 120 } });
