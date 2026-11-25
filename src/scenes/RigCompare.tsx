import { AbsoluteFill } from "remotion";
import { MomoRig } from "../characters/MomoRig";
import { ToriRig } from "../characters/ToriRig";

/** 원본 설정화와 비교용: 흰 배경, 정면, 평소 표정 (0프레임 = 숨쉬기·흔들림 0) */
export const RigCompare: React.FC<{ who: "tori" | "momo" }> = ({ who }) => (
  <AbsoluteFill style={{ background: "white" }}>
    {who === "tori" ? <ToriRig x={960} y={1040} size={0.9} /> : <MomoRig x={960} y={1040} size={0.9} />}
  </AbsoluteFill>
);
