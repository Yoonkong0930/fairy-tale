import { useCurrentFrame } from "remotion";
import { blink, wave } from "../lib/anim";

/**
 * 도토 (아기다람쥐, 주인공) — 임시 리그
 *
 * 지금은 단순한 도형으로 그린 "자리표시" 캐릭터입니다.
 * 파이어플라이로 만든 파츠 PNG가 준비되면 각 <g> 안의 도형을
 * <Img src={staticFile("assets/characters/doto/<파츠>.png")} /> 로 바꾸기만 하면
 * 움직임 코드는 그대로 재사용됩니다.
 *
 * 각 파츠는 "회전 기준점(pivot)"을 중심으로 움직입니다.
 */

export type Emotion = "happy" | "sad" | "angry" | "scared" | "surprised" | "neutral";

type Props = {
  x: number;
  y: number;
  scale?: number;
  emotion?: Emotion;
  talking?: boolean;
  waving?: boolean;
};

const C = {
  fur: "#D9894A",
  furDark: "#B86A32",
  belly: "#FBE3C4",
  line: "#5A3418",
  cheek: "#F7A6A0",
  acorn: "#8FC46B",
};

export const Doto: React.FC<Props> = ({
  x,
  y,
  scale = 1,
  emotion = "neutral",
  talking = false,
  waving = false,
}) => {
  const frame = useCurrentFrame();

  // 기본 생명감: 숨쉬기 + 꼬리 살랑 + 눈 깜빡임
  const breathe = 1 + wave(frame, 60) * 0.015;
  const tailSwing = wave(frame, 45) * 8;
  const headTilt = wave(frame, 80) * 3;
  const eyeClose = blink(frame, 100, 20);

  // 감정별 변화
  const scared = emotion === "scared" ? wave(frame, 4) * 2 : 0; // 덜덜
  const sad = emotion === "sad";
  const bounce = emotion === "happy" ? Math.abs(wave(frame, 20)) * -14 : 0;

  // 말할 때 입 벌림 (0~1)
  const mouthOpen = talking ? Math.max(0, wave(frame, 6)) : 0;

  // 손 흔들기
  const armAngle = waving ? -110 + wave(frame, 10) * 20 : -10;

  return (
    <svg
      width={400 * scale}
      height={480 * scale}
      viewBox="-200 -260 400 480"
      style={{
        position: "absolute",
        left: x - 200 * scale + scared,
        top: y - 380 * scale + bounce,
        overflow: "visible",
      }}
    >
      {/* 꼬리: 엉덩이 쪽이 pivot */}
      <g transform={`translate(70 140) rotate(${tailSwing})`}>
        <path
          d="M0 0 C 90 -10, 130 -120, 70 -200 C 30 -250, -30 -220, 0 -170 C 40 -120, 20 -60, -20 -20 Z"
          fill={C.furDark}
          stroke={C.line}
          strokeWidth={5}
        />
        <path d="M30 -60 C 60 -100, 70 -150, 50 -190" stroke={C.belly} strokeWidth={14} fill="none" strokeLinecap="round" />
      </g>

      {/* 몸통 */}
      <g transform={`translate(0 120) scale(1 ${breathe})`}>
        <ellipse cx={0} cy={0} rx={95} ry={90} fill={C.fur} stroke={C.line} strokeWidth={5} />
        <ellipse cx={0} cy={15} rx={60} ry={62} fill={C.belly} />
        {/* 발 */}
        <ellipse cx={-45} cy={85} rx={32} ry={16} fill={C.furDark} stroke={C.line} strokeWidth={4} />
        <ellipse cx={45} cy={85} rx={32} ry={16} fill={C.furDark} stroke={C.line} strokeWidth={4} />
      </g>

      {/* 왼팔 (화면 기준 왼쪽): 들고 있는 도토리 */}
      <g transform="translate(-70 80) rotate(15)">
        <ellipse cx={0} cy={40} rx={20} ry={40} fill={C.fur} stroke={C.line} strokeWidth={4} />
        <ellipse cx={5} cy={85} rx={18} ry={20} fill={C.acorn} stroke={C.line} strokeWidth={3} />
      </g>

      {/* 오른팔: 손 흔들기, 어깨가 pivot */}
      <g transform={`translate(70 80) rotate(${armAngle})`}>
        <ellipse cx={0} cy={40} rx={20} ry={40} fill={C.fur} stroke={C.line} strokeWidth={4} />
      </g>

      {/* 머리: 목이 pivot */}
      <g transform={`translate(0 30) rotate(${headTilt + (sad ? 8 : 0)}) translate(0 -110)`}>
        {/* 귀 */}
        <path d="M-95 -60 L-70 -150 L-35 -85 Z" fill={C.fur} stroke={C.line} strokeWidth={5} strokeLinejoin="round" />
        <path d="M95 -60 L70 -150 L35 -85 Z" fill={C.fur} stroke={C.line} strokeWidth={5} strokeLinejoin="round" />
        <ellipse cx={0} cy={0} rx={120} ry={105} fill={C.fur} stroke={C.line} strokeWidth={5} />
        <ellipse cx={0} cy={35} rx={70} ry={55} fill={C.belly} />

        {/* 볼터치 */}
        <ellipse cx={-75} cy={30} rx={20} ry={12} fill={C.cheek} opacity={0.8} />
        <ellipse cx={75} cy={30} rx={20} ry={12} fill={C.cheek} opacity={0.8} />

        {/* 눈 */}
        <Eye cx={-45} close={eyeClose} emotion={emotion} side={-1} />
        <Eye cx={45} close={eyeClose} emotion={emotion} side={1} />

        {/* 코 */}
        <ellipse cx={0} cy={18} rx={12} ry={8} fill={C.line} />

        {/* 입 */}
        <Mouth emotion={emotion} open={mouthOpen} />
      </g>
    </svg>
  );
};

const Eye: React.FC<{ cx: number; close: number; emotion: Emotion; side: 1 | -1 }> = ({
  cx,
  close,
  emotion,
  side,
}) => {
  const big = emotion === "surprised" || emotion === "scared";
  const r = big ? 24 : 20;
  // 화남: 눈썹이 안쪽으로 내려감, 슬픔: 바깥쪽으로 내려감
  const browTilt = emotion === "angry" ? 18 * side * -1 : emotion === "sad" ? -15 * side * -1 : 0;
  return (
    <g>
      <g transform={`translate(${cx} -15) scale(1 ${1 - close * 0.9})`}>
        <ellipse cx={0} cy={0} rx={r * 0.85} ry={r} fill={C.line} />
        <circle cx={-6} cy={-8} r={7} fill="white" />
      </g>
      <g transform={`translate(${cx} -55) rotate(${browTilt})`}>
        <rect x={-18} y={-4} width={36} height={8} rx={4} fill={C.line} />
      </g>
    </g>
  );
};

const Mouth: React.FC<{ emotion: Emotion; open: number }> = ({ emotion, open }) => {
  if (open > 0.05) {
    return <ellipse cx={0} cy={50} rx={18} ry={6 + open * 16} fill="#8A2E2E" stroke={C.line} strokeWidth={4} />;
  }
  switch (emotion) {
    case "happy":
      return <path d="M-25 40 Q 0 75, 25 40 Z" fill="#8A2E2E" stroke={C.line} strokeWidth={4} strokeLinejoin="round" />;
    case "sad":
      return <path d="M-20 60 Q 0 40, 20 60" fill="none" stroke={C.line} strokeWidth={5} strokeLinecap="round" />;
    case "angry":
      return <path d="M-20 55 L 20 55" stroke={C.line} strokeWidth={6} strokeLinecap="round" />;
    case "surprised":
    case "scared":
      return <ellipse cx={0} cy={52} rx={12} ry={16} fill="#8A2E2E" stroke={C.line} strokeWidth={4} />;
    default:
      return <path d="M-18 45 Q 0 60, 18 45" fill="none" stroke={C.line} strokeWidth={5} strokeLinecap="round" />;
  }
};
