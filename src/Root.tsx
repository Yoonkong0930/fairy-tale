import { Composition } from "remotion";
import { TestScene } from "./scenes/TestScene";
import { RigTest } from "./scenes/RigTest";
import { ToriDemo } from "./scenes/ToriDemo";
import { MomoRigTest } from "./scenes/MomoRigTest";
import { RigCompare } from "./scenes/RigCompare";
import { FPS, HEIGHT, WIDTH } from "./lib/video";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* 파이프라인 테스트용 장면: 임시 다람쥐가 인사하는 6초짜리 클립 */}
      <Composition
        id="TestScene"
        component={TestScene}
        durationInFrames={FPS * 6}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
      {/* 진짜 토리 리그 확인: 감정 6종(각 2초) → 말하기 → 손 흔들기 */}
      <Composition id="RigTest" component={RigTest} durationInFrames={FPS * 18} fps={FPS} width={WIDTH} height={HEIGHT} />
      {/* 걷기 + 인사 + 음성 입모양 데모 */}
      <Composition id="ToriDemo" component={ToriDemo} durationInFrames={FPS * 19} fps={FPS} width={WIDTH} height={HEIGHT} />
      {/* 모모 리그 확인 (토리와 크기 비교) */}
      <Composition id="MomoRigTest" component={MomoRigTest} durationInFrames={FPS * 18} fps={FPS} width={WIDTH} height={HEIGHT} />
      {/* 원본 설정화 비교용 정지 화면 */}
      <Composition id="RigCompare" component={RigCompare} durationInFrames={1} fps={FPS} width={WIDTH} height={HEIGHT} defaultProps={{ who: "tori" as const }} />
    </>
  );
};
