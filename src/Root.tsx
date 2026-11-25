import { Composition } from "remotion";
import { TestScene } from "./scenes/TestScene";
import { RigTest } from "./scenes/RigTest";
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
      {/* 진짜 도토 리그 확인: 감정 6종(각 2초) → 말하기 → 손 흔들기 */}
      <Composition id="RigTest" component={RigTest} durationInFrames={FPS * 18} fps={FPS} width={WIDTH} height={HEIGHT} />
    </>
  );
};
