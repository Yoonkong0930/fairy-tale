import { Composition } from "remotion";
import { TestScene } from "./scenes/TestScene";
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
    </>
  );
};
