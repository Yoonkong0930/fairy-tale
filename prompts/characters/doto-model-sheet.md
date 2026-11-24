# 도토 설정화(모델 시트) 프롬프트 (v3: 볼 줄임, 꼬리 키움, 팔 벌린 기본 자세)

목적: 도토의 기준 그림. 이후 모든 파츠·표정·포즈는 이 그림을 기준 이미지(--ref)로 넣어서 생성합니다.

포인트
- 2~4세 대상: 둥글고 단순한 형태, 큰 머리(2.5등신), 큰 눈
- 리깅용: 정면 / 3/4 측면 / 뒷모습, 팔다리가 몸에 가리지 않는 기본 자세
- 단순한 단색 채색 + 굵은 외곽선 (파츠 분리가 쉬움)
- 꼬리는 크지만 털 뭉치는 2~3개로 단순하게

```prompt
Character model sheet for a children's animated series for toddlers (ages 2-4).

Character: "Doto", a Korean baby chipmunk (Korean squirrel, "daramjwi"). Curious and brave, warm and friendly.
- Very round and soft shapes. Big round head, SMALL compact body (body clearly smaller than the head). Short little arms and legs.
- Chubby, soft round cheeks that puff out gently on both sides of the face (cute but moderate, not oversized; cheeks should NOT be visible from the back view).
- Eyes: large, perfectly round, glossy dark-brown eyes with two white highlights each. Thin, short, delicate eyebrows.
- Small round nose, tiny smiling mouth, soft pink blush on the cheeks.
- Fur: light warm tan / beige. Cream-white muzzle, cheeks and belly.
- Korean chipmunk markings: three thin darker-brown stripes on the top of the head and dark stripes running down the back.
- Small round ears (no ear tufts).
- BIG fluffy tail, about as large as the body, curling up behind the body in a clear S-curve and rising to head height, simplified into 2-3 soft clumps, with darker stripes. The tail should be clearly visible beside the body in the front view.
- Holds a small green acorn loosely in ONE hand at its side (the acorn must not cover the body).

Layout: three full-body views side by side on a plain white background — front view, three-quarter view, back view. Neutral standing T-pose-like rig pose: both arms hanging down and held slightly away from the body (small gap between arms and torso), legs slightly apart, so the arms, legs and tail do not overlap the torso. Same scale and same character in all three views.

Style: clean flat 2D vector illustration, bold smooth dark-brown outlines, flat colors with at most one soft shade, no texture, no gradients, no background scenery, no text. Cute, gentle, bright, like a premium preschool TV show.
```
