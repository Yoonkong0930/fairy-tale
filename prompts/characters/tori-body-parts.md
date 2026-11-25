# 토리 몸 파츠 시트 (리깅용, 정면)
- ref 1: art/final/tori-model-sheet.png (확정 디자인)
- ref 2: art/final/tori-expressions.png (머리 모양·볼터치 기준)
- 초록 단색 배경 → scripts/slice_parts.py 로 자름

배치 (4열 x 3행)
| 행 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 빈 얼굴 머리 | 왼쪽 귀 | 오른쪽 귀 | 애교머리 |
| 2 | 몸통 | 왼팔 | 오른팔 | 초커(띠만) |
| 3 | 왼다리 | 오른다리 | 꼬리 | 마음 도토리 펜던트 |

```prompt
Create a BODY PARTS SHEET for 2D cut-out rigging (like puppet animation) of the character "Tori", the Korean baby chipmunk in the reference images, FRONT VIEW. Every part must match the reference exactly: same colors (light warm tan fur, cream-white muzzle, cheeks and belly, dark-brown stripes), same bold smooth dark-brown outline, flat colors, same proportions and the same scale relative to each other as on the finished character.

Background: perfectly flat solid pure green (#00FF00) chroma-key background. No shadows, no gradients.

Layout: a neat grid of 4 columns x 3 rows. Each cell holds exactly ONE isolated part, centered, with generous green space around it. Parts never touch each other or the cell edges. No text, no labels.

Row 1:
 1. HEAD WITHOUT FACE FEATURES: the round head shape with the cream muzzle/cheek area, soft pink blush on the cheeks and the dark stripes on the forehead, but NO eyes, NO eyebrows, NO nose, NO mouth, NO ears and NO top tuft. The top of the head is a clean smooth curve.
 2. LEFT EAR only (small round ear with pink inner ear), with a little extra fur at its base for overlapping.
 3. RIGHT EAR only (mirror of the left ear).
 4. TOP TUFT only: the cute cowlick of 3 round puffy fur clumps, with a little extra fur at its base for overlapping.
Row 2:
 1. TORSO only: the small round body with the cream belly, including a short neck stub at the top. NO head, NO arms, NO legs, NO tail, NO collar.
 2. LEFT ARM only: short chubby arm hanging down, rounded hand, with a rounded shoulder end at the top for overlapping.
 3. RIGHT ARM only (mirror of the left arm).
 4. COLLAR BAND only: the snug soft brown collar as a short curved band, without the pendant.
Row 3:
 1. LEFT LEG only: short chubby leg with a small rounded foot, with a rounded hip end at the top for overlapping.
 2. RIGHT LEG only (mirror of the left leg).
 3. TAIL only: the big fluffy S-curved striped tail (dark-brown stripes, cream underside), with its base at the bottom-left.
 4. PENDANT only: the small mint-green glowing acorn-shaped pendant.
```
