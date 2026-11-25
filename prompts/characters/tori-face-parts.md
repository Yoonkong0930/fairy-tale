# 토리 얼굴 파츠 시트 (리깅용)
- ref 1: art/final/tori-expressions.png (확정 표정)
- ref 2: art/final/tori-model-sheet.png (확정 디자인)
- 초록 단색 배경 → 스크립트로 배경 제거 후 파츠별 PNG로 자름

배치 (4열 x 4행)
| 행 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 눈 | 뜬 눈 | 웃는 반달 눈 | 감은 눈(깜빡임) | 놀란 눈 |
| 2 눈썹 | 기본 | 화남 | 슬픔 | 걱정 |
| 3 입 | 기본(ω) | 활짝 웃음 | 말하기 작게 | 말하기 중간 |
| 4 입 | 말하기 크게 | 슬픔 | 화남(삐죽) | 놀람(o) |

```prompt
Create a FACE PARTS SHEET for rigging the character "Tori" (the Korean baby chipmunk in the reference images). Draw the parts exactly in the same style, line weight and colors as the reference: bold smooth dark-brown outlines, flat colors.

Background: a perfectly flat, solid pure green (#00FF00) chroma-key background filling the whole image. No shadows, no gradients, no texture on the background.

Layout: a neat grid of 4 columns x 4 rows. Each cell contains exactly ONE isolated part, centered, drawn at the same scale as it would appear on the character's face, with plenty of green space around it. Parts must not touch each other. No face, no fur, no skin behind the parts — only the part itself. No text, no labels.

Row 1 — single EYES (one eye only per cell, the character's left eye):
 1. open eye: large round glossy dark-brown pupil with exactly two white highlights — one large round highlight at the upper-left and one small round highlight at the lower-right.
 2. happy closed eye: an upward curved arc line (like ^).
 3. blink closed eye: a gentle downward curved line.
 4. surprised eye: white round eye with a thin dark outline and a small dark-brown pupil in the center.
Row 2 — single EYEBROWS (thin, short, dark brown, the character's left eyebrow):
 1. neutral: thin gentle arc.
 2. angry: thin line slanting down toward the center.
 3. sad: thin line slanting up toward the center.
 4. worried: thin wavy raised line.
Row 3 — MOUTHS (with the small dark-brown triangular nose on top of each mouth):
 1. neutral: small "ω"-shaped cat smile.
 2. happy: big open smile with a pink tongue.
 3. talking small: slightly open small mouth.
 4. talking medium: half-open rounded mouth showing a little pink tongue.
Row 4 — MOUTHS (with the same nose on top):
 1. talking big: wide open round mouth with pink tongue.
 2. sad: small downturned curve.
 3. angry: pouting wavy mouth.
 4. surprised: small round "o" mouth.
```
