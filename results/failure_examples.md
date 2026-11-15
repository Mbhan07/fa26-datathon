# Failure examples

## 1. The ground truth is wrong (label audit)
**P0591**: RR 16, SpO2 93% (scale 2, room air), SBP 109, HR 92, T 39.1, A
Dataset label: **Medium**. Published NEWS2: **Low** (total 4).
Scale-2 patients on room air with SpO2 >= 93 should score 0 for oxygen; the labels score them as if on oxygen. Any model trained on these labels learns to over-alarm for this group.

## 2. Same patient, different alarm (oximeter bias)
**P0600**: RR 20, SpO2 88% (scale 1, room air), SBP 134, HR 108, T 38.0, A. True NEWS2 risk: **Medium**.
Chance the oximeter reading downgrades this patient: **14%** under the light-skin error profile vs **27%** under the dark-skin profile.
No-SpO2 second check predicts **Medium**, so it would flag a downgrade to Low.

## 3. The ML model misses a High-risk patient the rule catches
**P0774**: RR 20, SpO2 89% (scale 1, room air), SBP 83, HR 99, T 39.2, P
Label: High (NEWS2 total 12). Gradient boosting predicts **Medium**. Out of 279 High-risk patients, the model misses 7; NEWS2 misses 0.

## 4. The second check fails
**P0440**: RR 21, SpO2 93% (scale 1, room air), SBP 105, HR 86, T 37.6, A. True risk Medium.
Chance of downgrade under dark-skin profile: **70%**. The no-SpO2 model predicts **Low**, so it would not flag the downgrade: this patient's risk lives almost entirely in the oxygen reading, which is exactly the number the second check ignores.

