# The Oxygen Gap

**Pulse oximeter bias and hospital triage** · Mahika Bhan · Datathon for Social Good 2026, Advanced Track

Pulse oximeters overestimate oxygen levels in patients with darker skin. That reading feeds early-warning scores like NEWS2, which decide who gets urgent care. This project measures the oximeter's error by skin tone on real paired data, pushes that error through NEWS2 and machine learning triage models, finds how much bias it takes to change who gets flagged, and tests a second check that does not use the oximeter or race.

**Live ward view:** https://mbhan07.github.io/fa26-datathon

## Data
| Dataset | Role |
|---|---|
| **Health Risk** (provided): 1,000 patients, 7 vital signs, risk level | The triage system being audited |
| **OpenOximetry Repository** (UCSF Hypoxia Lab, PhysioNet): 44,085 oximeter readings paired with arterial blood gas, 114 participants, skin tone measured on the Monk scale | How wrong the oximeter is, by skin tone and true oxygen level |
| **NEWS2** (Royal College of Physicians, 2017) | Published scoring rule, used as the baseline and to audit labels |

**How they are joined:** the datasets share no patients. They are linked on true oxygen level. OpenOximetry gives an error distribution for each skin-tone group and oxygen band; each Health Risk patient's SpO₂ (treated as true oxygen) is matched to its band, given simulated oximeter readings, and re-scored.

OpenOximetry is under the PhysioNet Restricted Health Data License, so its raw data is **not** in this repository. Only aggregate error percentiles are shared (`results/openox_error_model.json`).

## Reproduce
```
pip install -r requirements.txt
bash run_all.sh                       # every number in the slides, ~1 minute
bash run_all.sh /path/to/openox       # also refits the error model (needs PhysioNet access)
```

| Step | Script | Output |
|---|---|---|
| 1. Label audit | `src/label_audit.py` | `results/label_audit.txt`, `label_mismatches.csv` |
| 2. Oximeter error model | `src/openox_error_model.py` | `results/openox_error_model.json` |
| 3. Models vs. baselines | `src/models.py` | `results/model_results.csv` |
| 4. Bias stress test | `src/stress_test.py` | `results/stress_test_summary.csv` |
| 5. Where the bias matters | `src/where_bias_bites.py` | `results/same_oxygen_different_alarm.csv`, `oxygen_dependent.csv` |
| 6. Linked table | `src/linked_table.py` | `results/linked_patients.csv` |
| 7. Tipping point | `src/tipping_point.py` | `results/tipping_point.csv` |
| 8. Subgroups and failures | `src/subgroups.py` | `results/subgroup_performance.csv`, `subgroup_bias.csv`, `failure_examples.md` |
| 9. Ward view | `src/build_ward_view.py` | `docs/index.html` |

## Key results
- **The bias replicates.** Hidden low oxygen (true SaO₂ < 88% while the oximeter reads 92–96%): 3.3% light skin, 5.1% medium, 9.3% dark.
- **The rule beats ML on clean data.** NEWS2 macro-F1 0.98 and high-risk recall 1.00 on 200 held-out patients; gradient boosting 0.95 / 0.95; majority-class baseline 0.12.
- **13 dataset labels are wrong.** All are scale-2 patients on room air scored as if on oxygen; a sample was checked by hand.
- **Oximeter error hides about 1 in 5 medium-risk patients** (≈55 of 293), for every skin tone.
- **At lab-measured bias, triage did not differ by skin tone.** Half a point of extra bias on darker skin opens a gap; two points doubles the medium-risk patients hidden (116 vs. 54 per 1,000).
- **A second check without SpO₂** recovers 79% of hidden medium-risk patients and flags 11% of the ward. It is weakest at 88–91% oxygen.

## Advanced track checklist
| Requirement | Where |
|---|---|
| Model | Gradient boosting, logistic regression, no-SpO₂ second check (`src/models.py`, `src/stress_test.py`) |
| Baseline | Majority class and published NEWS2 rule (`src/news2.py`) |
| Evaluation | Macro-F1, high-risk recall, weighted kappa on a 200-patient held-out test set; tuning by cross-validation on training data only |
| Error and fairness analysis | Skin-tone error profiles, true oxygen level, SpO₂ scale, supplemental oxygen, consciousness; four failure examples (`results/failure_examples.md`) |
| Data decisions | Join on true oxygen band; labels audited against NEWS2; sample scored by hand (P0591, P0644, P0607); one join row (P0600) checked by hand |
| Reproducibility | `run_all.sh`, raw data to every reported number |

## Limitations
OpenOximetry participants are healthy volunteers in a lab, and the dark-skin group has 15 participants. Health Risk is a teaching dataset with rule-generated labels and no blood gas; its SpO₂ is itself an oximeter reading, so results likely understate the problem. This is a simulation, not a clinical validation.

## Sources
- Sjoding et al. Racial bias in pulse oximetry measurement. *N Engl J Med* 2020;383:2477–78.
- Bickler, Feiner, Severinghaus. Effects of skin pigmentation on pulse oximeter accuracy at low saturation. *Anesthesiology* 2005;102:715–9.
- Royal College of Physicians. National Early Warning Score (NEWS) 2. 2017.
- OpenOximetry Repository, PhysioNet: https://physionet.org/content/openox-repo/
