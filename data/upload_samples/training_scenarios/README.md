# Synthetic model-training scenarios

These CSVs are deterministic, synthetic examples that match the project's upload-training schema. Each has a distinct student-population pattern and includes both `Placed` and `Not Placed` outcomes.

| File | Scenario | Rows |
|---|---|---:|
| `01_general_balanced.csv` | General balanced cohort | 1,200 |
| `02_academic_support_cohort.csv` | Lower academic readiness and attendance | 1,350 |
| `03_internship_focused_cohort.csv` | More internships; experience is more predictive | 1,500 |
| `04_placement_training_cohort.csv` | High participation in placement training | 1,650 |
| `05_attendance_challenge_cohort.csv` | Lower attendance; attendance is more predictive | 1,800 |
| `06_technical_skills_cohort.csv` | Stronger technical-skill signal | 1,950 |
| `07_project_portfolio_cohort.csv` | More projects and hackathons; portfolio is more predictive | 2,100 |
| `08_department_mix_shift.csv` | Different department proportions | 1,250 |
| `09_high_achievement_cohort.csv` | Higher academic and skill scores | 1,450 |
| `10_noisy_borderline_cohort.csv` | More uncertain outcomes and intentionally noisy labels | 1,750 |

These files are for upload-flow demonstrations and controlled experiments, not real student records or evidence of real placement outcomes. Uploading a file trains and saves a model for that upload; separate uploads are not accumulated or used to update the project's default model. Synthetic data alone cannot establish that a model will perform better on real students.
