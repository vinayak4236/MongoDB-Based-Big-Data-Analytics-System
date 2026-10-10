import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import BASE_DIR


def generate_student_dataset(rows: int = 10000, output_path: str | Path | None = None, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    departments = ["CSE", "ECE", "EEE", "MECH", "IT", "CIVIL"]
    genders = ["Male", "Female", "Other"]
    dept_weights = [0.30, 0.22, 0.14, 0.12, 0.14, 0.08]

    target_path = Path(output_path) if output_path else BASE_DIR / "data" / "students.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for student_id in range(1, rows + 1):
        department = random.choices(departments, weights=dept_weights, k=1)[0]
        gender = random.choice(genders)

        dept_bias = {
            "CSE": 0.45,
            "IT": 0.35,
            "ECE": 0.20,
            "EEE": 0.10,
            "MECH": -0.10,
            "CIVIL": -0.20,
        }[department]

        tenth = float(np.clip(np.random.normal(86, 7), 55, 99))
        twelfth = float(np.clip(np.random.normal(84 + dept_bias * 4, 8), 50, 99))
        cgpa = float(np.clip(np.random.normal(7.8 + dept_bias * 0.5, 0.8), 4.0, 9.8))
        attendance = float(np.clip(np.random.normal(82 + dept_bias * 4, 10), 50, 99))
        backlogs = int(np.clip(np.random.poisson(0.8), 0, 6))

        programming_score = int(np.clip(np.random.normal(78 + dept_bias * 10, 12), 30, 100))
        aptitude_score = int(np.clip(np.random.normal(74 + dept_bias * 8, 12), 25, 100))
        communication_score = int(np.clip(np.random.normal(72 + dept_bias * 6, 12), 25, 100))
        projects = int(np.clip(np.random.poisson(1.8 + dept_bias), 0, 8))
        internships = int(np.clip(np.random.poisson(0.9 + max(dept_bias, 0)), 0, 5))
        certifications = int(np.clip(np.random.poisson(1.5), 0, 8))
        hackathons = int(np.clip(np.random.poisson(0.7), 0, 6))
        extracurricular_score = int(np.clip(np.random.normal(68, 18), 20, 100))
        placement_training = int(random.random() < 0.72)

        academic_signal = (
            (cgpa * 7)
            + (attendance * 0.5)
            + (programming_score * 0.7)
            + (aptitude_score * 0.5)
            + (communication_score * 0.4)
            + (projects * 4)
            + (internships * 8)
            + (certifications * 3)
            + (hackathons * 3)
            + (placement_training * 12)
            - (backlogs * 9)
        )

        placement_probability = 1 / (1 + np.exp(-(0.18 * academic_signal - 3.5 + dept_bias * 0.8)))
        placement_probability = float(np.clip(placement_probability, 0.08, 0.92))
        placement_status = "Placed" if random.random() < placement_probability else "Not Placed"

        if placement_status == "Placed":
            salary = round(float(np.random.uniform(4.5, 22.0) + (cgpa - 7) * 1.2 + (programming_score - 70) * 0.08), 2)
        else:
            salary = 0.0

        records.append(
            {
                "student_id": f"STU{student_id:05d}",
                "department": department,
                "semester": random.randint(1, 8),
                "gender": gender,
                "10th_percentage": round(tenth, 2),
                "12th_percentage": round(twelfth, 2),
                "cgpa": round(cgpa, 2),
                "attendance": round(attendance, 2),
                "backlogs": backlogs,
                "programming_score": programming_score,
                "aptitude_score": aptitude_score,
                "communication_score": communication_score,
                "projects": projects,
                "internships": internships,
                "certifications": certifications,
                "hackathons": hackathons,
                "extracurricular_score": extracurricular_score,
                "placement_training": placement_training,
                "placement_status": placement_status,
                "salary": salary,
            }
        )

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(target_path, index=False)
    return target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic student dataset for the analytics project.")
    parser.add_argument("--rows", type=int, default=10000, help="Total number of student records to generate.")
    parser.add_argument("--output", type=str, default=str(BASE_DIR / "data" / "students.csv"), help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    result = generate_student_dataset(rows=args.rows, output_path=args.output, seed=args.seed)
    print(f"Dataset generated: {result} ({args.rows} rows)")
