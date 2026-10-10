import argparse
from pathlib import Path

from config.config import DEFAULT_CSV_PATH, MODEL_DIR, MONGODB_COLLECTION, MONGODB_DB_NAME, MONGODB_URI, OUTPUT_DIR
from data.generate_dataset import generate_student_dataset
from database.import_data import import_csv_to_mongodb
from spark.analytics import run_main_analytics
from spark.data_ingestion import load_student_dataframe_from_csv, load_student_dataframe_from_mongodb
from spark.preprocessing import clean_student_dataframe, detect_invalid_values, encode_categorical_columns
from spark.spark_session import get_spark_session, stop_spark_session
from ml.predict import predict_placement
from ml.train_models import train_and_save_models


def run_phase_one(rows: int = 10000, csv_path: str | Path = DEFAULT_CSV_PATH, import_to_mongodb: bool = False):
    csv_file = generate_student_dataset(rows=rows, output_path=csv_path)
    print(f"Generated dataset at: {csv_file}")

    if import_to_mongodb:
        import_result = import_csv_to_mongodb(
            csv_path=csv_file,
            uri=MONGODB_URI,
            db_name=MONGODB_DB_NAME,
            collection_name=MONGODB_COLLECTION,
        )
        print(f"MongoDB import result: {import_result}")

    return csv_file


def run_phase_two(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    use_mongo: bool = False,
    mongo_uri: str | None = MONGODB_URI,
    db_name: str | None = MONGODB_DB_NAME,
    collection_name: str | None = MONGODB_COLLECTION,
):
    spark = get_spark_session(app_name="student-analytics-phase2")
    try:
        if use_mongo:
            raw_df = load_student_dataframe_from_mongodb(
                uri=mongo_uri,
                db_name=db_name,
                collection_name=collection_name,
                spark=spark,
            )
        else:
            raw_df = load_student_dataframe_from_csv(csv_path, spark=spark)

        print(f"Raw DataFrame schema:")
        raw_df.printSchema()
        print(f"Raw row count: {raw_df.count()}")

        cleaned_df = clean_student_dataframe(raw_df)
        print(f"Cleaned row count: {cleaned_df.count()}")
        print("Invalid value detection:")
        print(detect_invalid_values(cleaned_df))
        transformed_df = encode_categorical_columns(cleaned_df, ["department", "gender"])

        analytics = run_main_analytics(cleaned_df, spark)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # CSV output keeps local Windows runs independent of Hadoop winutils.
        cleaned_df.toPandas().to_csv(OUTPUT_DIR / "cleaned_students.csv", index=False)
        transformed_df.toPandas().to_csv(OUTPUT_DIR / "transformed_students.csv", index=False)
        for name, result in analytics.items():
            if hasattr(result, "write"):
                result.toPandas().to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

        print("Basic metrics:")
        print(analytics["basic_metrics"])
        print("Department-wise student counts:")
        analytics["department_counts"].show(10, truncate=False)
        print("Department-wise placement rate:")
        analytics["department_placement_rate"].show(10, truncate=False)
        print("Placed vs not placed:")
        analytics["placed_vs_non_placed"].show(10, truncate=False)
        print("CGPA by placement status:")
        analytics["cgpa_by_placement"].show(10, truncate=False)
        print("Internships by placement:")
        analytics["internships_by_placement"].show(10, truncate=False)

        return cleaned_df, analytics
    finally:
        stop_spark_session(spark)


def run_phase_three(csv_path: str | Path = DEFAULT_CSV_PATH):
    metadata = train_and_save_models(csv_path=csv_path, model_dir=MODEL_DIR)
    print("Phase 3 model evaluation:")
    for name, metrics in metadata["models"].items():
        print(f"{name}: {metrics}")
    print(f"Selected model: {metadata['best_model']}")
    print(f"Saved model: {metadata['model_path']}")

    sample = {
        "department": "CSE",
        "semester": 8,
        "gender": "Female",
        "10th_percentage": 90,
        "12th_percentage": 88,
        "cgpa": 8.7,
        "attendance": 90,
        "backlogs": 0,
        "programming_score": 88,
        "aptitude_score": 84,
        "communication_score": 82,
        "projects": 3,
        "internships": 2,
        "certifications": 3,
        "hackathons": 1,
        "extracurricular_score": 75,
        "placement_training": 1,
    }
    print(f"Sample prediction: {predict_placement(sample)}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate student analytics dataset, import to MongoDB, and run Spark Phase 2 processing.")
    parser.add_argument("--rows", type=int, default=10000, help="Number of records to generate for Phase 1.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH), help="CSV path to read for Spark processing.")
    parser.add_argument("--import-to-mongodb", action="store_true", help="Import generated data into MongoDB before Spark processing.")
    parser.add_argument("--phase2", action="store_true", help="Run the Spark ingestion and analytics pipeline.")
    parser.add_argument("--from-mongo", action="store_true", help="Load student data from MongoDB instead of the CSV file.")
    parser.add_argument("--phase3", action="store_true", help="Train and evaluate placement prediction models.")
    args = parser.parse_args()

    if args.phase3:
        run_phase_three(csv_path=args.csv)
    elif args.phase2:
        run_phase_two(csv_path=args.csv, use_mongo=args.from_mongo)
    else:
        run_phase_one(rows=args.rows, csv_path=args.csv, import_to_mongodb=args.import_to_mongodb)
