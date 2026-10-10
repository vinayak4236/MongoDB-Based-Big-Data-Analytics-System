from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

NUMERIC_COLUMNS = [
    "10th_percentage",
    "12th_percentage",
    "cgpa",
    "attendance",
    "backlogs",
    "programming_score",
    "aptitude_score",
    "communication_score",
    "projects",
    "internships",
    "certifications",
    "hackathons",
    "extracurricular_score",
    "salary",
    "semester",
]

CATEGORICAL_COLUMNS = ["department", "gender", "placement_status"]


def clean_student_dataframe(df: DataFrame) -> DataFrame:
    cleaned = df.dropDuplicates()

    for column in ["department", "gender"]:
        cleaned = cleaned.withColumn(column, F.when(F.col(column).isNull(), "Unknown").otherwise(F.trim(F.col(column))))

    cleaned = cleaned.withColumn("placement_status", F.when(F.col("placement_status").isNull(), "Not Placed").otherwise(F.col("placement_status")))
    cleaned = cleaned.withColumn("department", F.trim(F.col("department")))
    cleaned = cleaned.withColumn("gender", F.trim(F.col("gender")))

    for column in NUMERIC_COLUMNS:
        cleaned = cleaned.withColumn(column, F.when(F.col(column).isNull(), 0).otherwise(F.col(column).cast("double")))

    cleaned = cleaned.withColumn("salary", F.when(F.col("salary") < 0, 0.0).otherwise(F.col("salary")))
    cleaned = cleaned.withColumn("is_placed", F.when(F.col("placement_status") == "Placed", 1).otherwise(0).cast("int"))
    cleaned = cleaned.withColumn("semester", F.col("semester").cast("int"))

    return cleaned


def detect_invalid_values(df: DataFrame):
    checks = {
        "cgpa": lambda c: (F.col(c) < 0) | (F.col(c) > 10),
        "attendance": lambda c: (F.col(c) < 0) | (F.col(c) > 100),
        "10th_percentage": lambda c: (F.col(c) < 0) | (F.col(c) > 100),
        "12th_percentage": lambda c: (F.col(c) < 0) | (F.col(c) > 100),
        "semester": lambda c: (F.col(c) < 1) | (F.col(c) > 8),
        "salary": lambda c: F.col(c) < 0,
    }

    invalid_counts = {}
    for column, predicate in checks.items():
        invalid_counts[column] = df.filter(predicate(column)).count()

    return invalid_counts


def encode_categorical_columns(df: DataFrame, categorical_columns: list[str] | None = None):
    selected_columns = categorical_columns or CATEGORICAL_COLUMNS
    encoded_df = df

    for column in selected_columns:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_indexed", handleInvalid="keep")
        encoded_df = indexer.fit(encoded_df).transform(encoded_df)

        encoder = OneHotEncoder(inputCols=[f"{column}_indexed"], outputCols=[f"{column}_encoded"], handleInvalid="keep")
        encoded_df = encoder.fit(encoded_df).transform(encoded_df)

    return encoded_df
