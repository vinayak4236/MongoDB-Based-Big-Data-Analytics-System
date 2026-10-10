from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def get_basic_student_metrics(df: DataFrame):
    total_students = df.count()
    total_placed = df.filter(F.col("placement_status") == "Placed").count()
    total_not_placed = total_students - total_placed
    placement_rate = (total_placed / total_students * 100) if total_students else 0.0

    averages = df.agg(
        F.avg("cgpa").alias("avg_cgpa"),
        F.avg("salary").alias("avg_salary"),
    ).collect()[0]

    return {
        "total_students": total_students,
        "total_placed": total_placed,
        "total_not_placed": total_not_placed,
        "placement_rate": round(placement_rate, 2),
        "avg_cgpa": round(float(averages["avg_cgpa"] or 0.0), 2),
        "avg_salary": round(float(averages["avg_salary"] or 0.0), 2),
    }


def department_counts(df: DataFrame) -> DataFrame:
    return df.groupBy("department").count().orderBy(F.col("count").desc())


def department_average_cgpa(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("department")
        .agg(F.avg("cgpa").alias("avg_cgpa"))
        .orderBy(F.col("avg_cgpa").desc())
    )


def department_placement_rate(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("department")
        .agg(
            F.count("*").alias("total_students"),
            F.sum(F.when(F.col("placement_status") == "Placed", 1).otherwise(0)).alias("placed_students"),
        )
        .withColumn("placement_rate", F.col("placed_students") / F.col("total_students") * 100)
        .orderBy(F.col("placement_rate").desc())
    )


def placed_vs_non_placed(df: DataFrame) -> DataFrame:
    return df.groupBy("placement_status").count().orderBy(F.col("count").desc())


def attendance_by_placement(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("placement_status")
        .agg(F.avg("attendance").alias("avg_attendance"))
        .orderBy(F.col("avg_attendance").desc())
    )


def salary_by_department(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("department")
        .agg(F.avg("salary").alias("avg_salary"))
        .orderBy(F.col("avg_salary").desc())
    )


def correlation_pairs(df: DataFrame, numeric_columns: list[str]):
    result = []
    for index, left in enumerate(numeric_columns):
        for right in numeric_columns[index + 1 :]:
            corr_value = df.corr(left, right)
            result.append({"feature_1": left, "feature_2": right, "correlation": round(float(corr_value or 0.0), 4)})
    return result


def run_main_analytics(df: DataFrame, spark):
    df.createOrReplaceTempView("student_records")

    analytics = {
        "department_counts": spark.sql(
            "SELECT department, COUNT(*) AS total_students FROM student_records GROUP BY department ORDER BY total_students DESC"
        ),
        "department_avg_cgpa": spark.sql(
            "SELECT department, AVG(cgpa) AS avg_cgpa FROM student_records GROUP BY department ORDER BY avg_cgpa DESC"
        ),
        "department_placement_rate": spark.sql(
            "SELECT department, COUNT(*) AS total_students, SUM(CASE WHEN placement_status = 'Placed' THEN 1 ELSE 0 END) AS placed_students, AVG(CASE WHEN placement_status = 'Placed' THEN 1.0 ELSE 0.0 END) * 100 AS placement_rate FROM student_records GROUP BY department ORDER BY placement_rate DESC"
        ),
        "placed_vs_non_placed": spark.sql(
            "SELECT placement_status, COUNT(*) AS total_students FROM student_records GROUP BY placement_status ORDER BY total_students DESC"
        ),
        "avg_attendance_by_placement": spark.sql(
            "SELECT placement_status, AVG(attendance) AS avg_attendance FROM student_records GROUP BY placement_status ORDER BY avg_attendance DESC"
        ),
        "salary_by_department": spark.sql(
            "SELECT department, AVG(salary) AS avg_salary FROM student_records GROUP BY department ORDER BY avg_salary DESC"
        ),
        "cgpa_by_placement": spark.sql(
            "SELECT placement_status, AVG(cgpa) AS avg_cgpa FROM student_records "
            "GROUP BY placement_status ORDER BY avg_cgpa DESC"
        ),
        "internships_by_placement": spark.sql(
            "SELECT internships, COUNT(*) AS total_students, "
            "AVG(CASE WHEN placement_status = 'Placed' THEN 1.0 ELSE 0.0 END) * 100 AS placement_rate "
            "FROM student_records GROUP BY internships ORDER BY internships"
        ),
        "projects_by_placement": spark.sql(
            "SELECT projects, COUNT(*) AS total_students, "
            "AVG(CASE WHEN placement_status = 'Placed' THEN 1.0 ELSE 0.0 END) * 100 AS placement_rate "
            "FROM student_records GROUP BY projects ORDER BY projects"
        ),
        "certifications_by_placement": spark.sql(
            "SELECT certifications, COUNT(*) AS total_students, "
            "AVG(CASE WHEN placement_status = 'Placed' THEN 1.0 ELSE 0.0 END) * 100 AS placement_rate "
            "FROM student_records GROUP BY certifications ORDER BY certifications"
        ),
        "backlogs_by_placement": spark.sql(
            "SELECT backlogs, COUNT(*) AS total_students, "
            "AVG(CASE WHEN placement_status = 'Placed' THEN 1.0 ELSE 0.0 END) * 100 AS placement_rate "
            "FROM student_records GROUP BY backlogs ORDER BY backlogs"
        ),
    }

    analytics["basic_metrics"] = get_basic_student_metrics(df)
    return analytics
