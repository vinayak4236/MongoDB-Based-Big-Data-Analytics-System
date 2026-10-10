from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "student-analytics", master: str = "local[*]") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def stop_spark_session(spark: SparkSession | None = None) -> None:
    if spark is not None:
        spark.stop()
