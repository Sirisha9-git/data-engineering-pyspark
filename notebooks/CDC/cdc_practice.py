# Databricks notebook source
# input data sets.

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()

day1_data = [
    (1, "Alice", "Dallas", "2024-01-01"),
    (2, "Bob",   "Austin", "2024-01-01"),
    (3, "Charlie", "Houston", "2024-01-01")
]

day2_data = [
    (1, "Alice", "Frisco", "2024-01-02"),   # city changed
    (3, "Charlie", "Houston", "2024-01-02"),# same
    (4, "David", "Plano", "2024-01-02")     # new
    # customer_id =2 record is deleted on day 2
]

columns = ["customer_id", "name", "city", "updated_at"]

df_day1 = spark.createDataFrame(day1_data,columns)
df_day2 = spark.createDataFrame(day2_data,columns)

df_day1.show()
df_day2.show()

# COMMAND ----------

# Initial load of day1 data with SCD2 audit columns.
# initialize day 1 data with audit columns start_date, end_Date, is_current.

# day1 load on 2024-01-01
 # end date is null on first load
 # records are current when loaded
df_scd2_day1 = df_day1.withColumn("start_date",lit("2024-01-01")) \
                      .withColumn("end_date",lit(None).cast("string")) \
                      .withColumn("is_current",lit("True"))                

df_scd2_day1.show()

# COMMAND ----------

# Identify new/ updated/ unchanged record in day2 data.

# Bring everything from day2 and matching customer ids record from day 1.
# Result will all new, updated and unchanged records from day 2
df_join = df_day2.alias('s').join(df_day1.alias('t'),"customer_id","left")
df_join.show()

# COMMAND ----------

# Apply cdc tags to joined data.
# This tags day2 data as whether its an insert, update or unchanged record.

df_cdc = df_join.withColumn("tag",
                # when day1 city is null => new record in day 2. tagged as 'I'(Insert)
                when (col("t.city").isNull(),lit('I')) 

                # when day1 city and day 2 city are not same => updted record in day2 . Tagged as 'U' (Updated)
                .when (col("t.city") != col("s.city"), lit('U')) 

                # Rest of the records from join are new inserts in day 1. tagged as 'N' (No change)
                .otherwise(lit('N'))
        )

df_cdc.select (["customer_id","s.name","s.city","tag"]).show()

# COMMAND ----------

# filter only updates and inserts for SCD2 purpose.
# filtering cdc dataframe for updates and inserts only and adding SCD2 audit columns.

# updated and inserted dataset from day 2

df_updates = df_cdc.filter(col("tag").isin("U","I")) \
                    .select(col("customer_id"),
                            col("s.name").alias("name"),
                            col("s.city").alias("city"),
                            col("s.updated_at").alias("updated_at"),
                            col("tag")
                 )
                    
df_new_versions = df_updates.select("customer_id","name","city","updated_at") \
                    .withColumn("start_date",lit("2024-01-02")) \
                    .withColumn("end_date",lit(None).cast("String")) \
                    .withColumn("is_current",lit("True")) \
                    
df_updates.show()
df_new_versions.show()

# COMMAND ----------

# Update day1 data audit columns for deleted and changed records from day 2
# day1 closed dataset

df_to_close = df_scd2_day1.alias("t") \
            .join(df_updates.filter(col("tag") == lit("U")).alias("u"),"customer_id") \
            .select("customer_id","t.name","t.city","t.updated_at","t.start_date", 
                   lit("2024-01-02").alias("end_date"), 
                   lit("False").alias("is_current")
            )

df_to_close.show()

# COMMAND ----------

# unchanged dataset from day 1

df_unchanged = df_scd2_day1.alias("t").join(df_cdc.filter(col("tag") == lit("N")).alias("U") , "customer_id") \
.select("customer_id","t.name","t.city","t.updated_at","t.start_date","t.end_date","t.is_current")

df_unchanged.show()

# COMMAND ----------

# Final dataset 
# day1 unchaged + day1 closed + day2 updated+inserted

df_final = df_unchanged.unionByName(df_to_close).unionByName(df_new_versions).orderBy("customer_id")
df_final.show()

# COMMAND ----------

# when / otherwise

df_day2.show()

df_when_oth = df_day2.withColumn("area",when (col("city").isin("Dallas","Frisco","Plano"),lit("DFW area")) \
                                 .otherwise(lit("non DFW area"))
                            )

df_when_oth.show()  


# COMMAND ----------

# dropduplicates
df_dup = spark.createDataFrame([
    (1, "Alice", "Frisco", "2024-01-02"),
    (1, "Alice", "Frisco", "2024-01-02"),
    (3, "Charlie", "Houston", "2024-01-02")
], columns
)

df_dup.show()

#deduplicate by all columns:
df_unique = df_dup.dropDuplicates()
df_unique.show()

# deduplicate by customer_id only (keep first):
df_unique_cust = df_dup.dropDuplicates(["customer_id"])
df_unique_cust.show()

# COMMAND ----------

# Exploring currenttimestamp, lit

df_meta = df_day2.withColumn("current_ts",current_timestamp()) \
                .withColumn("source_system",lit("input feed"))

df_meta.show(truncate=False)

# COMMAND ----------

# remove duplicates using widow functions.
from pyspark.sql.window import Window
# input dataset
df_dup = spark.createDataFrame([
    (1, "Alice", "Dallas", "2024-01-01"),
    (1, "Alice", "Frisco", "2024-01-02"),
    (1, "Alice", "Frisco", "2024-01-03"),
    (3, "Charlie", "Houston", "2024-01-01")
],columns)

df_dup.show()

w = Window.partitionBy("customer_id").orderBy(col("updated_at").desc())

df_unique = df_dup.withColumn("rownum",row_number().over(w)) \
                    .filter(col("rownum") == lit(1)) \
                    .drop("rownum")

df_unique.show()