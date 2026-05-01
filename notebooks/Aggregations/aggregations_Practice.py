# Databricks notebook source
# PySpark UDFs
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, pandas_udf, upper, col
from pyspark.sql.types import IntegerType, StringType
import pandas as pd

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([(1, "alice"), (2, "bob"), (3, "charlie")], ["id", "name"])

# ========== 1. BUILT-IN (BEST — always try first) ==========
df.select(upper(col("name"))).show()

# ========== 2. REGULAR UDF (SLOW — row by row) ==========
@udf(IntegerType())
def plus_one(x):
    return x + 1

df.select("id", plus_one("id").alias("id_plus")).show()

# ========== 3. PANDAS UDF (FAST — vectorized) ==========
@pandas_udf("int")
def pd_plus_one(s: pd.Series) -> pd.Series:
    return s + 1

df.select("id", pd_plus_one("id").alias("id_plus")).show()

# ========== 4. mapInPandas (FULL DF — can change row count) ==========
def filter_func(iterator):
    for pdf in iterator:
        yield pdf[pdf.id > 1]

df.mapInPandas(filter_func, schema=df.schema).show()

# COMMAND ----------

# Grouping data
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

schema = StructType([
    StructField("color",StringType()),
    StructField("fruit",StringType()),
    StructField("price1",DoubleType()),
    StructField("price2",DoubleType())
])

color_df = spark.createDataFrame([("red","apple",12.0,34.0),
                            ('blue',  'banana', 2.0, 20.0),
                            ['red',   'carrot', 3.0, 30.0],
                            ['blue',  'grape',  4.0, 40.0],
                            ['red',   'carrot', 5.0, 50.0],
                            ['black', 'carrot', 6.0, 60.0],
                            ['red',   'banana', 7.0, 70.0],
                            ['red',   'grape',  8.0, 80.0]
                        ], schema = schema)
color_df.show()

# COMMAND ----------

# Basic aggregation - DataFrame Level Aggregation.  
# groupBy
# agg() collapses each group to one row (e.g., one avg per subject)
from pyspark.sql.functions import (
    sum, avg, count, max, min, collect_list, collect_set,
    countDistinct, first, last, stddev, variance,
    col, when, lit)

color_df.groupBy('color').avg().show()
# Applies avg() to all numeric columns automatically

# Group + Aggregate
# Single aggregation.
color_df.agg(avg("price1")).show() # Overall using agg()
color_df.select(avg("price1")).show()  # Overall using select()
color_df.groupBy("color").agg(avg("price1")).show() # Average per group using agg()
color_df.groupBy("color").avg("price1").show()  # Average per group

# Multiple agggregations on single group by column.
color_df.groupBy("color").agg(
            avg("price1").alias("avg_price1"),
            sum("price2").alias("sum_price2"),
            count("fruit").alias("friut_count")
        ).show()
        
# Multiple aggregations on nultiple group by columns.
color_df.groupBy("color","fruit").agg(
    avg("price1").alias("avg_price1"),
    sum("price2").alias("sum_price2")
).show()

# COMMAND ----------

# conditional aggregation
# SQL CASE statement

color_df.groupBy("color").agg(
    sum(when(col("fruit")=='banana', col("price1")).otherwise(0)).alias("banana_price1"),
    count(when (col("fruit")=='carrot', True)).alias("carrot_count")
).show()

# COMMAND ----------

# pivot()
# Convert Rows to columns.

color_df.groupBy("color").pivot("fruit").avg("price1").show()

# with specific pivot values
color_df.groupBy('color').pivot('fruit', ['banana', 'carrot']).sum('price1').show()

# COMMAND ----------

# applyInPandas()
# Custom logic for each row in a group
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.getOrCreate()

# INPUT DATA
df = spark.createDataFrame([
    ['Math',    'Alice', 90],
    ['Math',    'Bob',   70],
    ['Math',    'Charlie', 80],
    ['Science', 'Alice', 60],
    ['Science', 'Bob',   80],
], schema=['subject', 'student', 'score'])
# df.show()
# find how much each student scored above/below their class average for each subject.
def diff_from_mean(pdf):
    pdf["score"] = pdf["score"]-pdf["score"].mean()
    return pdf
df.groupBy("subject").applyInPandas(diff_from_mean,schema=df.schema).show()

# COMMAND ----------

# cogroup.applyInPandas()
# Eg : You have orders and returns in separate tables. You want to match them per customer and calculate net revenue.
# TABLE 1: Orders
orders = spark.createDataFrame([
    ['Alice', 100],
    ['Alice', 200],
    ['Bob',   150],
    ['Bob',    50],
], schema=['customer', 'amount'])

# TABLE 2: Returns
returns = spark.createDataFrame([
    ['Alice', 30],
    ['Bob',   50],
], schema=['customer', 'refund'])

#orders.show()
def net_revenue(orders_df, returns_df):
    Total_orders = orders_df['amount'].sum()
    Total_returns = returns_df['refund'].sum()
    net = Total_orders-Total_returns
    return pd.DataFrame({
        "customer": [orders_df['customer'].iloc[0]],
        "net_revenue":[net]
    })

orders.groupBy("customer").cogroup(returns.groupBy("customer")).applyInPandas(net_revenue,schema="customer String, net_revenue Int").show()

# COMMAND ----------

# Window Functions
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, lag, lead, sum, avg, count, col

# Window functions are applied to a group of rows that share the same values in specified partitioning columns.
w = Window.partitionBy("color").orderBy("price1")

color_df.withColumn('running_sum',sum("price1").over(w)).show()

color_df.select(
    'color', 'fruit', 'price1',
    row_number().over(w).alias('row_num'),     # 1,2,3,4...
    rank().over(w).alias('rank'),               # 1,2,2,4 (gaps)
    dense_rank().over(w).alias('dense_rank'),   # 1,2,2,3 (no gaps)
    lag('price1', 1).over(w).alias('prev_price1'),      # previous row's v1
    lead('price1', 1).over(w).alias('next_price1')      # next row's v1
).orderBy('color').show()

# COMMAND ----------

# Null Handling input data

# this contains Nulls -> Missing color,Missing fruit,Missing numeric values (v1, v2)
# this contains dirty values -> "N/A","unknown","?","null","NA","missing",empty string ""
# this contains mixed types -> Numeric columns with nulls, String columns with nulls and dirty values
# this contains realistic scenarios -> Rows with partial data, Rows with all missing data, Rows with inconsistent formatting, Entire row with all nulls except ID

data = [
    ("A101", "red",    "banana",   10,   1.5,   None),
    ("A102", None,     "banana",   None, None,  "N/A"),   
    ("A103", "blue",   None,       30,   3.2,   ""),     
    ("A104", "red",    "carrot",   None, 4.1,   "unknown"),
    ("A105", None,     None,       None, None,  None),   
    ("A106", "black",  "carrot",   60,   None,  "?"),      
    ("A107", "red",    "banana",   70,   7.0,   "null"),
    ("A108", "green",  "grape",    None, 8.5,   "NA"),
    ("A109", "",       "banana",   90,   None,  "missing"),
    ("A110", "blue",   "",         None, 2.2,   None)
]

columns = ["id", "color", "fruit", "v1", "v2", "notes"]
null_df = spark.createDataFrame(data,columns)
null_df.show()

# COMMAND ----------

# fillna()
# Replace null fruit with "Unknown"
null_df.fillna({"fruit":"Unknown"}).show()

# Replace null numeric values with 0 
# Numeric only
null_df.fillna(0).show()

# Replace null color with "NoColor"
null_df.fillna({"color":"Nocolor"}).show()

# COMMAND ----------

# dropna()
# Drop any row with at least one null
null_df.dropna().show()

# Drop only if ALL columns are null
null_df.dropna(how="all").show()

# Drop only if specific columns are null
null_df.dropna(subset=["fruit","color"]).show()

# Drop rows with fewer than N non‑null values
null_df.dropna(thresh=4).show()

# Drop rows where all values except ID are null
# Drop rows where fruit is null
null_df.dropna(subset=["fruit"]).show()

# Drop rows where both v1 and v2 are null
null_df.dropna(subset=["v1","v2"])

# COMMAND ----------

# na.replace()
# Replace dirty values: "N/A", "NA", "?", "unknown", "null", "missing", "" → None   (Replace multiple values)
# Replace "banana" → "BANANA"   (Replace single value)
# Replace with dictionary 

null_df.na.replace(["N/A","NA","?","unknown","null","missing",""],None).na.replace("banana","BANANA").show()
null_df.na.replace({"N/A":"None","NA":"None"}).show()

# COMMAND ----------

# Create clean columns   # when().otherwise()
# fruit_clean: If fruit is null → "Unknown", Else uppercase
# color_clean: Replace empty string with "Unknown"

from pyspark.sql.functions import col, when, upper
null_df.withColumn("fruit_clean",when(col("fruit").isNull(),"Unknown").otherwise(upper(col("fruit")))).show()

null_df.withColumn("color_clean",when(col("color")=="","Unknown").otherwise(col("color"))).show()

# COMMAND ----------

from pyspark.sql.functions import col, when, upper,count
# Count all rows
null_df.count()

# count nulls per column
null_df.filter(col("fruit").isNull()).count()

# Count not nulls per column
null_df.filter(col("fruit").isNotNull()).count()

# Count not nulls per column
null_df.select(count("fruit")).show()

# COMMAND ----------

from pyspark.sql.functions import lit,coalesce
# coalesce() returns the first non‑null value from the list of columns 
null_df.withColumn("not_null_col",coalesce("color","fruit",lit("NotKnown"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# nulls in joins
# 2nd dataset ( fruit_info)

data_fruit = [
  ("apple",  "fresh"),
    ("banana", "fresh"),
    ("orange", "citrus"),
    ("",       "unknown"),     # empty string
    (None,     "missing")      # null value
]
columns = ["fruit","category"]
fruit_df = spark.createDataFrame(data_fruit,columns)

join_df = null_df.join(fruit_df,"fruit","left").show()