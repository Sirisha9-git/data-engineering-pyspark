# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()

# input dataset
data = [
    ("red", "banana", 1, 10),
    ("blue", None, 2, None),
    (None, "carrot", None, 30),
    ("black", "carrot", 6, 60),
    ("red", None, 7, None)
]

columns = ["color", "fruit", "v1", "v2"]

window_df = spark.createDataFrame(data,columns)
window_df.show()

# COMMAND ----------

# Window.partitionBy
# Groups rows into partitions.
from pyspark.sql.functions import count
from pyspark.sql.window import Window

w=Window.partitionBy("color")
window_df.withColumn("cnt",count("*").over(w)).show()
window_df.select("color",count("*").over(w).alias("cnt")).show()

# COMMAND ----------

# Window.orderBy   -   Defines row order inside each partition.
# row_numer ()     -   Row number per fruit ordered by v1.
from pyspark.sql.functions import *
from pyspark.sql.window import Window

w= Window.partitionBy("color").orderBy("v1")
window_df.withColumn("ordered_v1",row_number().over(w)).show()

# COMMAND ----------

# Rank  - 
# dense_rank   - 

from pyspark.sql.functions import rank,dense_rank
from pyspark.sql.window import  Window

w=Window.partitionBy("color").orderBy("v1")
window_df.withColumn("v1_rank",rank().over(w)).show()
window_df.withColumn("v1_dense_rank",dense_rank().over(w)).show()

# COMMAND ----------

# lead - next value in the group
# lag - previous value in the group
from pyspark.sql.functions import lead,lag
from pyspark.sql.window import Window

w=Window.partitionBy("color").orderBy("v1")

window_df.withColumn("lead_v1",lead("v1",1).over(w)).show()
window_df.withColumn("lag_v1",lag("v1",1).over(w)).show()


# COMMAND ----------

# running Total
# Start at the first row.  Keep adding values row by row.   Reset when the partition changes

from pyspark.sql.functions import sum
from pyspark.sql.window import Window

w=Window.partitionBy("color").orderBy("v1") \
    .rowsBetween(Window.unboundedPreceding,Window.currentRow)
# Start from the first row in the partition.  End at the current row.  Add everything in between.

window_df.withColumn("running_total",sum("v1").over(w)).show()

# COMMAND ----------

# Rolling sum
# Window average
# Rolling window - Looks at a fixed number of rows around the current row. Computes a metric (sum, avg, min, max).
#                  Moves forward one row at a time

from pyspark.sql.functions import *
from pyspark.sql.window import Window

w = Window.partitionBy("color").rowsBetween(-2,0)
window_df.withColumn("rolling_sum",sum('v1').over(w)).show()
window_df.withColumn("window_avg",avg("v1").over(w)).show()

# COMMAND ----------

# Top‑N Per Group
# row_number() - Allows filtering for top 1, top 3, etc.  Works perfectly for “latest record per group”

w=Window.partitionBy("color").orderBy(col("v1").desc())
window_df.withColumn("Top_6",row_number().over(w)).filter("Top_6<=6").show()

# COMMAND ----------

# rangeBetween
# This is value based. not row based. Works only with numeric orderBy column.

from pyspark.sql.window import Window
from pyspark.sql.functions import sum

w= Window.partitionBy("color").orderBy("v1").rangeBetween(Window.unboundedPreceding,0)

window_df.withColumn("range_sum",sum("v1").over(w)).show()
