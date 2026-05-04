# Databricks notebook source
# input dataset for testing pyspark performance

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import random

spark = SparkSession.builder.getOrCreate()

skew_ids = [1]*50000 + [2]*30000 + [3]*10000+ list(range(4,1004))
# 1 appears 50,000 times + 2 appears 30,000 times + 3 appears 10,000 times + values from 4 to 011003 appears once.
# This creates heavy skew.

data = [
    (random.choice(skew_ids), 
     # Randomly picks a customer_id from the skewed list.1-ery often, 2-often, 3 - sometimes, 4to 1003 - rarely.
     # random.choice() — picks a value from a list

    random.randint(1,1000),                  
     # Random product_id between 1 and 1000. random.randint() — picks a number from a numeric range

    float(random.randint(10,500)),            # Random amount between 10 and 500 converted to float

    f"2024-01-{random.randint(1,28):02d}")   
    # Generates a date string:Year: 2024,Month:January,Day:01 to 28,:02d ensures 2digit formatting (e.g., 01, 02, 03)

    for _ in range(200000)                    # Creates 200,000 rows.
]
# this creates 200,000 rows of synthetic data.

schema = StructType([
    StructField("customer_id",IntegerType()),
    StructField("product_id",IntegerType()),
    StructField("amount",FloatType()),
    StructField("date",StringType())
])

skew_df = spark.createDataFrame(data,schema)
skew_df = skew_df.withColumn("date",to_date("date"))
skew_df.show(5)

# COMMAND ----------

# Catalyst Optimizer   - Projection pruning
# Spark removes unnecessary columns automatically.

df_new = skew_df.select("customer_id", "amount")
df_new.explain(True)

# ouput -> 'Project ['customer_id, 'amount]'. No other unnecessary columns are seen in plan.

# COMMAND ----------

# Catalyst Optimizer   - Predicate pushdown
# Spark pushes the filter down to the data source.

df_new = skew_df.filter(col("amount")>300)
df_new.explain(True)

# output ->  'Filter '`>`('amount, 300)' in plan. filtered data at source  where amount >300

# COMMAND ----------

# catalyst Optimizer - Join reordering
# reorder joins for efficiency.

df_new = skew_df.groupBy("customer_id").agg(sum("amount").alias("total"))
df_join = skew_df.join(df_new,"customer_id")
df_join.explain(True)

# output -> Join Inner, (customer_id#11234 = customer_id#11245). see if the smaller DF is used first.

# COMMAND ----------

# Tungsten engine - UDF Vs Built in function

from pyspark.sql.functions import udf

@udf("float")
def add_tax_udf(x):
    return x*1.1

df_new = skew_df.withColumn("amount_tax",add_tax_udf(col("amount")))
df_new_builtin = skew_df.withColumn("amount_tax1",col("amount")*1.1)

df_new.explain(True)
df_new_builtin.explain(True)

# COMMAND ----------

# Repartition   - Increase partitions

df_re = skew_df.repartition(50,"customer_id")
df_re.rdd.getNumPartitions()
# output - 50 and spark UI shows shuffle .

# COMMAND ----------

# Repartition - repartition before join
# Repartitioning before join reduces shuffle cost.

df_new = skew_df.groupBy("customer_id").agg(sum("amount").alias("total"))
df_new.repartition("customer_id")
df_join = skew_df.join(df_new,"customer_id")
df_join.explain(True)

# output - Look for ExchangeHashpartitioning

# COMMAND ----------

# coalesce 
# Reduce partitions without shuffle

df_new = skew_df.filter(col("amount")>300)
df_new.coalesce(1)
df_new.explain(True)

# output - No shuffle in plan.  Look for: Coalesce.

# COMMAND ----------

# Cache()    - cache a heavy transformation

df_big = skew_df.groupBy("customer_id").agg(sum("amount").alias("total"))
df_big.cache()

# First action.
df_big.count()

# second action.
df_big.count()

# output -> First action is slow (computes DAG) but second action is fast( reads from memory)

# COMMAND ----------

# persist()
# Both cache() and persist() store the computed DataFrame in memory/disk so Spark doesn’t recompute the lineage again. The only difference is the storage level.

from pyspark import StorageLevel

df_big.persist(StorageLevel.MEMORY_AND_DISK)
df_big.count()

# output -> Storage tab → MEMORY_AND_DISK.   If memory is insufficient → spills to disk.

# COMMAND ----------

# unpersist()
# Removes cached/persisted data.

df_big.unpersist()

# output -> Storage tab → entry disappears.

# COMMAND ----------

# checkpoint

# setup checkpoint dir
spark.sparkContext.setCheckPointDir("/tmp/checkpoints")

df_long = skew_df
for i in range(20):
    df_long = df_long.withColumn("amount_new",col("amount")+1)
# creates a very long lineage (20 transformations).
# the final result will be amount + 20 for every row.
# Each withColumn() call adds another step to the lineage (DAG).
# So after the loop, Spark has a chain like:amount + 1 + 1 + 1 + ... (20 times)
# Only when you call an action like:df_long.show()..Spark executes the entire chain and computes the final result.

df_long.checkpoint()
df_long.explain(True)

# output -> look for InMemoryFileIndex

# COMMAND ----------

# AQE - Adaptive Query Execution  - Skew join optimization

spark.conf.set("spark.sql.adaptive.enabled","true")

df_new = skew_df.groupby("customer_id").agg(sum("amount").alias("total"))
df_join = skew_df.join(df_new,"customer_id")
df_join.explain(True)

# output - Look for AdaptiveSparkPlan  and skewjoin

# COMMAND ----------

# AQE - Coalesced partitions
spark.conf.set("spark.sql.adaptive.enabled","true")
df_new = skew_df.groupby("customer_id").count()
df_new.explain(True)

# output - Look for Coalesced shuffle partitions.

# COMMAND ----------

# salting
# voids data skew problem.
from pyspark.sql.functions import rand
df_new  = skew_df.groupBy("customer_id").agg(sum("amount").alias("total"))

skew_df_salted = skew_df.withColumn("salt",(rand()*10).cast("int"))
df_new_salted = df_new.withColumn("salt",(rand()*10).cast("int"))

df_join = skew_df_salted.join(df_new_salted,["customer_id","salt"])
df_join.explain(True)

# more evenly distributed partitions.