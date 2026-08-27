# Databricks notebook source
# Datasets for joins .
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
spark = SparkSession.builder.getOrCreate()

# customers
df_customers = spark.createDataFrame([
    (1, "Alice", "TX"),
    (2, "Bob", "CA"),
    (3, "Charlie", None),
    (4, "David", "TX")
],schema = ["cust_id", "name", "state"])
df_customers.show()

# orders
df_orders = spark.createDataFrame([
    (1, "Laptop", 1200),
    (1, "Mouse", 25),
    (2, "Keyboard", 75),
    (5, "Monitor", 300)
], schema = ["cust_id", "product", "amount"])
df_orders.show()

# COMMAND ----------

# Basic join
df_customers.join(df_orders,"cust_id").show()

# COMMAND ----------

# inner join
# Returns rows where keys match in both tables.
df_customers.join(df_orders,"cust_id","inner").show()

# COMMAND ----------

# Left join
# Keeps all rows from left, matching from right.
df_customers.join(df_orders,"cust_id","left").show()

# COMMAND ----------

# Right join
# Keeps all rows from right, matching from left.
df_customers.join(df_orders,"cust_id","right").show()

# COMMAND ----------

# full outer join
# Keeps all rows from both sides.
df_customers.join(df_orders,"cust_id","full").show()

# COMMAND ----------

# Left semi join
# Returns rows from left only if a match exists in right. Does NOT return columns from right. 

df_customers.join(df_orders,"cust_id","left_semi").show()

# COMMAND ----------

# Left anti join
# Returns rows from left where NO match exists in right.

df_customers.join(df_orders,"cust_id","left_anti").show()

# COMMAND ----------

# Broadcast join
# Spark broadcasts the smaller table. No shuffle. Faster join
df_customers.join(broadcast(df_orders),"cust_id").show()

# COMMAND ----------

# Sort Merge join
df_customers.join(df_orders,"cust_id").explain(True)
# find SORTMERGEJOIN in Explain plan.

# COMMAND ----------

# comparing inner, broadcast and full joins
# inner - No broadcast hint.  Spark must shuffle
df_customers.join("df_orders","cust_id","inner").explain(True)    # SortMergeJoin in explain plan

# broadcast - broadcasts the smaller table.   No shuffle.    Fastest join
df_customers.join(broadcast(df_orders),"cust_id").explain(True)   # BroadcastHashJoin  in explain plan

# full - Full outer join cannot use broadcast.  Must shuffle both sides
df_customers.join("df_orders","full").explain(True)                # SortMergeJoin in explain plan


# COMMAND ----------

# Join with multiple conditions.
# join on cust_id and state(if column exists)
df_customers.join(df_orders,
    (df_customers.cust_id == df_orders.cust_id) &
    (df_customers.state == df_orders.state),
    "inner"
                  )

# COMMAND ----------

# Aggregations on joins
# Calculate total order amount per customer
joined_df = df_customers.join(df_orders,"cust_id","left")
joined_df.groupBy("cust_id","name").agg(sum("amount").alias("total_order_amount")).show()
