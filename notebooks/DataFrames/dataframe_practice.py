# Databricks notebook source
# importing SparkSession class from pyspark.sql module
from pyspark.sql import SparkSession

# COMMAND ----------

# Create or reuse the SparkSession and store it in a variable caleld spark
spark = SparkSession.builder.getOrCreate()

# more options to get or create or get spark session.
# spark = SparkSession.appName("dataframes_session") \ 
                    # .master("local[*]") \
                    # .getOrCreate

# COMMAND ----------

# create a PySpark DataFrame from a list of rows
from datetime import datetime, date
from pyspark.sql import Row

# way1
df1 = spark.createDataFrame([
    Row(id=1,name = 'john',dob=(date(1990,1,1,)),salary=10000),
    Row(id=2,name='mary',dob=date(1995,3,11),salary=5000)
])
df1.show()

# way2 - high performance comapred to way1
# need not import Row class
data = [(1,'john',date(1990,1,1),10000),(2,'mary',date(1995,3,11),5000)]
df2 = spark.createDataFrame(data,["id","name","dob","salary"])
df2.show()


# COMMAND ----------

# Create pyspark dataframe with an explicit schema
from datetime import date,datetime
from pyspark.sql.types import StructType,StructField,LongType,DoubleType,StringType,DateType,TimestampType

schema = StructType([
    StructField("a",LongType(),True),
    StructField("b",DoubleType(),True),
    StructField("c",StringType(),False),
    StructField("d",DateType(),True),
    StructField("e",TimestampType(),False)
    ])
df3 = spark.createDataFrame([(1,2.0,"str1",date(1990,1,1),datetime(2000,4,13,5,10,10)),
                             (5,6.0,"str2",date(1995,2,5),datetime(1997,6,10,10,10,15))
                             ], schema = schema
                            )
df3.show()
# Execution time : 458ms

# COMMAND ----------

# Create a PySpark DataFrame from a pandas DataFrame

# Import pandas library for creating a local pandas DataFrame
import pandas as pd

# Create a pandas DataFrame (lives in local Python memory, not in Spark)
pd_df = pd.DataFrame({"id":[1,2],"name":["John","Mary"]})
#pd_df.show()
# ❌ pandas DataFrame has no .show() method (only Spark DataFrames do)
# This will give 'AttributeError: 'DataFrame' object has no attribute 'show' error.'

# COMMAND ----------

# Convert pandas DataFrame → Spark DataFrame (Spark distributes the data across the cluster)
df4 = spark.createDataFrame(pd_df)
# Display the Spark DataFrame (now distributed)
df4.show()
# Execution time = 187ms

# COMMAND ----------

# viewing DataFrame data

# show()
df4.show () # Shows top rows from dataframe

# Show N rows - show(N)
df4.show(1) # Shows only one row from top

# COMMAND ----------

# Inspecting schema and column names

# columns
df4.columns

# Schema
df4.printSchema()
# ✔ Shows column names, types, and nullability

# COMMAND ----------

# View Summary Statistics

#describe()
df4.select("id").describe().show()


# COMMAND ----------

# Collecting Data to Driver

# collect()
df4.collect()

# Alternatives due to out of memory issue.
#tail()
df4.tail(1)  # take last one row.

# take()
df4.take(1)  # take first one row


# COMMAND ----------

# convert spark dataframe to pandas dataframe

# toPandas()

df3.toPandas()
# collects all data to driver memory.

# COMMAND ----------

# Selecting and Accessing Dataframe data

# Column object
from pyspark.sql import Column
df4.id
# No computation. Only returns a column object.

# COMMAND ----------

# Operations on Column types
from pyspark.sql.functions import upper
type(df4.id) == type(upper(df4.id))
# Everything stays a blueprint until an action (.show(), .collect(), .count()) triggers execution.

# COMMAND ----------

# select
# Returns a dataframe with only selected columns.
df4.select("id").show()

# COMMAND ----------

# select withColumn
# Creates a new column or overwrites using a transformation.
df4.withColumn("upper_name",upper(df4.name)).show()

# COMMAND ----------

# filter rows
# Returns a new DataFrame with only rows matching the condition.
df4.filter(df4.id ==1).show()