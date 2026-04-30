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

# COMMAND ----------

# explain() - Reading execution plans.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

explain_df = spark.createDataFrame([['red', 'banana', 1, 10],
    ['blue', 'banana', 2, 20],
    ['red', 'carrot', 3, 30],
    ],schema=['color','fruit','price1','price2'])

# simple plan:
explain_df.filter(col('color')=='red').explain()

# Detailed plan:
explain_df.filter(col('color')=='red').explain(True)

# Detialed plan for groupBy
explain_df.groupBy('color').avg('price1').explain(True)


# COMMAND ----------

# Removing duplicates

dup_df = spark.createDataFrame([
    ['red','banana'],
    ['red','banana'],
    ['blue','grape'],
    ['blue','grape'],
    ['blue','banana']
],schema=['color','fruit'])

# distinct() - Removes exact full row duplicates
# dup_df.distinct().show()
dup_df.distinct().explain() # - Shuffle - Exchange appears

# dropduplicates() - Removes whole row duplicates. Same as distinct(). no args.
# dup_df.dropDuplicates().show()
dup_df.dropDuplicates().explain() # - Shuffle - Exchange appears

# dropduplicates with args. Removes duplicates based on given column.
# dup_df.dropDuplicates(['color']).show()
# keeps the first occurence of each color
dup_df.dropDuplicates(['color']).explain() # - Shuffle - Exchange appears

# COMMAND ----------

# Narrow vs Wide Transformations 

# Narrow Transformation
from pyspark.sql.functions import col
df = spark.range(1,21).repartition(4)

narrow_df = df.filter("id>5") \
        .select((col("id")*2.0).alias("id_2")) \
        .withColumn("id_2_sum",col("id_2")+1)
#narrow_df.show()
narrow_df.explain()
# Only one stage in the Physical Plan. No Exchange keyword. All operations are pipelined

# COMMAND ----------

# Wide Transformation
df = spark.range(1,21).repartition(4)
df.groupBy().count().explain()
# Two stages( 2 EXCHANGE keywords) in the DAG. Shuffle boundary created.

# COMMAND ----------

# Narrow vs Wide Transformation
# Narrow.
df.filter("id>5").explain()
# Wide
df.groupBy("id").count().explain()


# COMMAND ----------

# Joins (Wide Transformation)
# shuffle vs Broadcast join
from pyspark.sql.functions import broadcast

df1=spark.range(1,1000).withColumn("key",(col("id")%10))
df2=spark.range(1,10).withColumn("key",col("id"))

# shuffle
df1.join(df2,"key","inner").explain()

# Broadcast join
df1.join(broadcast(df2),"key","inner").explain()


# COMMAND ----------

# Repartition vs Coalesce

df1 = spark.range(1,21).repartition(10)

# Repartition - shuffle - Exchange appears
df1.repartition(20).explain()
# Coalesce - no shuffle - No Exchange appears
df1.coalesce(2).explain()

# COMMAND ----------

# DAG building
# See how Spark pipelines operations.

df = spark.range(1,21).repartition(10)

df.filter("id>5") \
    .select("id") \   # stage 1 - Narrow Transformation
        .groupBy("id").count() \ - Stage 2 - Wide Transformation
            .explain()

# COMMAND ----------

# Getting Data IN/OUT

# Source file
df = spark.createDataFrame([
    ['red',   'banana', 1, 10],
    ['blue',  'banana', 2, 20],
    ['red',   'carrot', 3, 30],
    ['blue',  'grape',  4, 40],
    ['red',   'carrot', 5, 50],
    ['black', 'carrot', 6, 60],
    ['red',   'banana', 7, 70],
    ['red',   'grape',  8, 80]
    ],schema = ['color','fruit','price1','price2'])

df.show()
base = '/Volumes/workspace/default/pyspark_io'
# write to csv
#df.write.csv(f"{base}/basic_csv_write", header=True,mode ='overwrite')
df.write.csv(
    f"{base}/all_op_csv_write",
    header=True,         # Include column names in the first row.
    sep=',',             # Field delimiter (default is comma).
    quote = '"',         # Character used to quote fields.
    escape = '"',        # Escape character for quotes.
    encoding = 'UTF-8',  # Character encoding.
    mode='overwrite',    # overwrite-Replace existing folder,append-Add new files to existing folder,              # ignore-Do nothing if folder exists, error-Throw exception if folder exists
    compression='gzip'   # none, gzip, bzip2, snappy, lz4, deflate
)

# COMMAND ----------

# Write to single file (coalesce)
# Spark writes multiple part files by default. Forces to write to one CSV file
df.coalesce(1).write.csv(f"{base}/coalesce_csv_write",header=True,mode = "overwrite")

# Writing to multiple files.
df.repartition(10).write.csv(f"{base}/repartition_csv_write")

# COMMAND ----------

# write partitioned csv data
df.write.partitionBy("color").csv(f"{base}/partitioned_csv_write",mode = "overwrite")

# Read partitioned data
df = spark.read.csv(f"{base}/partitioned_csv_write")
df.filter("color = 'red'").show()

# COMMAND ----------

# Read csv file - Basic
spark.read.csv(f"{base}/basic_csv_write", header = True, inferSchema = True).show()

# Read csv file - all options
spark.read.csv(
        f"{base}/basic_csv_write",
        header = True,
        inferSchema = True
    ).show()

# COMMAND ----------

df = spark.createDataFrame([
    ['red',   'banana', 1, 10],
    ['blue',  'banana', 2, 20],
    ['red',   'carrot', 3, 30],
    ['blue',  'grape',  4, 40],
    ['red',   'carrot', 5, 50],
    ['black', 'carrot', 6, 60],
    ['red',   'banana', 7, 70],
    ['red',   'grape',  8, 80]
    ],schema = ['color','fruit','price1','price2'])

df.show()
base = '/Volumes/workspace/default/pyspark_io'

# write to parquet
df.write.parquet(f"{base}/parquet_data_write", mode='Overwrite',compression = 'snappy')

# Read from parquet
df_parquet = spark.read.parquet(f"{base}/parquet_data_write").show()

# COMMAND ----------

df = spark.createDataFrame([
    ['red',   'banana', 1, 10],
    ['blue',  'banana', 2, 20],
    ['red',   'carrot', 3, 30],
    ['blue',  'grape',  4, 40],
    ['red',   'carrot', 5, 50],
    ['black', 'carrot', 6, 60],
    ['red',   'banana', 7, 70],
    ['red',   'grape',  8, 80]
    ],schema = ['color','fruit','price1','price2'])

df.show()
base = '/Volumes/workspace/default/pyspark_io'

# write to orc
df.write.orc(f"{base}/orc_data_write",mode = "overwrite")

# Read from orc
df_orc = spark.read.orc(f"{base}/orc_data_write").show()

# COMMAND ----------

df = spark.createDataFrame([
    ['red',   'banana', 1, 10],
    ['blue',  'banana', 2, 20],
    ['red',   'carrot', 3, 30],
    ['blue',  'grape',  4, 40],
    ['red',   'carrot', 5, 50],
    ['black', 'carrot', 6, 60],
    ['red',   'banana', 7, 70],
    ['red',   'grape',  8, 80]
    ],schema = ['color','fruit','price1','price2'])

df.show()
base = '/Volumes/workspace/default/pyspark_io'

# write to json
df.write.json(f"{base}/json_data_write",mode="overwrite")

# Read from json
df_json = spark.read.json(f"{base}/json_data_write").show()

# Read from multiline json
df_multi_json = spark.read.json(f"{base}/json_data_write",multiLine=True).show()


# COMMAND ----------

# Write as delta
df.write.format("delta").mode("overwrite").save(f"{base}/delta_data_write")

# Read from delta
df_delta = spark.read.format("delta").load(f"{base}/delta_data_write").show()

# COMMAND ----------

# jdbc write
df.write.jdbc(
    url="jdbc:postgresql://host:5432/db",
    table="output_table",
    mode="append",
    properties={"user": "u", "password": "p"}
)

# jdbc read
df = spark.read.jdbc(
    url="jdbc:postgresql://host:5432/db",
    table="schema.table",
    properties={"user": "u", "password": "p"}
)

# jdbc paralell reads
df = spark.read.jdbc(
    url="jdbc:postgresql://host:5432/db",
    table="orders",
    column="id",
    lowerBound=1,
    upperBound=1000000,
    numPartitions=10,
    properties={"user": "u", "password": "p"}
)

# COMMAND ----------

# Reading from text file
df = spark.read.text(f"{base}/text_file")

# reading from binary file
df = spark.read.format("binaryFile").load(f"{base}/images")