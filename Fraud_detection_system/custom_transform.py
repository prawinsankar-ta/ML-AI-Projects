# This file is part of feature engineering in AWS Data Wrangler

# Table is available as variable `df`
import pyspark.sql.functions as F

# --- Categorical Encoding ---
df = df.withColumn("type_indexed", F.when(F.col("type") == "TRANSFER", 1)
                                     .when(F.col("type") == "CASH_OUT", 2)
                                     .otherwise(0))  

# --- Aggregated Features ---
df_orig = df.groupBy("nameOrig").agg(
    F.count("amount").alias("transaction_count_orig"),
    F.mean("amount").alias("transaction_mean_orig")
)

df_dest = df.groupBy("nameDest").agg(
    F.count("amount").alias("transaction_count_dest"),
    F.mean("amount").alias("transaction_mean_dest")
)

# Join the aggregated features back before dropping nameDest
df = df.join(df_orig, on="nameOrig", how="left") \
       .join(df_dest, on="nameDest", how="left")

# Drop categorical columns after using them
df = df.drop("nameOrig", "nameDest", "type")
