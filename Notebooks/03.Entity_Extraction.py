# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.18.0 mlflow==2.9.0
# MAGIC %pip install langchain_community
# MAGIC %pip install langchain langchain_core
# MAGIC %pip install better_profanity
# MAGIC %pip install "transformers>=4.40.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Load the data from the table "dss.imageocr.clean_data_updated" into a DataFrame named df
df = spark.sql("SELECT * FROM dss.imageocr.clean_data_updated")
print(df["ocr_text"])

# COMMAND ----------

# Get the values of the ocr_text
ocr_text_table = df.select("ocr_text").toPandas()
display(ocr_text_table)

# COMMAND ----------

# Test Databricks Foundation LLM model
from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

# COMMAND ----------

TEMPLATE = """Please respond to the following question with a {brand_type} response. 
Post:" {post}"
Comment: 
"""

# COMMAND ----------

#TEMPLATE = """ Please respond to the following question with the brand_type and the description. 
#[Post]: "{post}"
#[Brand_type]:"{brand_type}"
#[Description]: "{description}"
#"""

# COMMAND ----------

from langchain import PromptTemplate

prompt = PromptTemplate(template=TEMPLATE, input_variables=["brand_type", "post"])

#prompt = PromptTemplate(template=TEMPLATE, input_variables=["post", "brand_type","description"])

brand_prompt = prompt.format(brand_type="", post =ocr_text_table.values[0])

#brand_prompt = prompt.format(post =ocr_text_table.values[0], brand_type="Conditioner",description="")

print(f"brand prompt:{brand_prompt}")

# COMMAND ----------

from langchain.chains import LLMChain

from langchain_community.chat_models import ChatDatabricks

product_chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    output_key="output",
    verbose=False,
) 

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

# Create SparkSession
spark = SparkSession.builder.getOrCreate()

# Convert text_values to Spark DataFrame
text_values_spark = spark.createDataFrame(ocr_text_table)

# Define UDF to apply example_output to each row of ocr_text_table
apply_example_output = udf(lambda text: product_chain.run({"brand_type": "", "post": text}), StringType())

#apply_example_output = udf(lambda text: product_chain.run({"post": text,"brand_type": text,"description":text}), StringType())

# Apply UDF to ocr_text_table
text_values_spark = text_values_spark.withColumn("ocr_text_description", apply_example_output("ocr_text"))


# COMMAND ----------

text_values_spark.toPandas().values[0]

# COMMAND ----------


from better_profanity import profanity

# Optionally, you can remove any profanity from the ocr_text column
cleaned_text = udf(lambda text: profanity.censor(text))
text_values_spark = text_values_spark.withColumn("ocr_text_description_cleaned", cleaned_text("ocr_text_description"))


# COMMAND ----------

text_values_pandas = text_values_spark.toPandas()
text_values_pandas.head()

# COMMAND ----------

display(text_values_pandas)

# COMMAND ----------

spark_df = spark.createDataFrame(text_values_pandas)
spark_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("dss.imageocr.ocr_text_Description")


# COMMAND ----------


