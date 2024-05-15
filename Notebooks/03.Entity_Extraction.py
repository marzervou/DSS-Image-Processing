# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.18.0 mlflow==2.9.0
# MAGIC %pip install langchain_community
# MAGIC %pip install langchain langchain_core
# MAGIC %pip install better_profanity
# MAGIC %pip install "transformers>=4.40.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Load the data from the table "dss.imageocr.clean_data_updated" into a DataFrame named df
df = spark.sql("SELECT * FROM dss.imageocr.clean_data")
print(df["ocr_text"])

# COMMAND ----------

# Get the values of the ocr_text
ocr_text_table = df.select("ocr_text")
display(ocr_text_table)

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 100)

# COMMAND ----------

# DBTITLE 1,Prompt Version with json output format
# TEMPLATE = """Please respond to the following question by based on the provided text.
# USER: {input}
# ASSISTANT = "Sure, I'm ready to help you with your NER task. Please provide me with the necessary information to get started."
# USER: 
# Entity Definition:\n"
#     "1. brand_name: Short name or full name of of the product brand, choose between L'OREAL ELVIVE OR L'OREAL .\n"
#     "2. product_type: Type of product CHOOSE BETWEEN shampoo or conditioner or cream OR ELIXIR or mask.\n"
#     "3. product_description: Description of the product and the hair challenges it solves like detangling, REPAIRING and etc.\n"
#     "\n"
#     "Output Format:\n"
#     "{{'brand_name': [first of entities present], 'product_type': [first of entities present], 'product_description': [create a phrase up to 10 words of combining the entities present]
#     "If no entities are presented in any categories keep it None\n"
#     "\n"
#   "output: "
#   """

# COMMAND ----------

# DBTITLE 1,Prompt Version with Few Shot Learning
TEMPLATE = """Please respond to the following question by extracting all the entities from the provided text.
USER: {input}
ASSISTANT = "Sure, I'm ready to help you with your NER task. Please provide me with the necessary information to get started."
USER: 
Entity Definition:\n"
    "1. brand_name: Short name or full name of of the product brand, choose between L'OREAL ELVIVE OR L'OREAL .\n"
    "2. product_type: Type of product CHOOSE BETWEEN shampoo or conditioner or cream OR ELIXIR or mask.\n"
    "3. product_description: Description of the product and the hair challenges it solves like detangling, REPAIRING and etc.\n"
    "\n"
  Input:
  "Protect yourtips! Does the appearance of split ends and h oreakage make you feel like-youneed a haircut Discover Elvive Dream Lengths NO HAIRCUT CREAM It's reinforcing formula helps to reduce breakage from brushingSay no to cutting appearance reduo yourends long hairtip Apply before using heat appliances to protect yor halrup to 180C and add shine back to long lengths DIRECTIONs FOR USE:apply to the lengths and ends of wer dry hair. thoroughly In case of contact- witn eyes. rinse immediately ""Instrumental test. Breakage from brushing K08000304031RO18008186 1199552 INGREDIENTS:EA WIS RSA:0860102491 GUAR-NYOROCES RONSMNSSEO/CASEDO INAEN. SARCHXCOPES. 200mle LOREAL O0SSSB4STOUENCEDEXER LONDONWS SAZ TSA7500"

  Output:
  "ELVIVE DREAM LENGTHS CREAM It's reinforcing formula helps to reduce breakage from brushing"

  Input:
  "STRENGTHENS LENGTHS& LORE AL LOREAL ELVIVE REDUCESTHE ELVIVE Dreamlengths Dreamlengths APPEARANCE NEW OF SPLIT ENDS DETANGLING CONDITIONER WEOTALCEADNOL RESTORING WSTANTLY OETANOCE SHAMPOO LONG.GAMAGD FOR LONG, RCESFRAGRELENGT DAMAGED LONG,GAMAGEHA HAIR"
  Output:
  "LOREAL ELVIVE Dream lengths CONDITIONER RESTORING DETANGLING FOR LONG DAMANGED HAIR"

  "output: "
  """

# COMMAND ----------

from langchain import PromptTemplate

prompt = PromptTemplate(template=TEMPLATE, input_variables=["input"])

print(prompt)

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

apply_example_output = udf(lambda text: product_chain.run({ "input": text}), StringType())

# Apply UDF to ocr_text_table
text_values_spark = ocr_text_table.withColumn("ocr_text_description", apply_example_output("ocr_text"))

# COMMAND ----------

from better_profanity import profanity

# Optionally, you can remove any profanity from the ocr_text column
cleaned_text = udf(lambda text: profanity.censor(text))
text_values_spark = text_values_spark.withColumn("ocr_text_description_cleaned", cleaned_text("ocr_text_description"))

# COMMAND ----------

display(text_values_spark)

# COMMAND ----------

text_values_spark.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("dss.imageocr.ocr_text_description")


# COMMAND ----------


