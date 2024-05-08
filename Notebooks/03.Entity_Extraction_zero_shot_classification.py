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
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 20)

# COMMAND ----------

from langchain import PromptTemplate

# create a example template
example_template = """ Please respond to the following question with a brand type response:
[Question]: LOREAL PARiS ELVIVE COLORVIBRANCY up to 60DAYS VIBRANCYE withsystem COLORPROTECTING SHAMPOO WITH LINSEED ELIXIR +ANTI OXIDANTSUV ANTI-FADE, HIGH SHINE &PROTECTS COLOR TREATED HAIR 28FL.OZ.-828ml
[Answer]: L'oreal Paris Shampoo
###
[Question]: L'OREAL ARiS ELVIVE COLORVIBRANCY PROTECTING CONDITIONER EWLOOK SVIBRANCYE DELIXIR PROTECTS AIR OZ-375ml WITH LINSEED ELIXER + ANTI-OXIDANTS [UV]
[Answer]: L'oreal Paris Conditioner
###
[Question]: {query}
[Answer]: """

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query"],
    template=example_template
)

# COMMAND ----------

brand_prompt = example_prompt.format(query=ocr_text_table.values[1])

# COMMAND ----------

from langchain import PromptTemplate

prompt = PromptTemplate(template=example_template, input_variables=["query"])

brand_prompt = prompt.format(query =ocr_text_table.values[1])
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
apply_example_output = udf(lambda text: product_chain.run({"query": text}), StringType())

# Apply UDF to ocr_text_table
text_values_spark = text_values_spark.withColumn("ocr_text_description", apply_example_output("ocr_text"))


# COMMAND ----------

text_values_spark.toPandas().values[2]

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

from transformers import pipeline
# We will limit the response length for our few-shot learning tasks.
few_shot_pipeline = pipeline(
    #task="text-generation",
    model="databricks/dbrx-instruct",
    max_new_tokens=10,
)

# COMMAND ----------

# Get the token ID for "###", which we will use as the EOS token below.
eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]

# COMMAND ----------

from langchain import FewShotPromptTemplate

# create our examples
examples = [
    {
        "query": "LOREAL PARiS ELVIVE COLORVIBRANCY up to 60DAYS VIBRANCYE withsystem COLORPROTECTING SHAMPOO WITH LINSEED ELIXIR +ANTI OXIDANTSUV ANTI-FADE, HIGH SHINE &PROTECTS COLOR TREATED HAIR 28FL.OZ.-828ml",
        "answer": "L'oreal Paris Shampoo"
    }, {
        "query": "L'OREAL ARiS ELVIVE COLORVIBRANCY PROTECTING CONDITIONER EWLOOK SVIBRANCYE DELIXIR PROTECTS AIR OZ-375ml WITH LINSEED ELIXER + ANTI-OXIDANTS [UV]",
        "answer": "L'oreal Paris Conditioner"
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
#prefix = """The following provides the responses from the assistant regarding the brand type of the provided product: 
#"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    #prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

# COMMAND ----------


query = ocr_text_table.values[0]

print(few_shot_prompt_template.format(query=query))

# COMMAND ----------

print(few_shot_pipeline(
    few_shot_prompt_template.format(query=query)
))

# COMMAND ----------

temp = (few_shot_pipeline(
    few_shot_prompt_template.format(query=query)
))

# COMMAND ----------

print(temp[0]["generated_text"])

# COMMAND ----------

import pandas as pd

# Convert temp list to pandas DataFrame
temp_dataframe = pd.DataFrame(temp)

# Display the DataFrame
temp_dataframe

# COMMAND ----------

print(temp_dataframe['generated_text'])

# COMMAND ----------

# Split the 'generated_text' column into different columns separated by '\nUser' and '\nAI'
temp_split = temp_dataframe['generated_text'].str.split('\nUser', expand=True)
temp_split.columns = ['user_column', 'ai_column']

# COMMAND ----------

user_column

# COMMAND ----------

from transformers import AutoTokenizer, DbrxForCausalLM

model = DbrxForCausalLM.from_pretrained("databricks/dbrx-instruct")
tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."

# COMMAND ----------


