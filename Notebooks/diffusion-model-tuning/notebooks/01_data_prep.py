# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at https://github.com/databricks-industry-solutions.
# MAGIC

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC #Prepare your images for fine-tuning
# MAGIC  Tailoring the output of a generative model is crucial for building a successful application. This applies to use cases powered by an image generation model as well. For example, a furniture designer wants to see their previous designs reflected on a newly generated image. But they also want to see some modifications, for example in material or color. In such case, it is important that the model is aware of their previous products and can apply new styles to generate new product designs. Customization is necessary in a case like this. We can do this by fine-tuning a pre-trained model on our own images.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage your images in Unity Catalog Volumes

# COMMAND ----------

# MAGIC %md
# MAGIC This solution accelerator uses the 25 training images stored in the subfolders of ```/images/chair/``` to fine-tune a model. We copy the images to Unity Catalog (UC) and managed them as volume files. To adapt this solution to your use case, you can directly upload your images in UC volumes.

# COMMAND ----------

df_links = spark.read.table("dss.imageocr.clean_data")
df_prompts = spark.read.table("dss.imageocr.ocr_text_description")

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col

df_prompts_final = df_prompts.withColumn(
    "prompt",
    regexp_replace(
        col("ocr_text_description_cleaned"),
        r'(1\. Brand Name:|1\. brand_name:|2\. Product Type:|2\. product_type:|3\. Product Description:.*|3\. product_description:.*)|Input:.*|".*',
        ''
    )
)

# COMMAND ----------

display(df_prompts_final)

# COMMAND ----------

df_prompts_final.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("dss.imageocr.ocr_text_with_prompts")


# COMMAND ----------

volumes_dir="/Volumes/dss/imageocr/shampoo_data"

# COMMAND ----------

import glob

# Display images in Volumes
img_paths = f"{volumes_dir}/*.jpg"
imgs = [PIL.Image.open(path) for path in glob.glob(img_paths)]
num_imgs_to_preview = 25
show_image_grid(imgs[:num_imgs_to_preview], 3, 3) # Custom function defined in util notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage Dataset in UC Volumes
# MAGIC We create a Hugging Face Dataset object and store it in Unity Catalog Volume.

# COMMAND ----------



# COMMAND ----------

import pandas as pd

image_paths_df = pd.DataFrame(df_links.select(col('image_link')).collect()).rename(columns={0: "image_links"})
caption_df = pd.DataFrame(df_prompts_final.select(col('prompt')).collect()).rename(columns={0: "caption"})
display(image_paths_df)

# COMMAND ----------

from datasets import Dataset, Image
import PIL
from pyspark.sql.functions import col

catalog = "dss"
theme = "imageocr"
volume = "diffusion_data"

# Iterate over the list and replace the characters
content = [caption.replace("\\r\\n", "").replace("\n", "") for caption in caption_df["caption"].to_list()]
d = {
    "image": image_paths_df["image_links"].to_list(),
    "caption": content,
}

dataset = Dataset.from_dict(d).cast_column("image", Image())
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.{volume}")
dataset.save_to_disk(f"/Volumes/{catalog}/{theme}/{volume}")

# COMMAND ----------

d

# COMMAND ----------


