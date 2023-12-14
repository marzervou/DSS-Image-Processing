# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-computer-vision-pcb-maria_zervou` from the dropdown menu ([open cluster configuration](https://e2-demo-emea.cloud.databricks.com/#setting/clusters/1207-133528-7hybql3y/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('computer-vision-pcb')` or re-install the demo: `dbdemos.install('computer-vision-pcb')`*

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # PCB - Ingestion Data Pipeline 
# MAGIC
# MAGIC This is the pipeline we will be building. We ingest 2 datasets, namely:
# MAGIC
# MAGIC * The raw images (jpg) containing PCB
# MAGIC * The label, the type of anomalies saved as CSV files
# MAGIC
# MAGIC We will first focus on building a data pipeline to incrementally load this data and create a final Gold table.
# MAGIC
# MAGIC This table will then be used to train a ML Classification model to learn to detect anomalies in our images in real time!
# MAGIC
# MAGIC *Note that this demo leverage the standard spark API. You could also implement this same pipeline in pure SQL leveraging Delta Live Tables. For more details on DLT, install `dbdemos.install('dlt-loans)`*
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=3488470091230896&notebook=%2F01-ingestion-and-ETL&demo_name=computer-vision-pcb&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fcomputer-vision-pcb%2F01-ingestion-and-ETL&version=1">

# COMMAND ----------

# MAGIC %run ./_resources/00-init $reset_all_data=false $db=dbdemos $catalog=manufacturing_pcb

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Reviewing the incoming dataset
# MAGIC
# MAGIC The dataset was downloaded for you automatically and is available in cloud your dbfs storage folder. Let's explore the data:

# COMMAND ----------

# MAGIC %fs ls /dbdemos/manufacturing/pcb/Images/Normal/

# COMMAND ----------

# MAGIC %fs ls /dbdemos/manufacturing/pcb/labels

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCB images inspection
# MAGIC
# MAGIC We can display images with `matplotlib` in a native python way.
# MAGIC
# MAGIC Let us investigate what a normal image looks like, and then one with an anomaly.

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=300):
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.imshow(img, interpolation="nearest", aspect="auto")


display_image("/dbfs/dbdemos/manufacturing/pcb/Images/Normal/0000.JPG")
display_image("/dbfs/dbdemos/manufacturing/pcb/Images/Anomaly/000.JPG")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Ingesting raw images with Databricks Autoloader
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-1.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC The first step is to load the individual JPG images. This can be quite challenging at scale, especially for incremental load (consume only the new one).
# MAGIC
# MAGIC Databricks Autoloader can easily handle all type of format and make it very easy to ingest new datasets.
# MAGIC
# MAGIC Autoloader will garantee that only new files are being processed while scaling with millions of individual images. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load binary files with Auto Loader
# MAGIC
# MAGIC We can now use the Auto Loader to load images, and spark function to create the label column. Autoloader will automatically create the table and tune it accordingly, disabling compression for binary among other.
# MAGIC
# MAGIC We can also very easily display the content of the images and the labels as a table.

# COMMAND ----------

(spark.readStream.format("cloudFiles")
                 .option("cloudFiles.format", "binaryFile")
                 .option("pathGlobFilter", "*.JPG")
                 .option("recursiveFileLookup", "true")
                 .option("cloudFiles.schemaLocation", "/dbdemos/manufacturing/pcb/stream/pcb_schema")
                 .load(f"/dbdemos/manufacturing/pcb/Images/")
    .withColumn("filename", F.substring_index(col("path"), "/", -1))
    .writeStream.trigger(availableNow=True)
                .option("checkpointLocation", f"/dbdemos/manufacturing/pcb/stream/pcb_checkpoint")
                .toTable("pcb_images").awaitTermination())

spark.sql("ALTER TABLE pcb_images OWNER TO `account users`")
display(spark.table("pcb_images"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load CSV label files with Auto Loader
# MAGIC CSV files can easily be loaded using Databricks [Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html), including schema inference and evolution.

# COMMAND ----------

(spark.readStream.format("cloudFiles")
                 .option("cloudFiles.format", "csv")
                 .option("header", True)
                 .option("cloudFiles.schemaLocation", "/dbdemos/manufacturing/pcb/stream/labels_schema")
                 .load(f"/dbdemos/manufacturing/pcb/labels/")
      .withColumn("filename", F.substring_index(col("image"), "/", -1))
      .select("filename", "label")
      .withColumnRenamed("label", "labelDetail")
      .writeStream.trigger(availableNow=True)
                  .option("checkpointLocation", "/dbdemos/manufacturing/pcb/stream/labels_checkpoint")
                  .toTable("pcb_labels").awaitTermination())

spark.sql("ALTER TABLE pcb_labels OWNER TO `account users`")
display(spark.table("pcb_labels"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's now merge the labels and the images tables
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-2.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC Note that we're working with delta tables to make the ingestion simple. 
# MAGIC
# MAGIC You don't have to worry about individual small images anymore.
# MAGIC
# MAGIC We can do the join operation either in python or SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE training_dataset AS 
# MAGIC   (SELECT 
# MAGIC     *, CASE WHEN labelDetail = 'normal' THEN 'normal' ELSE 'damaged' END as label
# MAGIC     FROM pcb_images INNER JOIN pcb_labels USING (filename)) ;
# MAGIC
# MAGIC ALTER TABLE training_dataset OWNER TO `account users`;
# MAGIC
# MAGIC SELECT * FROM training_dataset LIMIT 10;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Final step: Preparing and augmenting our image dataset for DL Fine Tuning
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-3.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC Note that we're working with tables. We can do this transformation in python or SQL.
# MAGIC
# MAGIC Some transformation on image can be expensive. We can leverage Spark to distribute some image pre-processing first.
# MAGIC
# MAGIC In this example, we will do the following:
# MAGIC - crop the image in the center to make them square (the model we use for fine-tuning take square images)
# MAGIC - resize our image to smaller a resolution (256x256) as our models won't take use image in high resolution. 
# MAGIC
# MAGIC We will also augment our dataset to add more "damaged" items as we have here something fairly imbalanced (only 1 on 10 item has an anomaly). <br/>
# MAGIC It looks like our system takes pcb pictures upside/down without preference and that's how our inferences will be. Let's then flip all the damaged images horizontally and add them back in our dataset.
# MAGIC
# MAGIC *Note: if you're using deltatorch, you can directly split your test/training dataset and add an id column for each here directly. For more detail open the 04-ADVANCED-pytorch-training-and-inference notebook*

# COMMAND ----------

# DBTITLE 1,Crop and resize our images
from PIL import Image
import io
from pyspark.sql.functions import pandas_udf
IMAGE_RESIZE = 256

#Resize UDF function
@pandas_udf("binary")
def resize_image_udf(content_series):
  def resize_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    width, height = image.size   # Get dimensions
    new_size = min(width, height)
    # Crop the center of the image
    image = image.crop(((width - new_size)/2, (height - new_size)/2, (width + new_size)/2, (height + new_size)/2))
    #Resize to the new resolution
    image = image.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(resize_image)


# add the metadata to enable the image preview
image_meta = {"spark.contentAnnotation" : '{"mimeType": "image/jpeg"}'}

(spark.table("training_dataset")
      .withColumn("sort", F.rand()).orderBy("sort").drop('sort') #shuffle the DF
      .withColumn("content", resize_image_udf(col("content")).alias("content", metadata=image_meta))
      .write.mode('overwrite').saveAsTable("training_dataset_augmented"))

spark.sql("ALTER TABLE training_dataset_augmented OWNER TO `account users`")

# COMMAND ----------

# DBTITLE 1,Flip and add damaged images
import PIL
@pandas_udf("binary")
def flip_image_horizontal_udf(content_series):
  def flip_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    #Flip
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(flip_image)

(spark.table("training_dataset_augmented")
    .filter("label == 'damaged'")
    .withColumn("content", flip_image_horizontal_udf(col("content")).alias("content", metadata=image_meta))
    .write.mode('append').saveAsTable("training_dataset_augmented"))

# COMMAND ----------

# DBTITLE 1,Final dataset has now 20% damaged images
# MAGIC %sql select label, count(*) from training_dataset_augmented group by label

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Our dataset is ready for our Data Science team
# MAGIC
# MAGIC That's it! We have now deployed a production-ready ingestion pipeline.
# MAGIC
# MAGIC Our images are incrementally ingested and joined with our label dataset.
# MAGIC
# MAGIC Let's see how this data can be used by a Data Scientist to [build our Computer Vision model]($./02-huggingface-model-training).
