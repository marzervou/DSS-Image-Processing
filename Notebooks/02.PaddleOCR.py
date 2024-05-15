# Databricks notebook source
# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install paddlepaddle-gpu
# MAGIC %pip install paddleocr

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from paddleocr import PaddleOCR,draw_ocr

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the paddle OCR cod

# COMMAND ----------

import paddle
import requests
import io
import cv2
import time
from PIL import Image
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR

# COMMAND ----------

pddl_ocr = PaddleOCR(lang='en', use_gpu=False)

# COMMAND ----------

link = '/Volumes/dss/imageocr/shampoo_data/L_Oreal_Paris_Elvive_Dream_Lengths_Conditioner_Cream_back.jpg'
# response = requests.get(link)
image = Image.open(link)
np_img = np.array(image)
image

# COMMAND ----------

result = pddl_ocr.ocr(np_img, cls=False)
result

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
import torch

# COMMAND ----------

class CustomPaddleOCRModel(PythonModel):
    from paddleocr import PaddleOCR
    import numpy as np
    import requests
    import pandas as pd
    import io
    from PIL import Image
    from time import time
    import paddle
    import torch

    def __init__(self):
        super().__init__()

    def load_context(self, context):
        #self.is_gpu = torch.cuda.is_available()
        #paddle.utils.run_check()
        self.paddleocr = PaddleOCR(lang='en', use_gpu=False)

    def _preprocess(self, link):
        # response = requests.get(link)
        image = Image.open(link)
        np_img = np.array(image)
        return np_img
    
    def get_paddleocr_text(self, np_arry):
        result = self.paddleocr.ocr(np_arry, cls=False)
        output = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                output.append(line[1][0])
        output = ' '.join(output)
        return output  
    
    def predict(self, context, input_data):
        ocr_text_list=[]
        for item in range(0,len(input_data)):
            np_image = self._preprocess(input_data['image_link'][item])
            ocr_text = self.get_paddleocr_text(np_image)
            ocr_text_list.append(ocr_text)

        return pd.DataFrame({"ocr_text":ocr_text_list})


# COMMAND ----------

custom_ocr = CustomPaddleOCRModel()

# COMMAND ----------

from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import DataType, Schema, ColSpec

link = '/Volumes/dss/imageocr/shampoo_data/L_Oreal_Paris_Elvive_Dream_Lengths_Conditioner_Cream_back.jpg'
input_data = pd.DataFrame({'image_link':[link]})
# print(input_data)
custom_ocr.load_context(None)
output_data = custom_ocr.predict(None, input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

ocr_signature = infer_signature(input_data, output_data)


# COMMAND ----------

#pip_requirements = [f'numpy=={np.__version__}','paddlepaddle-gpu==2.5.1','paddleocr','cudatoolkit==11.7','torch==1.13.1']

pip_requirements = [f'numpy=={np.__version__}','paddleocr','torch==1.13.1']
default_conda_env = mlflow.pyfunc.get_default_conda_env()
default_conda_env['channels'].append("https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/")
#default_conda_env['dependencies'].append('cudatoolkit=11.7')
default_conda_env['dependencies'][2]['pip'].append(f'numpy=={np.__version__}')
default_conda_env['dependencies'][2]['pip'].append('torch==1.13.1')
#default_conda_env['dependencies'][2]['pip'].append('paddlepaddle-gpu==2.5.1')
default_conda_env['dependencies'][2]['pip'].append('paddleocr')
default_conda_env

# COMMAND ----------

with mlflow.start_run(run_name="shampoo_paddle_ocr") as run:
    mlflow.pyfunc.log_model(
            "model",
            python_model=CustomPaddleOCRModel(),
            input_example=input_data,
            signature=ocr_signature,
            conda_env=default_conda_env)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
# loaded_model = mlflow.pyfunc.load_model('runs:/8d2a26c33be94d0dbcdcac7043d8fe9d/model_wheels')


# COMMAND ----------

loaded_model.predict(input_data)

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
reg_model_info = mlflow.register_model(f"runs:/{run.info.run_id}/model", 'dss.imageocr.mz_shampoo_paddle_ocr')

# COMMAND ----------

reg_model_info

# COMMAND ----------

from mlflow.models.utils import add_libraries_to_model
add_libraries_to_model(f"models:/dss.imageocr.mz_shampoo_paddle_ocr/{reg_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC Apply the modle to the bacth of data

# COMMAND ----------

from pyspark.sql.functions import input_file_name, regexp_replace

df = spark.read.format("image").load("/Volumes/dss/imageocr/shampoo_data/")
df = df.withColumn("image_link", input_file_name())
df = df.withColumn("image_link", regexp_replace("image_link", "dbfs:", ""))
input_data = df.toPandas()
custom_ocr.load_context(None)
output_data = custom_ocr.predict(None, input_data)
input_data['ocr_text'] = output_data


# COMMAND ----------

spark_df = spark.createDataFrame(input_data)
spark_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("dss.imageocr.clean_data")


# COMMAND ----------

spark_df.show()
