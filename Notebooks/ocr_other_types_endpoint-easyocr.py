# Databricks notebook source
# MAGIC %md
# MAGIC #### EasyOCR python packages

# COMMAND ----------

# MAGIC %pip install easyocr

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the Easy OCR code

# COMMAND ----------

import requests
import io
import cv2
import time
from PIL import Image
import numpy as np
import pandas as pd
import easyocr

# COMMAND ----------

ocr_reader = easyocr.Reader(['en'], gpu=True, detector=True, recognizer=True)

# COMMAND ----------

link = '/Volumes/dss/imageocr/training_data/images/0000971160.png'
# response = requests.get(link)
image = Image.open(link)
np_img = np.array(image)
image

# COMMAND ----------

result = ocr_reader.readtext(image=np_img)
result

# COMMAND ----------

# MAGIC %md
# MAGIC #### create paddle model

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
import torch

# COMMAND ----------

class CustomEasyOCRModel(PythonModel):
    import numpy as np
    import requests
    import pandas as pd
    import io
    from PIL import Image
    from time import time
    import torch

    def __init__(self):
        super().__init__()

    def load_context(self, context):
        self.is_gpu = torch.cuda.is_available()
        self.ocr = easyocr.Reader(['en'], gpu=self.is_gpu, detector=True, recognizer=True)

    def _preprocess(self, link):
        image = Image.open(link)
        np_img = np.array(image)
        return np_img
    
    def get_paddleocr_text(self, np_arry):
        result = self.ocr.readtext(np_arry)
        output = []
        for idx in range(len(result)):
            res = result[idx][1]
            output.append(res)
        output = ' '.join(output)
        return output  
    
    def predict(self, context, input_data):
        np_image = self._preprocess(input_data['image_link'][0])
        ocr_text = self.get_paddleocr_text(np_image)
        return pd.DataFrame({"ocr_text":[ocr_text]})


# COMMAND ----------

custom_ocr = CustomEasyOCRModel()

# COMMAND ----------

from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import DataType, Schema, ColSpec

link = '/Volumes/dss/imageocr/training_data/images/0000971160.png'
input_data = pd.DataFrame({'image_link':[link]})
print("input_data: ",input_data)

custom_ocr.load_context(None)
output_data = custom_ocr.predict(None, input_data)
print("output_data: ",output_data)

ocr_signature = infer_signature(input_data, output_data)
print("ocr_signature: ",ocr_signature)

pip_requirements = [f'numpy=={np.__version__}','cudatoolkit==11.7','torch==1.13.1']

default_conda_env = mlflow.pyfunc.get_default_conda_env()
default_conda_env['dependencies'].append('cudatoolkit=11.7')
default_conda_env['dependencies'][2]['pip'].append(f'numpy=={np.__version__}')
default_conda_env['dependencies'][2]['pip'].append('torch==1.13.1')
default_conda_env

# COMMAND ----------

with mlflow.start_run(run_name="easy_ocr_gpu") as run:
    mlflow.pyfunc.log_model(
            "model",
            python_model=CustomEasyOCRModel(),
            input_example=input_data,
            signature=ocr_signature,
            conda_env=default_conda_env)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

# COMMAND ----------

loaded_model.predict(input_data)

# COMMAND ----------

reg_model_info = mlflow.register_model(f"runs:/{run.info.run_id}/model", 'mz_easy_ocr')

# COMMAND ----------

mlflow.models.utils.add_libraries_to_model(f"models:/mz_easy_ocr/{reg_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Serve Real Time with MLflow Serving

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
import time
import datetime 

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  
  token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)  
  url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/mz_easy_ocr/invocations'
  
  headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  resp_json = response.json()
  print(response.elapsed.total_seconds())
  
  return resp_json['predictions'][0]

# COMMAND ----------

response = score_model(input_data)
response

# COMMAND ----------

# MAGIC %md
# MAGIC #### Serve in Batch/Streaming Mode

# COMMAND ----------


