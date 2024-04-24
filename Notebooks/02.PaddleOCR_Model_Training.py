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

link = '/Volumes/dss/imageocr/training_data/images/0000971160.png'
# response = requests.get(link)
image = Image.open(link)
np_img = np.array(image)
image

# COMMAND ----------

#link = 'https://images.meesho.com/images/products/142209432/gfowe_512.jpg'
#response = requests.get(link)
#image = Image.open(io.BytesIO(response.content)).convert("RGB")
#np_img = np. array(image).astype(np.float32)
#image

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
        response = requests.get(link)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        np_img = np. array(image).astype(np.float32)
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
        np_image = self._preprocess(input_data['image_link'][0])
        ocr_text = self.get_paddleocr_text(np_image)
        return pd.DataFrame({"ocr_text":[ocr_text]})


# COMMAND ----------

custom_ocr = CustomPaddleOCRModel()

# COMMAND ----------

from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import DataType, Schema, ColSpec

link = 'https://images.meesho.com/images/products/142209432/gfowe_512.jpg'
input_data = pd.DataFrame({'image_link':[link]})
print(input_data)
custom_ocr.load_context(None)
output_data = custom_ocr.predict(None, input_data)
print(output_data)



# COMMAND ----------

output_data

# COMMAND ----------

ocr_signature = infer_signature(input_data, output_data)
print(ocr_signature)

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

with mlflow.start_run(run_name="paddle_ocr") as run:
    mlflow.pyfunc.log_model(
            "model",
            python_model=CustomPaddleOCRModel(),
            input_example=input_data,
            signature=ocr_signature,
            conda_env=default_conda_env)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

# COMMAND ----------

loaded_model.predict(input_data)

# COMMAND ----------

reg_model_info = mlflow.register_model(f"runs:/{run.info.run_id}/model", 'dss_signature_paddle_ocr')

# COMMAND ----------

mlflow.models.utils.add_libraries_to_model(f"models:/dss_signature_paddle_ocr/{reg_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### realtime testing

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
  url = ''
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
