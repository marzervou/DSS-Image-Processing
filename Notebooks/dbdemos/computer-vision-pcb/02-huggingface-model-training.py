# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-computer-vision-pcb-maria_zervou` from the dropdown menu ([open cluster configuration](https://e2-demo-emea.cloud.databricks.com/#setting/clusters/1207-133528-7hybql3y/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('computer-vision-pcb')` or re-install the demo: `dbdemos.install('computer-vision-pcb')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Building a Computer Vision model with hugging face
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-4.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC
# MAGIC Our next step as Data Scientist is to implement a ML model to run image segmentation.
# MAGIC
# MAGIC We'll re-use the gold table built in our previous data pipeline as training dataset.
# MAGIC
# MAGIC Building such a model is greatly simplified by the use of <a href="https://huggingface.co/docs/transformers/index">huggingface transformer library</a>.
# MAGIC  
# MAGIC
# MAGIC ## MLOps steps
# MAGIC
# MAGIC While building an image segmentation model can be easily done, deploying such model in production is much harder.
# MAGIC
# MAGIC Databricks simplify this process and accelerate DS journey with the help of MLFlow by providing
# MAGIC
# MAGIC * Auto experimentation tracking to keep track of progress
# MAGIC * Simple, distributed hyperparameter tuning with hyperopt to get the best model
# MAGIC * Model packaging in MLFlow, abstracting our ML framework
# MAGIC * Model registry for governance
# MAGIC * Batch or real time serving (1 click deployment)
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=3488470091230896&notebook=%2F02-huggingface-model-training&demo_name=computer-vision-pcb&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fcomputer-vision-pcb%2F02-huggingface-model-training&version=1">

# COMMAND ----------

# DBTITLE 1,Demo Initialization
# MAGIC %run ./_resources/00-init $reset_all_data=false $db=dbdemos $catalog=manufacturing_pcb

# COMMAND ----------

# DBTITLE 1,Review our training dataset
#Setup the training experiment
init_experiment_for_batch("computer-vision-dl", "pcb")

df = spark.read.table("training_dataset_augmented")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create our Dataset from the delta table
# MAGIC
# MAGIC Hugging face makes this step very easily. All it takes is calling the `Dataset.from_spark` function. 
# MAGIC
# MAGIC Read the <a href="https://www.databricks.com/blog/contributing-spark-loader-for-hugging-face-datasets">blogbost</a> for more detail on the new Delta Loader.

# COMMAND ----------

# DBTITLE 1,Create the transformer dataset from a spark dataframe (Delta Table)   
from datasets import Dataset

dataset = Dataset.from_spark(df).rename_column("content", "image")

splits = dataset.train_test_split(test_size=0.2, seed = 42)
train_ds = splits['train']
val_ds = splits['test']

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Transfer learning with Hugging Face
# MAGIC
# MAGIC Transfer learning is the process of taking an existing model trained for another task on thousands of images, and transfering its knowledge to our domain. Hugging Face provides helper class to make transfer learning very easy to implement.
# MAGIC
# MAGIC
# MAGIC The classic process is to re-train the model or part of the model (typically the last layer) using our custom dataset.
# MAGIC
# MAGIC This provides an the best tradeoff between training cost and efficiency, especially when our training dataset is limited.

# COMMAND ----------

# DBTITLE 1,Define the base model
import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor

# pre-trained model from which to fine-tune
# Check the hugging face repo for more details & models: https://huggingface.co/google/vit-base-patch16-224
model_checkpoint = "google/vit-base-patch16-224"

#Check GPU availability
if not torch.cuda.is_available(): # is gpu
  raise Exception("Please use a GPU-cluster for model training, CPU instances will be too slow")

# COMMAND ----------

# DBTITLE 1,Define image transformations for train & validation
from PIL import Image
import io
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, Resize, ToTensor, Lambda

#Extract the model feature (contains info on pre-process step required to transform our data, such as resizing & normalization)
#Using the model parameters makes it easy to switch to another model without any change, even if the input size is different.
model_def = AutoFeatureExtractor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=model_def.image_mean, std=model_def.image_std)
byte_to_pil = Lambda(lambda b: Image.open(io.BytesIO(b)).convert("RGB"))

#Transformations on our training dataset. we'll add some crop here
train_transforms = Compose([byte_to_pil,
                            RandomResizedCrop((model_def.size['height'], model_def.size['width'])),
                            ToTensor(), #convert the PIL img to a tensor
                            normalize
                           ])
#Validation transformation, we only resize the images to the expected size
val_transforms = Compose([byte_to_pil,
                          Resize((model_def.size['height'], model_def.size['width'])),
                          ToTensor(),  #convert the PIL img to a tensor
                          normalize
                         ])

# Add some random resiz & transformation to our training dataset
def preprocess_train(batch):
    """Apply train_transforms across a batch."""
    batch["image"] = [train_transforms(image) for image in batch["image"]]
    return batch

# Validation dataset
def preprocess_val(batch):
    """Apply val_transforms across a batch."""
    batch["image"] = [val_transforms(image) for image in batch["image"]]
    return batch
  
#Set our training / validation transformations
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# COMMAND ----------

# DBTITLE 1,Build our model from 
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

#Mapping between class label and value (huggingface use it during inference to output the proper label)
label2id, id2label = dict(), dict()
for i, label in enumerate(set(dataset['label'])):
    label2id[label] = i
    id2label[i] = label
    
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine tuning our model 
# MAGIC
# MAGIC Our dataset and model is ready. We can now start the training step to fine-tune the model.
# MAGIC
# MAGIC *Note that for production-grade use-case, we would typically to do some [hyperparameter](https://huggingface.co/docs/transformers/hpo_train) tuning here. We'll keep it simple for this first example and run it with fixed settings.*
# MAGIC

# COMMAND ----------

# DBTITLE 1,Training parameters
model_name = model_checkpoint.split("/")[-1]
batch_size = 32 # batch size for training and evaluation

args = TrainingArguments(
    f"/tmp/huggingface/pcb/{model_name}-finetuned-leaf",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False
)

# COMMAND ----------

# DBTITLE 1,define our evaluation metric
import numpy as np
import evaluate
# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.

# Let's evaluate our model against a F1 score. Keep it as binary for this demo (we don't classify by default type)
accuracy = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# COMMAND ----------

# DBTITLE 1,Start our Training and log the model to MLFlow
import mlflow
import torch
from transformers import pipeline, DefaultDataCollator, EarlyStoppingCallback

def collate_fn(examples):
    pixel_values = torch.stack([e["image"] for e in examples])
    labels = torch.tensor([label2id[e["label"]] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}

#Make sure the model is trained on GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#mlflow.autolog(log_models=False)
with mlflow.start_run(run_name="hugging_face") as run:
  early_stop = EarlyStoppingCallback(early_stopping_patience=10)
  trainer = Trainer(model, args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=model_def, compute_metrics=compute_metrics, data_collator=collate_fn, callbacks = [early_stop])

  train_results = trainer.train()

  #Build our final hugging face pipeline
  classifier = pipeline("image-classification", model=trainer.state.best_model_checkpoint, tokenizer = model_def, device_map='auto')
  #log the model to MLFlow
  reqs = mlflow.transformers.get_default_pip_requirements(model)
  mlflow.transformers.log_model(artifact_path="model", transformers_model=classifier, pip_requirements=reqs)
  mlflow.set_tag("dbdemos", "pcb_classification")
  mlflow.log_metrics(train_results.metrics)

# COMMAND ----------

# DBTITLE 1,Let's try our model to make sure it works as expected
test = spark.read.table("training_dataset_augmented").where("filename = '010.JPG'").toPandas()
img = Image.open(io.BytesIO(test.iloc[0]['content']))
print(f"predictions: {classifier(img)}")
img

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model deployment
# MAGIC
# MAGIC Our model is now trained. All we have to do is save it in our Model Registry and move it as Production ready. <br/>
# MAGIC For this demo we'll use our lastes run, but we could also search the best run with ` mlflow.search_runs` (based on the metric we defined during training).

# COMMAND ----------

#Save the model in the registry & move it to Production
model_registered = mlflow.register_model("runs:/"+run.info.run_id+"/model", "dbdemos_pcb_classification")
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
#Move the model as Production
client.transition_model_version_stage(name = "dbdemos_pcb_classification", version = model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Inference in batch and real-time 
# MAGIC
# MAGIC Our model is now trained and registered in MLflow Model Registry. Databricks mitigates the need for a lot of the anciliary code to train a model, so that you can focus on improving your model performance.
# MAGIC
# MAGIC The next step is now to use this model for inference - in batch or real-time behind a REST endpoint.
# MAGIC
# MAGIC Open the next [03-running-cv-inferences notebook]($./03-running-cv-inferences) to see how to leverage Databricks serving capabilities.
