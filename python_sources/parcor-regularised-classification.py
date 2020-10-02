#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %pip install transformers
# %pip install tensor2tensor


# In[ ]:





# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
interativeEnvironment=False
localEnvironment=False
IN_COLAB=False
try:
    from kaggle_datasets import KaggleDatasets
except ImportError as e:
  try:
    import google.colab
    IN_COLAB = True
  except:
    IN_COLAB = False
    localEnvironment=True
import transformers
from transformers import TFAutoModel, AutoTokenizer
# from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig

from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import os
os.environ['XLA_USE_BF16'] = "1"
localEnvironment
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', None)
import warnings
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
# from pandarallel import pandarallel

# pandarallel.initialize(nb_workers=4, progress_bar=True)
from collections import Counter


# In[ ]:


def is_interactive():
   return 'runtime' in get_ipython().config.IPKernelApp.connection_file

print('Interactive?', is_interactive())


# In[ ]:


# from pandarallel import pandarallel

# pandarallel.initialize(nb_workers=2, progress_bar=True)


# In[ ]:


tf_env="tensorflow"


# In[ ]:


fit_verbocity=1
model=tf_env
nrows=None
interativeEnvironment=is_interactive()
interativeEnvironment=False
if localEnvironment:
#     nrows=200
    warnings.warn("Nrows limited")
    pass
elif interativeEnvironment:
    pass
else:
    fit_verbocity=2
pooling_mode_cls_token="CLS_TOKEN"
pooling_mode_fc="FC"
pooling_mode=pooling_mode_fc
fc_dims=[]
trainable_transformer=False
use_dann=False
dann_lambda=-1
use_parcor=False
parcor_lambda=1
use_augmented_data=False
use_lowercase_data=False
use_finetuning=True
use_validation_during_pretraining=True
use_translated_data=False
use_pretraining=False
load_model="../input/ver-22-parcor-output/21_06_2020_05_03_28_zero_shot_pro.h5"
use_english_validation=False
use_label_rounding=True
use_label_filtering=True


# In[ ]:


SEED = 42

# BACKBONE_PATH = '../input/multitpu-inference'
# CHECKPOINT_PATH = '../input/multitpu-inference/checkpoint-xlm-roberta.bin'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# seed_everything(SEED)    


# In[ ]:


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return tf.math.scalar_mul(-1,dy)
    return y, custom_grad
class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

# Data access
if not localEnvironment  and not IN_COLAB:
    GCS_DS_PATH = KaggleDatasets().get_gcs_path("jigsaw-multilingual-toxic-comment-classification")

# Configuration
NEXAMPLESPEREPOCH=240000
EPOCHS = 10
if strategy.num_replicas_in_sync==1:
    BATCH_SIZE = 4 * strategy.num_replicas_in_sync
else:
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192
# MAX_LEN = 512

MODEL = 'jplu/tf-xlm-roberta-large'
if localEnvironment:
    MODEL='jplu/tf-xlm-roberta-base'
if IN_COLAB:
  MODEL='jplu/tf-xlm-roberta-base'
print("BATCH_SIZE: ", BATCH_SIZE)


# In[ ]:


if IN_COLAB:
  project_id="global-sun-279412"


# In[ ]:


get_ipython().system('gcloud config set project {project_id}')


# In[ ]:


bucket_name="jigsaw-tfrecords"


# In[ ]:


if IN_COLAB:
  from google.colab import auth
  auth.authenticate_user()


# In[ ]:


# if IN_COLAB:
#   from google.colab import drive
#   drive.mount('/content/gdrive')


# In[ ]:


get_ipython().system('gsutil ls gs://{bucket_name}/')


# In[ ]:


# # First load the real tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL)


# In[ ]:


if localEnvironment:
    root =".."
elif IN_COLAB:
    root='gs://jigsaw-tfrecords/'
else:
    root="/kaggle"
root


# In[ ]:


# validation_fold_index=random.choice(range(5))
validation_fold_index=4
validation_fold_index


# In[ ]:


def validationFoldRemoveFilter(validation_fold_index):
    def _filter(features):
#         label=example["toxic"]
#         print(label)
        return tf.math.not_equal(features["ShardIndex"],validation_fold_index)
    return _filter


# In[ ]:


def validationFoldOnlyFilter(validation_fold_index):
    def _filter(features):
#         label=example["toxic"]
#         print(label)
        return tf.math.equal(features["ShardIndex"],validation_fold_index)
    return _filter


# In[ ]:


if IN_COLAB:
    TF_DS_PATH=root+"jigsaw-official"
elif not localEnvironment:
    TF_DS_PATH = KaggleDatasets().get_gcs_path("jigsawaugmented")+"/tfrecords"
#     if use_augmented_data:
#         TF_DS_PATH = KaggleDatasets().get_gcs_path("jigsawaugmented")+"/tfrecords"
#     else:
#         TF_DS_PATH = KaggleDatasets().get_gcs_path("jigsaw-as-tf-record")
else:
    TF_DS_PATH = "../tfrecords"
TF_DS_PATH


# In[ ]:


file="en1_train_non_toxic_PropercaseSentences_0_4.tfrecord"
("Sentences" not in file) or ("LowercaseSentences" in file)


# In[ ]:



data_files=tf.io.gfile.glob(TF_DS_PATH + "/*.tfrecord")
if use_augmented_data:
    pass
elif use_lowercase_data:
    data_files=list(filter(lambda file:("Sentences" not in file) or ("LowercaseSentences" in file),data_files))
else:
    data_files=list(filter(lambda file:"Sentences" not in file,data_files))
len(data_files),data_files


# In[ ]:


toxic_en_files=list(filter(lambda file:"nen" not in file and "toxic" in file and "non_toxic" not in file and "valid" not in file and "sharded" not in file, data_files))
toxic_en_files


# In[ ]:


non_toxic_en_files=list(filter(lambda file:"nen" not in file and "non_toxic" in file and "valid" not in file and "sharded" not in file, data_files))
non_toxic_en_files


# In[ ]:


non_toxic_nen_files=list(filter(lambda file:"non_toxic" in file and "nen1" in file, data_files))
non_toxic_nen_files


# In[ ]:


toxic_nen_files=list(filter(lambda file:"non_toxic" not in file and "nen1" in file, data_files))
toxic_nen_files


# In[ ]:


unknown_nen_files=list(filter(lambda file: "test" not in file and "nen2" in file, data_files))
unknown_nen_files


# In[ ]:


valid_files=list(filter(lambda file:"test" not in file and "nen1" in file and "Sentences" not in file, data_files)) 
valid_files


# In[ ]:



if IN_COLAB:
  TRANLSATED_DATA_PATH_ONE=root+"/translated-train-bias-all-langs"
  TRANLSATED_DATA_PATH_TWO=root+"/jigsaw-train-multilingual-coments-google-api"
elif not localEnvironment:
    TRANSLATED_DATA_PATH = KaggleDatasets().get_gcs_path("jigsawtranslated")+"/tfrecords-translated"

else:
    pass
TRANSLATED_DATA_PATH


# In[ ]:


translated_files=[]

translated_files=tf.io.gfile.glob(TRANSLATED_DATA_PATH + "/*tfrecord")
translated_files[:3]


# In[ ]:


toxic_translated_files=list(filter(lambda file:"nen" not in file and "toxic" in file and "non" not in file and "valid" not in file and "sharded" not in file, translated_files))
random.choice(toxic_translated_files)


# In[ ]:


non_toxic_translated_files=list(filter(lambda file:"nen" not in file and "toxic" in file and "non" in file and "valid" not in file and "sharded" not in file, translated_files))
random.choice(non_toxic_translated_files)


# In[ ]:


# tf.data.experimental.CsvDataset(
#     "my_file*.csv",
#     [tf.float32,  # Required field, use dtype or empty tensor
#      tf.constant([0.0], dtype=tf.float32),  # Optional field, default to 0.0
#      tf.int32,  # Required field, use dtype or empty tensor
#      ],
#     select_cols=[1,2,3]  # Only parse last three columns
# )


# In[ ]:


opus_langs=[['en', 'it'],
 ['en', 'ru'],
 ['en', 'tr'],
 ['en', 'pt_br'],
 ['en', 'fr'],
 ['en', 'pt'],
 ['en', 'es'],
 ['en', 'hi']]
if IN_COLAB:
  OPUS_DS_PATH=root+"opus"
elif not localEnvironment:
    OPUS_DS_PATH = KaggleDatasets().get_gcs_path("opus-for-jigsaw-sharded")
else:
    OPUS_DS_PATH = "../opus-processed"
OPUS_DS_PATH


# In[ ]:


if IN_COLAB:
  extension=".gz"
else:
  extension=""
opus_files=tf.io.gfile.glob(OPUS_DS_PATH + "/*.tfrecord"+extension)
if interativeEnvironment:
  print(opus_files)
    


# In[ ]:


# Create a description of the features.
feature_description = {
    'tokens': tf.io.FixedLenFeature([192], tf.int64),
    'mask': tf.io.FixedLenFeature([192], tf.int64),
#     'types': tf.io.FixedLenFeature([192], tf.int64, default_value=[0]*192),
    'toxic': tf.io.FixedLenFeature([], tf.float32),
    's_toxic': tf.io.FixedLenFeature([], tf.float32),
    'source':tf.io.FixedLenFeature([], tf.int64),
    'toxicity_annotator_count':tf.io.FixedLenFeature([], tf.int64),
    'lang':tf.io.FixedLenFeature([], tf.string),
    'ShardIndex':tf.io.FixedLenFeature([], tf.int64),
}

def _parse_proto(example_proto):
#     tf.print(example_proto)
  # Parse the input `tf.Example` proto using the dictionary above.
    example= tf.io.parse_single_example(example_proto, feature_description)
#     return example
    return example


def _form_tuple(example):
  y={"toxic":tf.cast(example['s_toxic'],tf.float32)}
  if use_dann:
    y["lang"]=tf.cast(tf.math.equal(example['lang'],'en'),tf.float32)
  if use_parcor:
    y["is_same"]=example["is_same"]


  return (example['tokens'],example['mask']),y

def smoother(smoothing=0.01):
    def smooth(example):

        label=tf.cast(example['s_toxic'],tf.float32)
#         tf.print(label)
        label=tf.cond(label>=1.0-smoothing,lambda :tf.ones_like(label)-smoothing,lambda :tf.cond(label<0.0+smoothing,lambda :tf.zeros_like(label)+smoothing,lambda :label))
        example['s_toxic']=label
        return example
    return smooth
def isValidLabel(example):
    example['isLabelValid']=tf.cond(example['toxic']==-1,lambda :tf.constant(0.0),lambda :tf.constant(1.0))
    return example

def _filter_non_confident(example):
    return (example["s_toxic"]>0.5) or (example["s_toxic"]<0.25)


# In[ ]:


def roundLabels(example):
    label=tf.cast(example['s_toxic'],tf.float32)
#         tf.print(label)
    label=tf.cond(label>=0.5,lambda :tf.ones_like(label),lambda :tf.zeros_like(label))
    example['s_toxic']=label
    return example


# In[ ]:


def perFileDataset(filename):
    tf.print("Retrieving from filename",filename)
    return tf.data.TFRecordDataset(filename).map(_parse_proto)
def perFileDatasetShuffled(filename):
    return perFileDataset(filename).shuffle(200000)

def unsqueezeTargetDimensions(features,labels):
    labels=tf.cast(labels,tf.float32)
    return features,tf.stack([tf.math.subtract(tf.constant([1.0]),labels),labels],axis=-1),


# In[ ]:


def toxic_train_en_dataset(shardIndex=-1):
    return tf.data.Dataset.from_tensor_slices(toxic_en_files).shuffle(2048).interleave(perFileDataset, num_parallel_calls=AUTO,cycle_length=12).map(smoother())
def toxic_train_translated_dataset(shardIndex=-1):
    return tf.data.Dataset.from_tensor_slices(toxic_translated_files).shuffle(2048*4).interleave(perFileDataset, num_parallel_calls=AUTO,cycle_length=12).map(smoother(0.1))


# In[ ]:


def non_toxic_train_en_dataset(shardIndex=-1):
   return tf.data.Dataset.from_tensor_slices(non_toxic_en_files).shuffle(2048).interleave(perFileDataset, num_parallel_calls=AUTO,cycle_length=12).map(smoother())
def non_toxic_train_translated_dataset(shardIndex=-1):
    return tf.data.Dataset.from_tensor_slices(non_toxic_translated_files).shuffle(2048*4).interleave(perFileDataset, num_parallel_calls=AUTO,cycle_length=12).map(smoother(0.1))


# In[ ]:


def toxic_train_nen_dataset(shardIndex=-1):
    return tf.data.Dataset.from_tensor_slices(toxic_nen_files).shuffle(2048).interleave(perFileDataset).filter(validationFoldRemoveFilter(validation_fold_index)).map(smoother()).shuffle(8000)


# In[ ]:


def non_toxic_train_nen_dataset():
    return tf.data.Dataset.from_tensor_slices(non_toxic_nen_files).shuffle(2048).interleave(perFileDataset).filter(validationFoldRemoveFilter(validation_fold_index)).map(smoother()).shuffle(8000)


# In[ ]:


def unknown_nen_dataset():
    return tf.data.Dataset.from_tensor_slices(unknown_nen_files).shuffle(2048).interleave(perFileDataset,num_parallel_calls=AUTO)


# In[ ]:


def valid_nen_dataset():
    return tf.data.Dataset.from_tensor_slices(valid_files).interleave(perFileDataset).filter(validationFoldOnlyFilter(validation_fold_index))#.map(_form_tuple)


# In[ ]:


def finetune_nen_dataset():
    return tf.data.Dataset.from_tensor_slices(valid_files).shuffle(2048).interleave(perFileDataset).filter(validationFoldRemoveFilter(validation_fold_index))#.map(_form_tuple)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if interativeEnvironment:\n    total_by_source=Counter()\n    total_by_toxic=Counter()\n    total=0\n    for i,l in tqdm(enumerate(non_toxic_train_nen_dataset())):\n        total+=1\n        total_by_source.update([l["source"].numpy()])\n        total_by_toxic.update([l["s_toxic"].numpy()])\n    total_nen_train_non_toxic=total\n    total,total_by_toxic.most_common(),total_by_source.most_common()\nelse:\n    total_nen_train_non_toxic=5956')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if interativeEnvironment:\n    total_by_source=Counter()\n    total_by_toxic=Counter()\n    total=0\n    for i,l in tqdm(enumerate(toxic_train_nen_dataset())):\n        total+=1\n        total_by_source.update([l["source"].numpy()])\n        total_by_toxic.update([l["s_toxic"].numpy()])\n    total_nen_train_toxic=total\n    total,total_by_toxic.most_common(),total_by_source.most_common()\nelse:\n    total_nen_train_toxic=1061')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if interativeEnvironment:\n\n    total_by_source=Counter()\n    total_by_toxic=Counter()\n    total=0\n    for i,l in tqdm(enumerate(unknown_nen_dataset())):\n        total+=1\n        total_by_source.update([l["source"].numpy()])\n        total_by_toxic.update([l["s_toxic"].numpy()])\n    total_nen_unknown=total\n    total,total_by_toxic.most_common(),total_by_source.most_common()\nelse:\n    total_nen_unknown=71078')


# In[ ]:


n_steps = NEXAMPLESPEREPOCH // BATCH_SIZE
num_nen=0
if use_validation_during_pretraining:
    num_nen+=total_nen_train_toxic+total_nen_train_non_toxic
if use_dann:
    num_nen += total_nen_unknown
# n_steps=20
# valid_dataset=list(valid_dataset)[:20]
n_steps,num_nen


# In[ ]:


def applyOptions(ds):
    if use_label_filtering:
        ds=ds.filter(_filter_non_confident)
    if use_label_rounding:
        ds=ds.map(roundLabels)
        ds=ds.map(smoother())
    return ds
num_dataset_section=2
if use_translated_data:
    num_dataset_section+=2
def f0():
    return unknown_nen_dataset().repeat().take(total_nen_unknown)
def f1():
    return applyOptions(toxic_train_en_dataset().repeat().take(int((NEXAMPLESPEREPOCH-num_nen)/num_dataset_section)))
def f2():
    return applyOptions(non_toxic_train_en_dataset().repeat().take(int((NEXAMPLESPEREPOCH-num_nen)/num_dataset_section)))
def f3():
    return applyOptions(toxic_train_translated_dataset().repeat().take(int((NEXAMPLESPEREPOCH-num_nen)/num_dataset_section)))
def f4():
    return applyOptions(non_toxic_train_translated_dataset().repeat().take(int((NEXAMPLESPEREPOCH-num_nen)/num_dataset_section)))
def f5():
    return toxic_train_nen_dataset()
def f6():
    return non_toxic_train_nen_dataset()
branches={0:f0,1:f1,2:f2,3:f3,4:f4,5:f5,6:f6}
    
    

mixedDatasetParams=[1,2]
if use_dann:
    mixedDatasetParams.append(0)
    # mixedDatasetParams=[["en","toxic"]]#,["en","nontoxic"]]#,["nen","nontoxic"],["nen","toxic"]]
#     mixedDatasetParams=[["nen","unknown"]]
if use_validation_during_pretraining:
    mixedDatasetParams.extend([5,6])
if use_translated_data:
    mixedDatasetParams.extend([3,4])
def mapParamsToDataset(params):
#     tf.print("Params",params)
    return tf.switch_case(params,branches)


# In[ ]:


mixedDatasetParams


# In[ ]:


for key,item in branches.items():
    if key in mixedDatasetParams:
        print(key,item())


# In[ ]:


cycle_length=len(mixedDatasetParams)
def non_parallel_dataset(params=mixedDatasetParams):
  print(params)
  return tf.data.Dataset.from_tensor_slices(params).interleave(mapParamsToDataset, num_parallel_calls=AUTO,cycle_length=cycle_length).take(NEXAMPLESPEREPOCH)
if interativeEnvironment:
  train_dict_dataset=non_parallel_dataset()
  train_dict_dataset


# In[ ]:


non_parallel_dataset()


# In[ ]:


a=[1,2,3,4]
b=[1,2,3,4]
a==b


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from collections import Counter\n# interativeEnvironment=True\nif interativeEnvironment:\n    dlist=non_parallel_dataset().shuffle(NEXAMPLESPEREPOCH).repeat()\n    epoch_metrics={}\n    shard_metrics={}\n    signature_examples_source={}\n    signature_examples_count={}\n    for i,l in tqdm(enumerate(dlist)):\n#         if i==200:\n#             break\n        epoch=str(i//NEXAMPLESPEREPOCH)\n        shard=str(i//40000)\n        if epoch=="20":\n            break\n        if epoch in epoch_metrics.keys():\n            pass\n        else:\n            total_by_source=Counter()\n            total_by_toxic=Counter()\n            total_by_toxic_source=Counter()\n            total_by_s_toxic=Counter()\n            total=0\n            epoch_metrics[epoch]={"by_toxic":total_by_toxic,"by_source":total_by_source,"total":total,"by_both":total_by_toxic_source,"by_s_toxic":total_by_s_toxic}\n        if shard in shard_metrics.keys():\n            pass\n        else:\n            stotal_by_source=Counter()\n            stotal_by_toxic=Counter()\n            stotal_by_toxic_source=Counter()\n            stotal_by_s_toxic=Counter()\n            stotal=0\n            shard_metrics[shard]={"by_toxic":stotal_by_toxic,"by_source":stotal_by_source,"total":stotal,"by_both":stotal_by_toxic_source,"by_s_toxic":stotal_by_s_toxic}    \n\n        total+=1\n        if l["source"].numpy() not in signature_examples_source:\n            signature_examples_source[l["source"].numpy()]=l\n            signature_examples_count[l["source"].numpy()]=0\n        if all(signature_examples_source[l["source"].numpy()]["tokens"].numpy()==l["tokens"].numpy()) :\n            signature_examples_count[l["source"].numpy()]+=1\n        total_by_source.update([l["source"].numpy()])\n        total_by_toxic.update([-1 if l["toxic"] ==-1 else 1 if l["toxic"].numpy()>=0.5 else 0])\n        total_by_s_toxic.update([-1 if l["s_toxic"] ==-1 else 1 if l["s_toxic"].numpy()>=0.5 else 0])\n        total_by_toxic_source.update([(round(l["toxic"].numpy()),l["source"].numpy())])\n        \n        \n        stotal+=1\n        stotal_by_source.update([l["source"].numpy()])\n        stotal_by_toxic.update([-1 if l["toxic"] ==-1 else 1 if l["toxic"].numpy()>=0.5 else 0])\n        stotal_by_s_toxic.update([-1 if l["s_toxic"] ==-1 else 1 if l["s_toxic"].numpy()>=0.5 else 0])\n        stotal_by_toxic_source.update([(round(l["toxic"].numpy()),l["source"].numpy())])\n\n#         print(list(map(lambda a:a.numpy(),[l["source"],l["ShardIndex"],l["s_toxic"]]))) \n    for epoch,epoch_metric in epoch_metrics.items():\n        print("Epoch:-",epoch,"Toxic:-",epoch_metric["by_toxic"].most_common(),"By SToxic:",epoch_metric["by_s_toxic"].most_common(),"By source:-",epoch_metric["by_source"].most_common(),"\\n")\n    print(signature_examples_count)')


# In[ ]:


if interativeEnvironment:
    for epoch,epoch_metric in shard_metrics.items():
            print("Shard:-",epoch,"Toxic:-",epoch_metric["by_toxic"].most_common(),"Stoxic:",epoch_metric["by_s_toxic"].most_common(),"By source:-",epoch_metric["by_source"].most_common(),"\n")


# In[ ]:


def opus_files(lang):
    return tf.io.gfile.glob(OPUS_DS_PATH + "/{}_*.tfrecord".format(lang)+extension)
opus_files("es")


# In[ ]:


if use_parcor:
  parallel_feature_description = {
      'tokens_en': tf.io.FixedLenFeature([192], tf.int64),
      'en_attn_mask': tf.io.FixedLenFeature([192], tf.int64),
  #     'types': tf.io.FixedLenFeature([192], tf.int64, default_value=[0]*192),
      'tokens_nen': tf.io.FixedLenFeature([192], tf.int64),
      'nen_attn_mask': tf.io.FixedLenFeature([192], tf.int64),
      'lang':tf.io.FixedLenFeature([], tf.string),
      'source':tf.io.FixedLenFeature([], tf.int64,default_value=[5]),
      's_toxic':tf.io.FixedLenFeature([2], tf.float32,default_value=[-1.0,-1.0]),
      'is_same':tf.io.FixedLenFeature([], tf.float32,default_value=[1.0]),
  }
  def _parse_parallel_proto(example_proto):
  #     tf.print(example_proto)
    # Parse the input `tf.Example` proto using the dictionary above.
      example= tf.io.parse_single_example(example_proto, parallel_feature_description)
  #     return example
      return example
  if IN_COLAB:
    compression_type="GZIP"
  else:
    compression_type=""
  def perFileParallelDataset(filename,feature_description=feature_description):
      tf.print("Retrieving from filename",filename)
      return tf.data.TFRecordDataset(filename,compression_type=compression_type).map(_parse_parallel_proto)

  def make_parallel_pairs(example):
      ret={}
      # r=tf.random.uniform(shape=[])
      # ret["tokens"]=tf.cond(r>=0.5,lambda:tf.stack([example["tokens_en"],example["tokens_nen"]]),lambda :tf.stack([example["tokens_nen"],example["tokens_en"]])
      ret["tokens"]=tf.stack([example["tokens_en"],example["tokens_nen"]])
      ret["mask"]=tf.stack([example["en_attn_mask"],example["nen_attn_mask"]])
      ret["s_toxic"]=example["s_toxic"]
      ret["is_same"]=example["is_same"]
      ret["lang"]=example["lang"]
      return ret
  def parallel_dataset(files):
      
      return tf.data.Dataset.from_tensor_slices(files).shuffle(2048).interleave(perFileParallelDataset).map(make_parallel_pairs)
  def parallel_labelled_dataset(labelled_dataset):
      return labelled_dataset.batch(2).map(make_parallel_labelled_pairs)
  def make_parallel_labelled_pairs(example):
      example["is_same"]=tf.constant(0.0)
      return example
  langs=["es","fr","pt","tr","ru","it"]
  langFiles=list(map(opus_files,langs))
  def all_parallel_dataset():
    return tf.data.Dataset.from_tensor_slices(langFiles).interleave(parallel_dataset, num_parallel_calls=AUTO,cycle_length=len(langs))
  def parallel_nonparallel_mixer(params):
      return tf.cond(params=="parallel",lambda :all_parallel_dataset().take(int(NEXAMPLESPEREPOCH/4)).map(_form_tuple),lambda :parallel_labelled_dataset(non_parallel_dataset()).take(int(NEXAMPLESPEREPOCH/4)).map(_form_tuple))
  def parcor_dataset():
      return tf.data.Dataset.from_tensor_slices(["parallel","nonparallel"]).interleave(parallel_nonparallel_mixer,num_parallel_calls=2)


# In[ ]:


if interativeEnvironment:
  ds=list(all_parallel_dataset().take(7))
  print(list(map(lambda a:a["is_same"],ds)))
  # print(ds)
  # print(ds[0].keys(),list(map(lambda a:a.shape,ds[0])))


# In[ ]:


if interativeEnvironment:
  ds=list(parallel_labelled_dataset(valid_nen_dataset()).take(7))
  print(list(map(lambda a:a["is_same"],ds)))


# In[ ]:


if interativeEnvironment:
  list(parcor_dataset().take(2))


# In[ ]:


valid_en_files=list(filter(lambda file:"valid"  in file and "nen" not in file and "Sentences" not in file, data_files))
valid_en_files


# In[ ]:


def valid_en_dataset():
    return tf.data.Dataset.from_tensor_slices(valid_en_files).interleave(perFileDataset).shuffle(200000).map(roundLabels)


# In[ ]:



if use_parcor:
  train_dataset=parcor_dataset().shuffle(NEXAMPLESPEREPOCH).repeat().batch(int(BATCH_SIZE/2)).prefetch(AUTO)

  valid_dataset=parallel_labelled_dataset(valid_nen_dataset()).map(_form_tuple).batch(int(BATCH_SIZE/2)).prefetch(AUTO)
  finetune_dataset=parallel_labelled_dataset(finetune_nen_dataset()).map(_form_tuple).batch(int(BATCH_SIZE/2)).prefetch(AUTO)

else:
  if use_english_validation:
    valid_dataset = valid_en_dataset().batch(BATCH_SIZE).map(_form_tuple).prefetch(AUTO)
  else:
    valid_dataset = valid_nen_dataset().map(_form_tuple).batch(BATCH_SIZE).prefetch(AUTO)



  train_dataset=non_parallel_dataset().shuffle(NEXAMPLESPEREPOCH).map(_form_tuple).repeat().batch(BATCH_SIZE).prefetch(AUTO)
  finetune_dataset=finetune_nen_dataset().shuffle(NEXAMPLESPEREPOCH).map(_form_tuple).batch(BATCH_SIZE).prefetch(AUTO)
valid_dataset


# In[ ]:


# for ex,label,weight in valid_dataset:
#     print(tf.shape(weight[0]))
#     print(tf.shape(weight[1]))
#     print(tf.shape(label["toxic"]))
#     print(tf.shape(label["lang"]))
#     break


# In[ ]:


from tensorflow.python.keras.utils import losses_utils
class WeightedBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
    def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,lossWeight=0.1,
               name='binary_crossentropy'):
        """Initializes `BinaryCrossentropy` instance.
        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
              assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
              **Note - Using from_logits=True may be more numerically stable.
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
            we compute the loss between the predicted labels and a smoothed version
            of the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial]
            (https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: (Optional) Name for the op. Defaults to 'binary_crossentropy'.
        """
        super(WeightedBinaryCrossentropy, self).__init__(
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)
        self.lossWeight=lossWeight
    def call(self, y_true, y_pred):
        l= tf.math.scalar_mul(self.lossWeight,super().call(y_true,y_pred))
#         tf.print(tf.shape(l))
        return l


# In[ ]:


from tensorflow.python.keras.utils import losses_utils
class MaskedBinaryCrossentropy(tf.keras.losses.BinaryCrossentropy):
    def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy'):
        """Initializes `BinaryCrossentropy` instance.
        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
              assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
              **Note - Using from_logits=True may be more numerically stable.
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
            we compute the loss between the predicted labels and a smoothed version
            of the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial]
            (https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: (Optional) Name for the op. Defaults to 'binary_crossentropy'.
        """
        super(MaskedBinaryCrossentropy, self).__init__(
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)
    def call(self, y_true, y_pred):
        # tf.print("toxic",y_true)

        if use_parcor:
          raise("Unimplemented")
          y_true=tf.reshape(y_true,[-1,1])
          y_pred=tf.reshape(y_pred,[-1,1])
#         tf.print("ytrue",tf.shape(y_true))
#         tf.print("ytrue_val",y_true)
#         tf.print("ypred",tf.shape(y_pred))
#         tf.print("ypred_val",y_pred)
        mask=tf.math.greater_equal(y_true,0)
        mask=tf.cast(mask,y_true.dtype)
        
#         tf.print("mask shape",tf.shape(mask))
        mask=tf.squeeze(mask,axis =-1)
#         tf.print("squeezed mask shape",tf.shape(mask))
        masked_y_true=mask * y_true
#         tf.print("masked_y_true",tf.shape(masked_y_true))
        l=super().call(masked_y_true,y_pred)
#         tf.print("pre mask loss",tf.shape(l))
#         tf.print("pre mask loss val",l)
        l=mask*l
#         tf.print("post mask loss",tf.shape(l))
#         tf.print("toxic loss",l)
#         tf.print(tf.shape(l))
        if use_parcor:
          l=tf.reshape(l,[-1,2])
          l=l[:,0]+l[:,1]
        return l


# In[ ]:


class SquareLoss(tf.keras.losses.Loss):
  def __init__(self,reduction=losses_utils.ReductionV2.AUTO, name="squareLoss",lossWeight=0):
    super().__init__(reduction=reduction,name=name)
    self.lossWeight=lossWeight
  def call(self, y_true, y_pred):
    # tf.print("is same pred",y_pred)
    # tf.print("is same true",y_true)
    is_same_masked= y_pred*y_true
    loss=tf.math.scalar_mul(self.lossWeight,is_same_masked*is_same_masked)
    # tf.print("is same loss",loss)
    return loss


# In[ ]:



def build_model(transformer, max_len=512):
    if use_parcor:
      pair_input_word_ids = Input(shape=(2,max_len,), dtype=tf.int32, name="input_word_ids")
      pair_attention_masks = Input(shape=(2,max_len,), dtype=tf.int32, name="attention_mask")
      input_word_ids = tf.keras.backend.reshape(pair_input_word_ids,shape=(-1,max_len))
      attention_masks= tf.keras.backend.reshape(pair_attention_masks,shape=(-1,max_len))
      inputs=[pair_input_word_ids,pair_attention_masks]
    else:
      input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
      attention_masks = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
      inputs=[input_word_ids,attention_masks]
#     token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="token_type_ids")
#     transformer.trainable = trainable_transformer
#     cls_token=transformer((input_word_ids,attention_masks))[1]
#     out = Dense(2, activation='sigmoid')(cls_token)
    transformer_output = transformer((input_word_ids,attention_masks))
    sequence_output = transformer_output[0]
    transformer_output
    if pooling_mode == pooling_mode_cls_token:
        transformed_features = sequence_output[:, 0, :]
        cls_token=transformed_features
    else:
        float_mask=tf.cast(tf.expand_dims(attention_masks,axis=-1),tf.float32)
        lengths=tf.reduce_sum(float_mask,axis=-2)

        masked_vals=tf.multiply(sequence_output,float_mask)
        sums=tf.reduce_sum(masked_vals,axis=-2)
        avgs=tf.math.divide(sums,lengths)
        maxes=tf.math.reduce_max(masked_vals,axis=-2)
        transformed_features=tf.concat([avgs,maxes],axis=-1)
        cls_token=transformed_features
    for dims in fc_dims:
        cls_token=Dense(dims, activation='relu')(cls_token)
    if use_dann:
        lang_token=GradReverse()(transformed_features)
        for dims in fc_dims:
            lang_token=Dense(dims, activation='relu')(lang_token)
        lang=Dense(1, activation="sigmoid",name="lang")(lang_token)
        
    
    
    if use_parcor:
      toxic = Dense(1, activation="sigmoid",name="unpair_toxic")(cls_token)
      toxic=tf.keras.backend.reshape(toxic,shape=(-1,2,1))
      toxic=tf.keras.layers.Layer(name="toxic")(toxic)
      is_same=tf.keras.layers.Subtract(name="is_same")([toxic[:,1,:],toxic[:,0,:]])
    else:
      toxic = Dense(1, activation="sigmoid",name="toxic")(cls_token)
    out=[toxic]
    if use_parcor:
      out.append(is_same)
    celoss=MaskedBinaryCrossentropy(
        from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
        name='crossentropy_loss'
    )
    loss={"toxic":celoss}
    if use_dann:
        domainGap=WeightedBinaryCrossentropy(
        from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
        name='domaingap_loss',lossWeight=dann_lambda
        )
        out.append(lang)
        loss["lang"]=domainGap
    if use_parcor:
      loss["is_same"]=SquareLoss(lossWeight=parcor_lambda)
    model = Model(inputs=inputs, outputs=out)
    model.compile(Adam(lr=5e-6), loss=loss, metrics={"toxic":['accuracy',tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.AUC(name="pr-auc",curve="PR")]})
    
    return model


# In[ ]:


def setTrainableLayers(transformer,embeddingsTrainable=False,poolerTrainable=False,hiddenTrainable=8):
    transformer.layers[0].pooler.trainable=poolerTrainable
    transformer.layers[0].embeddings.trainable=embeddingsTrainable
    trainableHiddens=transformer.layers[0].encoder.layer[-1*hiddenTrainable:]
    for layer in transformer.layers[0].encoder.layer:
        if layer in trainableHiddens:
            layer.trainable=True
        else:
            layer.trainable=False


# In[ ]:


get_ipython().run_cell_magic('time', '', '# tf.keras.backend.clear_session()\nwith strategy.scope():\n#     tf.keras.backend.clear_session()\n\n    transformer_layer = TFAutoModel.from_pretrained(MODEL)\n#     setTrainableLayers(transformer_layer)\n\n    model = build_model(transformer_layer, max_len=MAX_LEN)\n    if load_model is not None:\n            model.load_weights(load_model)\n\nmodel.summary()')


# In[ ]:


MODEL


# In[ ]:


from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
 
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_")
out_path=dt_string+'zero_shot_pro.h5'
write_root=""
if IN_COLAB:
  write_root=root+"outputs/"
  out_path=write_root+out_path
print(out_path)


# In[ ]:


#checkpointing best
if use_pretraining:
    metric_prefix=""
    if use_parcor  or use_dann:
        metric_prefix="_toxic"
    mc = ModelCheckpoint(out_path, monitor="val"+metric_prefix+"_auc", mode='max',verbose=1,save_weights_only=True,save_best_only=True)
    es = EarlyStopping(monitor="val"+metric_prefix+"_auc", mode='max',patience=2,verbose=1,restore_best_weights=True)
    lrScheduler=ReduceLROnPlateau(
        monitor="val"+metric_prefix+"_loss",
        factor=0.2,
        patience=1,
        verbose=1,
        mode="min",
        min_delta=0,
        cooldown=0,
        min_lr=1e-10,
    )


# In[ ]:


# if interativeEnvironment:
# n_steps=4
# EPOCHS=2
if use_pretraining:
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[mc,es,lrScheduler]
    )


# In[ ]:


if use_pretraining:

    if use_parcor or use_dann:
        metric_prefix="toxic_"
    else:
        metric_prefix=""
    plt.plot(train_history.history[metric_prefix+'auc'])
    plt.plot(train_history.history['val_'+metric_prefix+'auc'])

    plt.title('Model auc')
    plt.ylabel('Auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[ ]:


if use_pretraining:

    if use_parcor or use_dann:
        plt.plot(train_history.history['toxic_loss'])
        plt.plot(train_history.history['val_toxic_loss'])
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Toxic Train', 'Toxic Test','Train', 'Test'], loc='upper left')
    plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n\nsub = pd.read_csv(root+'/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv',nrows=nrows)")


# In[ ]:


test_files=list(filter(lambda file:"test"  in file and "nen" in file, data_files))
test_files


# In[ ]:


if use_parcor:
    test_dataset=parallel_labelled_dataset(tf.data.Dataset.from_tensor_slices(test_files).interleave(perFileDataset,cycle_length=1)).map(_form_tuple).batch(int(BATCH_SIZE/2)).prefetch(AUTO)
else:
    test_dataset=tf.data.Dataset.from_tensor_slices(test_files).interleave(perFileDataset,cycle_length=1).batch(BATCH_SIZE).map(_form_tuple).prefetch(AUTO)


# In[ ]:


if use_pretraining:
    l=model.predict(test_dataset, verbose=1)
    if use_dann:
        l=l[0]
    if use_parcor:
        l=np.reshape(l,(-1,1))
    sub['toxic'] = l[:,-1]
    # sub['toxic'] = (l[:,1]+(1-l[:,0]))/2
    if use_finetuning:
        write_file=write_root+'submission.unfinetuned.csv'
    else:
        write_file=write_root+'submission.csv'
    sub.to_csv(write_file, index=False)


# In[ ]:


if use_finetuning:
    # datetime object containing current date and time
    now = datetime.now()

    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_")
    out_path=dt_string+'zero_shot_pro.h5'
    write_root=""
    if IN_COLAB:
      write_root=root+"outputs/"
      out_path=write_root+out_path
    print(out_path)


# In[ ]:


if use_finetuning:
    #checkpointing best
    metric_prefix=""
    if use_parcor  or use_dann:
        metric_prefix="_toxic"
    mc = ModelCheckpoint(out_path, monitor="val"+metric_prefix+"_auc", mode='max',verbose=1,save_weights_only=True,save_best_only=True)
    es = EarlyStopping(monitor="val"+metric_prefix+"_auc", mode='max',patience=2,verbose=1,restore_best_weights=True)
    lrScheduler=ReduceLROnPlateau(
        monitor="val"+metric_prefix+"_loss",
        factor=0.2,
        patience=1,
        verbose=1,
        mode="min",
        min_delta=0,
        cooldown=0,
        min_lr=1e-10,
    )


# In[ ]:


if use_finetuning:
    # if interativeEnvironment:
    # n_steps=4
    # EPOCHS=2
    train_history = model.fit(
        finetune_dataset,
#         steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[mc,es,lrScheduler]
    )


# In[ ]:


if use_finetuning:
    l=model.predict(test_dataset, verbose=1)
    if use_dann:
        l=l[0]
    if use_parcor:
        l=np.reshape(l,(-1,1))
    sub['toxic'] = l[:,-1]
    # sub['toxic'] = (l[:,1]+(1-l[:,0]))/2
    write_file=write_root+'submission.csv'
    sub.to_csv(write_file, index=False)


# In[ ]:


from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import gc

# Reset Keras Session
def reset_keras():
#     del model
    tf.keras.backend.clear_session()
    print(gc.collect()) # if it's done something you should see a number being outputted


# In[ ]:


# reset_keras()


# In[ ]:




