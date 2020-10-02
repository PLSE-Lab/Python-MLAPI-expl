#!/usr/bin/env python
# coding: utf-8

# # [Model-Pipelines](http://https://github.com/aponte411/model_pipelines)
# 

# In[ ]:


import os

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
CREDENTIALS = {}
CREDENTIALS['aws_access_key_id'] = user_secrets.get_secret("aws_access_key_id")
CREDENTIALS['aws_secret_access_key'] = user_secrets.get_secret("aws_secret_access_key")
CREDENTIALS['bucket'] = user_secrets.get_secret("bucket")

# download repo and install requirements
get_ipython().system('git clone https://github.com/aponte411/model_pipelines.git')
os.chdir('/kaggle/working/model_pipelines/model_factory')
os.mkdir('trained_models')
get_ipython().system('pip install -r requirements.txt')


# # Create Cross-Validation Folds
# 

# In[ ]:


from cross_validators import BengaliCrossValidator

cv = BengaliCrossValidator(
    input_path='/kaggle/input/bengaliai-cv19/train.csv', 
    output_path='/kaggle/working/train-folds.csv', 
    target=[
        "grapheme_root", 
        "vowel_diacritic", 
        "consonant_diacritic"
    ]
)

train = cv.apply_multilabel_stratified_kfold(save=True)
train.head()


# # Train Models & Conduct Inference
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'kaggle_training_run.py', '\nfrom typing import Optional\n\nimport click\n\nfrom engines import BengaliEngine\nfrom trainers import BengaliTrainer\nimport utils\nfrom dispatcher import MODEL_DISPATCHER\n\n\nTRAINING_PARAMS = {\n    1: {\n        "train": [0, 1, 2, 3],\n        "val": [4]\n    },\n    2: {\n        "train": [0, 1, 2, 4],\n        "val": [3]\n    },\n    3: {\n        "train": [0, 1, 3, 4],\n        "val": [2]\n    },\n    4: {\n        "train": [0, 2, 3, 4],\n        "val": [1]\n    }\n}\n\n\n@click.command()\n@click.option(\'--model-name\', type=str, default=\'resnet50\')\n@click.option(\'--train\', type=bool, default=True)\n@click.option(\'--inference\', type=bool, default=True)\n@click.option(\'--train-path\',\n              type=str,\n              default=\'/kaggle/working/train-folds.csv\')\n@click.option(\'--test-path\', type=str, default=\'/kaggle/input/bengaliai-cv19\')\n@click.option(\'--pickle-path\',\n              type=str,\n              default=\'/kaggle/input/bengaliai-image-pickles/image_pickles/kaggle_dataset/image_pickles\')\n@click.option(\'--model-dir\', type=str, default=\'trained_models\')\n@click.option(\'--submission-dir\', type=str, default=\'/kaggle/working\')\n@click.option(\'--train-batch-size\', type=int, default=64)\n@click.option(\'--test-batch-size\', type=int, default=32)\n@click.option(\'--epochs\', type=int, default=5)\ndef run_bengali_engine(model_name: str, train: bool, inference: bool, train_path: str,\n                       test_path: str, pickle_path: str, model_dir: str,\n                       train_batch_size: int, test_batch_size: int,\n                       epochs: int, submission_dir: str) -> Optional:\n    timestamp = utils.generate_timestamp()\n    print(f\'Training started {timestamp}\')\n    if train:\n        for loop, fold_dict in TRAINING_PARAMS.items():\n            print(f\'Training loop: {loop}\')\n            ENGINE_PARAMS = {\n                "train_path": train_path,\n                "test_path": test_path,\n                "pickle_path": pickle_path,\n                "model_dir": model_dir,\n                "submission_dir": submission_dir,\n                "train_folds": fold_dict[\'train\'],\n                "val_folds": fold_dict[\'val\'],\n                "train_batch_size": train_batch_size,\n                "test_batch_size": test_batch_size,\n                "epochs": epochs,\n                "image_height": 137,\n                "image_width": 236,\n                "mean": (0.485, 0.456, 0.406),\n                "std": (0.229, 0.239, 0.225),\n                # 1 loop per test parquet file\n                "test_loops": 5,\n            }\n            model = MODEL_DISPATCHER.get(model_name)\n            trainer = BengaliTrainer(model=model, model_name=model_name)\n            bengali = BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)\n            bengali.run_training_engine()\n        print(f\'Training complete!\')\n    if inference:\n        ENGINE_PARAMS = {\n                "train_path": train_path,\n                "test_path": test_path,\n                "pickle_path": pickle_path,\n                "model_dir": model_dir,\n                "submission_dir": submission_dir,\n                "train_folds": [0],\n                "val_folds": [4],\n                "train_batch_size": train_batch_size,\n                "test_batch_size": test_batch_size,\n                "epochs": epochs,\n                "image_height": 137,\n                "image_width": 236,\n                "mean": (0.485, 0.456, 0.406),\n                "std": (0.229, 0.239, 0.225),\n                # 1 loop per test parquet file\n                "test_loops": 5,\n            }\n        timestamp = utils.generate_timestamp()\n        print(f\'Inference started {timestamp}\')\n        model = MODEL_DISPATCHER.get(model_name)\n        trainer = BengaliTrainer(model=model, model_name=model_name)\n        bengali = BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)\n        submission = bengali.run_inference_engine(\n            model_name=model_name,\n            model_dir=ENGINE_PARAMS[\'model_dir\'],\n            to_csv=True,\n            output_dir=ENGINE_PARAMS[\'submission_dir\'])\n        print(f\'Inference complete!\')\n        print(submission)\n\n\nif __name__ == "__main__":\n    run_bengali_engine()')


# In[ ]:


get_ipython().system('python kaggle_training_run.py')


# # Submit Predictions

# In[ ]:


import pandas as pd

submission = pd.read_csv("/kaggle/working/submission_March-08-2020-21:52")
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission


# In[ ]:




