#!/usr/bin/env python
# coding: utf-8

# ![](https://www.charlescountymd.gov/Home/ShowImage?id=4496)
# # COVID-19 CXR Classifier using [RADTorch](https://www.radtorch.com)

# ### Install

# In[ ]:


get_ipython().system('pip install https://repo.radtorch.com/archive/v0.1.3-beta.zip -q')


# ### Import

# In[ ]:


from radtorch import pipeline, datautils
import pandas as pd


# ### Mix dataset and create a new label dataframe

# In[ ]:


# mix and match the data
normal_files_1 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL/')
normal_files_2 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/NORMAL/')
normal_list = normal_files_1+normal_files_2

covid_file_1 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/PNEUMONIA/')
covid_files_2 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/PNEUMONIA/')
covid_list = covid_file_1+covid_files_2

normal_label = ['normal']*len(normal_list)
covid_label = ['covid']*len(covid_list)

all_files = normal_list+covid_list
all_labels = normal_label+covid_label

label_df = pd.DataFrame(list(zip(all_files, all_labels)), columns=['IMAGE_PATH', 'IMAGE_LABEL'])


#shuffle
label_df = label_df.sample(frac=1).reset_index(drop=True)

label_df


# ### Create the pipeline instance to compare different model architectures

# In[ ]:


model_comparison = pipeline.Compare_Image_Classifier(
    data_directory='/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/',
    is_dicom=False,
    label_from_table=True,
    table_source=label_df,
    model_arch=['alexnet', 'resnet50', 'vgg16', 'vgg16_bn', 'wide_resnet50_2'],
    train_epochs=[20],
    balance_class=[True],
    normalize=[False],
    valid_percent=[0.1],
    learning_rate = [0.00001]

)


# ### Display list of created image classifiers to be tested

# In[ ]:


model_comparison.grid()


# ### Show Dataset info

# In[ ]:


model_comparison.dataset_info()


# ### Show sample from dataset

# In[ ]:


model_comparison.sample()


# ### Train and Analyze the image classifiers

# In[ ]:


model_comparison.run()


# ### Display training metrics

# In[ ]:


model_comparison.metrics()


# ### Display models ROC

# In[ ]:


model_comparison.roc()


# ### Show best classifier

# In[ ]:


model_comparison.best()


# ### Export best classifier

# In[ ]:


model_comparison.best(path='best_classifier', export_classifier=True)
model_comparison.best(path='best_model', export_model=True)


# In[ ]:




