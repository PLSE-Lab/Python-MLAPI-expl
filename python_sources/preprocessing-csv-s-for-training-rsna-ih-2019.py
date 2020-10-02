#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing CSV's for training

# ![](https://www.rsna.org/-/media/Images/RSNA/Menu/logo_sml.ashx?w=100&la=en&hash=9619A8238B66C7BA9692C1FC3A5C9E97C24A06E1)

# Are you working a lot with Data Generators (for example Keras' ".flow_from_dataframe") and competing in the [RSNA Intercranial Hemorrhage 2019 competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)? 
# 
# I've created a function that creates a simple preprocessed DataFrame with a column for ImageID and a column for each label in the competition. ('epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any') 
# 
# I also made a function which translates your predictions into the correct submission format.
# 
# If you are interested in getting the metadata as CSV files also you can check out [this Kaggle kernel](https://www.kaggle.com/carlolepelaars/converting-dicom-metadata-to-csv-rsna-ihd-2019). 
# 
# I hope this can be of help to you in the competition!

# ## Preparation

# In[ ]:


# We will only need OS and Pandas for this one
import os
import pandas as pd

# Path names
BASE_PATH = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"
TRAIN_PATH = BASE_PATH + 'stage_2_train.csv'
TEST_PATH = BASE_PATH + 'stage_2_sample_submission.csv'

# All labels that we have to predict in this competition
targets = ['epidural', 'intraparenchymal', 
           'intraventricular', 'subarachnoid', 
           'subdural', 'any']


# In[ ]:


# File sizes and specifications
print('\n# Files and file sizes')
for file in os.listdir(BASE_PATH)[2:]:
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(BASE_PATH + file) / 1000000, 2))))


# ## Preprocessing CSV's

# We read in the CSV's remove duplicates and create a column for every label we have to predict. We therefore get one row for each image and this works nicely with datagenerator of popular frameworks like Keras and PyTorch.

# In[ ]:


train_df = pd.read_csv(TRAIN_PATH)
train_df['ImageID'] = train_df['ID'].str.rsplit('_', 1).map(lambda x: x[0]) + '.png'
label_lists = train_df.groupby('ImageID')['Label'].apply(list)


# In[ ]:


train_df[train_df['ImageID'] == 'ID_0002081b6.png']


# In[ ]:


def prepare_df(path, train=False, nrows=None):
    """
    Prepare Pandas DataFrame for fitting neural network models
    Returns a Dataframe with two columns
    ImageID and Labels (list of all labels for an image)
    """ 
    df = pd.read_csv(path, nrows=nrows)
    
    # Get ImageID and type for pivoting
    df['ImageID'] = df['ID'].str.rsplit('_', 1).map(lambda x: x[0]) + '.png'
    df['type'] = df['ID'].str.split("_", n = 3, expand = True)[2]
    # Create new DataFrame by pivoting
    new_df = df[['Label', 'ImageID', 'type']].drop_duplicates().pivot(index='ImageID', 
                                                                      columns='type', 
                                                                      values='Label').reset_index()
    return new_df


# In[ ]:


# Convert dataframes to preprocessed format
train_df = prepare_df(TRAIN_PATH, train=True)
test_df = prepare_df(TEST_PATH)


# In[ ]:


print('Training data: ')
display(train_df.head())

print('Test data: ')
test_df.head()


# In[ ]:


# Save to CSV
train_df.to_csv('clean_train_df.csv', index=False)
test_df.to_csv('clean_test_df.csv', index=False)


# ## Creating submission file

# To convert the DataFrame back to the original submission format you can use this function.

# In[ ]:


def create_submission_file(IDs, preds):
    """
    Creates a submission file for Kaggle when given image ID's and predictions
    
    IDs: A list of all image IDs (Extensions will be cut off)
    preds: A list of lists containing all predictions for each image
    
    Returns a DataFrame that has the correct format for this competition
    """
    sub_dict = {'ID': [], 'Label': []}
    # Create a row for each ID / Label combination
    for i, ID in enumerate(IDs):
        ID = ID.split('.')[0] # Remove extension such as .png
        sub_dict['ID'].extend([f"{ID}_{target}" for target in targets])
        sub_dict['Label'].extend(preds[i])
    return pd.DataFrame(sub_dict)


# In[ ]:


# Finalize submission files
train_sub_df = create_submission_file(train_df['ImageID'], train_df[targets].values)
test_sub_df = create_submission_file(test_df['ImageID'], test_df[targets].values)


# In[ ]:


print('Back to the original submission format:')
train_sub_df.head(6)


# That's all! You can find the clean CSV's in the "output files" of this kernel.
# 
# If you like this Kaggle kernel, feel free to give an upvote and leave a comment! I will do my best to implement your suggestions!
