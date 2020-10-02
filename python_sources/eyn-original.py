#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
print(os.listdir("../input"))


# In[2]:


df_train = pd.read_csv("../input/ey-nextwave/data_train/data_train.csv")
df_test = pd.read_csv("../input/ey-nextwave/data_test/data_test.csv")


# In[3]:


df_train.tail()


# In[4]:


df_test.tail()


# In[5]:


# normalising location information
X_MIN = 3750901.5068
X_MAX = 3770901.5068
X_MID = X_MIN + 0.5 * (X_MAX - X_MIN)
Y_MIN = -19268905.6133
Y_MAX = -19208905.6133
Y_MID = Y_MIN + 0.5 * (Y_MAX - Y_MIN)

def normalise_X(arr):
    return (arr - X_MID) / 10000

def normalise_Y(arr):
    return (arr - Y_MID) / 100000  
    # extra zero by design, seems to make figure to be in proportion
    # looking for evidence that the x-axis and y-axis fulfil some ratio

x_min, x_max = normalise_X(X_MIN), normalise_X(X_MAX)
y_min, y_max = normalise_Y(Y_MIN), normalise_Y(Y_MAX)
print("Borders:")
print("{:.4f} < X < {:.4f}".format(x_min, x_max))
print("{:.4f} < Y < {:.4f}".format(y_min, y_max))

df_train['x_entry'], df_train['x_exit'] = normalise_X(df_train['x_entry']), normalise_X(df_train['x_exit'])
df_train['y_entry'], df_train['y_exit'] = normalise_Y(df_train['y_entry']), normalise_Y(df_train['y_exit'])
df_test['x_entry'], df_test['x_exit'] = normalise_X(df_test['x_entry']), normalise_X(df_test['x_exit'])
df_test['y_entry'], df_test['y_exit'] = normalise_Y(df_test['y_entry']), normalise_Y(df_test['y_exit'])


# In[6]:


# normalising time information
def convert_time(time_sting):
    hms = time_sting.split(":")
    seconds = int(hms[0])*60*60 + int(hms[1])*60 + int(hms[2])
    seconds = (seconds-15*60*60)/(10*60*60)
    return seconds

df_train["t_entry"] = df_train["time_entry"].apply(lambda x: convert_time(x))
df_train["t_exit"] = df_train["time_exit"].apply(lambda x: convert_time(x))
df_test["t_entry"] = df_test["time_entry"].apply(lambda x: convert_time(x))
df_test["t_exit"] = df_test["time_exit"].apply(lambda x: convert_time(x))


# In[7]:


# obtaining metadata from IDs
df_train['tid_0'] = [tid.split("_")[-1] for tid in df_train['trajectory_id']]
df_train['tid_1'] = [tid.split("_")[-2] for tid in df_train['trajectory_id']]
df_test['tid_0'] = [tid.split("_")[-1] for tid in df_test['trajectory_id']]
df_test['tid_1'] = [tid.split("_")[-2] for tid in df_test['trajectory_id']]
df_train['tid_0'], df_test['tid_0'] = df_train['tid_0'].astype(int), df_test['tid_0'].astype(int)
df_train['tid_1'], df_test['tid_1'] = df_train['tid_1'].astype(int), df_test['tid_1'].astype(int)


# In[8]:


# extract relevant infromation and rearrange
columns = ['hash','tid_0',
           't_entry','t_exit',
           'x_entry','y_entry','x_exit','y_exit',
           'vmax','vmin','vmean',
           'time_entry','time_exit',
           'trajectory_id','tid_1']
df_train = df_train[columns]
df_test = df_test[columns]


# In[9]:


# tid_1 is likely the day of the month, this information may be useful
print(max([int(x) for x in df_test['tid_0']]), max([int(x) for x in df_test['tid_1']]))


# In[10]:


hash_most_freq = df_train['hash'].mode().tail(1).item()
df_train.loc[df_train['hash'] == hash_most_freq]


# # BASELINE SUBMISSION

# In[11]:


df_test_1st_traj_only = df_test[df_test['x_exit'].isnull()]
df_submit = df_test_1st_traj_only[['trajectory_id']].copy()
df_submit = df_submit.rename(columns = {'trajectory_id':'id'})

# helper function to determine if point is inside
def is_inside(arr_x, arr_y):
    return ((arr_x > x_min) & 
            (arr_x < x_max) & 
            (arr_y > y_min) & 
            (arr_y < y_max)).astype(float)

df_submit['target'] = is_inside(df_test_1st_traj_only['x_entry'],
                                df_test_1st_traj_only['y_entry'])
df_submit.to_csv('submission.csv', index=False)
df_submit.tail()


# # DATASET PIVOTING

# In[12]:


p_train = df_train.pivot('hash', 'tid_0')
p_train.tail()


# In[13]:


p_test = df_test.pivot('hash', 'tid_0')
p_test.tail()


# In[14]:


def obtain_matrix(row):
    df_hash = row.stack().iloc[::-1].reset_index()
    trajectory_id = df_hash.loc[0,"trajectory_id"]
    df_hash = df_hash[['t_entry','t_exit',
                       'x_entry','y_entry','x_exit','y_exit',
                       'vmax','vmin','vmean','tid_0','tid_1']]
    targets = df_hash.loc[0,"x_exit"], df_hash.loc[0,"y_exit"]

    df_hash.loc[0,"x_exit"] = np.nan
    df_hash.loc[0,"y_exit"] = np.nan
    embeds = np.transpose(df_hash.values)
    df_hash = df_hash.append(pd.DataFrame([[np.nan]*df_hash.shape[1]], 
                                            columns=list(df_hash),
                                            index=[99]*(21-df_hash.shape[0])))
    return {"targets" : targets, 
            "df_hash" : df_hash,
            "matrix" : df_hash.values,
            "trajectory_id" : trajectory_id}

print(np.shape(obtain_matrix(p_train.iloc[[323]])["matrix"]))
print(obtain_matrix(p_train.iloc[[323]])["targets"])
obtain_matrix(p_train.iloc[[323]])["df_hash"]
# note that x_exit and y_exit is removed from matrix


# In[15]:


test_data = []
test_ids = []

for i in tqdm(range(p_test.shape[0])):
    output = obtain_matrix(p_test.iloc[[i]])
    test_data.append(output["matrix"])
    test_ids.append(output["trajectory_id"])
#     if i>100:
#         break


# In[16]:


print(np.shape(test_data))
print(np.shape(test_ids))
np.save("test_data", test_data)
np.save("test_ids", test_ids)


# In[17]:


train_data = []
train_targets = []

for i in tqdm(range(p_train.shape[0])):
    output = obtain_matrix(p_train.iloc[[i]])
    train_data.append(output["matrix"])
    train_targets.append(output["targets"])
#     if i>100:
#         break


# In[18]:


# evaluate if the targets are inside
train_targets = np.array(train_targets)
train_targets_inside = is_inside(train_targets[:,0], train_targets[:,1])


# In[19]:


print(np.shape(train_data))
print(np.shape(train_targets))
print(np.shape(train_targets_inside))
np.save("train_data", train_data)
np.save("train_targets", train_targets)
np.save("train_targets_inside", train_targets_inside)


# # TRAIN-TEST SPLIT INDICES

# In[20]:


# standardised 4-fold train-test split for clustering purposes
from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(train_data)),
                                      train_targets_inside.astype(int)):
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
    
np.save("trn_index_list",trn_index_list)
np.save("val_index_list",val_index_list)


# In[21]:


get_ipython().system('ls')


# In[ ]:


# to document: specifications for all if not clear enough
# might not care to do: make the pivot table to 3D array faster

