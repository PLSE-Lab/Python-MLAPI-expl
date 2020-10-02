#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set()
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


KAGGLE_DIR = '../input/'

train = pd.read_csv(KAGGLE_DIR + "train/train.csv")
test = pd.read_csv(KAGGLE_DIR + "test/test.csv")
full=train.append(test)


# In[ ]:


print(train.shape)
print(test.shape)
print(full.shape)


# We can appreciate that the split between trainning and test is 26%

# In[ ]:


train.columns


# ### Health

# In[ ]:


plt.figure()
health_states =['Not Specified','Healthy', 'Minor Injury','Serious Injury']
pos = np.arange(len(health_states))
health=[np.sum(full.Health==0),np.sum(full.Health==1),np.sum(full.Health==2),np.sum(full.Health==3)]
# change the bar color to be less bright blue
bars = plt.bar(pos, health, align='center', linewidth=0, color='lightslategrey')
# soften all labels by turning grey
plt.xticks(pos, health_states, alpha=0.8)
plt.title('Health States of the Animals', alpha=3)
# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1, str(int(bar.get_height())), 
                 ha='center', color='k', fontsize=11)
plt.show()


# We can appreciate that all animals in the train set have a health report, and most of them are healthy

# ### Age

# In[ ]:


plt.figure()
sns.distplot(full.Age)
plt.ylabel('% of times')
plt.title('Histogram of the age of the different pets')
plt.show()
full.Age.describe()


# We can appreciate that the pets are normally young, understanding that Age=0 is less than a month and not a Nan
# 

# ### Rescuer ID

# In[ ]:


from collections import Counter
rescuer_counts = Counter(full.RescuerID)


# In[ ]:


plt.figure()
sns.distplot(list(rescuer_counts.values()))
plt.ylabel('% of times')
plt.title('Histogram of the number of rescues by rescuers')
plt.show()
pd.Series(list(rescuer_counts.values())).describe()


# We can appreciate that most of the the Rescuers just report a dog, we can assume that this is random people that give only a founded dog or have a litter and they are turning the dogs
# There is a rescuer that gave more than 459 dogs in rescue

# ### Predictions

# In[ ]:


plt.figure()
sns.distplot(train.AdoptionSpeed)
plt.ylabel('No of times')
plt.title('Histogram of the Adoption Speed of the different pets')
plt.show()
pd.Series(train.AdoptionSpeed.describe())


# 0 - Pet was adopted on the same day as it was listed. 
# 
# 1 - Pet was adopted between 1 and 7 days (1st week) after being listed. 
# 
# 2 - Pet was adopted between 8 and 30 days (1st month) after being listed. 
# 
# 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed. 
# 
# 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).

# We observe that the number of pets adopted in the same day is minimum while the maximum is pets that have not been adopted after +90 days

# ### Vaccinated

# In[ ]:


plt.figure()
sns.distplot(train.Vaccinated)
plt.ylabel('No of times')
plt.title('Histogram of the age of the different pets')
plt.show()
pd.Series(train.Vaccinated.describe())
# Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)


# Surprisingly there are more  not vaccinated dogs than vaccinated. I guess that would be good to assume that people that dont know that if the dog is vaccinated put them in the no vaccinated group 

# # Data Cleaning

# In[ ]:


#Easy DataCleaning to avoid categorical features
from sklearn.preprocessing import LabelEncoder

cols=[]

for i in full:
    if type(full[i][4])==str: #I put a 4. Works any other but 0
        cols.append(i)        

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(full[c].values)) 
    full[c] = lbl.transform(list(full[c].values))

# shape        
print('Shape all_data: {}'.format(full.shape))


# In[ ]:


test=full.query('AdoptionSpeed =="NaN"')
train=full.query('AdoptionSpeed !="NaN"')
target=train['AdoptionSpeed']
train.drop(columns=['AdoptionSpeed','Description','Name','PetID','RescuerID'],inplace=True)
test.drop(columns=['AdoptionSpeed','Name', 'RescuerID', 'Description', 'PetID'],inplace=True)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# Seed for reproducability
seed = 12345
np.random.seed(seed)


# In[ ]:


tic=time.time()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 800, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
toc=time.time()
print('The time elapsed is {}s'.format(np.round(np.abs(tic-toc))))


# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(train, target)
# rf_random.best_params_

# In[ ]:


{'n_estimators': 800,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': 14,
 'bootstrap': True}


# In[ ]:


rf=RandomForestClassifier(n_estimators=800,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',max_depth=14,bootstrap=True)
rf.fit(train,target)


# In[ ]:


# Metric used for this competition (Quadratic Weigthed Kappa aka Cohen Kappa Score)
def metric(y1,y2):
    return cohen_kappa_score(y1,y2, weights='quadratic')


# In[ ]:


metric(rf.predict(train), target)


# In[ ]:


# Get and store predictions
predictions = rf.predict(test)
test = pd.read_csv(KAGGLE_DIR + "test/test.csv")
submission_df = pd.DataFrame(data={"PetID" : test["PetID"], 
                                   "AdoptionSpeed" : predictions})


# In[ ]:


submission_df['AdoptionSpeed']=submission_df['AdoptionSpeed'].astype('int32');
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


submission_df.head()


# In[ ]:




