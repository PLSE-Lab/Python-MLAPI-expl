#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser


# ## Data

# In[ ]:


dataset = pd.read_csv('../input/appdata10.csv')


# ## EDA

# In[ ]:


dataset.head()


# In[ ]:


dataset.describe()


# ### Data Cleaning

# In[ ]:


dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)


# ### Plotting

# In[ ]:


dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
dataset2.head()


# ### Histograms

# In[ ]:


plt.figure(figsize = (20, 20))
plt.suptitle('Histogram of Numerical Columns', fontsize = 20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])
    
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins = vals, color = '#3F5D7D')


# ### Coorelation with Response

# In[ ]:


dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20, 10), 
                                             title = 'Coorelation with Response Variable', 
                                             fontsize = 15, rot = 45,
                                             grid = True)


# ### Correlation Matrix

# In[ ]:


sns.set(style="white", font_scale=2)

corr = dataset2.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ## Feature Engineering

# ### Response

# In[ ]:


dataset.dtypes


# In[ ]:


dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]


# In[ ]:


dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]


# In[ ]:


dataset.dtypes


# In[ ]:


dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')


# In[ ]:


plt.figure(figsize = (20, 20))
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D')
plt.title("Distribution of Time-Since-Enrolled")
plt.show()


# In[ ]:


plt.figure(figsize = (20, 20))
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range = [0, 100])
plt.title("Distribution of Time-Since-Enrolled")
plt.show()


# In[ ]:


dataset.loc[dataset.difference > 48, 'enrolled'] = 0


# In[ ]:


dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])


# ### Screens

# In[ ]:


top_screens = pd.read_csv('CS3_Data/top_screens.csv').top_screens.values
top_screens


# In[ ]:


dataset['screen_list'] = dataset.screen_list.astype(str) + ','

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+',','')


# In[ ]:


dataset['Other'] = dataset.screen_list.str.count(',')
dataset = dataset.drop(columns = ['screen_list'])


# #### Funnels

# In[ ]:


savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns=savings_screens)


# In[ ]:


cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)


# In[ ]:


cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)


# In[ ]:


loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)


# In[ ]:


dataset.head()


# In[ ]:


dataset.columns


# In[ ]:


dataset.to_csv('../input/new_appdata10.csv', index = False)


# ## MODEL

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


# In[ ]:


dataset = pd.read_csv('../input/new_appdata10.csv')


# ## Data Preprocessing

# In[ ]:


response = dataset['enrolled']
dataset = dataset.drop(columns = 'enrolled')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset, response, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# In[ ]:


train_identifier = X_train['user']
X_train = X_train.drop(columns = 'user')
test_identifier = X_test['user']
X_test = X_test.drop(columns = 'user')


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# ## Model Building

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


# In[ ]:


precision_score(y_test, y_pred)


# In[ ]:


recall_score(y_test, y_pred)


# In[ ]:


f1_score(y_test, y_pred)


# In[ ]:


df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10, 7))
sns.set(font_scale = 1.4)
sns.heatmap(df_cm, annot = True, fmt = 'g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Logestic Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() *2))


# ### Formating the Final Results

# In[ ]:


final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()


# In[ ]:


final_results['predicted_results'] = y_pred


# In[ ]:


final_results[['user', 'enrolled', 'predicted_results']].reset_index(drop = True)


# In[ ]:




