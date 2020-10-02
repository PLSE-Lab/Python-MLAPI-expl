#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 


# In[2]:


train = pd.read_csv('../input/train_file.csv')
test = pd.read_csv('../input/test_file.csv')


# In[3]:


train.head()


# In[4]:


train.isna().sum()


# In[5]:


train.dtypes


# In[6]:


train['YEAR'].value_counts().plot.bar()


# Number of asked questions is higher in recent years

# In[7]:


train['LocationDesc'].value_counts()


# We can see that survey is perfomed on quite a large scale 

# In[8]:


train['Subtopic'].value_counts(normalize = True).plot.bar()


# Maximum number of questions asked in the survey was about the non-alcoholic drugs, one inference we can draw out of this is that people taking non-alcoholic drugs are more prone to addiction

# Let's check the distribution of sample size

# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[10]:


plt.figure(1,figsize=(16,6))

plt.subplot(121)
sns.distplot(train['Sample_Size'])

plt.subplot(122)
sns.boxplot(y=train['Sample_Size'])

plt.show()


# We can see the significant number of outliers are present in our dataset for this column. Hence, transdormation will be required

# In[11]:


plt.figure(1,figsize=(16,6))

plt.subplot(121)

sns.distplot(np.log(train['Sample_Size']))

plt.subplot(122)
sns.boxplot(y=np.log(train['Sample_Size']))

plt.show()


# Now its better

# In[12]:


train.Sex.value_counts().plot.bar()


# ??? what is a total gender ??? 

# Let's dig a bit

# In[13]:


train.loc[train['Sex']=='Total'].head()


# We can see columns 'Race' and 'Sex' is having values as 'Total', it could be because some people might not be wanting to share their details

# In[14]:


train['Race'].value_counts().plot.bar(figsize = (16,6))


# In[15]:


train['Grade'].value_counts().plot.bar(figsize = (10,6))


# Most people who were surveyed have comlpeted education upto grade 4

# One inference we can draw from the nature of the dataset is statergy adopted to prevent the addiction will certainly affect 'Greater_Risk_Probability'

# In[16]:


train['StratificationType'].value_counts().plot.bar()


# Lets check year wise of average of Greater_Risk_Probability

# In[17]:


def make_plot(df,col_name,figsize=(16,6)):
    
    plt.figure(1,figsize=figsize)
    dic = {}
    for val in df[col_name].value_counts().keys():
        dic[val] = np.mean(df['Greater_Risk_Probability'].loc[df[col_name]==val])
    
    plt.bar(range(len(dic)),dic.values(),align='center')
    plt.xticks(range(len(dic)),dic.keys())
    plt.show()


# In[18]:


make_plot(train,'YEAR')


# We can see that average of 'Greater_Risk_Probability' has decreased with years

# Let's check Location-wise average of 'Greater_Risk_Probability'

# In[19]:


dic = {}
for val in train['LocationDesc'].value_counts().keys():
    dic[val] = np.mean(train['Greater_Risk_Probability'].loc[train['LocationDesc']==val])

dic = sorted(dic.items(),key=lambda x: x[1])[::-1]

dic


# One inference we can draw from above is  different locations have "greater_risk_probabilty". This  feature might be useful for our model

# In[20]:


make_plot(train,col_name='Subtopic',figsize=(8,4))


# **Intresting!!** People addicted to alchols have higher average probability of staying addicted, this could be because alchols are legal commodity and is easily available

# In[21]:


plt.figure(1,(16,8))
plt.scatter(x=train['Greater_Risk_Probability'],y=train['Sample_Size'])


# Some abnoramlly large sample size is present in our datset

# In[22]:


make_plot(train,'Sex',(8,4))


# Average of greater_probabilty risk is higher in Males

# In[23]:


make_plot(train,'Race',(16,4))


# In[24]:


make_plot(train,'Grade',(8,4))


# In[25]:


make_plot(train,'StratificationType',(8,4))


# Average greater_risk_probabilty is quite uniform for differernt StratificationType

# Let's Process ahead with some preprocessing feature selection 

# In[26]:


X = train.copy()
X_test = test.copy()

y= X['Greater_Risk_Probability']
X = X.drop(labels = 'Greater_Risk_Probability',axis=1)


# In[27]:


from sklearn.preprocessing import LabelEncoder


# In[28]:


label = LabelEncoder()
X['LocationDesc'] = label.fit_transform(X['LocationDesc'])
X_test['LocationDesc'] = label.fit_transform(X_test['LocationDesc'])

label = LabelEncoder()
X['Sex'] = label.fit_transform(X['Sex'])
X_test['Sex'] = label.fit_transform(X_test['Sex'])

label = LabelEncoder()
X['Race'] = label.fit_transform(X['Race'])
X_test['Race'] = label.fit_transform(X_test['Race'])

label = LabelEncoder()
X['StratificationType'] = label.fit_transform(X['StratificationType'])
X_test['StratificationType'] = label.fit_transform(X_test['StratificationType'])

label = LabelEncoder()
X['QuestionCode'] = label.fit_transform(X['QuestionCode'])
X_test['QuestionCode'] = label.fit_transform(X_test['QuestionCode'])

X['Sample_Size'] = np.log(X['Sample_Size'])
X_test['Sample_Size'] = np.log(X_test['Sample_Size'])


# In[29]:


# drop = ['Patient_ID','Greater_Risk_Question','Description','GeoLocation','QuestionCode']
drop = ['Patient_ID','Greater_Risk_Question','Description','GeoLocation']
X = X.drop(labels = drop,axis=1)
X_test = X_test.drop(labels = drop,axis=1)


# In[30]:


from sklearn.feature_selection import f_classif


# In[31]:


fval,p_val = f_classif(X,y)

print('F-values for different features')
print(fval)

print('P-values for different features')
print(p_val)


# * Let's drop the StratificationType, Race, Grade

# In[32]:


X = X.drop(labels = ['StratificationType','Race','Grade'],axis=1)
X_test = X_test.drop(labels = ['StratificationType','Race','Grade'],axis=1)

fval,p_val = f_classif(X,y)

print('F-values for different features')
print(fval)

print('P-values for different features')
print(p_val)


# In[33]:


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# In[34]:


continuous_cols = ['YEAR','Sample_Size']

categorical_cols = ['LocationDesc','Sex','StratID1','StratID2','StratID3','QuestionCode']

mapper = DataFrameMapper(  
    [([continuous_col], StandardScaler()) for continuous_col in continuous_cols] +
    [([categorical_col], OneHotEncoder()) for categorical_col in categorical_cols])

pipe = Pipeline([('mapper',mapper)])

pipe.fit(X)


# In[35]:


X = pipe.transform(X)
X_test = pipe.transform(X_test)


# In[36]:


X.shape


# In[37]:


y = (y/100).values


# Let's create a baseline model and perform a k-fold cross validation on it
# 

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,mean_squared_error


# In[39]:


kf = KFold(n_splits=10,shuffle=True,random_state=42)
cv_scores = []

for train_index, val_index in kf.split(X):
    
    X_train, X_val = X[train_index],X[val_index]
    y_train, y_val = y[train_index],y[val_index]
    
    regressor = LinearRegression(n_jobs=-1)
    regressor.fit(X_train,y_train)

    pred = regressor.predict(X_val)
    mean = mean_squared_error(y_val,pred)
    cv_scores.append(mean)
    print(mean)
print(f"Mean Score {np.mean(cv_scores)}")


# In[40]:


from keras.layers import Dense, Dropout
from keras.models import Sequential


# In[41]:


model = Sequential()
model.add(Dense(256,input_dim=132,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(rate=0.7))
model.add(Dense(1,activation='relu'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])


# In[42]:


model.summary()


# In[43]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# In[44]:


checkpoints = ModelCheckpoint('model.h5',monitor='val_mean_squared_error',mode='min',save_best_only='True',verbose=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, patience=2, verbose=1, min_lr=0.000001)


# In[45]:


epochs = 50
batch_size = 64


# In[46]:


history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                    validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])


# In[47]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(history.history['mean_squared_error'], label='Train MSE')
plt.plot(history.history['val_mean_squared_error'], label='Test MSE')
plt.legend(('Train MSE', 'Val MSE'))
plt.show()


# In[48]:


model.load_weights('model.h5')


# In[51]:


pred_test = (model.predict(X_test)*100).round(4)
test['Greater_Risk_Probability'] = pred_test
test.head()


# In[52]:


df_sub = test.loc[:,['Patient_ID','Greater_Risk_Probability']]
df_sub.head()


# In[53]:


df_sub.to_csv(path_or_buf = 'submission.csv',index=False)


# In[ ]:




