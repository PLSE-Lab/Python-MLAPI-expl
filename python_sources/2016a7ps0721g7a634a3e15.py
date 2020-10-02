#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 


# In[ ]:


np.random.seed(0)


# In[ ]:


train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")


# In[ ]:


pd.set_option('display.max_rows', 5400)
# "The maximum width in characters of a column"


# In[ ]:


train.head(120)


# In[ ]:


train.info()


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


# In[ ]:


df_tr = train
#TODO
from sklearn.preprocessing import MinMaxScaler
m = MinMaxScaler()
#m.fit_transform(df_tr)
X = df_tr.drop(['class'],axis=1)
y = train['class'].tolist()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#y=train['class']
#x=train.drop('class',axis=1)
##Oversampling
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import ADASYN
#import random

#oversampler=SMOTE(kind='regular',k_neighbors=3,random_state=random.randint(1,100000))

#x_resampled, y_resampled = oversampler.fit_resample(x, y)


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


# In[ ]:


param_test5={'booster':['gbtree'],
'eta':[0,0.02,0.04,0.06],
'gamma':np.arange(0,0.6,0.1),
'learning_rate':np.arange(0.01,0.07,0.01),
'n_estimators':[120,140,160,180],
'max_depth':range(3,7,1),
'min_child_weight':range(1,6,2),
'objective':['multi:softprob','multi:softmax'] }

gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(), 
 param_grid = param_test5, cv = 3, n_jobs = -1,verbose=2)
gsearch1.fit(X,y)
gsearch1.best_params_


# In[ ]:


from xgboost import XGBClassifier
# fit model no training data
model = xgb.XGBClassifier(booster='gbtree',eta=0,gamma=0.4,learning_rate =0.03, max_depth=4,
 min_child_weight=3,n_estimators=140,objective='multi:softprob')
model.fit(X, y)


# In[ ]:


# make predictions for test data
y_pred = model.predict(X_test)
print(y_pred)
predictions = [round(value) for value in y_pred]
print(predictions)


# In[ ]:


y_pred = model.predict(test)
y_ans = [round(value) for value in y_pred]


# In[ ]:


import pandas as pd
c1=pd.DataFrame(test, columns=['id'])
c2 = pd.DataFrame(y_ans, columns = ['class'])
c2 = c2.astype(int)
data_frame = pd.merge(c1, c2, right_index=True, left_index=True)
data_frame


# In[ ]:


y_ans


# In[ ]:


data_frame.to_csv('submission.csv',columns=['id','class'],index=False)


# In[ ]:





# In[ ]:





# # Second Python Notebook

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


pd.set_option('display.max_rows', 5400)
# "The maximum width in characters of a column",to ensure we can see all data via .head() function


# In[ ]:


train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")


# In[ ]:


train.isnull().values.any()


# In[ ]:


test.isnull().values.any()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train = train.drop(['id'], axis=1)
tester = test.drop(['id'],axis=1)


# # Oversampling via imblearn

# 

# In[ ]:


#y=train['class']
#x=train.drop('class',axis=1)
##Oversampling
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import ADASYN
#import random

#oversampler=SMOTE(kind='regular',k_neighbors=3,random_state=random.randint(1,100000))

#x_resampled, y_resampled = oversampler.fit_resample(x, y)


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(train.drop(['class'],axis=1), train['class'])


# In[ ]:


afterLDATransform = pd.DataFrame(lda.transform(tester))
tester['lda_0'] = afterLDATransform[0]
tester['lda_1'] = afterLDATransform[1]
tester['lda_2'] = afterLDATransform[2]
tester['lda_3'] = afterLDATransform[3]
tester['lda_4'] = afterLDATransform[4]


# In[ ]:


tester.head()


# In[ ]:


new_features = pd.DataFrame(lda.transform(train.drop(['class'],axis=1)))
train['lda_0'] = new_features[0]
train['lda_1'] = new_features[1]
train['lda_2'] = new_features[2]
train['lda_3'] = new_features[3]
train['lda_4'] = new_features[4]


# In[ ]:


train.head()


# In[ ]:


train['factor'] = (train['chem_3']*train['attribute']*train['chem_4'])
tester['factor'] = (tester['chem_3']*tester['attribute']*tester['chem_4'])


# In[ ]:


X = train.drop(['class'],axis=1)
y = train['class']


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr, center=0)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
# Compute the correlation matrix
corr = train.corr(method="kendall")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

plt.show()


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)


# In[ ]:


#Constructing a scatter plot of the data
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='g')


# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trainScaled = train.copy()
testScaled = tester.copy()
columns = tester.columns
trainScaled[columns]=scaler.fit_transform(train[columns])
testScaled[columns]=scaler.transform(tester[columns])


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)
plt.xlabel('row0')
plt.ylabel('row1')
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='g')


# In[ ]:


trainScaled.head()


# In[ ]:


testScaled.head()


# In[ ]:


from sklearn.decomposition import PCA

model=PCA(n_components=2)
fittedData = model.fit(trainScaled.drop('class',axis=1)).transform(trainScaled.drop('class',axis=1))


# In[ ]:


plt.figure(figsize=(10,8))
plt.xlabel('row0')
plt.ylabel('row1')
plt.legend()
plt.scatter(fittedData[:,0],fittedData[:,1],label = train['class'],c=train['class'])
plt.show()


# In[ ]:


x = trainScaled.drop(['class'],axis=1)
y = trainScaled['class']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

clf = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")
clf.fit(x_train,y_train)
y_ans = clf.predict(x_test)
accuracy_score(y_test, y_ans)


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")
clf.fit(x,y)
test['class'] = clf.predict(testScaled)
data_frame = test[['id','class']]
data_frame.info()


# In[ ]:


data_frame.head(120)


# In[ ]:


data_frame.to_csv('submission.csv',columns=['id','class'],index=False)


# In[ ]:




