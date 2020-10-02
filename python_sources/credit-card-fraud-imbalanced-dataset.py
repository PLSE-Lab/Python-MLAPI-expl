#!/usr/bin/env python
# coding: utf-8

# This dataset is imbalanced dataset. We have to carefull to pick a technique to process and evaluate.
# Here the link https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
# 
# 1. use the evaluation metrics : not use the accuracy but precision, recall, f1 score (harmonic mean of precision and recall), mcc or auc
# 2. resampling training set : undersampling or oversampling
# 3. cluster the abudant class
# 4. the famous **XGBoost** is already a good starting point if the classes are not skewed too much, because it internally takes care that the bags it trains on are not imbalanced.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 1. CHECK descriptive analysis 
# 2. Check null / missing value
# 3. check the imbalanced dataset
# 4. check correlation dataset, pick the potent variable
# 5. pake algoritma apa
# 6. evaluasi

# In[ ]:


data_df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data_df.head()


# In[ ]:


data_df.head()


# In[ ]:


data_df.describe()


# Looking to the Time feature, we can confirm that the data contains 284,807 transactions, during 2 consecutive days (or 172792 seconds).

# **Data Unbalanced**

# In[ ]:


temp = data_df["Class"].value_counts()
df = pd.DataFrame({'Class': temp.index,'values': temp.values})

trace = go.Bar(
    x = df['Class'],y = df['values'],
    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
    marker=dict(color="Red"),
    text=df['values']
)

data = [trace]
layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
          xaxis = dict(title = 'Class', showticklabels=True), 
          yaxis = dict(title = 'Number of transactions'),
          hovermode = 'closest',width=600
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='class')


# In[ ]:


data_df.head()


# In[ ]:


data_df[data_df['Class'] == 1].head(1)


# In[ ]:


data_df[data_df['Class'] == 0].head(1)


# In[ ]:


corr = data_df.corr(method='pearson').head()


# In[ ]:


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[ ]:


corr = data_df.corr().unstack().sort_values()
print_full(corr)


# In[ ]:


component_var = {}
for i in range(1,28):
    pca = PCA(n_components=i)
    res = pca.fit(data_df)
    component_var[i] = sum(pca.explained_variance_ratio_)
    
print(component_var)


# In[ ]:


plt.matshow(data_df.corr())
plt.show()


# In[ ]:


f = plt.figure(figsize=(19,15))
plt.matshow(data_df.corr(),fignum=f.number)
plt.xticks(range(data_df.shape[1]),data_df.columns,fontsize=14, rotation=45)
plt.yticks(range(data_df.shape[1]),data_df.columns,fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)


# In[ ]:


corr = data_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


plt.hist(data_df['Class'], color='blue', edgecolor='black', bins = int(185/5))
sns.distplot(data_df['Class'], hist=True, kde=False, bins=int(180/5), color='blue', hist_kws={'edgecolor':'black'})

plt.title('histo')
plt.xlabel('delay ')
plt.ylabel('flight')


# In[ ]:


print("Normal :", data_df['Class'][data_df['Class'] == 0].count())

print("Fraud :", data_df['Class'][data_df['Class'] == 1].count())


# In[ ]:


# separate classes into different dataset
classNormal = data_df.query('Class == 0')
classFraud = data_df.query('Class == 1')

#randomize the dataset
classNormal = classNormal.sample(frac=1)
classFraud = classFraud.sample(frac=1)


# In[ ]:


classNormaltrain = classNormal.iloc[0:6000]
classFraudtrain = classFraud

#combine become one
train = classNormaltrain.append(classFraudtrain, ignore_index=True).values


# In[ ]:


X = train[:,0:30].astype(float)
Y = train[:,30]


# In[ ]:


model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)

scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:


mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for(train,test), color in zip(kfold.split(X,Y), colors) :
    probas_ = model.fit(X[train], Y[train]).predict_proba(X[test])
    
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:,1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i +=1
    
plt.plot([0,1],[0,1], linestyle='--', lw=lw, color='k', label='luck')

mean_tpr /= kfold.get_n_splits(X,Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label ='Mean ROC(area = %0.20f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


corrmat = df.corr()
f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(corrmat,vmax=0.8,square=True,annot=True,annot_kws={'size':8})


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

y_t = np.array(data_df['Class'])
X_t = data_df
X_t = data_df.drop(['Class'],axis=1)
X_t = np.array(X_t)

print("shape of Y :"+str(y_t.shape))
print("shape of X :"+str(X_t.shape))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_t = scaler.fit_transform(X_t)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X_t,y_t,test_size=.20,random_state=42)
print("shape of X Train :"+str(X_train.shape))
print("shape of X Test :"+str(X_test.shape))
print("shape of Y Train :"+str(Y_train.shape))
print("shape of Y Test :"+str(Y_test.shape))


# In[ ]:


for this_C in [1,3,5,10,40,60,80,100]:
    clf = SVC(kernel='linear',C=this_C).fit(X_train,Y_train)
    scoretrain = clf.score(X_train,Y_train)
    scoretest  = clf.score(X_test,Y_test)
    print("Linear SVM value of C:{}, training score :{:2f} , Test Score: {:2f} \n".format(this_C,scoretrain,scoretest))


# In[ ]:




