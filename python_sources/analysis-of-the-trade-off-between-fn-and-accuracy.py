#!/usr/bin/env python
# coding: utf-8

# # Diagnosis of COVID-19 - Analysis of the trade-off between FN and Accuracy
# > ## AI and Data Science supporting clinical decisions (Task #1)
# > #### Rodrigo Fragoso

# #### The main objective on this notebook is to explore and find how to deal with the NaN values and implement some Classifiers, sampling and feature engineering techniques to understand how the unbalance results can affect our models.

# <a id='top'></a>
# 
# 1. [Importings](#t1)
# 
# 2. [Exploratory Data Analysis](#t2)
#     
# 3. [Feature Engineering](#t3)
# 
# 4. [Model Selection](#t4)
# 
# 5. [Resampling and fitting the train data](#t5)
# 
# 6. [Sequential Feature Selection](#t6)
# 
# 7. [Final Conclusions](#t7)
# 
# 
# 
# 
# 

# <a id='t1'></a>
# # <div>1. Importings</div>
# [Summary](#top)
# 
# [Next](#t2)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data= pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
print('Dimensions: ',data.shape[0],'rows','x',data.shape[1],'columns')
data.head()


# #### With a brief look, we can see that are lots of missing values on this dataset

# In[ ]:


sns.set(font_scale=1.5)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white','figure.figsize':(10,5)})
sns.set_style("white")

tg_values=data['SARS-Cov-2 exam result'].value_counts()
tg_values.plot.barh(color=tuple(["r", "black"]))
plt.title('SARS-Cov-2 exam result')
print("Negative exam results: "+"{:.2%}".format(tg_values[0]/tg_values.sum())+' ('+str(tg_values[0])+' records)')
print("Positive exam results: "+"{:.2%}".format(tg_values[1]/tg_values.sum())+'  ('+str(tg_values[1])+' records)')
print('')


# #### Obs: we can see a very unbalanced dataset

# <a id='t2'></a>
# # <div>2. Exploratory Data Analysis</div>
# [Summary](#top)
# 
# [Back](#t1)
# 
# [Next](#t3)

# In[ ]:


#Labeling encode the target variable
def positive_bin(x):
    if x == 'positive':
        return 1
    else:
        return 0
data['SARS-Cov-2 exam result_bin']=data['SARS-Cov-2 exam result'].map(positive_bin)


# In[ ]:


nulls=(data.isnull().sum()/len(data))*100
print('Percentage (%) of nulls for each feature:')
nulls.sort_values(ascending=False)


# ## Kernel Density Estimate (KDE) - Probability density function of % of nulls of features

# ### Positive and negative,separatedly

# In[ ]:


sns.set(font_scale=1.5)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white','figure.figsize':(8,5)})
sns.set_style("white")

pos=data[data['SARS-Cov-2 exam result_bin']==1]
neg=data[data['SARS-Cov-2 exam result_bin']==0]

nulls_neg=(neg.isnull().sum().sort_values(ascending=False)/len(neg))*100
nulls_pos=(pos.isnull().sum().sort_values(ascending=False)/len(pos))*100

ax=sns.distplot(nulls_pos[nulls_pos>0],color='red',bins=20,kde_kws={"color": "red", "label": "Positive"})
ax=sns.distplot(nulls_neg[nulls_neg>0],color='black',bins=20,kde_kws={"color": "black", "label": "Negative"})
# ax=sns.distplot(nulls_neg[nulls_neg>0],color='black',kde=False, norm_hist=False,bins=20) # histogram
ax.set(xlabel='% of Nulls',title='Features nulls KDE (% Nulls > 0)',label='3')
plt.grid(False)
plt.show()


# ### All exam results

# In[ ]:


ax=sns.distplot(nulls[nulls>0],color='blue',bins=20,kde_kws={"color": "blue", "label": "All Exam Results"})
ax.set(xlabel='% of Nulls',title='Variables Nulls KDE (% Nulls > 0)')
plt.grid(False)
plt.show()


# ## Corrrelation with SARS-Cov-2 exam result, for features with less than 90% nulls

# In[ ]:


variables_corr=nulls.loc[nulls<90].index.tolist()


# In[ ]:


corr = data[variables_corr].corr()['SARS-Cov-2 exam result_bin'].abs().sort_values(ascending=False)
corr


# ### Correlation Heat Map, for features with less than 90% nulls

# In[ ]:


corr=data[variables_corr].corr().abs()

fig, ax = plt.subplots(figsize=(15, 15))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, annot=True, fmt='.2f')
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


# ### Platelets and Leukocytes shows higher correlation, than the others features, with the target variable. Even so, that isn't a strong correlation

# In[ ]:


data['SARS-Cov-2 exam result_Baseline']=0
print("Baseline accuracy: "+"{:.2%}".format((data['SARS-Cov-2 exam result_Baseline']==data['SARS-Cov-2 exam result_bin']).sum()/len(data['SARS-Cov-2 exam result_Baseline'])))


# ### As a baseline, we'll be using a prediction off all exams as Negative. That give us 90,11% accuracy

# ### Conclusions from this section:
# - As we can see there a lot of missing records;
# - Most of the variables have at least 80% of NaNs;
# - We are dealing with a very unbalanced dataset 9:1 negative/positive results.

# > #### Assuming all results as negative will be a good baseline

# <a id='t3'></a>
# # <div>3. Feature Engineering</div>
# [Summary](#top)
# 
# [Back](#t2)
# 
# [Next](#t4)

# ## Based on the **KDE**, I'm selecting features that have a maximum of 90% as missing values

# In[ ]:


nulls.drop(['SARS-Cov-2 exam result','Patient ID','SARS-Cov-2 exam result_bin'],inplace=True)


# In[ ]:


selecting_features=nulls.loc[nulls<90].index.tolist()
features=selecting_features
features.append('SARS-Cov-2 exam result_bin')


# In[ ]:


print(features)


# ## Encode categorical features using label encoder

# In[ ]:


df=data[features]

def bins(x):
    if x == 'detected' or x=='positive':
        return 1
    elif x=='not_detected' or x=='negative':
        return 0
    else:
        return x
    
for col in df.columns:
    df[col]=df[col].apply(lambda row: bins(row))


# ## Analysing some statistical attributes from each feature, to solve out what to do with the NaNs values

# In[ ]:


pd.set_option('display.max_columns', None)
df.describe()


#  ### Variables with too long range, not good to substitute **NaNs** with 0.
#  ### Using KNN imputer to substitute then, another approach could be substituting the nulls with the feature mean:

# In[ ]:


from sklearn.impute import KNNImputer

X=df.drop(['SARS-Cov-2 exam result_bin'],axis=1)

temp = X
imputer = KNNImputer(n_neighbors=3)
temp = imputer.fit_transform(X.values)

X = pd.DataFrame(temp, columns=X.columns)
y = df['SARS-Cov-2 exam result_bin']
X.head()


# ## Conclusions:
# - Only features that have a maximum of 90% as missing values were selected;
# - Label encoder was used for categorical values;
# - KNN imputer was used to fill the NaN values.

# <a id='t4'></a>
# # <div>4. Model Selection</div>
# [Summary](#top)
# 
# [Back](#t3)
# 
# [Next](#t5)

# ## First of all, the dataset will be splitted in Train(80%) and Test(20%)

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=5)
print('Train shape:',X_train.shape)
print('Test  shape:',X_test.shape)


# ## Training a Random Forest Classifier, Logistic Regression and a Decision Tree (10 folds CV)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import log_loss, accuracy_score

resultados1=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in kf.split(X_train):
    
    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]
    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]
    
    rf= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    rf.fit(Xtr,ytr)
    
    p=rf.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados1.append(acc)


# In[ ]:


from sklearn.linear_model import LogisticRegression

resultados2=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in kf.split(X_train):
    
    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]
    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]
    
    lr= LogisticRegression(max_iter=50)
    lr.fit(Xtr,ytr)
    
    p=lr.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados2.append(acc)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

resultados3=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in kf.split(X_train):
    
    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]
    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]
    
    dt= DecisionTreeClassifier(random_state=3, max_depth=8)
    dt.fit(Xtr,ytr)
    
    p=dt.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados3.append(acc)


# ### Results:

# In[ ]:


print('')
p1=rf.predict(X_test)
p1[:]=rf.predict(X_test)
acc=accuracy_score(y_test,p1)
print("Vanilla Random Forest          Test accuracy: "+"{:.2%}".format(acc)+' / Train accuracy: '+"{:.2%}".format(np.mean(resultados1)))
p2=lr.predict(X_test)
p2[:]=lr.predict(X_test)
acc=accuracy_score(y_test,p2)
print("Vanilla Logistic Regression    Test accuracy: "+"{:.2%}".format(acc)+' / Train accuracy: '+"{:.2%}".format(np.mean(resultados2)))
p3=dt.predict(X_test)
p3[:]=dt.predict(X_test)
acc=accuracy_score(y_test,p3)
print("Vanilla Decision Tree          Test accuracy: "+"{:.2%}".format(acc)+' / Train accuracy: '+"{:.2%}".format(np.mean(resultados3)))
print('')


# ## In spite of having a high accuracy, we are dealing with a very low recall
# ### This is a real problem. In my opinion, the worse result that we should mitigate are the **False Negatives(FN)**, as they can bring thee COVID back to society instead of putting them in quarentine

# In[ ]:


visual=pd.concat([X_test,y_test],axis=1)
visual['predict']=p2
visual2=visual[visual['SARS-Cov-2 exam result_bin']==visual['predict']]

print('Positive results in the test sample: ',visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0])
print('Positive results correctly predicted: ',visual2[visual2['predict']==1].shape[0])
print('Only positives accuracy: ',"{:.2%}".format(visual2[visual2['predict']==1].shape[0]/visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0]))


# ## Looking at the precision - recall curve, we can see that changing our threshold can improve our recall, but it will reduce the accuracy and the precision will fall very hardly.
# ## What is the most important metric here?

# In[ ]:


sns.set(font_scale=1.5)
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white','figure.figsize':(8,5)})
sns.set_style("white")

pred_prob=lr.predict_proba(X_test)

from sklearn.metrics import precision_recall_curve,roc_curve

scores=pred_prob[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, scores)

plt.rcParams["axes.grid"] = True

plt.plot([1,0],[0,1],linestyle = '--',lw = 2,color = 'grey')
plt.plot(recall[:-1],precision[:-1],label='Logistic Regression', color='red',lw=2, alpha=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall curve')
plt.legend(loc="upper right")
plt.show()

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')
scores=pred_prob[:,1]
fpr, tpr, thresholds = roc_curve(y_test,scores)
plt.plot(fpr,tpr,label='Logistic Regression', color='black',lw=2, alpha=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


# ### The confusion matrix reforces that this model has lots of FNs

# In[ ]:


from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(lr, X_test, y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')
disp.ax_.set_title('Confusion Matrix - Exam Results')
disp.ax_.grid(False)


# ## Conclusions:
# - In spite of having a "high" accuracy, there are lots of False Negatives(FN);
# - In this case, accuracy could not a good metric, but it still is important;
# - Maybe this is also happening because the data is too unbalanced;
# - Looking at the precision/recall curve, we can see that changing the threshold will bring different results.

# <a id='t5'></a>
# # <div>5. Resampling and fitting the train data</div>
# [Summary](#top)
# 
# [Back](#t4)
# 
# [Next](#t6)

# ## Two teechniques were used:
# 

# ![](https://miro.medium.com/max/1400/1*ENvt_PTaH5v4BXZfd-3pMA.png)
# [source](https://miro.medium.com/max/1400/1*ENvt_PTaH5v4BXZfd-3pMA.png)

# *  **Only the train set will be resampled**
# *  Oversampling with Synthetic Minority Oversampling Technique(SMOTE)
# *  Undersampling with Random UnderSampler
# *  1:1 Positive / Negative Ratio
# *  Random Forest Classifier

# In[ ]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2,n_jobs=-1,sampling_strategy=1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# In[ ]:


resultados3=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in kf.split(X_train_res):
    
    Xtr, Xvld = X_train_res.iloc[train], X_train_res.iloc[valid]
    ytr, yvld = y_train_res.iloc[train], y_train_res.iloc[valid]
    
    rf_os= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    rf_os.fit(Xtr,ytr)
    
    p=rf_os.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados3.append(acc)


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=27,sampling_strategy=1)
X_train_res2, y_train_res2 = rus.fit_resample(X_train, y_train)


# In[ ]:


resultados3=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in kf.split(X_train_res2):
    
    Xtr, Xvld = X_train_res2.iloc[train], X_train_res2.iloc[valid]
    ytr, yvld = y_train_res2.iloc[train], y_train_res2.iloc[valid]
    
    rf_us= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    rf_us.fit(Xtr,ytr)
    
    p=rf_us.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados3.append(acc)
p1=rf_us.predict(X_test)
p1[:]=rf_us.predict(X_test)
acc=accuracy_score(y_test,p1)


# ## Both of the resampling techniques improves our recall (less FN) and worsens the accuracy

# In[ ]:


disp = plot_confusion_matrix(rf_us, X_test, y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')
disp.ax_.set_title('UnderSampling Confusion Matrix - Exam Results')
disp.ax_.grid(False)

disp = plot_confusion_matrix(rf_os, X_test, y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')
disp.ax_.set_title('OverSampling Confusion Matrix - Exam Results')
disp.ax_.grid(False)


# ## Switching our thresold will bring different scenarios, but always trading-off between FN and Accuracy

# In[ ]:


pred_prob=rf_us.predict_proba(X_test)

scores=pred_prob[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, scores)

plt.rcParams["axes.grid"] = True

plt.plot([1,0],[0,1],linestyle = '--',lw = 2,color = 'grey')
plt.plot(recall[:-1],precision[:-1],label='Random Forest + Random Undersampling', color='red',lw=2, alpha=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.legend(loc="upper right")
plt.show()

pred_prob=rf_os.predict_proba(X_test)

scores=pred_prob[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, scores)

plt.plot([1,0],[0,1],linestyle = '--',lw = 2,color = 'grey')
plt.plot(recall[:-1],precision[:-1],label='Random Forest + SMOTE Oversampling', color='black',lw=2, alpha=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.legend(loc="upper right")
plt.show()


# <a id='t6'></a>
# # <div>6. Sequential Feature Selection</div>
# [Summary](#top)
# 
# [Back](#t5)
# 
# [Next](#t7)

# ## Trying to improve the model using Backward selection
# ### It should work better with a Custom Cost function, balancing accuracy and the recall

# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs

model=sfs(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0),k_features=8,forward=False,verbose=2,cv=10,n_jobs=-1,scoring='accuracy')
model.fit(X_train_res2,y_train_res2)


# In[ ]:


var=list(model.k_feature_names_)
var


# In[ ]:


X_train_res3=X_train_res2[var]
y_train_res3=y_train_res2

resultados3=[]
kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)
for train,valid in kf.split(X_train_res3):
    
    Xtr, Xvld = X_train_res3.iloc[train], X_train_res3.iloc[valid]
    ytr, yvld = y_train_res3.iloc[train], y_train_res3.iloc[valid]
    
    dt= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    dt.fit(Xtr,ytr)
    
    p=dt.predict(Xvld)
    acc=accuracy_score(yvld,p)
    resultados3.append(acc)


# ### Similar results to the original model

# In[ ]:


disp = plot_confusion_matrix(dt, X_test[var], y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')
disp.ax_.set_title('Confusion Matrix - Exam Results')
disp.ax_.grid(False)


# <a id='t7'></a>
# # <div>7. Final Conclusions</div>
# [Summary](#top)
# 
# [Back](#t6)

# -  In this study, we should choose what is the most important metric;
# -  Maybe reducing FN is more important than improving the accuracy, but we can't have a very low accuracy: will be the same as predicting all pacients as positive.
# -  With more data and model improvings, I think we can make a very helpful model that will assist doctors to make decisions.

# ## Stay safe!

# In[ ]:




