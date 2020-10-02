#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from statsmodels.api import Logit


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve


# # Load dataset

# In[ ]:


backpain = pd.read_csv('../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')
backpain.head()


# In[ ]:


backpain.drop('Unnamed: 13',axis=1,inplace=True)


# In[ ]:


column_name = ['pelvic_incidence','pelvictilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','status']


# In[ ]:


backpain.columns = column_name


# In[ ]:


backpain.head()


# In[ ]:


backpain.info()


# In[ ]:


sns.countplot(backpain.status)


# # Encode

# In[ ]:


backpain['status'] = backpain['status'].map({'Abnormal':1,'Normal':0})


# # Check skewness

# In[ ]:


fig = plt.figure(figsize=(15,15))
for i in range(backpain.shape[1]):
    ax = fig.add_subplot(5,3,i+1)
    sns.distplot(backpain[backpain.columns[i]],ax=ax)


# # Train test split

# In[ ]:


x = backpain.drop(['status','scoliosis_slope','lumbar_lordosis_angle','thoracic_slope','pelvic_incidence','sacrum_angle','pelvic_slope','Direct_tilt'],axis=1)
y = backpain.status


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.30,random_state=1)


# # Apply Classification Model 

# In[ ]:


lor = LogisticRegression()


# In[ ]:


lor.fit(xtrain,ytrain)


# In[ ]:


ypred = lor.predict(xtest)
ypred


# # Accuracy

# In[ ]:


cm = confusion_matrix(ytest,ypred)
cm


# In[ ]:


accuracy = accuracy_score(ytest,ypred)
accuracy


# # Feature Selection

# ## using correlarion

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(backpain.corr(),annot=True)


# ## forward selection

# In[ ]:


temp = backpain.drop(['scoliosis_slope','lumbar_lordosis_angle'],axis=1)


# In[ ]:


vif = [variance_inflation_factor(temp.values,i) for i in range(temp.shape[1])]


# In[ ]:


pd.DataFrame({'colums':temp.columns,'vif':vif})


# ## backward elimination

# In[ ]:


result = Logit(temp['status'],temp.drop(['status','thoracic_slope','pelvic_incidence','sacrum_angle','pelvic_slope','Direct_tilt'],axis=1)).fit()
result.summary()


# # Conclusion

# In[ ]:


conf = np.exp(result.conf_int())
conf['odds-ratio'] = round(result.params,2)
conf['pvalue'] = round(result.pvalues,3)
conf.rename({0:'ci_2.5%',1:'ci_97.5%'},inplace=True,axis=1)
conf


#     holding all other features constant ,odd of having backpain is:
#     - 11% increase for every 1 unit increase in pelvictilt
#     - 6% decrease for every 1 unit increase in sacral_slope
#     - 2% decrease for every 1 unit increase in pelvic_radius
#     - 15% increase for every 1 unit increase in degree_spondylolisthesis
#     - 15% increase for every 1 unit increase in cervical_tilt

# # Model statistics

# In[ ]:


TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
misclassification = 1 - accuracy
sensitivity = TP/float(TP+FN)
specificity = TN/float(TN+FP)
print('accuracy, misclassification, sensitivity, specificity')
print(accuracy,misclassification,sensitivity,specificity)


# In[ ]:


positive_predicted_value = TP/float(TP+FP)
negative_predicted_value = TN/float(TN+FN)
print('positive_predicted_value, negative_predicted_value')
print(positive_predicted_value,negative_predicted_value)


# In[ ]:


positive_likelyhood_ratio = sensitivity/(1-specificity)
negative_likelyhood_ratio = (1-sensitivity)/specificity
print('positive_likelyhood_ratio, negative_likelyhood_ratio')
print(positive_likelyhood_ratio,negative_likelyhood_ratio)


# # ROC-AUC curve

# In[ ]:


fpr,tpr,thresolds = roc_curve(ytest,ypred)
plt.plot(fpr,tpr)
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')


# In[ ]:


roc_auc_score(ytest,ypred)


# In[ ]:




