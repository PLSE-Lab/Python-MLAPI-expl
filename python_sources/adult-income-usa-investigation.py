#!/usr/bin/env python
# coding: utf-8

# # How to earn over $50K  a year in the US
# An investigation into predicting high-income earners.
# 
# This notebook develops an XGBoost Classifier model on the data set "Adult Incomes in the United States". Then investigates the effect of each feature on predicting if someone earns over $50K annually.
# This data is old and needed almost no cleaning. However, I believe the insights are extensive.
# 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
import eli5 
from eli5.sklearn import PermutationImportance 
from eli5 import show_weights 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Importing the data
# The data is contained in a .data, which is still readable by the panda's method `.read_csv` though the features are not included in the file.

# In[ ]:


features=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain","capital-loss", "hours-per-week", "native-country", "class"]
df = pd.read_csv("/kaggle/input/adult-incomes-in-the-united-states/adult.data", names=features)
y = df["class"]
X = df.drop(["class"], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)


# ## Check for missing data

# In[ ]:


train_X.isna().sum()


# The column `fnlwgt` is an estimation of the number of people this data point represents. We will remove it as previous investigations have found it has little to no effect on machine learning models. 
# 

# In[ ]:


train_X = train_X.drop(["fnlwgt"],axis=1)
val_X = val_X.drop(["fnlwgt"],axis=1)


# ## Encoding Categorical Data

# In[ ]:


high_cardinality = [col for col in train_X.columns if train_X[col].nunique() > 25]
high_cardinality


# We will remove `native-country` as it has a large cardinality, the other columns above are numerical.

# In[ ]:


train_X = train_X.drop(["native-country"],axis=1)
val_X = val_X.drop(["native-country"],axis=1)


# In[ ]:


train_X = pd.get_dummies(train_X, prefix_sep='_', drop_first=True)
val_X = pd.get_dummies(val_X, prefix_sep='_', drop_first=True)
train_X.columns


# In[ ]:


train_y = pd.Series(np.where(train_y.values == ' >50K', 1, 0), train_y.index)
val_y = pd.Series(np.where(val_y.values == ' >50K', 1, 0), val_y.index)


# In[ ]:


train_y.value_counts()


# ## Building the Model

# In[ ]:


my_model = XGBClassifier()

my_model.fit(train_X, train_y,  
             early_stopping_rounds=5,  
             eval_set=[(val_X, val_y)], 
             eval_metric="auc",
             verbose=False)


# ## Evaluating the Model

# In[ ]:


lr_probs = my_model.predict_proba(val_X)
lr_probs = lr_probs[:, 1]
yhat = my_model.predict(val_X)
lr_precision, lr_recall, _ = precision_recall_curve(val_y, lr_probs)
lr_f1, lr_auc = f1_score(val_y, yhat), auc(lr_recall, lr_precision)
print('XGBoost: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
no_skill = len(val_y[val_y==1]) / len(val_y)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='-', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='GXBoost')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.show()


# In[ ]:


ns_probs = [0 for _ in range(len(val_y))]
ns_auc = roc_auc_score(val_y, ns_probs)
lr_auc = roc_auc_score(val_y, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('XGBoost: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(val_y, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(val_y, lr_probs)
pyplot.plot(ns_fpr, ns_tpr, linestyle='-', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='XGBoost')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()


# So we have a model that looks reasonable by these two metrics, area under the receiver operating characteristic and precision recall curves.

# ## Investigation
# 

# We shall first look at Permuatation Importance figues as a start.

# In[ ]:


perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y) 
show_weights(perm, feature_names = val_X.columns.tolist()) 


# It seems that having a larger gain on capital investments is a large contributor to earning above $50,000 in annual income.
# 
# Interestingly, being married is also an important factor in determining higher-income classification. This could be because those who are married or have a partner can make bigger investments and thus bigger returns on capital.
# 
# Age and number of years of education are also important features, this makes sense as more years spent educating yourself typically leads to earning a higher wage. In addition, the older you become, the more likely you are to receive a higher income.
# 
# Capital-loss is also an important feature that will need to be investigated further.
# 
# Finally, we can see hours-per-week is also an important factor.
# 
# We will now look at a SHAP summary plot:
# 
# 
# 

# In[ ]:


import shap
explainer = shap.TreeExplainer(my_model) 
shap_values = explainer.shap_values(val_X) 
shap.summary_plot(shap_values, val_X) 


# It is clear making small capital gains doesn't affect the prediction of higher-income earners, however, high capital gain increases the prediction by a considerable margin, though in some cases, the opposite effect is observed.
# 
# `capital-loss` shows us something similar, in that your income is also likely to be higher if you make large losses on investments. This could be data from the same people, making lots of large investments that go up, and down. Further investigation is needed.
# 
# Being married or in a partnership compared to other relationship statuses also seems to have an equal effect however in opposite directions.
# 
# Age has a very negative effect on our prediction, but only if you are young. From middle-age to retirement, the change in prediction is not as strong.
# 
# 

# In[ ]:


shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index='capital-loss') 


# It seems there is no relation between those making capital gains and losses.
# 
# Curiously, values above only 10,000 in capital gains have a positive effect on our prediction.
# 
# However, it  is fascinating that some capital gains below 20,000 had a more positive effect on the prediction than making 100,000 in capital gains.
# In addition, those only making small capital gains have less of a chance of making over 50K a year.
# 
# I will search for an interaction between `capital gain` and other features.
# Please fork the notebook to see if I've missed anything interesting.

# In[ ]:


shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index='marital-status_ Married-civ-spouse') 


# Those who are married or in a civil partnership who make capital gains have a positive and negative effect on our prediction. However, it is interesting that a large number of those who make 100,000 in capital gains are also married.
# 
# 

# In[ ]:


#for col in train_X.columns:
#    shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index=col) 


# In[ ]:


shap.dependence_plot('capital-loss', shap_values, val_X, interaction_index="occupation_ Exec-managerial")


# In[ ]:


shap.dependence_plot('capital-gain', shap_values, val_X, interaction_index="occupation_ Exec-managerial")


# It seems like those in executive managerial roles, mostly have capital-gains above 10,000 and have a more positive effect on our prediction. Similarily their capital losses, for the most part, have a positive SHAP value. Lets further investigate if there is a correlation between this occupation and our other continuous variables.
# 
# Interestingly there is a lot of noise in our `capital-loss` SHAP value distribution plot, meaning that another variable is interacting with it. Further investigation is needed here as well.

# In[ ]:


train = train_X[["age","capital-gain","capital-loss","education-num", "hours-per-week","occupation_ Exec-managerial", "marital-status_ Married-civ-spouse"]]
correlations = train.corr()
fig, ax = pyplot.subplots(figsize=(10,10))
sns.heatmap(correlations,vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
pyplot.show();


# There are no significant correlations between these variables. Please fork the notebook yourself to view some comparison statistics on those in different occupations earning above and below 50K to see if Exec-managerial staff earn more. I will show the plot for those earning above and below 50K working in both Exec-managerial roles and other industries.
# 

# In[ ]:


"""
occup_cols= ['occupation_ Adm-clerical',
       'occupation_ Armed-Forces', 'occupation_ Craft-repair',
       'occupation_ Exec-managerial', 'occupation_ Farming-fishing',
       'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct',
       'occupation_ Other-service', 'occupation_ Priv-house-serv',
       'occupation_ Prof-specialty', 'occupation_ Protective-serv',
       'occupation_ Sales', 'occupation_ Tech-support',
       'occupation_ Transport-moving']
exe_mang_income = pd.concat([train_y, train_X[occup_cols]], axis=1)
exe_mang_income.columns = ['Income >50K', 'occupation_ Adm-clerical',
       'occupation_ Armed-Forces', 'occupation_ Craft-repair',
       'occupation_ Exec-managerial', 'occupation_ Farming-fishing',
       'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct',
       'occupation_ Other-service', 'occupation_ Priv-house-serv',
       'occupation_ Prof-specialty', 'occupation_ Protective-serv',
       'occupation_ Sales', 'occupation_ Tech-support',
       'occupation_ Transport-moving']

for col in occup_cols:
    sns.countplot(x = col, hue="Income >50K", data=exe_mang_income)
    pyplot.show()
"""
cols_of_interest=["occupation_ Exec-managerial","marital-status_ Married-civ-spouse"]
exe_mang_income = pd.concat([train_y, train_X[cols_of_interest]], axis=1)
exe_mang_income.columns = ['Income >50K','occupation_ Exec-managerial', "marital-status_ Married-civ-spouse"]
for col in cols_of_interest:
    sns.countplot(x=col, hue="Income >50K", data=exe_mang_income)
    pyplot.show()


# Above we see exec-managerial staff have the highest number of people earning above 50K, however that number is still below 2500 and there is an equal number of people in the same industry earning under 50K. 
# 
# So there is a higher number of people earning larger salaries in the Exec-managerial industry, but working in that industry doesn't guarantee you will be earning above 50K.
# 
# There is a significantly larger amount of people who are married and make above 50K, roughly 5000, than compared to those who are not married and make above 50K, approximately 1000. So we could say that those who are married are more likely to earn over 50K however it is not for certain.
# 
# The last thing I will look at will the SHAP values of 100 rows and choose an interesting prediction to see the effects of each feature on it. 

# In[ ]:


data_for_prediction = val_X.iloc[:100, :] 
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)


# On the far right at x=95, we see a prediction change from our baseline of 7.15. Lets view this prediction in isolation.

# In[ ]:


data_for_prediction = val_X.iloc[34, :] 
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)


# Here we see the driving factor as `capital-gain`, which is much larger than years of education, the factor with the second largest effect. Of interest is also that the capital gain is only $7,688, which is relatively small.

# ## Conclusion
# 
# The dataset is very old, though it does seem to give some interesting insights. Capital gains, being married and the number of years of education are the most important factors. Though more investigation should be conducted to see if these have a correlation. It would also be interesting to see if these findings are present in current datasets.
# 
# This is my first submission to Kaggle, any feedback on the interpretation and/or modelling would be much appreciated.
