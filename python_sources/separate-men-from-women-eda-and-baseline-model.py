#!/usr/bin/env python
# coding: utf-8

# # Goal of the kernel 
# 
# I built this kernel for three main reasons: 1) to dust my vizualization skills while waiting for a Deep Neural Network to finish training, 2) because I thought that the particular dataset is really interesting for further inclusion in cognitive systems, and 3) to show to early-stage aspiring data scientists the power of visuals in telling a story. 
# 
# I then performed some basic modelling to see if I could reach already accomplished accuracy levels. 
# 
# Enjoy and keep those thumbs up! 

# ## Part 1. Data importation and basic info

# In[ ]:


#Import needed libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Input data
df = pd.read_csv('../input/voice.csv')


# In[ ]:


df.head()


# In[ ]:


#Check ratio of classes in dependent variable
gender_dict = {'male': 0,'female': 1}
gender_list = [gender_dict[item] for item in df.label]

print('Ratio between two classes is:', np.mean(gender_list))


# All right, so our data set contains 50% men and 50% women. A balanced dataset is always a nice thing that makes our life easier when modelling.  

# ## Part 1. Visuallization 

# ### kde plots - a nice alternative to histograms

# In[ ]:


#Lets check the distribution of meanfreq (mean vs women)
plt.figure()
sns.kdeplot(df['meanfreq'][df['label']=='male'], shade=True);
sns.kdeplot(df['meanfreq'][df['label']=='female'], shade=True);
plt.xlabel('meanfreq value')
plt.show()


# In[ ]:


#Print mean of each category
print('Male mean frequency:', np.mean(df['meanfreq'][df['label']=='male']))
print('Female mean frequency:', np.mean(df['meanfreq'][df['label']=='female']))


# ### A quick statistical test: Student's t-test for mean equality

# I wonder if these two means are statistically different....Lets check this out. We are going to use Student's t-test for this goal. The null hypothesis here is that the two means are equal. If the p-value of the test falls below a significance level (say 5%) we reject the null hypothesis in favor of the alternative.

# In[ ]:


#Run the student's t-test
male_df = df[df['label']=='male']
female_df = df[df['label']=='female']
t2, p2 = stats.ttest_ind(male_df['meanfreq'], female_df['meanfreq'])

print('P-value:', p2)


# The p-value is really close to zero which provides evidence against our null hypothesis of mean equality. This probably means that this feature will have explanatory power on our dependent variable

# ### I enjoyed that! Lets run another one!

# In[ ]:


#Check the distribution of meanfun (mean vs women)
plt.figure()
sns.kdeplot(df['meanfun'][df['label']=='male'], shade=True);
sns.kdeplot(df['meanfun'][df['label']=='female'], shade=True);
plt.show()


# Hmm, in this case the dissimilarity between means is eye-catching; no need for statistical tests. I bet this variable will have great importance on our model!

# ### Heatmaps for correlation visualization

# In[ ]:


#Lets construct a heatmap to see which variables are very correlated
no_label_data = df.drop(['label'], axis = 1)
cors = no_label_data.corr()

plt.figure(figsize=(12,7))
sns.heatmap(cors, linewidths=.5, annot=True)
plt.show()


# ### Boxplots are also nice because they contain SO much info

# In[ ]:


plt.figure(figsize=(8,7))
sns.boxplot(x="label", y="dfrange", data=df)
plt.show()


# Many outliers there outside the top whisker for males. That means that there quite many male persons that sound female-like, since they have high frequency ranges.

# ### Violinplot: a 'modern' alternative to boxplots that better describes the data distribution

# In[ ]:


plt.figure(figsize=(8,7))
sns.violinplot(x="label", y="meanfun", data=df)
plt.show()


# There is a great difference between the medians of the two distributions thus is fair to believe that meanfun will be a very powerful explanatory variable. 

# ### Scatterplot: all-time classic way of depicting the relation between two numerical variables

# In[ ]:


sns.lmplot( x="sfm", y="meanfreq", data=df, fit_reg=False, hue='label', legend=False)
plt.show()


# Highly negative correlation between average of fundamental frequency and spectral flatness. 

# ### Pairplots: a quick and easily digestible way to dive into your numerical data

# In[ ]:


#Lets select fewer variables
no_label_data_red = no_label_data[['median', 'skew', 'kurt', 'sp.ent', 
                                   'sfm', 'centroid', 'dfrange', 'modindx']]
sns.pairplot(no_label_data_red)
plt.show()


# ### Jointplots: check data correlation and distribution in one chart

# In[ ]:


sns.jointplot(x=df["centroid"], y=df["sfm"], kind='scatter',
              color='m', edgecolor="skyblue", linewidth=1)

plt.show()


# In[ ]:


#Jointplot alternative: 'hex'. Easily identify the area where two variables are forming a 'cloud' (the peak of their distributions)
sns.jointplot(x=df["centroid"], y=df["sfm"], kind='hex',
              color='m', edgecolor="skyblue", linewidth=1)

plt.show()


# # Part 2. Modelling

# In[ ]:


#Shuffle data 
df = df.sample(frac=1, random_state = 42)


# In[ ]:


#Check missing 
df.isnull().sum().sum()


# Missing value count says there are no missing values. But you gotta be smarter than that! Maybe another value is where null should have been.  A carefull examination of the dataset and u will see that absolute zeros should actually pertain to missing values that have been filled in. So lets replace them back to nulls and lets leave them that way since we are going to use boosted trees that can handle this situation. If we were using some other algorithm(SVMs or Neural Nets for example) we would have to fill them in properly (perhaps with median or some other technique)

# In[ ]:


#Replace zeros with null
df.replace(0, np.nan, inplace=True)


# In[ ]:


#Now lets count nulls again 
df_null = df.isnull().sum()


# In[ ]:


plt.figure(figsize=(8,7))
g = sns.barplot(df_null.index, df_null.values)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.ylabel('No of missing values')
plt.show()


# Now thats more like it. Mode seems to have the most missing values with about 250 values. Anyway still very little comared to our data so it should not make a great difference.

# In[ ]:


#Convert text target variable to number
gender_dict = {'male': 0,'female': 1}
gender_list = [gender_dict[item] for item in df.label]

df_final = df.copy() 
df_final['label'] = gender_list


# In[ ]:


#Split in train and test
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state = 14)


# In[ ]:


#Get input and output variables
train_x = train_df.drop(['label'], axis = 1)
train_y = train_df['label']

test_x = test_df.drop(['label'], axis = 1)
test_y = test_df['label']


# In[ ]:


#LGB model
lgb_train = lgb.Dataset(train_x, train_y)

# Specify hyper-parameters as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 16,
    'max_depth': 6,
    'learning_rate': 0.1,
    #'feature_fraction': 0.95,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    #'reg_alpha': 0.1,
    #'reg_lambda': 0.1,
    #'is_unbalance': True,
    #'num_class': 1,
    #'scale_pos_weight': 3.2,
    'verbose': 1,
}

# Train LightGBM model
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=90,
                #valid_sets= lgb_valid,
                #early_stopping_rounds=40,
                verbose_eval=20
                )


# In[ ]:


# Plot Importances
print('Plot feature importances...')
importances = gbm.feature_importance(importance_type='gain')  # importance_type='split'
model_columns = pd.DataFrame(train_x.columns, columns=['features'])
feat_imp = model_columns.copy()
feat_imp['importance'] = importances
feat_imp = feat_imp.sort_values(by='importance', ascending=False)
feat_imp.reset_index(inplace=True)

plt.figure()
plt.barh(np.arange(feat_imp.shape[0] - 1, -1, -1), feat_imp.importance)
plt.yticks(np.arange(feat_imp.shape[0] - 1, -1, -1), (feat_imp.features))
plt.title("Feature Importances")
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# It looks like by far the most important variable is meanfun which corresponds to how high the frequency of the voice is to us people. This is expectable both from analysis but also intuitively: we expect men to have thicker, fatter voices and women thinner, high frequency voices. Lets see how that performs in new data: 

# In[ ]:


pred_lgb = gbm.predict(test_x,num_iteration=50)
pred_01 = np.where(pred_lgb > 0.5, 1, 0)


# In[ ]:


recall_pred = recall_score(test_y, pred_01)
precision_pred = precision_score(test_y, pred_01)
accuracy_pred = accuracy_score(test_y, pred_01)


# In[ ]:


print('Recall score: %0.2f' %recall_pred)
print('Precision score: %0.2f' %precision_pred)
print('Overall Accuracy: %0.2f' %accuracy_pred)


# In[ ]:


confusion_matrix(test_y, pred_01)


# # Part 3. Conclusion 
# 
# From a quick and simple LightGBM model we got amazing results! All metrics reach levels close to 99%! The confusion matrix shows that we missclassified only 3 men as women and 5 women as men. By going through a parameter fine-tuning process (e.g. gridsearch) we might have done even better than that! 
# 
# I hope that you enjoyed this kernel, learnt something new and that the visuals here are useful at your own work!

# In[ ]:




