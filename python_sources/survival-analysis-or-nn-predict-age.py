#!/usr/bin/env python
# coding: utf-8

# Hello Kaggle,nice to meet you.This is my first kernel and i want to try the lifelines library.Don't hesitate to give me some advice if you found mistakes. This will not be a convantionnal approach in kaggle because it's not really machine learning or deep learning approach. Next i will use a  NN to compare the two aproach.  And then i will stop focussing on age and i will predict my survivability with lightgbm.

# 1. **Problem and approach**
# 2. **Create Survival model**
# 3. **Predict my survivability**
# 4. **Neural network to predict the age my heart diseases**
# 5. **Lightgbm to predict if someone have heart diseases **
# 
# 
# 
# 
# 

# Don't hesitate to leave a comment if you don't like or like my kernel. (a little upvote if you are fan is welcome )

#  **Problem and approach**

# I started this project to predict what is the probability you will have heart diseases. I want to test the lifelines library for survival analysis. I will start with a simple model. Next i will use a more complete model in order to extract the weight of variable. Then i will  extract weight as bases of my analyze.At the end i will try to predict when i will have an heart disease!!

# 

# In[ ]:



import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import seaborn as sns
sns.set(rc={'figure.figsize':(18,10)})


# In[ ]:


data=pd.read_csv('../input/heart.csv')


# In[ ]:


data.head()


# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
equilibre=data['target'].value_counts()
ax=equilibre.plot.bar(title='target_count')
ax.set(xlabel='Notrhing/heart_disease', ylabel='Count')
print(equilibre)


# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
import matplotlib as plt
corr=data.corr()
ax = sns.heatmap(corr,cmap='coolwarm')


# In[ ]:


import lifelines
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter


# I'm going to start with a simple model with no attribute. The model just estimate the percentage you stay in good health based on your age. (It's the Kaplan Meir fitter. I will try to made a more complex model after this one). 

# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
kmf = KaplanMeierFitter()
kmf.fit(data['age'], data['target'], label="kmf.plot()")
ax=kmf.plot()
ax.set(xlabel='Age', ylabel='Probability_nothing')


# **Create Survival model**

# Here you can see the 'survival curve'. It represent the propability you stay in good health in function of the time. This is a simple estimation based only on the age. To continue i will train a model to predict in how much time you risk to have  heart diseases. This one will be based on CoxPH fitter. The main advantage is the fact this fitter will take many arguments and is able to give us some weight of features . I will use this weight as a base of my analysis.

# In[ ]:


cph = CoxPHFitter()
cph.fit(data, duration_col='age',  event_col='target')

estimation=cph.baseline_survival_
 
hazard=cph.baseline_cumulative_hazard_
print(cph.score_)
print(cph.summary)


# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
hazard['curve']=estimation.values
hazard['curve1']=hazard['curve']+(hazard['baseline hazard']/100)
hazard['curve2']=hazard['curve']-(hazard['baseline hazard']/100)

ax=hazard['curve'].plot(color='r',label='main_curve')
hazard['curve1'].plot(color='b',alpha=0.5,ax=ax,label='error_sup')
hazard['curve2'].plot(color='b',alpha=0.5,ax=ax,label='error_inf')
ax.set(xlabel='Age', ylabel='Probability_nothing')
ax.legend()


# Here you can see the survival curve. In red you have the estimation and in blue the confidence interval.This curve is based on all  different variables. like the previous curve it is the average with all values. Just under i will extract the hazard rates ( what i call weight of the variables). For instance :  in a study, men receiving the same treatment may suffer a certain complication ten times more frequently per unit time than women, giving a hazard ratio of 10.

# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
cph.plot()


# I see that Fbs seems to be a huge factor. I will start with this one. 

# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
total_1=data['fbs'].loc[data['target']==0].value_counts()
total_2=data['fbs'].loc[data['target']==1].value_counts()
df=pd.DataFrame({'nothing':total_2,'heart_disease':total_1})
ax=df.plot.bar(title='Target functiun of fbs',colormap='Accent')
ax.set(xlabel='Fbs', ylabel='Count')


# we see a difference between the two class for fbs : if fbs< 120 mg/dl you have more chance to avoid heart diseases. The ratio is near one when you are higher than 120 mg/dl.  It can be consider as a huge augmentation of the risk.  i have search a little and 120 mg/dl represent the limit you don't want to exceed. So it's seem logic.

# In[ ]:


sns.set(rc={'figure.figsize':(18,10)})
total_1=data['exang'].loc[data['target']==0].value_counts()
total_2=data['exang'].loc[data['target']==1].value_counts()
df=pd.DataFrame({'Nothing':total_2,'Heart_diseases':total_1})

a=df.plot.pie(subplots=True,colormap='Set1',autopct='%.0f%%',label='',title='angina')


# angina also known as angina pectoris, is the chest pain or pressure, usually due to not enough blood flow for the heart muscle. Th is is the variable i study here
# The ratio of angina in heart_diseases ( grey part) is as expected higher than when you have nothin.  You multiplied by 4 the risk to have something if you have an angina.  If i have one advice when i see this graph, angina have not to take lightly!
# 

# In[ ]:


total_1=data['slope'].loc[data['target']==0].value_counts()
total_2=data['slope'].loc[data['target']==1].value_counts()
df=pd.DataFrame({'Nothing':total_2,'Heart_diseases':total_1})
sns.set(rc={'figure.figsize':(18,10)})
a=df.plot.pie(subplots=True,colormap='Set1',autopct='%.0f%%',label='',title ='slope')


# I don't find yet the meaning of the value 0,1,2(i will investigate later). But it's clear that the factor 1 change.It is a factor of risk for heart diseases. 

# In[ ]:


total_1=data['sex'].loc[data['target']==0].value_counts()
total_2=data['sex'].loc[data['target']==1].value_counts()
df=pd.DataFrame({'nothing':total_2,'Heart_diseases':total_1})

ax=df.plot.bar(colormap='Accent',label='',title ='male vs female')
ax.set(xlabel='male/female', ylabel='Count')


# I have heard that female are more resistant than men. I want with this graph investigate this myth. In the graph, this myth seems to be a reality, but i want to go further. 

# In[ ]:


total_1=data[['sex','age','target']].loc[data['sex']==0]
total_2=data[['sex','age','target']].loc[data['sex']==1]

kmf = KaplanMeierFitter()
kmf.fit(total_1['age'], total_1['target'], label="kmf.plot()")
ax=kmf.plot(label='female')

kmf = KaplanMeierFitter()
kmf.fit(total_2['age'], total_2['target'], label="kmf.plot()")
kmf.plot(color='g',title='male vs female',ax=ax,label='male')

ax.set(xlabel='age', ylabel='Probability of good health')


# here i'm surpised, i thinked women had less probabilit to contract heart diseases than men, but these curves say the contrary. I need to investigate more. I will watch the distribution per age of men and women because i have the intuition women leave older and that why they have higher probability to contract heart diseases.

# In[ ]:



total=data[['age','sex']].loc[data['target']==1]
total_1=data['age'].loc[data['sex']==0].value_counts()
total_2=data['age'].loc[data['sex']==1].value_counts()
sns.set(rc={'figure.figsize':(20,12)})


df=pd.DataFrame({'Male':total_2,'Female':total_1})
df.plot.bar(title='Distribution')
#total_2=data[['age','target']].loc[data['sex']==1]
#group1=total_2.groupby('age').size()
#group1.plot.bar(colormap='Accent',label='',title ='male and female by age',ax=ax,color='r',stack=True)
ax.set(xlabel='age', ylabel='Count')


# here i underlign that i have more old women than men old men .It can explain the previous curve. Due to the fact women leave older, they have more probability to have a diseases. 

# **Predict my survivability**

# I will try to predict my survivability. i'm a young man with no real health problem. the vector will be this one:
# 01. age: 24
# 02. sex=1 male
# 03. chest pain=0
# 04. trestbps=130
# 05. chol=170 ( i take an average value)
# 06. fbs=0 (<120)
# 07. restecg=0 ( don't really know the meaning of 0 and 1 here)
# 08. thalach=120
# 09. exang=0 (no)
# 10. oldpeak=0.62
# 11. slope=1 (random value don't know what it is)
# 12. ca=0
# 13. thal=3

# In[ ]:


data.head()
me=np.array([24,1,0,130,170,0,0,120,0,0.62,1,0,3,0])
data.loc[-1] = me


# In[ ]:


cph = CoxPHFitter()
cph.fit(data, duration_col='age',  event_col='target')
censored_subjects = data.loc[data['target'] == 0]


# In[ ]:


unconditioned_sf = cph.predict_survival_function(censored_subjects)
print(unconditioned_sf.head())


# In[ ]:


ax=unconditioned_sf[-1].plot(label='me')
unconditioned_sf[167].plot(color='r',ax=ax,label='random')
ax.set(xlabel='age', ylabel='probability of good health')
ax.legend()


# Blue curve  is my survival curve. The red one is another random one for exemple. Mine seems not so cool but i have used random value in some variable. I hope it come from that !!!  in the next one i will take some threshold to define when my heart will have problem. 

# In[ ]:


from lifelines.utils import median_survival_times, qth_survival_times
predictions_75 = qth_survival_times(0.75, unconditioned_sf)
predictions_25 = qth_survival_times(0.25, unconditioned_sf)
predictions_50 = median_survival_times(unconditioned_sf)


# In[ ]:


import matplotlib.pyplot as plt

ax=unconditioned_sf[-1].plot(label='me')
unconditioned_sf[167].plot(color='y',label='random',ax=ax)

plt.axvline((predictions_75[-1].values), 0,1,color='g',label='75%')
plt.axvline((predictions_50[-1].values), 0,1,color='b',label='50%')
plt.axvline((predictions_25[-1].values), 0,1,color='r',label='25%')
ax.set(xlabel='age', ylabel='probability of good health')
ax.legend()


# here i see that i have a big probabilitie to contract a heart disease near my 65 year old. Not the best news for me ! . My random friend seems to be luckier than me. 
# 

# **Neural network to predict my heart diseases**

# In this part i will train a very simple NN to predict my age when i will have a heart diseases. 
# I want to compare this result with my previous ones ( 65 years old)

# In[ ]:


data_train=(data.loc[data['target']==1]).copy()
data_train.drop(['target'],1,inplace=True)
feature=[c for c in data_train.columns if c not in ['age']]
target=['age']

data_train.head()


# In[ ]:


train=data_train[:130]
val=data_train[130:]


# In[ ]:


from keras.layers import Activation, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential

NN_model = Sequential()
NN_model.add(Dense(32, kernel_initializer='normal'))
NN_model.add(Dense(1, kernel_initializer='normal'))    
NN_model.add(Activation('linear'))
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

history=NN_model.fit(train[feature].values, train[target].values, epochs=19, batch_size=10)


preds = NN_model.predict(val[feature].values) 
score = mean_absolute_error(val[target].values, preds)
print(score)


# In[ ]:


fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['mean_absolute_error'])
plt.show()


# In[ ]:


me=np.array([1,0,130,170,0,0,120,0,0.62,1,0,3,0])
data_test=(data.loc[data['target']==0]).copy()
data_test.drop(['target'],1,inplace=True)
data_test.loc[-1] = me
estimation = NN_model.predict(data_test[feature].values)


# In[ ]:


estimation[-1]
print("A heart diseases will append at the age of: {}".format(estimation[-1]))


# i have lose some years with this NN!! 

# **Predict if someone have heart diseases**

# In[ ]:


import lightgbm as lgb
feature=[c for c in data_train.columns if c not in ['target']]
target=data['target']


# In[ ]:


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
predict = np.zeros(len(data))
feature_importance_df = pd.DataFrame()


# In[ ]:


param={
       'bagging_fraction': 0.33,
       'boost_from_average':'false',
       'boost': 'gbdt',
       'max_depth': -1,
       'metric':'auc',
       'objective': 'binary',
       'verbosity': 1
    }


# In[ ]:


from sklearn.metrics import roc_auc_score
for fold_, (trn_idx, val_idx) in enumerate(folds.split(data.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(data.iloc[trn_idx][feature], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(data.iloc[val_idx][feature], label=target.iloc[val_idx])

    num_round = 500
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)
    predict[val_idx] = clf.predict(data.iloc[val_idx][feature], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = feature
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    

print("CV score: {:<8.5f}".format(roc_auc_score(target, predict)))


# In[ ]:


feature_importance_df.head()


# In[ ]:


try1=feature_importance_df.groupby(['Feature'],as_index=False).mean()
try1.drop(['fold'],1,inplace=True)


# In[ ]:



sns.barplot(x="importance", y="Feature", data=try1.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (average_for_all_fold)')


# In[ ]:


me=np.array([[24,1,0,130,170,0,0,120,0,0.62,1,0,3]])
random=data.iloc[1][feature]
disease_me=clf.predict(me,num_iteration=clf.best_iteration)
disease_random=clf.predict(random,num_iteration=clf.best_iteration)
print("Your result is: {}".format(disease_me))
print("other result is: {}".format(disease_random))


# 
