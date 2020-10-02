#!/usr/bin/env python
# coding: utf-8

# In[78]:


#import all necessory libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading training , testing and gender submission csv file

# In[79]:


df = pd.read_csv("../input/train.csv")
df.head()


# In[80]:


df2 = pd.read_csv("../input/test.csv")
df2.head()


# In[81]:


gender_sub = pd.read_csv("../input/gender_submission.csv")
gender_sub.head()


# In[ ]:





# In[82]:


df['Age'].hist()


# In[83]:


trace0 = go.Scatter(
    x = df['Age'],
    y = df['Survived'],
    mode = 'markers',
    name = 'markers'
)

layout = go.Layout({
        "title" : 'Age wise Survived rate'
})


data = [trace0]
fig = go.Figure(data,layout)
py.offline.iplot(fig)


#  ## Data Wrangling

# In[84]:


#Creating new family_size column
df['Family_Size']=df['SibSp']+df['Parch']
df2['Family_Size']=df2['SibSp']+df2['Parch']


# In[85]:


#replace Nan value to 0 
df['Age'].fillna(0 , inplace = True)
df2['Age'].fillna(0 , inplace = True)


# In[86]:


#segregate age in various groups
def age_classification(age):
    
    if age >= 1 and age < 15:
        return 1
    elif age >= 15 and age <= 30:
        return 2
    elif age > 30  and age <= 50:
        return 3
    elif age > 50  and age <= 70:
        return 4
    elif age > 70  and age <= 90:
        return 5
    return 0


# In[87]:


df['age_class'] = df['Age'].apply(lambda x :  age_classification(int(x)))
df2['age_class'] = df2['Age'].apply(lambda x :  age_classification(int(x)))


# ## Label encoding of gender 

# In[88]:


from sklearn.preprocessing import LabelEncoder
label_sex = LabelEncoder()
label_cabin = LabelEncoder()

df['gender'] = label_sex.fit_transform(df['Sex'])
df2['gender'] = label_sex.fit_transform(df2['Sex'])


# ## Creating actual training and testing dataframe with necessory fields

# In[89]:


train_titanic_data = pd.DataFrame({
    'pclass' : df['Pclass'],
    'age' : df['age_class'],
    'family' : df['Family_Size'],
    'gender' : df['gender'],
    'survived' : df['Survived']
})


# In[90]:


train_titanic_data.head()


# In[91]:


test_titanic_data = pd.DataFrame({
    'pclass' : df2['Pclass'],
    'age' : df2['age_class'],
    'family' : df2['Family_Size'],
    'gender' : df2['gender']
})


# In[92]:


test_titanic_data.head()


# ## Creating Model through Estimator Api 

# In[93]:


X_train = train_titanic_data.drop('survived',axis=1)


# In[94]:


Y_train = train_titanic_data['survived']


# In[95]:


import tensorflow as tf


# In[96]:


#creating feature columns
fea_age = tf.feature_column.categorical_column_with_vocabulary_list('age', [0,1,2,3,4,5,6])
fea_family = tf.feature_column.categorical_column_with_vocabulary_list('family' , [0,1,2,3,4,5,6,7,8,9,10])
fea_gender = tf.feature_column.categorical_column_with_vocabulary_list('gender' , [0,1])
fea_pclass = tf.feature_column.categorical_column_with_vocabulary_list('pclass' , [1,2,3])


# In[97]:


fea_age_embed = tf.feature_column.embedding_column(fea_age , dimension=6)
fea_family_embed = tf.feature_column.embedding_column(fea_family , dimension=11)
fea_gender_embed = tf.feature_column.embedding_column(fea_gender , dimension=2)
fea_pclass_embed = tf.feature_column.embedding_column(fea_pclass , dimension=3)


# In[98]:


feat_cols = [fea_age_embed ,fea_family_embed,fea_gender_embed,fea_pclass_embed]


# ### building a input function 

# In[99]:


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,batch_size=10,num_epochs=1000,shuffle=True)


# ## DNNClassifier Model

# In[100]:


model = tf.estimator.DNNClassifier(feature_columns=feat_cols,hidden_units=[5,5,10,10],n_classes=2,optimizer='Adagrad' , activation_fn=tf.nn.sigmoid)


# In[101]:


model.train(input_fn=input_func,steps=5000)


# In[102]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=test_titanic_data,batch_size=100,shuffle=False)


# In[103]:


predictions = list(model.predict(input_fn=pred_fn))


# In[104]:


predictions[0]


# In[105]:


final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])


# In[106]:


final_preds[:10]


# In[107]:


gender_sub.Survived[:10]


# ## Analysis of model

# In[108]:


predict_submission = pd.DataFrame({
    'pid' : df2['PassengerId'],
    'survived' : final_preds,
    'gender' : df2['Sex']
})


# In[109]:


pmale = predict_submission[predict_submission.gender == 'male']


# In[110]:


len(pmale[pmale.survived==1])


# In[111]:


fmale = predict_submission[predict_submission.gender == 'female']


# In[112]:


len(fmale[fmale.survived==0])


# In[113]:


from sklearn.metrics import classification_report , confusion_matrix


# In[114]:


print(classification_report(gender_sub.Survived,final_preds))


# In[115]:


cm = confusion_matrix(gender_sub.Survived, final_preds)


# In[116]:


import seaborn as sn
plt.figure(figsize =(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[119]:


submission = pd.DataFrame({
        "PassengerId":df2["PassengerId"],
        "Survived": final_preds
    })
submission.to_csv('submission.csv', index=False)

