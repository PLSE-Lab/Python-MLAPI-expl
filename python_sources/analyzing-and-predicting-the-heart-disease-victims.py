#!/usr/bin/env python
# coding: utf-8

# ![](http://)In this kernel, I am exploring all features, and analyzing both categorical and non-categorical data. So, let's see how these features gonna influence the Heart Disease.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Reading the "heart.csv" file

# In[ ]:


df = pd.read_csv("../input/heart.csv")


# let's see few samples of data

# In[ ]:


df.sample(3)


# seems like all the fields are in numeric.

# As said in description,
# Let's understand the data,
# > 1. age 
# > 2. sex 
# > 3. chest pain type (4 values) 
# > 4. resting blood pressure 
# > 5. serum cholestoral in mg/dl 
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved 
# > 9. exercise induced angina 
# > 10. oldpeak = ST depression induced by exercise relative to rest 
# > 11. the slope of the peak exercise ST segment 
# > 12. number of major vessels (0-3) colored by flourosopy 
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# So,** "cp","fbs", "restecg", "ca" and "thal"** are categorical attributes.

# ***PART - I***
# 
# ***analyzing Categorical Data*** :

# Let's plot these categorical values, based on target

# In[ ]:


"""
Normalizing the values and then making it as a DataFrame and then plotting using sns.barplot.
"""
temp = (df.groupby(['target']))['cp'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "cp", data = temp).set_title("Chest Pain vs Heart Disease")


# ***Insights : ***
# * From the above plot, we can understand that,
# * * chest pain of **type - 2** constitutes most of the chest Pain Category for Heart Disease.

# Let's check how other features are affecting Heart Disease

# In[ ]:


temp = (df.groupby(['target']))['fbs'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "fbs", data = temp).set_title("FBS vs Target")


# Seems like there's no such difference.I guess, We can eliminate this feature while building model

# In[ ]:


temp = (df.groupby(['target']))['restecg'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "restecg", data = temp).set_title("resting electrocardiographic results vs Heart Disease")
                     


# As we see,
# * "restecg" of **type 1** are more prone to Heart Disease compared to that of other types.

# In[ ]:


temp = (df.groupby(['target']))['ca'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "ca", data = temp).set_title("ca vs Heart Disease")
                     


# From the above plot we can understand that,
# * ca having of **type 0** are high in Heart Disease patients.

# In[ ]:


temp = (df.groupby(['target']))['thal'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "thal", data = temp).set_title("thal vs Heart Disease")
                     


# As we see,
# * **Type - 3** is common in people who are not affected with Heart Disease 
# * **Type - 2** is common in Heart Disease victims.

# Finished of Categorical Data, let's check about other attributes.

# In[ ]:


df.info()


# Let's plot boxplot and check whether there are any outliers in any columns

# In[ ]:


df.boxplot()
plt.xticks(rotation = 90)


# seems like some of the columns have worst outliers. Let's see correlation. How these columns are affecting target column

# In[ ]:


import seaborn as sns
sns.heatmap(df.corr())


# Let's remove outliers.
# * Outliers are those which are in (Q1 - 1.5 * IQR ) and (Q3 + 1.5 * IQR).
# * where IQR is Inter Quartile Range (Q3 - Q1), where Q1 is 25%ile and Q3 is 75percintile

# In[ ]:


plt.boxplot(df.trestbps)


# In[ ]:


Q3 = df.trestbps.quantile(.75)
Q1 = df.trestbps.quantile(.25)
IQR = Q3 - Q1
df = df[~((df.trestbps < Q1 - 1.5*IQR) | (df.trestbps > Q3 + 1.5*IQR))]


# In[ ]:


Q3 = df.chol.quantile(.75)
Q1 = df.chol.quantile(.25)
IQR = Q3 - Q1
df = df[~((df.chol < Q1 - 1.5*IQR) | (df.chol > Q3 + 1.5*IQR))]


# In[ ]:


df.boxplot()
plt.xticks(rotation = 90)


# In[ ]:


sns.countplot(x="target", data=df,hue = 'sex').set_title("GENDER - Heart Diseases")


# In[ ]:


df.sample()


# **Let's start  Analyzing the data and find some insights.**

# 1. average age of **MALE** who got stroke

# In[ ]:


df[(df.target ==  1) & (df.sex == 1)].age.mean()


# 2. average age of **FEMALE** who got stroke

# In[ ]:


df[(df.target ==  1) & (df.sex == 0)].age.mean()


# 3. Average values of features that are responsible for disease for **female**

# In[ ]:


df[(df.target ==  1) & (df.sex == 0)].describe()[1:2]


# 4. Average values of features that are responsible for disease for **male**

# In[ ]:


df[(df.target ==  1) & (df.sex == 1)].describe()[1:2]


# 5. Average values of features that are responsible for not having disease for **female**

# In[ ]:


df[(df.target ==  0) & (df.sex == 0)].describe()[1:2]


# 6. Average values of features that are responsible for not having disease for **male**

# In[ ]:


df[(df.target ==  0) & (df.sex == 0)].describe()[1:2]


# But in the above data,
# **we are applying describe() on categorical data too, which is wrong. we Should use "mode" on Categorical data. So let's do it.** Before that, let's generalize the Age attribute to Age-group

# In[ ]:


x = df.age.tolist()
after_x = []
for i in x:
    if i < 20:
        after_x.append("teenager")
    elif i < 30:
        after_x.append("20 - 30")
    elif i < 40:
        after_x.append("30 - 40")
    elif i < 50:
        after_x.append("40 - 50")
    elif i < 60:
        after_x.append("50 - 60")
    else:
        after_x.append("senior citizen")
df["age_category"] = after_x


# In[ ]:


df.sample()


# In[ ]:


for_analyzing = df.groupby(["age_category","sex","target"]).agg({"age":"mean", "trestbps":"mean", "chol":"mean", "thalach":"mean",     "exang":"mean","oldpeak":"mean", "slope":"mean","fbs" : pd.Series.mode,     "cp" : pd.Series.mode, "restecg": pd.Series.mode,"ca":pd.Series.mode,"thal":pd.Series.mode})
for_analyzing


# So, above are the generalized values based on grouping by age-group, Gender, Heart Disease. So. let's draw some insights from it.
# * **Categorical attributes insights**
# * Each and every Age Group has , mode, **"thal" - type - 2 for Heart Disease victims** irrespective of GENDER
# * Each and every Age Group has , mode, **"ca" - type - 0 for Heart Disease victims** irrespective of GENDER
# * Each and every Age Group has , mode, **"cp" - type - 0 for NON - Heart Disease victims** irrespective of GENDER
# *  **Non - categorical attributes insights**
# * **old peak value**,  is always **less for Heart Disease victims** compared to that that of those who doesn't have heart disease. (irrespective of GENDER).
# * Similarly, even** exang value** is also **less for Heart Disease victims**.
# * **thalach value** is **more for Heart Disease victims**.
# 
# 
# 
# 

# Let's build the model

# In[ ]:


df.sample()


# In[ ]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(df.iloc[:,:-2],df.iloc[:,-2],)


# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
dpen = []
for i in range(5,11):
    model = XGBClassifier(max_depth = i)
    model.fit(train_x,train_y)
    target = model.predict(test_x)
    dpen.append(accuracy_score(test_y, target))
    print("accuracy : ",dpen[i-5])
print("Best accuracy: ",max(dpen))


# * The accuracy can be improved by feature engineering.
# * I would like to know what's the ideal way of seleceting the best features for fitting the model. Is there any way other than correlation for finding the relation?
# * Let me know if there are any suggestions regarding the model building. How to know what model to be used?
