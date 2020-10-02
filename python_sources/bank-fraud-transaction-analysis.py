#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns


# In[ ]:


df = pd.read_excel("/kaggle/input/Case Study - Fraud analysis.xlsx")
df.head(1000)


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


df.size


# In[ ]:


df.values


# In[ ]:


df.isnull().sum()


# In[ ]:


df.columns


# In[ ]:


print('No Frauds', round(df['Genuine'].value_counts()["Genuine"]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Genuine'].value_counts()["Fraud"]/len(df) * 100,2), '% of the dataset')


# In[ ]:


import plotly.graph_objects as go
x = ['Genuine', 'Fraud']
y = [df['Genuine'].value_counts()["Genuine"], df['Genuine'].value_counts()["Fraud"] ]
fig = go.Figure([go.Bar(x=x , y= y )])
fig.update_layout(title_text = "Fraud Distributions")
fig.show()


# In[ ]:


df_M = df[df["Gender"] == "M"]
X_1 = df_M.groupby(["Genuine"]).count()
Male_FG = (X_1["Gender"])

df_F = df[df["Gender"] == "F"]
Y_1 = df_F.groupby(["Genuine"]).count()
Female_FG = (Y_1["Gender"])

print(Male_FG)
print(Female_FG)


# In[ ]:


import plotly.graph_objects as go
x = ['Genuine_Male', 'Fraud_Male', 'Genuine_Female', 'Fraud_Female']
y = [332 , 8 ,660,  0 ]
fig = go.Figure([go.Bar(x=x , y = y )])
fig.update_layout(title_text = "Fraud Distributions")
fig.show()


# In[ ]:


df_M = df[df["Nationality"] == "UK"]
X_1 = df_M.groupby(["Genuine"]).count()
UK_FG = (X_1["Nationality"])
print("Fraud and Genuine in UK are:", UK_FG)

df_M = df[df["Nationality"] == "China"]
X_1 = df_M.groupby(["Genuine"]).count()
China_FG = (X_1["Nationality"])
print("Fraud and Genuine in China are:",China_FG)

df_M = df[df["Nationality"] == "India"]
X_1 = df_M.groupby(["Genuine"]).count()
UK_FG = (X_1["Nationality"])
print("Fraud and Genuine in India are:", UK_FG)

df_M = df[df["Nationality"] == "Italy"]
X_1 = df_M.groupby(["Genuine"]).count()
China_FG = (X_1["Nationality"])
print("Fraud and Genuine in Italy are:",China_FG)

df_M = df[df["Nationality"] == "Nigeria"]
X_1 = df_M.groupby(["Genuine"]).count()
UK_FG = (X_1["Nationality"])
print("Fraud and Genuine in Nigeria are:", UK_FG)

df_M = df[df["Nationality"] == "South Africa"]
X_1 = df_M.groupby(["Genuine"]).count()
China_FG = (X_1["Nationality"])
print("Fraud and Genuine in Sounth africa are:",China_FG)

df_M = df[df["Nationality"] == "Spain"]
X_1 = df_M.groupby(["Genuine"]).count()
UK_FG = (X_1["Nationality"])
print("Fraud and Genuine in Spain are:", UK_FG)

df_M = df[df["Nationality"] == "USA"]
X_1 = df_M.groupby(["Genuine"]).count()
China_FG = (X_1["Nationality"])
print("Fraud and Genuine in USA are:",China_FG)


# In[ ]:


import plotly.graph_objects as go
x = ['Fraud_UK','Fraud_China', 'Fraud_India','Fraud_Italy', 'Fraud_Nigeria', 'Fraud_SounthAfrica', 'Fraud_Spain','Fraud_USA']
y = [2, 0 , 1 , 3 , 2, 1, 1, 1 ]
fig = go.Figure([go.Bar(x=x , y = y )])
fig.update_layout(title_text = "Fraud Distributions")
fig.show()


# In[ ]:



for i in range (20,41):
    df_M = df[df["Age"]  == i]
    X_1 = df_M.groupby(["Genuine"]).count()
    L30_FG = (X_1["Age"])
    print(L30_FG)


# In[ ]:


import plotly.graph_objects as go
x = ['G_20','F_20','G_21','F_21','G_22','F_22','G_23','F_23','G_24','F_24','G_25','F_25','G_26','F_26','G_27','F_27','G_28','F_28','G_29','F_29','G_30','F_30','G_31','F_31','G_32','F_32','G_33','F_33','G_34','F_34','G_35','F_35','G_36','F_36','G_37','F_37','G_38','F_38','G_39','F_39' ,'G_40','F_40']
y = [50,0,42,8,34,0,46,0,46,0,58,0,53,0,45,0,43,0,50,0,44,0,55,0,42,0,58,0,43,0,45,0,45,0,47,0,43,0,52,0,51]
fig = go.Figure([go.Bar(x=x , y = y )])
fig.update_layout(title_text = "Fraud Distributions")
fig.show()

