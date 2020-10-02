#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/insurance/insurance.csv")
data1 = data.copy()


# # General Information about the data
# 
# - The data give information about the profile of the medical insurance beneficiaries and their charged medical costs. The final aim is to predict potential medical costs of a beneficiary.
# 
# - The aim of this study is to make initial EDA and data visualization in order to have a better understanding of the data. 
# 
# - There are 7 features/variables in the data namely:
#     - __age__: age of primary beneficiary
#     - __sex__: insurance contractor gender (female/male)
#     - __bmi__: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
#     - __children__: Number of children covered by health insurance (Number of dependents)
#     - __smoker__: Smoking (whether the beneficiary is a smoker or not)
#     - __region__: the beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
#     - __charges__: Individual medical costs billed by health insurance
# 
# 
# - First of all we can start identifing the types of our variables. 
#     - The age, bmi index, children and charges are ratio variables.
#     - The sex, smoker and region are nominal categorical variables.

# # Main subjects to investigate
# ### Since the final goal of this data is to predict the medical costs, we mainly focus on the potential relations between this variable and the other variables. 
# 
# 1. What is the general outlook, statistics of the data?
# 1. Are there any NaN values?
# 1. What is the frequency of categorical variables? Is there any significant difference in frequencies?
# 1. What is the dispersion of medical costs vis-a-vis other variables? Is there any finding that stands out?
# 1. Would there any significant change in dispersion if we add a third variable into the picture?
# 1. What is the correlation between the quantitative variables? Is there any significant potential relation?
# 1. Is there any important observation as we analyze the quantitative variables and also we add categorical variables to the analysis?

# ## 1. What is the general outlook, statistics of the data?

# ### We have 7 columns and 1338 rows in total.

# In[ ]:


display(data1.head(10))
display(data1.tail(10))


# ### The categorical variables are object; age and number of children are integer; bmi and charges are float.

# In[ ]:


data1.info()


# ### When we look at the descriptive statistics, at first glance the medical costs have a relatively large std. deviation and outliers as well. This can also be seen from the comparison of mean and median which are 13,270 and 9,382 respectively. The min and max values are 1,121 and 63,770 respectively. We will look into these deeper through data visualization tools.
# ### The age of the beneficiaries range from 18 to 64. The mean and median are almost similar as they are 39.
# ### The value of bmi ranges from 15.96 to 53.13. The mean and median are more or less same as they are 30.

# In[ ]:


data1.describe().T


# ### We have the following unique values for the variables of sex, children, region, smoker:
# - sex: male/female
# - children: 0/1/2/3/4/5
# - smoker: yes/no
# - region: southwest/southeast/northwest/northeast

# In[ ]:


display(data1.sex.unique())
display(data1.children.unique())
display(data1.smoker.unique())
display(data1.region.unique())


# ### We have very few observations of beneficiaries who have 4 or 5 children relative to the rest. So, we may want to be cautious about our assessment over the beneficiaries with 4-5 children.

# In[ ]:


display(data1.sex.value_counts())
display(data1.children.value_counts())
display(data1.smoker.value_counts())
display(data1.region.value_counts())


# ## 2. Are there any NaN values?

# ### We have 0 NaN values.

# In[ ]:


data1.isna().sum()


# ## 3. What is the frequency of categorical variables? Is there any significant difference in frequencies?
# - Although the male beneficiaries are a bit higher, gender variable looks balanced.
# - There are relatively more beneficiaries from southeast region. 
# - The highest number of children is 0. The following number is 1. 
# - Most of the beneficiaries are non-smokers. 

# In[ ]:


plt.subplots(1,1)
sns.countplot(data1.sex)
plt.title("gender",color = 'blue',fontsize=15)
plt.show()

plt.subplots(1,1)
sns.countplot(data1.children)
plt.title("number of children",color = 'blue',fontsize=15)
plt.show()

plt.subplots(1,1)
sns.countplot(data1.smoker)
plt.title("smoker",color = 'blue',fontsize=15)
plt.show()

plt.subplots(1,1)
sns.countplot(data1.region)
plt.title("region",color = 'blue',fontsize=15)
plt.show()


# ## 4.What is the dispersion of medical costs vis-a-vis other variables? Is there any observation that stands out?
# 
# ## 5. Would there be any significant change if we add a third variable into the picture?

# ### Men tend to have higher medical costs.

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.sex, y=data1.charges);
plt.xticks(rotation= 45)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()


# ### Beneficaries with 2-4 children tend to have relatively higher medical costs and the std. deviation of medical costs with 4 children is worth noting. It is relatively high.

# In[ ]:


plt.figure(figsize=(10, 5))
sns.barplot(x=data1.children, y=data1.charges);
plt.xticks(rotation= 0)
plt.xlabel('Number of Children', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Number of Children', color = 'blue', fontsize=15)
plt.show()


# ### Smokers have relatively very high medical costs compared to non-smokers.

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.smoker, y=data1.charges);
plt.xticks(rotation= 45)
plt.xlabel('Smoker', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()


# ### Region southeast has a relatively higher medical costs. Northeast comes the second.

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.region, y=data1.charges);
plt.xticks(rotation= 45)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Region', color = 'blue', fontsize=15)
plt.show()


# ### We divide the age into three categories as 18-35/36-50/51-64 to see whether age makes a difference in medical costs. 
# ### We see that beneficiaries have higher costs, as they are older.

# In[ ]:


age_cat = pd.cut(data1.age, [17,35,51,64])
age_cat
data1['age_cat'] = age_cat


# In[ ]:


sns.countplot(data1.age_cat)
plt.title("age_cat",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.age_cat, y=data1.charges);
plt.xticks(rotation= 0)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Age Categories', color = 'blue', fontsize=15)
plt.show()


# ### We divide the bmi into four categories as 15-18.4/18.4-24.9/24.9-40/40-55 to see whether bmi makes a difference in medical costs.
# ### Most of the beneficiaries fall into the category of 25-40 regarding BMI. 
# ### It looks that beneficiaries have higher medical costs, as their bmi increases. But, when we look at the swarmplots below, we realize that we need to look deeper. 

# In[ ]:


bmi_cat = pd.cut(data1.bmi, [15,18.4,24.9,40,55])
data1['bmi_cat'] = bmi_cat
data1.bmi_cat.value_counts()


# In[ ]:


sns.countplot(data1.bmi_cat)
plt.title("bmi_cat",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.bmi_cat, y=data1.charges);
plt.xticks(rotation= 0)
plt.xlabel('BMI', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by BMI Categories', color = 'blue', fontsize=15)
plt.show()


# ### Among beneficiaries who have 0-3 children, men have more; who have 4 children both men and women have somewhat same, but when it comes to 4 chidren women have more medical costs.

# In[ ]:


plt.figure(figsize=(10, 5))
sns.barplot(x=data1.children, y=data1.charges, hue=data1.sex);
plt.xticks(rotation= 0)
plt.xlabel('Number of Children', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Number of Children and Gender', color = 'blue', fontsize=15)
plt.show()


# ### In all regions men tend to have higher medical costs except northwest. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.region, y=data1.charges, hue=data1.sex);
plt.xticks(rotation= 45)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Region and Gender', color = 'blue', fontsize=15)
plt.show()


# ### When we look at the medical costs of beneficiaries across different regions and number of children, we have diverse averages. 
# ### Also, the standard deviation of beneficiaries with 4 children who live in southeast and southwest have higher standard deviation compared to the other two regions. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.region, y=data1.charges, hue=data1.children);
plt.xticks(rotation= 45)
plt.xlabel('Region', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Region and Number of Children', color = 'blue', fontsize=15)
plt.show()


# ### Some might expect that a smoker with more children should have more medical costs. It is true if we compare smokers and non-smokers but when it comes to beneficiaries among smokers, we have a different picture. But we need to be cautious about this since we have very few observations for beneficiaries who have 4-5 children as we mentioned above. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data1.smoker, y=data1.charges, hue=data1.children);
plt.xticks(rotation= 45)
plt.xlabel('Smoker', fontsize=14)
plt.ylabel('Medical Costs', fontsize=14)
plt.title('Medical Costs by Smoking and Number of Children', color = 'blue', fontsize=15)
plt.show()


# ### When we look at the distribution of medical costs across different categorical variables and number of children, first of all we observe people under the same group have different levels of medical costs. This leads us to look at simultenaously more than one factor.
# ### In addition, we see that age and number of children has effect on medical costs. 
# ### But, the most striking factor is the smoking habit of the beneficiary. It has a significant effect on the medical costs as we look at it by itself and together with other variables.
# ### When we look at three variables together, there is a relatively mixed distribution between 10-30K but the split is more obvious in 0-10K and >30K with respect to medical costs. Thus, we may think that other factors may have an effect at 10-30K level.

# In[ ]:


sns.swarmplot(x="age_cat", y="charges", data=data1)
plt.title('Medical Costs by Age Categories', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="bmi_cat", y="charges", data=data1)
plt.title('Medical Costs by BMI', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="sex", y="charges", data=data1)
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="children", y="charges", data=data1)
plt.title('Medical Costs by Number of Children', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="charges", data=data1)
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="region", y="charges", data=data1)
plt.title('Medical Costs by Region', color = 'blue', fontsize=15)
plt.show()


# In[ ]:


sns.swarmplot(x="age_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Age Categories and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="bmi_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by BMI and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="sex", y="charges",hue="smoker", data=data1)
plt.title('Medical Costs by Gender and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="children", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Number of Children and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="region", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Region and Smoking', color = 'blue', fontsize=15)
plt.show()


# ### When we look at the distribution of BMI across different variables, we can observe that beneficiaries who live in the Southeast tend to have higher BMI. We may want to keep in mind this considering the fact that beneficiaries who live in the Southeast have the highest average medical costs across the regions. 

# In[ ]:


sns.swarmplot(x="age_cat", y="bmi", data=data1)
plt.title('BMI by Age Categories', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="sex", y="bmi", data=data1)
plt.title('BMI by Gender', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="children", y="bmi", data=data1)
plt.title('BMI by Number of Children', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="smoker", y="bmi", data=data1)
plt.title('BMI by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.swarmplot(x="region", y="bmi", data=data1)
plt.title('BMI by Region', color = 'blue', fontsize=15)
plt.show()


# ### We see outliers across the categories regarding the medical costs. But, part of outliers disappears as smokers come into the picture. On the other hand, there are still outliers for non-smokers between 20-40K. 

# In[ ]:


sns.boxplot(x="age_cat", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Age Categories', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="bmi_cat", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by BMI', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="sex", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Gender', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="children", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Number of Children', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="smoker", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="region", y="charges", data=data1, palette="PRGn")
plt.title('Medical Costs by Region', color = 'blue', fontsize=15)
plt.show()


# In[ ]:


sns.boxplot(x="age_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Age Categories and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="bmi_cat", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by BMI Categories and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="sex", y="charges",hue="smoker", data=data1)
plt.title('Medical Costs by Gender and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="children", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Number of Children and Smoking', color = 'blue', fontsize=15)
plt.show()

sns.boxplot(x="region", y="charges", hue='smoker', data=data1)
plt.title('Medical Costs by Region and Smoking', color = 'blue', fontsize=15)
plt.show()


# ## 5. What is correlation values between the quantitative variables? Is there any significant potential relation?
# - We convert sex,smoker,region features into quantitative variables by replacing the categories with values of integer values to look into more possible linear relationships. 

# In[ ]:


data3 = data1.copy()
data3['smoker'].replace('yes',1,inplace=True)
data3['smoker'].replace('no',0,inplace=True)
data3['sex'].replace('male',1,inplace=True)
data3['sex'].replace('female',0,inplace=True)
data3['region'].replace('southwest',0,inplace=True)
data3['region'].replace('southeast',1,inplace=True)
data3['region'].replace('northwest',2,inplace=True)
data3['region'].replace('northeast',3,inplace=True)


# In[ ]:


data3.corr()


# ### When we look at the correlations, our observations above are supported with the correlation coefficients. 
# ### We see that there is a strong correlation between smoking and medical costs. 
# ### The following correlation coefficient is 0.3 with age, and 0.2 with bmi. They are less than 0.5, but it may be still worthwhile to note. 

# In[ ]:


f,ax = plt.subplots(figsize=(10, 5))
sns.heatmap(data3.corr(), annot=True, linewidths=0.5, linecolor="red", fmt= '.3f',ax=ax)
plt.show()


# ## 7. Is there any important observation as we analyze the quantitative variables together with categorical variables?

# ### When we look at the graphs, we see that medical costs are not normally distributed and positively skewed to the right. 

# In[ ]:


sns.distplot(data1.charges, bins = 20, kde = True);


# In[ ]:


sns.distplot(data1.bmi, bins = 20, kde = True);


# In[ ]:


sns.pairplot(data1[['bmi','charges','age']],kind='reg')
plt.show()


# ### When we look at the lmplot of bmi and medical charges, we see that they do not fit to the regression line as the dots are not dispersed around the regression line and there is not a linear relationship between them. 
# ### But, when we add smoking as a third factor, the dots tend to disperse around smokers and non-smokers. When we look at blue line in the second graph, medical costs tend to increase as bmi increases for the beneficiarie who smoke.

# In[ ]:


sns.lmplot(x='bmi', y='charges', data=data1)
plt.show()


# In[ ]:


sns.lmplot(x='bmi', y='charges', hue='smoker', data=data1)
plt.show()


# In[ ]:


sns.lineplot(x='bmi',y='charges', data = data1)
plt.show()


# In[ ]:


sns.lineplot(x='bmi',y='charges', hue='smoker', data = data1)
plt.show()


# ### We see through graphs above that the medical costs increase as the beneficiaries gets older. 

# In[ ]:


sns.lmplot(x='age', y='charges', hue='smoker', data=data1)
plt.show()


# In[ ]:


sns.lmplot(x='bmi', y='charges', hue='smoker', col='age_cat', data=data1)
plt.show()


# In[ ]:


sns.lineplot(x='age',y='charges',data = data1)
plt.show()


# ### When we divide beneficiaries into two groups as smokers and non-smokers and recalculate the correlation between bmi and charges, this time we come up with the score of 0.80 and this is a significant value since it is >0.5.

# In[ ]:


data3.groupby('smoker')[['charges','bmi']].corr()


# ### Other visualization methods for the same purposes to see and work on different types of graphs. 

# ### jointplot

# In[ ]:


g = sns.jointplot(data1.bmi, data1.charges, kind="kde", size=7)
plt.show()


# ### Kdeplot with a second dimension

# In[ ]:


(sns
 .FacetGrid(data1,
              hue = "smoker",
              height = 5,
              xlim = (0, 70000))
 .map(sns.kdeplot, "charges", shade= True)
 .add_legend()
);


# In[ ]:


(sns
 .FacetGrid(data1,
              hue = "age_cat",
              height = 5,
              xlim = (0, 70000))
 .map(sns.kdeplot, "charges", shade= True)
 .add_legend()
);


# ### Catplot, kind = point

# In[ ]:


sns.catplot(x = "bmi_cat", y = "charges", hue = "smoker", kind = "point", data = data1);


# In[ ]:


sns.catplot(x = "age_cat", y = "charges", hue = "smoker", kind = "point", data = data1);


# In[ ]:


data_bmi = data1.bmi/data1.bmi.max()
data_charges = data1.charges/data1.charges.max()
data_com = pd.concat([data_bmi,data_charges,data1.age,data.region],axis=1)
data_com.sort_values("age", ascending=True, inplace=True)
new_index = np.arange(len(data_com))
data_com = data_com.set_index(new_index)
data_com


# ### Scatter plot with 4 diemensions

# In[ ]:


sns.scatterplot(x = "bmi", y = "charges", hue= "smoker", size = "age", data = data1);


# ### Pointplot

# In[ ]:


f,ax1 = plt.subplots(figsize =(15,10))
sns.pointplot(x='age',y='charges',data=data_com,color='lime',alpha=0.8)
sns.pointplot(x='age',y='bmi',data=data_com,color='red',alpha=0.8)
plt.text(40,0.15,'bmi ratio',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.10,'charges ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Age',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Charges vs. Bmi',fontsize = 20,color='blue')
plt.grid()


# ## We display similar results with graphs in Plotly.

# In[ ]:


charges_age = data_com.groupby('age')['charges'].mean()
bmi_age = data_com.groupby('age')['bmi'].mean()
ages = data_com.groupby('age')['bmi'].mean().index

trace1 = go.Scatter(
                    x = ages,
                    y = charges_age,
                    mode = "lines",
                    name = "charges",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= data_com.index)
# Creating trace2
trace2 = go.Scatter(
                    x = ages,
                    y = bmi_age,
                    mode = "lines+markers",
                    name = "bmi",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= data_com.index)
data = [trace1, trace2]
layout = dict(title = 'Medical Costs vs BMI by Age',
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# ### Histogram

# In[ ]:


trace1 = go.Histogram(
    x=data1.charges,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='medical costs distribution',
                   xaxis=dict(title='charges'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Boxplot

# In[ ]:


trace0 = go.Box(
    y=data1.charges,
    name = 'medical costs',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

data = [trace0]
iplot(data)


# ### Scatterplot Matrix

# In[ ]:


data4 = data1.loc[:,["age","charges", "bmi"]]
data4["index"] = np.arange(1,len(data4)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data4, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# ### Bubble Chart

# In[ ]:


bmi_size  = [each for each in data1.bmi]
smoker_color = [float(each) for each in data3.smoker]
data = [
    {
        'y': data1.charges,
        'x': data1.age,
        'mode': 'markers',
        'marker': {
            'color': smoker_color,
            'size': bmi_size,
            'showscale': True
        },
        "text" :  data1.index    
    }
]
iplot(data)


# In[ ]:


age_size  = [each for each in data1.age]
smoker_color = [float(each) for each in data3.smoker]
data = [
    {
        'y': data1.charges,
        'x': data1.bmi,
        'mode': 'markers',
        'marker': {
            'color': smoker_color,
            'size': age_size,
            'showscale': True
        },
        "text" :  data1.index    
    }
]
iplot(data)


# In[ ]:


trace1 = go.Scatter3d(
    x=data1.bmi,
    y=data1.age,
    z=data1.charges,
    mode='markers',
    marker=dict(
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="3D ScatterPlot",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## Conclusion
# ### Based on our findings as a result of EDA and data visualization, we conclude that smoking is the most important factor that has effect on the Individual medical costs billed by health insurance among the factors provided in the data.
# ### When we look at the correlation between medical costs and bmi, it is <0.5. But, when we divide beneficiaries into two groups as smokers and nonsmokers, the correlation between medical costs and bmi of smokers goes up to 0.8. Then, bmi needs to be considered in the following stages. 
# ### Age has less than 0.5 correlation coefficient scores but it is still noteworthy to consider it during the following stages of the analysis, as the graphs show that people tend to pay higher costs as they get older. 
# ### There are 3 layers of medical insurance costs; 0-10K, 10-30K and >30K. Smoking (and partly age and bmi) explains the split between 0-10K and >30K but the 10-30K level is rather mixed. There may be other factors at this level besides the variables provided in this data. 
# ### Medical costs are not normally distributed and there are some outliers. These issues need to be handled in the next stage before working on the regression model.

# In[ ]:




