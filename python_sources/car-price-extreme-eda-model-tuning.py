#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


def plot_feature(df,col):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        mean = df.groupby(col)['price'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels,inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    plt.subplot(1,2,2)
    
    if df[col].dtype == 'int64' or col == 'price':
        mean = df.groupby(col)['price'].mean()
        std = df.groupby(col)['price'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)),mean.values-std.values,mean.values + std.values,                         alpha=0.1)
    else:
        sns.boxplot(x = col,y='price',data=df)
    plt.xticks(rotation=45)
    plt.ylabel('Sales')
    plt.show()


# In[ ]:


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt


# # Data

# In[ ]:


df = pd.read_csv('/kaggle/input/auto-data-car-price-prediction-regression/AutoData.csv')


# In[ ]:


df.head()


# In[ ]:


pd.options.display.max_rows=None
pd.options.display.max_columns = None


# In[ ]:


df.head()


# In[ ]:


# back up data
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()


# In[ ]:


df1.head()


# In[ ]:


sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


import missingno as msno
msno.matrix(df1,figsize=(12,5))
plt.show()


# In[ ]:


total = df1.isnull().sum().sort_values(ascending=False)
percent = ((df1.isnull().sum())*100)/df1.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)


# In[ ]:


missing_data


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.info()


# In[ ]:


df1.describe()


# In[ ]:


df1.corr()


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df1.corr(),annot=True,cmap='Blues')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df1.corr(),annot=False,cmap='viridis')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df1.corr(),annot=True,cmap='Reds')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
df1.corr().style.background_gradient(cmap='coolwarm')


# In[ ]:


corr = df1.corr()
cut_off = 0.1
corr = corr[(corr > cut_off) | (corr < -cut_off)]
plt.subplots(figsize=(16,16));
sns.heatmap(corr, cmap="RdBu", square=True, annot=False, cbar_kws={"shrink": .6}, )
plt.title(f"Correlation of features greater than +/-{cut_off}")
plt.tight_layout()
plt.show()


# In[ ]:


#plt.style.use('ggplot')
corr = df1.corr()
plt.figure(figsize=(16,9));
corr["price"].sort_values(ascending=True)[:-1].plot(kind="barh")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()


# In[ ]:


corrMatt = df1.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
plt.show()


# In[ ]:


from scipy.stats import spearmanr
labels = []
values = []
for col in df1:
    if col not in ["price"]:
        labels.append(col)
        values.append(spearmanr(df1[col].values, df1["price"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='g')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# In[ ]:


sns.clustermap(df1.corr(), annot=True, linewidths=.6, fmt= '.2f', figsize=(20, 15))
plt.show()


# In[ ]:


plt.figure(figsize=(30,9))
df1.corr()['price'].sort_values(ascending = False).plot('bar')
plt.show()


# In[ ]:


asset_name = df1.groupby('make')['price'].count().sort_values(ascending=False)[:10]
plt.barh(y=asset_name.index, width=asset_name.values, height=0.5, color='green')
plt.gca().invert_yaxis()
plt.xlabel('cars')


# In[ ]:


msno.matrix(df1.sample(200),figsize=(35, 60), width_ratios=(10, 1), color=(.0, 0.5, 0.5),fontsize=16)
plt.show()


# In[ ]:


msno.dendrogram(df1,fontsize=14)
plt.show()


# In[ ]:


msno.bar(df1.sample(100))
plt.show()


# In[ ]:


df1.plot(subplots=True,figsize=(18,18))
plt.show()


# In[ ]:


for col in df1.drop(['price'],axis=1):
    plot_feature(df1,col)


# In[ ]:


df2.head()


# In[ ]:


df2.drop(['make'],axis=1,inplace=True)


# In[ ]:


cat = []
num = []
for col in df2.columns:
    if df2[col].dtype=='O':
        cat.append(col)
    else:
        num.append(col)


# In[ ]:


cat


# In[ ]:


num


# In[ ]:


plt.style.use('ggplot')
for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.barplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.boxplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.boxenplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.violinplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.barplot(x=col,y='price',data=df2)
    sns.pointplot(x=col,y='price',data=df2,color='Black')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.boxplot(x=col,y='price',data=df2)
    sns.pointplot(x=col,y='price',data=df2,color='Black')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.boxenplot(x=col,y='price',data=df2)
    sns.pointplot(x=col,y='price',data=df2,color='Black')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.boxplot(x=col,y='price',data=df2)
    sns.stripplot(x=col,y='price',data=df2,color='Black')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    sns.boxenplot(x=col,y='price',data=df2)
    sns.stripplot(x=col,y='price',data=df2,color='Black')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


import plotly.express as px
for col in df2[num].drop(['price'],axis=1):
    import plotly.express as px
    type_count = df2[col].value_counts().reset_index()
    total = len(df3)
    type_count['percent'] = round((type_count[col]/total)*100,2)
    print(type_count)
    fig = px.bar(type_count, x='index', y=col,
             hover_data=['index','percent'], color=col,
             labels={'type':'Type Count'}, height=400)
    fig.show()


# In[ ]:


for col in df2[cat]:
    import plotly.express as px
    type_count = df2[col].value_counts().reset_index()
    total = len(df3)
    type_count['percent'] = round((type_count[col]/total)*100,2)
    print(type_count)
    fig = px.bar(type_count, x='index', y=col,
             hover_data=['index','percent'], color=col,
             labels={'type':'Type Count'}, height=400)
    fig.show() 


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.jointplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.jointplot(x=col,y='price',data=df2,kind='reg')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.jointplot(x=col,y='price',data=df2,color='g',kind='hex')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.jointplot(x=col,y='price',data=df2,color='r',kind='kde')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.jointplot(x=col,y='price',data=df2,color='magenta',kind='kde')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.lineplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.factorplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.stripplot(x=col,y='price',data=df2,jitter=True,edgecolor='gray',size=10,palette='winter',orient='v')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(13,7))
    sns.stripplot(x=col,y='price',data=df2,jitter=True,edgecolor='gray',size=10,palette='winter',orient='v')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.residplot(x=col,y='price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.distplot(df2[col],color='r')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    plt.plot(col,'price',data=df2)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    plt.bar(col,'price',data=df2,color='b')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    plt.hist(df2[col],color='g',bins=50)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    df2.plot(col,'price')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


import scipy.stats as st
for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    st.probplot(df2[col],plot=plt)
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(13,7))
    sns.kdeplot(data=df2[num])
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


plt.style.use('dark_background')
for col in df2[num].drop(['price'],axis=1):
    plt.figure(figsize=(12,7))
    plt.plot(df2[col].value_counts(),color='Red')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(12,7))
    plt.plot(df2[col].value_counts(),color='R')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.tight_layout()
    plt.show()


# In[ ]:


plt.style.use('ggplot')

def plot_count(x,fig):
    plt.subplot(23,2,fig)
   
    sns.countplot(df2[x],palette=("magma"))
    plt.subplot(23,2,(fig+1))
    sns.boxplot(x=df2[x], y=df2.price, palette=("magma"))
    
plt.figure(figsize=(15,65))

plot_count('symboling',1)
plot_count('fueltype',3)
plot_count('aspiration',5)
plot_count('doornumber',7)
plot_count('carbody',9)
plot_count('drivewheel',11)
plot_count('enginelocation',13)
plot_count('wheelbase',15)
plot_count('carlength',17)
plot_count('carwidth',19)
plot_count('carheight',21)
plot_count('curbweight',23)
plot_count('enginetype',25)
plot_count('cylindernumber',27)
plot_count('enginesize',29)
plot_count('fuelsystem',31)
plot_count('boreratio',33)
plot_count('stroke',35)
plot_count('compressionratio',37)
plot_count('horsepower',39)
plot_count('peakrpm',41)
plot_count('citympg',43)
plot_count('highwaympg',45)



plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(23,2,fig)
   
    sns.boxenplot(df2[x],y=df2.price,palette=("Blues_d"))
    plt.subplot(23,2,(fig+1))
    sns.lineplot(x=df2[x], y=df2.price)
    
plt.figure(figsize=(15,65))

plot_count('symboling',1)
plot_count('fueltype',3)
plot_count('aspiration',5)
plot_count('doornumber',7)
plot_count('carbody',9)
plot_count('drivewheel',11)
plot_count('enginelocation',13)
plot_count('wheelbase',15)
plot_count('carlength',17)
plot_count('carwidth',19)
plot_count('carheight',21)
plot_count('curbweight',23)
plot_count('enginetype',25)
plot_count('cylindernumber',27)
plot_count('enginesize',29)
plot_count('fuelsystem',31)
plot_count('boreratio',33)
plot_count('stroke',35)
plot_count('compressionratio',37)
plot_count('horsepower',39)
plot_count('peakrpm',41)
plot_count('citympg',43)
plot_count('highwaympg',45)



plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(23,2,fig)
   
    plt.plot(df2[x],color='Blue')
    plt.subplot(23,2,(fig+1))
    sns.lineplot(x=df2[x], y=df2.price,color='Black')
    
plt.figure(figsize=(15,65))

plot_count('symboling',1)
plot_count('fueltype',3)
plot_count('aspiration',5)
plot_count('doornumber',7)
plot_count('carbody',9)
plot_count('drivewheel',11)
plot_count('enginelocation',13)
plot_count('wheelbase',15)
plot_count('carlength',17)
plot_count('carwidth',19)
plot_count('carheight',21)
plot_count('curbweight',23)
plot_count('enginetype',25)
plot_count('cylindernumber',27)
plot_count('enginesize',29)
plot_count('fuelsystem',31)
plot_count('boreratio',33)
plot_count('stroke',35)
plot_count('compressionratio',37)
plot_count('horsepower',39)
plot_count('peakrpm',41)
plot_count('citympg',43)
plot_count('highwaympg',45)



plt.tight_layout()
plt.show()


# In[ ]:


def plot_count(x,fig):
    plt.subplot(23,2,fig)
   
    plt.plot(df2[x],color='Orange')
    plt.subplot(23,2,(fig+1))
    plt.plot(df2[x],'price',data=df2)
    
plt.figure(figsize=(15,65))

plot_count('symboling',1)
plot_count('fueltype',3)
plot_count('aspiration',5)
plot_count('doornumber',7)
plot_count('carbody',9)
plot_count('drivewheel',11)
plot_count('enginelocation',13)
plot_count('wheelbase',15)
plot_count('carlength',17)
plot_count('carwidth',19)
plot_count('carheight',21)
plot_count('curbweight',23)
plot_count('enginetype',25)
plot_count('cylindernumber',27)
plot_count('enginesize',29)
plot_count('fuelsystem',31)
plot_count('boreratio',33)
plot_count('stroke',35)
plot_count('compressionratio',37)
plot_count('horsepower',39)
plot_count('peakrpm',41)
plot_count('citympg',43)
plot_count('highwaympg',45)



plt.tight_layout()
plt.show()


# In[ ]:


for col in df2[num]:
    x = pd.DataFrame()
    x[col] = df3.groupby(col)['price'].mean()
    x.plot(secondary_y='price', figsize=(20, 10),color='Black')


# In[ ]:


plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
sns.countplot(df2.fueltype,palette=("Blues_d"))

plt.subplot(1,2,2)
sns.countplot(df2.enginetype,palette=("PuBuGn"))


# In[ ]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.countplot(df2.doornumber)

plt.subplot(1,2,2)
sns.countplot(df2.carbody)

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.countplot(df2.fuelsystem)

plt.subplot(1,2,2)
sns.countplot(df2.cylindernumber)

plt.show()


# In[ ]:


plt.figure(figsize=(30,10))
sns.countplot(df1.make.value_counts())
plt.xlabel('cars')
plt.ylabel('price')
plt.show()


# In[ ]:


plt.style.use('default')
plt.figure(figsize=(20,8))
dft = pd.DataFrame(df1.groupby(['make'])['price'].mean().sort_values(ascending=False)).plot.bar()

dft = pd.DataFrame(df1.groupby(['aspiration'])['price'].mean().sort_values(ascending=False)).plot.bar()

dft = pd.DataFrame(df1.groupby(['enginelocation'])['price'].mean().sort_values(ascending=False)).plot.bar()
plt.show()


# In[ ]:


for col in df2[num]:
    plt.figure(figsize=(15,8))
    dft = pd.DataFrame(df2.groupby([col])['price'].mean().sort_values(ascending=False)).plot.bar()
    plt.xlabel(col)
    plt.ylabel('price')
    plt.show()


# In[ ]:


for col in df2[cat]:
    plt.figure(figsize=(15,8))
    dft = pd.DataFrame(df2.groupby([col])['price'].mean().sort_values(ascending=False)).plot.bar()
    plt.xlabel(col)
    plt.ylabel('price')
    plt.show()


# In[ ]:


plt.figure(figsize=(15,6))

dft2 = pd.DataFrame(df2.groupby(['carlength','doornumber'])['price'].mean().unstack(fill_value=0))
df2.plot.bar()


# In[ ]:


plt.figure(figsize=(15,6))

dft2 = pd.DataFrame(df2.groupby(['fuelsystem','carlength','drivewheel'])['price'].mean().unstack(fill_value=0))
df2.plot.bar()


# In[ ]:


plt.figure(figsize=(15,6))

sns.scatterplot(x=df3['fuelsystem'],y=df3['price'],hue=df3['drivewheel'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))

sns.scatterplot(x=df3['fueltype'],y=df3['price'],hue=df3['enginesize'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))

sns.scatterplot(x=df3['make'],y=df3['price'],hue=df3['doornumber'])
plt.show()


# In[ ]:


plt.style.use('ggplot')
fig,ax=plt.subplots(1,2,figsize=(18,8))
df3['carbody'].value_counts().plot.pie(explode=[0,0.05,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('carbody')
ax[0].set_ylabel('carbody')
sns.countplot('carbody',data=df3,ax=ax[1],order=df3['carbody'].value_counts().index)
ax[1].set_title('carbody')
plt.show()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(18,8))
df3['doornumber'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('doornumber')
ax[0].set_ylabel('doornumber')
sns.countplot('doornumber',data=df3,ax=ax[1],order=df3['doornumber'].value_counts().index)
ax[1].set_title('doornumber')
plt.show()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(18,8))
df3['aspiration'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('aspiration')
ax[0].set_ylabel('aspiration')
sns.countplot('aspiration',data=df3,ax=ax[1],order=df3['aspiration'].value_counts().index)
ax[1].set_title('aspiration')
plt.show()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(18,8))
df3['drivewheel'].value_counts().plot.pie(explode=[0,0.05,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('drivewheel')
ax[0].set_ylabel('drivewheel')
sns.countplot('drivewheel',data=df3,ax=ax[1],order=df3['drivewheel'].value_counts().index)
ax[1].set_title('drivewheel')
plt.show()


# In[ ]:


# check with all columns

plt.style.use('fivethirtyeight')
for col in df3:
    fig,ax=plt.subplots(1,2,figsize=(18,8))
    df3[col].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title(col)
    ax[0].set_ylabel(col)
    sns.countplot(col,data=df3,ax=ax[1],order=df3[col].value_counts().index)
    ax[1].set_title(col)
    plt.show()


# In[ ]:


for col in df3:
    a = df3[df3.price <50000]
    plt.figure(figsize=(10,6))
    sns.boxplot(y="price",x =col ,data = a)
    plt.show()    


# In[ ]:


for col in df3:
    a = df3[df3.price > 18920]
    plt.figure(figsize=(10,6))
    sns.boxplot(y="price",x =col ,data = a)
    plt.show() 


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'carbody',hue = "fueltype",data = df3)
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'enginelocation',hue = "fueltype",data = df3)
plt.show()


# In[ ]:


cat


# In[ ]:


for col in df2[cat]:
    fig,ax=plt.subplots(1,2,figsize=(15,8))
    clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
    df3[col].value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
    ax[0].set_title("aspiration",size=20)
    ax[0].set_xlabel('aspiration',size=18)


    count=df3[col].value_counts()
    groups=list(df3[col].value_counts().index)[:10]
    counts=list(count[:10])
    counts.append(count.agg(sum)-count[:10].agg('sum'))
    groups.append('Other')
    type_dict=pd.DataFrame({"group":groups,"counts":counts})
    clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
    qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
    plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
    plt.subplots_adjust(wspace =0.5, hspace =0)
    plt.ioff()
    plt.ylabel('')
    plt.show()


# In[ ]:


num


# In[ ]:


for col in df2[num]:
    fig,ax=plt.subplots(1,2,figsize=(15,8))
    clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
    df3[col].value_counts().sort_values(ascending=False).sort_values().plot(kind='barh',color=clr,ax=ax[0])
    ax[0].set_title("wheelbase",size=20)
    ax[0].set_xlabel('wheelbase',size=18)
    count=df3[col].value_counts()
    groups=list(df3[col].value_counts().index)
    counts=list(count)
    counts.append(count.agg(sum)-count.agg('sum'))
    groups.append('Other')
    type_dict=pd.DataFrame({"group":groups,"counts":counts})
    clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
    qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
    plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
    plt.subplots_adjust(wspace =0.5, hspace =0)
    plt.ioff()
    plt.ylabel('')
    plt.show()
    


# In[ ]:


a = df3['carlength'].values[:100]
f = df3['carwidth'].values[:100]

def plot_acc_ttf_data(a,f, title="Car Length and width"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(a, color='r')
    ax1.set_ylabel('carlength', color='r')
    plt.legend(['carlength'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(f, color='b')
    ax2.set_ylabel('carwidth', color='b')
    plt.legend(['carwidth'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(a,f)


# In[ ]:


a = df3['enginesize'].values[:300]
f = df3['peakrpm'].values[:500]

def plot_acc_ttf_data(a,f, title="Engine size and peakrpm"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(a, color='r')
    ax1.set_ylabel('enginesize', color='r')
    plt.legend(['enginesize'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(f, color='b')
    ax2.set_ylabel('peakrpm', color='b')
    plt.legend(['peakrpm'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(a,f)


# In[ ]:


for i in [0.05, 0.1, 0.25, 0.5 ,0.7, 0.9, 0.95]:
    price_df = df2.groupby('wheelbase')['carlength'].quantile(i).reset_index()
    plt.plot(price_df['wheelbase'], price_df['carlength'], label='%.2f quantile' %i)
plt.legend(loc='best')
plt.xlabel('Wheelbase')
plt.ylabel('Price')
plt.show()


# In[ ]:


for i in [0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    price_df = df2.groupby('carlength')['carwidth'].quantile(i).reset_index()
    plt.plot(price_df['carlength'], price_df['carwidth'], label='%.2f quantile' %i)
plt.legend(loc='best')
plt.xlabel('Wheel')
plt.ylabel('Price')
plt.show()


# In[ ]:


for i in [0.05,0.2,0.6,0.9,0.95]:
    price_df = df3.groupby('citympg')['highwaympg'].quantile(i).reset_index()
    plt.plot(price_df['citympg'], price_df['highwaympg'], label='%.2f quantile' %i)
plt.legend(loc='best')
plt.xlabel('highwaympg')
plt.ylabel('Price')
plt.show()


# In[ ]:


for i in [0.05,0.2,0.6,0.9,0.95]:
    price_df = df3.groupby('peakrpm')['price'].quantile(i).reset_index()
    plt.plot(price_df['peakrpm'], price_df['price'], label='%.2f quantile' %i)
plt.legend(loc='best')
plt.xlabel('peakrpm')
plt.ylabel('Price')
plt.show()


# In[ ]:


df3.head()


# In[ ]:


plt.style.use('default')
plt.figure(figsize = (20, 15))
sns.pointplot(df3["carlength"], df3["carwidth"], linestyles="-")
sns.pointplot(df3["wheelbase"], df3["carheight"], color = "green", linestyles="--")
plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize = (20, 15))
sns.pointplot(df3["boreratio"], df3["stroke"], linestyles="-")
sns.pointplot(df3["citympg"], df3["highwaympg"], color = "green", linestyles="--")
plt.grid()
plt.show()


# In[ ]:


count = df3.groupby(['wheelbase','carlength'])['price'].count().reset_index()
fig = px.line(count, x="wheelbase", y="carlength", color='price')
fig.show()


# In[ ]:


count = df3.groupby(['carwidth','carheight'])['price'].count().reset_index()
fig = px.line(count, x="carwidth", y="carheight", color='price')
fig.show()


# In[ ]:


count = df3.groupby(['curbweight','enginesize'])['price'].count().reset_index()
fig = px.line(count, x="curbweight", y="enginesize", color='price')
fig.show()


# In[ ]:


count = df3.groupby(['boreratio','stroke'])['price'].count().reset_index()
fig = px.line(count, x="boreratio", y="stroke", color='price')
fig.show()


# In[ ]:


count = df3.groupby(['compressionratio','horsepower'])['price'].count().reset_index()
fig = px.line(count, x="compressionratio", y="horsepower", color='price')
fig.show()


# In[ ]:


count = df3.groupby(['peakrpm','symboling'])['price'].count().reset_index()
fig = px.line(count, x="peakrpm", y="price", color='symboling')
fig.show()


# In[ ]:


count = df3.groupby(['make'])['price'].count().reset_index()
count.sort_values(by=['price'], inplace = True, ascending = True)
print(count)
fig = px.line(count, x="make", y="price")
fig.show()


# In[ ]:


plt.style.use('ggplot')
for col in df3[num]:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    g = sns.distplot((df3[col] + 1))
    plt.subplot(1,2,2)
    plt.scatter(range(df3.shape[0]), np.sort(np.log(df3[col]+1).values))
    plt.xticks(rotation=45)
    plt.show() 


# In[ ]:


for col in df3[num]:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    g = sns.distplot(df3[col].values, bins=100, kde=False)
    plt.subplot(1,2,2)
    plt.scatter(range(df3.shape[0]), np.sort(np.log(df3[col]+1).values))
    plt.xticks(rotation=45)
    plt.show() 


# In[ ]:


plt.figure(figsize = (12,18))
plt.subplot(421)
g4 = sns.distplot(np.log(df3['boreratio']),
                  hist=False)
g4 = sns.distplot(np.log(df3['stroke']), 
                  hist=False)

plt.subplot(422)
g5 = sns.distplot(np.log(df3['curbweight']), 
                  hist=False)
g5 = sns.distplot(np.log(df3['enginesize']), 
                  hist=False)

plt.show()


# In[ ]:


plt.figure(figsize = (12,18))

plt.subplot(423)
g1 = sns.distplot(np.log(df3['wheelbase']), 
                  hist=False)
g1 = sns.distplot(np.log(df3['carlength']), 
                  hist=False)


plt.subplot(424)
g2 = sns.distplot(np.log(df3['carwidth']),
                  hist=False)
g2 = sns.distplot(np.log(df3['carheight']), 
                  hist=False)


plt.subplot(425)
g3 = sns.distplot(np.log(df3['curbweight']), 
                  hist=False)
g3 = sns.distplot(np.log(df3['enginesize']), 
                  hist=False)




plt.show()


# In[ ]:


df3['carlength'] = np.log(df3['carlength'])

(sns.FacetGrid(df3, hue='carbody', size=5, aspect=2)
  .map(sns.kdeplot, 'carlength', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


df3['wheelbase'] = np.log(df3['wheelbase'])

(sns.FacetGrid(df3, hue='aspiration', size=5, aspect=2)
  .map(sns.kdeplot, 'wheelbase', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


df3['carheight'] = np.log(df3['carheight'])

(sns.FacetGrid(df3, hue='drivewheel', size=5, aspect=2)
  .map(sns.kdeplot, 'carheight', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


df3['price'] = np.log(df3['price'])

(sns.FacetGrid(df3, hue='make', size=5, aspect=2)
  .map(sns.kdeplot, 'price', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


df3['carlength'] = np.log(df3['carlength'])

(sns.FacetGrid(df3, hue='carbody', size=5, aspect=2)
  .map(sns.kdeplot, 'carlength', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


df3['carwidth'] = np.log(df3['carwidth'])

(sns.FacetGrid(df3, hue='doornumber', size=5, aspect=2)
  .map(sns.kdeplot, 'carwidth', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


df3['peakrpm'] = np.log(df3['peakrpm'])

(sns.FacetGrid(df3, hue='fuelsystem', size=5, aspect=2)
  .map(sns.kdeplot, 'peakrpm', shade=True)
 .add_legend()
)
plt.show()


# In[ ]:


for col in df3[num]:
    df3[col] = np.log(df3[col])
    (sns.FacetGrid(df3, hue='fueltype', size=5, aspect=2)
    .map(sns.kdeplot, col, shade=True)
    .add_legend()
    )
    plt.show()    


# In[ ]:


for col in df3[num]:
    df3[col] = np.log(df3[col])
    (sns.FacetGrid(df3, hue='aspiration', size=5, aspect=2)
    .map(sns.kdeplot, col, shade=True)
    .add_legend()
    )
    plt.show()  


# In[ ]:


df4 = df.copy()
for col in df4[num]:
    fig,axes=plt.subplots(nrows=2,ncols=1)
    df4.plot(kind='hist',y=col,bins=50,range=(0,100),density=True,ax=axes[0])
    df4.plot(kind='hist',y=col,bins=100,density=True,ax=axes[1],cumulative=True)
    plt.show()    


# In[ ]:


plt.figure(figsize=(20,25))
i = 0

for item in df4[num].columns:
    i += 1
    plt.subplot(8, 2, i)
    sns.distplot(df4[item], rug=True, rug_kws={"color": "b"},kde=True,
                 kde_kws={"color": "blue", "lw": 5, "label": "KDE"},
                 hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "blue"},label="{0}".format(item))
plt.show()


# In[ ]:


df4.head()


# In[ ]:


import plotly.express as px
fig = px.histogram(df4.dropna(), x='make', y="price", color="fueltype",
                   marginal="box", 
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='carlength', y='price', color='aspiration',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='carbody', y='carlength', color='carlength',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='doornumber', y='carheight', color='enginetype',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='citympg', y='highwaympg', color='price',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='make', y='wheelbase', color='price',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='carwidth', y='curbweight', color='fuelsystem',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


fig = px.histogram(df4, x='enginetype', y='fuelsystem', color='wheelbase',
                   marginal="box",
                   hover_data=df4.columns)
fig.show()


# In[ ]:


sns.pairplot(df4)
plt.show()


# In[ ]:


sns.pairplot(df4,hue='fueltype')
plt.show()


# In[ ]:


plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(df4.price)
plt.subplot(1,2,2)
sns.distplot(df4.price,bins=20)
plt.show()


# In[ ]:


q = df2.price.describe()
print(q)
IQR    = q['75%'] - q['25%']
Upper  = q['75%'] + 1.5 * IQR
Lower  = q['25%'] - 1.5 * IQR
print("the upper and lower outliers are {} and {}".format(Upper,Lower))


# In[ ]:


rows =7

cols = 2

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,35))

col = df4[num].columns

index = 0

for i in range(rows):
    for j in range(cols):
        sns.distplot(df4[col[index]],ax=ax[i][j])
        index = index + 1
        
plt.tight_layout()


# In[ ]:


rows = 7
cols = 2

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,35))

col = df4[num].columns

index = 0

for i in range(rows):
    for j in range(cols):
        sns.regplot(x=df4[col[index]],y=df4['price'],ax=ax[i][j])
        index = index + 1
        
plt.tight_layout()


# In[ ]:


df.head()


# In[ ]:


df.rename(columns={'make':'cars_name'},inplace=True)


# In[ ]:


df['cars_name'].unique()


# In[ ]:


def change_name(a,b):
    df.cars_name.replace(a,b,inplace=True)
    
change_name('maxda rx3','mazda rx3')
change_name('maxda glc deluxe','mazda glc deluxe')
change_name('porcshce panamera','porsche panamera')
change_name('toyouta tercel','toyota tercel')
change_name('vokswagen rabbit','volkswagen rabbit')
change_name('vw dasher','volkswagen dasher')
change_name('vw rabbit','volkswagen rabbit')


# In[ ]:


df['cars_name'].unique()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE


# In[ ]:


df.head()


# In[ ]:


df.drop(['cars_name'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


cat = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']


# In[ ]:


dfn = pd.get_dummies(df,cat,drop_first=True)


# In[ ]:


dfn.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
df_columns = dfn.columns
dfn = scalar.fit_transform(dfn)
dfn = pd.DataFrame(dfn)
dfn.columns = df_columns
dfn.head()


# In[ ]:


X = dfn.drop(['price'],axis=1)


# In[ ]:


y = df[['price']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[ ]:


cross_val_score(LinearRegression(),X_train,y_train).mean()


# In[ ]:


cross_val_score(RandomForestRegressor(n_estimators=100),X_train,y_train).mean()


# In[ ]:


cross_val_score(SVR(),X_train,y_train).mean()


# In[ ]:


cross_val_score(GradientBoostingRegressor(),X_train,y_train).mean()


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


gb = GradientBoostingRegressor()
gb.fit(X_train,y_train)
pred = gb.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


svm = SVR()
svm.fit(X_train,y_train)
pred = svm.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True,cmap='Blues')
plt.show()


# In[ ]:


rf.feature_importances_


# In[ ]:


feature_importance = pd.Series(rf.feature_importances_,index=X_train.columns)
feature_importance.sort_values()
feature_importance.plot(kind='barh',figsize=(15,25))
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(7,8))
features = X_train.columns
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='Black', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


from sklearn.feature_selection import SelectFromModel


# In[ ]:


fe = SelectFromModel(RandomForestRegressor(n_estimators=100,random_state=100))
fe.fit(X_train,y_train)


# In[ ]:


fe.get_support()


# In[ ]:


a = X_train.columns[fe.get_support()]


# In[ ]:


a


# In[ ]:


lr = LinearRegression()
lr.fit(X_train[a],y_train)
pred = lr.predict(X_test[a])
r2_score(y_test,pred)


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train[a],y_train)
pred = rf.predict(X_test[a])
r2_score(y_test,pred)


# In[ ]:


gb = GradientBoostingRegressor()
gb.fit(X_train[a],y_train)
pred = gb.predict(X_test[a])
r2_score(y_test,pred)


# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


def run_Linear(X_train, X_test, y_train, y_test):
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('R2 Score: ', r2_score(y_test, pred))
    print('MSE:',metrics.mean_squared_error(pred,y_test))
    rms = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print('RMSE:',rms)


# In[ ]:


def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestRegressor(n_estimators=100, random_state=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('R2 Score: ', r2_score(y_test, pred))
    print('MSE:',metrics.mean_squared_error(pred,y_test))
    rms = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print('RMSE:',rms)


# In[ ]:


def run_Gradient(X_train, X_test, y_train, y_test):
    clf = GradientBoostingRegressor(n_estimators=100, random_state=100)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('R2 Score: ', r2_score(y_test, pred))
    print('MSE:',metrics.mean_squared_error(pred,y_test))
    rms = np.sqrt(metrics.mean_squared_error(pred, y_test))
    print('RMSE:',rms)


# In[ ]:


X_train.shape


# In[ ]:


for index in range(1,44):
    fe = RFE(LinearRegression(), n_features_to_select = index)
    fe.fit(X_train, y_train)
    X_train_b = fe.transform(X_train)
    X_test_b = fe.transform(X_test)
    print('Selected Feature: ', index)
    run_Linear(X_train_b, X_test_b, y_train, y_test)
    print()


# In[ ]:


for index in range(1,44):
    fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = index)
    fe.fit(X_train, y_train)
    X_train_b = fe.transform(X_train)
    X_test_b = fe.transform(X_test)
    print('Selected Feature: ', index)
    run_randomForest(X_train_b, X_test_b, y_train, y_test)
    print()


# In[ ]:


for index in range(1,44):
    fe = RFE(GradientBoostingRegressor(n_estimators=100, random_state=100), n_features_to_select = index)
    fe.fit(X_train, y_train)
    X_train_b = fe.transform(X_train)
    X_test_b = fe.transform(X_test)
    print('Selected Feature: ', index)
    run_Gradient(X_train_b, X_test_b, y_train, y_test)
    print()


# In[ ]:


fe = RFE(LinearRegression(), n_features_to_select = 26)
fe.fit(X_train, y_train)
X_train_b = fe.transform(X_train)
X_test_b = fe.transform(X_test)
print('Selected Feature: ', 26)
run_randomForest(X_train_b, X_test_b, y_train, y_test)
print()


# In[ ]:


fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = 12)
fe.fit(X_train, y_train)
X_train_b = fe.transform(X_train)
X_test_b = fe.transform(X_test)
print('Selected Feature: ', 12)
run_randomForest(X_train_b, X_test_b, y_train, y_test)
print()


# In[ ]:


fe = RFE(GradientBoostingRegressor(n_estimators=100, random_state=100), n_features_to_select = 4)
fe.fit(X_train, y_train)
X_train_b = fe.transform(X_train)
X_test_b = fe.transform(X_test)
print('Selected Feature: ', 4)
run_Gradient(X_train_b, X_test_b, y_train, y_test)
print()


# In[ ]:





# In[ ]:


fe = RFE(LinearRegression(), n_features_to_select = 26)
fe.fit(X_train, y_train)
X_train_b = fe.transform(X_train)
X_test_b = fe.transform(X_test)
print('Selected Feature: ', 26)
run_randomForest(X_train_b, X_test_b, y_train, y_test)
print()


# In[ ]:


a = X_train.columns[fe.get_support()]


# In[ ]:


a


# In[ ]:


lr = LinearRegression()
lr.fit(X_train[a],y_train)
pred = lr.predict(X_test[a])
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train[a])
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[a].columns
vif['VIF'] = [variance_inflation_factor(X_train[a].values, i) for i in range(X_train[a].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


fe = RFE(RandomForestRegressor(n_estimators=100, random_state=100), n_features_to_select = 12)
fe.fit(X_train, y_train)
X_train_b = fe.transform(X_train)
X_test_b = fe.transform(X_test)
print('Selected Feature: ', 12)
run_randomForest(X_train_b, X_test_b, y_train, y_test)
print()


# In[ ]:


b = X_train.columns[fe.get_support()]


# In[ ]:


b


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train[b],y_train)
pred = rf.predict(X_test[b])
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train[b])
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[b].columns
vif['VIF'] = [variance_inflation_factor(X_train[b].values, i) for i in range(X_train[b].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


fe = RFE(GradientBoostingRegressor(n_estimators=100, random_state=100), n_features_to_select = 4)
fe.fit(X_train, y_train)
X_train_b = fe.transform(X_train)
X_test_b = fe.transform(X_test)
print('Selected Feature: ', 4)
run_Gradient(X_train_b, X_test_b, y_train, y_test)
print()


# In[ ]:


c= X_train.columns[fe.get_support()]


# In[ ]:


c


# In[ ]:


gb = GradientBoostingRegressor()
gb.fit(X_train[c],y_train)
pred = gb.predict(X_test[c])
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train[c])
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[c].columns
vif['VIF'] = [variance_inflation_factor(X_train[c].values, i) for i in range(X_train[c].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:





# In[ ]:


lr = LinearRegression()
lr.fit(X_train[a],y_train)
pred = lr.predict(X_test[a])
r2_score(y_test,pred)


# In[ ]:


plt.style.use('default')
title = "(Linear Regression)"
cv= 5
plot_learning_curve(lr, title, X_train[a], 
                    y_train,cv=cv ,ylim=(0.2,1.0));


# In[ ]:


plt.scatter(y_test,pred)
plt.show()


# In[ ]:


sns.distplot(y_test-pred)
plt.show()


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train[b],y_train)
pred = rf.predict(X_test[b])
r2_score(y_test,pred)


# In[ ]:


title = "(Random Forest)"
cv= 5
plot_learning_curve(rf, title, X_train[b], 
                    y_train,cv=cv ,ylim=(0.2,1.0));


# In[ ]:


plt.scatter(y_test,pred)
plt.show()


# In[ ]:


gb = GradientBoostingRegressor()
gb.fit(X_train[c],y_train)
pred = gb.predict(X_test[c])
r2_score(y_test,pred)


# In[ ]:


title = "(Gradient Boosting)"
cv= 5
plot_learning_curve(gb, title, X_train[c], 
                    y_train,cv=cv ,ylim=(0.2,1.0));


# In[ ]:


plt.scatter(y_test,pred)
plt.show()


# In[ ]:


lr = LinearRegression()
rfe = RFE(lr, 14)             
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)          
print(rfe.ranking_) 


# In[ ]:


b=X_train.columns[rfe.get_support()]
b


# In[ ]:


lr = LinearRegression()
lr.fit(X_train[b],y_train)
pred = lr.predict(X_test[b])
r2_score(y_test,pred)


# In[ ]:


X_train.shape


# In[ ]:


folds = KFold(n_splits = 5, shuffle = True, random_state = 100)


hyper_params = [{'n_features_to_select': list(range(1, 43))}]

lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)             

model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

model_cv.fit(X_train, y_train) 


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[ ]:


plt.figure(figsize=(16,6))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop('symboling',axis=1,inplace=True)
X_test.drop('symboling',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop('wheelbase',axis=1,inplace=True)
X_test.drop('wheelbase',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('carlength',axis=1,inplace=True)
X_test.drop('carlength',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('carheight',axis=1,inplace=True)
X_test.drop('carheight',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('curbweight',axis=1,inplace=True)
X_test.drop('curbweight',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('compressionratio',axis=1,inplace=True)
X_test.drop('compressionratio',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('horsepower',axis=1,inplace=True)
X_test.drop('horsepower',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('citympg',axis=1,inplace=True)
X_test.drop('citympg',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('highwaympg',axis=1,inplace=True)
X_test.drop('highwaympg',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('fueltype_gas',axis=1,inplace=True)
X_test.drop('fueltype_gas',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('doornumber_two',axis=1,inplace=True)
X_test.drop('doornumber_two',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('carbody_hardtop',axis=1,inplace=True)
X_test.drop('carbody_hardtop',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('carbody_hatchback',axis=1,inplace=True)
X_test.drop('carbody_hatchback',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop(['carbody_sedan','carbody_wagon','drivewheel_fwd','drivewheel_rwd','enginetype_dohcv'],axis=1,inplace=True)
X_test.drop(['carbody_sedan','carbody_wagon','drivewheel_fwd','drivewheel_rwd','enginetype_dohcv'],axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop(['enginelocation_rear','enginetype_l','enginetype_ohcf','cylindernumber_five','cylindernumber_four','cylindernumber_five'],axis=1,inplace=True)
X_test.drop(['enginelocation_rear','enginetype_l','enginetype_ohcf','cylindernumber_five','cylindernumber_four','cylindernumber_five'],axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop(['cylindernumber_six','fuelsystem_2bbl','fuelsystem_4bbl','fuelsystem_idi','fuelsystem_mfi','fuelsystem_mpfi','fuelsystem_spdi'],axis=1,inplace=True)
X_test.drop(['cylindernumber_six','fuelsystem_2bbl','fuelsystem_4bbl','fuelsystem_idi','fuelsystem_mfi','fuelsystem_mpfi','fuelsystem_spdi'],axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


X_train.drop('fuelsystem_spfi',axis=1,inplace=True)
X_test.drop('fuelsystem_spfi',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop('enginetype_rotor',axis=1,inplace=True)
X_test.drop('enginetype_rotor',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop('stroke',axis=1,inplace=True)
X_test.drop('stroke',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop('carwidth',axis=1,inplace=True)
X_test.drop('carwidth',axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop(['boreratio'],axis=1,inplace=True)
X_test.drop(['boreratio'],axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop(['enginetype_ohc'],axis=1,inplace=True)
X_test.drop(['enginetype_ohc'],axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


X_train.drop(['cylindernumber_three'],axis=1,inplace=True)
X_test.drop(['cylindernumber_three'],axis=1,inplace=True)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()


# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


gb = GradientBoostingRegressor()
gb.fit(X_train,y_train)
pred = gb.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


svm = SVR()
svm.fit(X_train,y_train)
pred = svm.predict(X_test)
r2_score(y_test,pred)


# In[ ]:





# In[ ]:


plt.style.use('default')
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)
title = "Learning Curve (Linear Regression)"
cv = 5
plot_learning_curve(lr, title, X_train, 
                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1);


# In[ ]:


plt.figure(figsize=(10,5))
plt.style.use('default')
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)
plt.scatter(y_test,pred)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(y_test-pred)
plt.show()


# In[ ]:


title = "Learning Curve (Random Forest)"
cv = 5
plot_learning_curve(rf, title, X_train, 
                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1);


# In[ ]:


plt.figure(figsize=(10,5))
plt.style.use('default')
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)
plt.scatter(y_test,pred)
plt.show()


# In[ ]:


title = "Learning Curve (Gradient Boosting)"
cv = 5
plot_learning_curve(gb, title, X_train, 
                    y_train, ylim=(0.1, 1.0), cv=cv, n_jobs=-1);


# In[ ]:


plt.figure(figsize=(10,5))
plt.style.use('default')
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)
plt.scatter(y_test,pred)
plt.show()


# In[ ]:





# In[ ]:


rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
r2_score(y_test,pred)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5
parameters = {'max_depth': [1,2,3,4,5]}
dtree = DecisionTreeRegressor(criterion='mse',
                               random_state = 100)
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds,
                   return_train_score=True)
tree.fit(X_train, y_train)
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[ ]:


plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


n_folds = 5
parameters = {'min_samples_leaf': [1,2,3,4,5]}
dtree = DecisionTreeRegressor(criterion='mse',
                               random_state = 100)
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds,
                   return_train_score=True)
tree.fit(X_train, y_train)
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[ ]:


plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


n_folds = 5
parameters = {'min_samples_split': [2,3,4,5]}
dtree = DecisionTreeRegressor(criterion='mse',
                               random_state = 100)
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds,
                   return_train_score=True)
tree.fit(X_train, y_train)
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[ ]:


plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[ ]:


results = []
n_estimator_options = [10,20,30,40,50,60,70,80,90,100,110,120]
for trees in n_estimator_options:
    model = RandomForestRegressor(trees,n_jobs=-1,random_state=100)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print (trees,"trees")
    roc = r2_score(y_test,pred)
    print("accuary",roc)
    results.append(roc)
    print ("")
        
pd.Series(results,n_estimator_options).plot() 
plt.show()


# In[ ]:


result = []
max_features_options = ["auto",None,"sqrt","log2",0.9,0.2]
for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=50,n_jobs=-1,random_state=100,max_features=max_features)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print (max_features,"option")
    roc = r2_score(y_test,pred)
    print("accuary",roc)
    result.append(roc)
    print ("")
pd.Series(result,max_features_options).plot() 
plt.show()


# In[ ]:


result = []
min_samples_split_options = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for min_split in min_samples_split_options:
    model = RandomForestRegressor(n_estimators=100,
                                  n_jobs=-1,
                                  random_state=100,
                                  max_features=0.9,
                                  min_samples_split=min_split)
    model.fit(X_train,y_train)
    print(min_split,"min_sample_split")
    pred = model.predict(X_test)
    roc = r2_score(y_test,pred)
    print("min_sample_split",roc)
    result.append(roc)
    print("")
pd.Series(result,min_samples_split_options).plot() 
plt.show()


# In[ ]:


result = []
min_samples_leaf_options = [1,2,3,4]

for min_sample in min_samples_leaf_options:
    model = RandomForestRegressor(n_estimators=100,
                                  n_jobs=-1,
                                  random_state=100,
                                  max_features=0.9,
                                  min_samples_leaf=min_sample)
    model.fit(X_train,y_train)
    print(min_sample,"min_sample")
    pred = model.predict(X_test)
    roc = r2_score(y_test,pred)
    print("min_sample",roc)
    result.append(roc)
    print("")
pd.Series(result,min_samples_leaf_options).plot() 
plt.show()


# In[ ]:


rf = RandomForestRegressor()

scores = cross_val_score(rf, X_train, y_train, cv=5)

print(scores.mean())

rf.fit(X_train, y_train)
train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)

print('Before Parameter Tuning')
print('The Training R2 Score is', r2_score(y_train,train_predictions))
print('The Testing R2 Score is', r2_score(y_test,test_predictions))

parameters = {   
              'max_depth':[1,2,3,4,5], 
              'min_samples_leaf':[1,2,3,4,5], 
              'min_samples_split':[2,3,4,5],
              'n_estimators': [50,100,150],
              'max_features' : ['auto'],
              
              
    
              }


def generate_model_from_search(grid_or_random,model, parameters,X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(model, parameters)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(model, parameters,cv=5,n_jobs=-1,verbose=5,random_state=100)
    fit_obj = search_obj.fit(X, y)
    best_model = fit_obj.best_estimator_
    return best_model

best_model_random = generate_model_from_search("Random", 
                                           rf, 
                                           parameters, 
                                           X_train, 
                                           y_train,
                                            )

scores = cross_val_score(best_model_random, X_train, y_train, cv=5,n_jobs=-1, verbose=5)
print(scores.mean())

best_model_random.fit(X_train, y_train)
best_train_predictions = best_model_random.predict(X_train)
best_test_predictions = best_model_random.predict(X_test)
print('After Parameter Tuning')
print('The training R2 Score is', r2_score(y_train,best_train_predictions))
print('The testing R2 Score is', r2_score(y_test,best_test_predictions))


# In[ ]:


rf = RandomForestRegressor()

scores = cross_val_score(rf, X_train, y_train, cv=5)

print(scores.mean())

rf.fit(X_train, y_train)
train_predictions = rf.predict(X_train)
test_predictions = rf.predict(X_test)

print('Before Parameter Tuning')
print('The Training R2 Score is', r2_score(y_train,train_predictions))
print('The Testing R2 Score is', r2_score(y_test,test_predictions))

parameters = {   
              'max_depth':[5], 
              'min_samples_leaf':[1], 
              'min_samples_split':[2],
              'n_estimators': [150],
              'max_features' : ['auto'],
              
              
    
              }


def generate_model_from_search(grid_or_random,model, parameters,X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(model, parameters)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(model, parameters,cv=5,n_jobs=-1,verbose=5,random_state=100)
    fit_obj = search_obj.fit(X, y)
    best_model = fit_obj.best_estimator_
    return best_model

best_model_random = generate_model_from_search("Random", 
                                           rf, 
                                           parameters, 
                                           X_train, 
                                           y_train,
                                            )

scores = cross_val_score(best_model_random, X_train, y_train, cv=5,n_jobs=-1, verbose=5)
print(scores.mean())

best_model_random.fit(X_train, y_train)
best_train_predictions = best_model_random.predict(X_train)
best_test_predictions = best_model_random.predict(X_test)
print('After Parameter Tuning')
print('The training R2 Score is', r2_score(y_train,best_train_predictions))
print('The testing R2 Score is', r2_score(y_test,best_test_predictions))


# In[ ]:





# In[ ]:


gb = GradientBoostingRegressor()

scores = cross_val_score(gb, X_train, y_train, cv=5)

print(scores.mean())

gb.fit(X_train, y_train)
train_predictions = gb.predict(X_train)
test_predictions = gb.predict(X_test)

print('Before Parameter Tuning')
print('The Training R2 Score is', r2_score(y_train,train_predictions))
print('The Testing R2 Score is', r2_score(y_test,test_predictions))

parameters = {   
              'max_depth':[5], 
              'min_samples_leaf':[1], 
              'min_samples_split':[2],
              'n_estimators': [150],
              'max_features' : ['auto'],
            }



def generate_model_from_search(grid_or_random,model, parameters,X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(model, parameters)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(model, parameters,cv=5,n_jobs=-1,verbose=5,random_state=100)
    fit_obj = search_obj.fit(X, y)
    best_model = fit_obj.best_estimator_
    return best_model

best_model_random = generate_model_from_search("Random", 
                                           gb, 
                                           parameters, 
                                           X_train, 
                                           y_train,
                                            )

scores = cross_val_score(best_model_random, X_train, y_train, cv=5,n_jobs=-1, verbose=5)
print(scores.mean())

best_model_random.fit(X_train, y_train)
best_train_predictions = best_model_random.predict(X_train)
best_test_predictions = best_model_random.predict(X_test)
print('After Parameter Tuning')
print('The training R2 Score is', r2_score(y_train,best_train_predictions))
print('The testing R2 Score is', r2_score(y_test,best_test_predictions))


# In[ ]:




