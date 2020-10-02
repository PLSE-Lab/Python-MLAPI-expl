#!/usr/bin/env python
# coding: utf-8

# #### Importing Required Package

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ### Task 1: To read the fundamentals.csv file into a data frame

# In[ ]:


df=pd.read_csv("/kaggle/input/fundamentals.csv")


# ### Task 2: To remove the special charecters from columnnames.

# In[ ]:



oldcolumns=df.columns
newcolumns=[]
for i in oldcolumns:
    i=i.replace(" ","_")
    i=i.replace("/","_")
    i=i.replace("'","_")
    i=i.replace("-","_")
    i=i.replace(".","_")
    i=i.replace(",","_")
    i=i.replace("&","_")
    i=i.replace(":","_")
    i=re.sub("_+","_",i)
    newcolumns.append(i)
    #Check and print if any other charecter apart from alphabets,digits and _ are present
    check=re.search("^[a-zA-Z_0-9]+$",i)
    if(not check):
        print(i)
df.columns=newcolumns


# In[ ]:


df.columns


# ### Task 3: Group data by Ticker Symbols and take a mean of all numeric variables

# In[ ]:


grp_tickers=df.groupby("Ticker_Symbol")
df_tickers=grp_tickers.mean()


# In[ ]:


df_tickers.columns


# #### Displaying Dist plots of various important parameters

# #### Task 4: Using ungrouped and grouped data, perform visual analysis of data using seaborn and plotly-express

# In[ ]:


parameters=['Capital_Expenditures','Goodwill','Research_and_Development',
                   'Total_Assets','Earnings_Per_Share','Cash_Ratio', 'Current_Ratio', 'Quick_Ratio']
fig = plt.figure(figsize = (15,15))
for i in range(len(parameters)):
    plt.subplot(3,3,i+1)
    sns.distplot(df_tickers[parameters[i]])


# In[ ]:


plt.figure(figsize=(100,20))
plot=sns.barplot(x=df_tickers.index,y='Earnings_Per_Share',data=df_tickers)
plot.set_xticklabels(plot.get_xticklabels(),rotation=90,fontsize=15)
plot.set_title("Bar Plot of EPS across tickers",fontsize=30)
plot.set_xlabel('Ticker Symbol',fontsize=20)
plot.set_ylabel('Earings Per Share',fontsize=20)


# ### Creating Categorical variables from numerical values

# #### Profitability

# In[ ]:


lowesteps=min(df_tickers.Earnings_Per_Share)
highesteps=max(df_tickers.Earnings_Per_Share)
df_tickers['Profitability']=pd.cut(
                       df_tickers['Earnings_Per_Share'],
                       bins = [lowesteps-1,0,2,4,highesteps],
                        labels= ["Loss", "Average", "Good","High"]
                      )
df_tickers['Profitability'].value_counts()
df_tickers['Earnings_Per_Share'].describe()


# #### Company Size

# In[ ]:


lowestsize=min(df_tickers.Total_Assets)
highestsize=max(df_tickers.Total_Assets)
df_tickers['Company_Size']=pd.qcut(
                       df_tickers['Total_Assets'],
                       #bins = [82433501,(lowestsize+highestsize)/100,(lowestsize+highestsize)/10,highestsize],     # Else devise your bins: [0,20,60,110]
                       q=3,
                       labels= ["s", "m", "l"]
                      )
df_tickers['Company_Size'].value_counts()
#df_tickers['Total_Assets']
df_tickers[df_tickers['Company_Size']=='l'].Total_Assets.describe()


# #### Liquidity

# In[ ]:


lowestliq=min(df_tickers.Cash_Ratio)
highestliq=max(df_tickers.Cash_Ratio)
df_tickers['Liquidity']=pd.cut(
                       df_tickers['Cash_Ratio'],
                       bins = [0,(lowestliq+highestliq)/8,(lowestliq+highestliq)/4,highestliq],           # Else devise your bins: [0,20,60,110]
                       labels= ["Low", "Medium", "High"]
                      )
df_tickers['Liquidity'].value_counts()
lowestliq


# In[ ]:



parameters2=['Company_Size','Liquidity']
fig = plt.figure(figsize = (15,8))
for i in range(len(parameters2)):
    plt.subplot(1,2,i+1)
    sns.boxplot(x=parameters2[i],y="Earnings_Per_Share",data=df_tickers)


# In[ ]:


sns.boxplot(x="Profitability",y="Cash_Ratio",data=df_tickers, notch=True)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x="Profitability",y="Current_Ratio",data=df_tickers,notch=True)


# In[ ]:


plt.figure(figsize=(10,8))
ax=sns.barplot(x="Profitability",y="Research_and_Development",hue='Company_Size',data=df_tickers)


# ### Task 5: Dataset has a number of NaNs. Either remove rows that have NaNs or remove columns that have NaNs.So that dataset has no nulls.

# In[ ]:


df_tickers_copy=df_tickers.copy()
for i in df_tickers:
    ncount=df_tickers[i].isna().sum()
    if(ncount>70):
        print(i,ncount)
        df_tickers.drop(i,axis='columns',inplace=True)
        


# In[ ]:


df_tickers.drop(["For_Year","Unnamed_0"],axis='columns',inplace=True)
df_tickers=df_tickers.dropna(axis='rows')
print(df_tickers.columns[df_tickers.isna().any()])
print(df_tickers.shape)


# In[ ]:


print(df_tickers.columns)


#  ### Task 6 Normalise the data using sklearn's StandardScaler()

# In[ ]:


scaler = StandardScaler()


# In[ ]:


#tdata temporarily holds numeric only columns from df_tickers
tdata = df_tickers.select_dtypes(include = ['float64', 'int64']).copy()
#ndata is transformed numpy array of the numeric data
ndata=scaler.fit_transform(tdata)
#xdata is a dataframe of the numeric data from numpy array
Xdata=pd.DataFrame(ndata,columns=tdata.columns)
Xdata.head()
#ydata is a dataframe of the categorical variables
ydata=df_tickers[["Profitability","Company_Size"]]
#pdata combines xdata and ydata
pdata=Xdata.copy()
pdata['Profitability']=df_tickers.reset_index()['Profitability']
pdata['Company_Size']=df_tickers.reset_index()['Company_Size']
pdata


# In[ ]:


#profitability parallel plot
fig2 = plt.figure(figsize=(20,15))

ax = pd.plotting.parallel_coordinates(pdata.drop(["Company_Size"],axis="columns"),
                                 'Profitability',
                                  colormap= plt.cm.spring
                                  )
plt.xticks(rotation=90)


# ### Tas 7: Perform clustering and check Silhoutte score.
#      

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "none"


# In[ ]:


maxscore=0
optimumkval=0
wss=[]
silscore=[0]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    y_means=kmeans.fit(Xdata)
    wss.append(kmeans.inertia_)
    if(i>1):
        silscore.append(silhouette_score(Xdata, kmeans.labels_))
plt.plot(range(1,11),wss,marker='*')
print("silhouette scores", silscore)


# In[ ]:


print(y_means.labels_)


# In[ ]:


#From above plot and silhoeutte score, 3 seems to be an optimum k value
plt.figure(figsize=(15,15))
kmeans=KMeans(n_clusters=3)
y_means=kmeans.fit(Xdata)
colorseries=pd.Series(y_means.labels_).map({0:"green",1:"blue",2:"red",3:"yellow",4:"grey"})
for i in range(len(ndata)):
    plt.plot(ndata[i,0],ndata[i,1],marker='o',color=colorseries[i])


# In[ ]:


from yellowbrick.cluster import SilhouetteVisualizer
sv = SilhouetteVisualizer(kmeans)
sv.fit(Xdata)        
sv.show()          


# ### Task 8: Perform TSNE visualization (of the dataset) and color points with the clusters discovered above.

# In[ ]:


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(Xdata)
df1 = pd.DataFrame(X_embedded, columns=['X','Y'])
colorlist = ["#FF1111", "#11FF11", "#1111FF"]
sns.relplot(x = "X",
            y = "Y",
            hue = y_means.labels_,
            data = df1,
            palette=sns.color_palette(colorlist)
            )


# In[ ]:




