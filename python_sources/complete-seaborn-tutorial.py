#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Bar Plots

# In[ ]:


tips = sns.load_dataset('tips')
tips.groupby('smoker').tip.agg([min,max])


# ### Height levels are the mean values plotted
# Default estimator is 'mean'

# In[ ]:


sns.barplot(x='day',y='tip',data=tips)


# In[ ]:


sns.barplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.barplot(x='day',y='tip',data=tips, hue='sex')


# In[ ]:


sns.barplot(x='day',y='tip',data=tips, hue='sex', palette='summer_r')


# In[ ]:


sns.barplot(x='day',y='tip',data=tips, hue='smoker')


# In[ ]:


sns.barplot(x='total_bill',y='day',data=tips, palette='spring')


# In[ ]:


sns.barplot(x='day',y='tip',data=tips,palette='spring',order=['Sat','Fri','Sun','Thur'])


# In[ ]:


sns.barplot(x='day',y='total_bill',data=tips)


# ### estimator changed to median

# In[ ]:


from numpy import median
sns.barplot(x='day',y='total_bill',data=tips, estimator=median, palette='spring_r')


# In[ ]:


tips.total_bill[tips.day == 'Fri'].median()


# In[ ]:


from numpy import mean
sns.barplot(x='smoker',y='tip', data=tips, estimator=median)


# In[ ]:


sns.barplot(x='smoker',y='tip',data=tips,estimator=median,hue='sex',palette='coolwarm')


# In[ ]:


sns.barplot(x='smoker',y='tip',data=tips,ci=99)


# In[ ]:


sns.barplot(x='smoker',y='tip',data=tips,ci=68,palette='winter_r', estimator=median)


# In[ ]:


sns.barplot(x='smoker',y='tip',data=tips,ci=34)


# In[ ]:


sns.barplot(x='day',y='total_bill',data=tips,palette='spring',capsize=0.1)


# In[ ]:


sns.barplot(x='day',y='total_bill',data=tips,palette='spring',capsize=0.5)


# In[ ]:


sns.barplot(x='day',y='total_bill',data=tips,palette='husl',hue='sex',capsize=0.1)


# In[ ]:


sns.barplot(x='size',y='tip',data=tips,capsize=0.5,palette='autumn')


# In[ ]:


sns.barplot(x='size',y='tip',data=tips,capsize=0.15,palette='husl')


# In[ ]:


sns.barplot(x='size',y='tip',data=tips,capsize=0.15,color='green')


# In[ ]:


sns.barplot(x='size',y='tip',data=tips,capsize=0.15,color='red', saturation=0.7)


# # Dist Plots

# In[ ]:


num = np.random.randn(150)


# In[ ]:


sns.distplot(num)


# In[ ]:


sns.distplot(num, color='red')


# In[ ]:


label_dist = pd.Series(num, name='variable x')
sns.distplot(label_dist)


# In[ ]:


sns.distplot(label_dist, vertical=True)


# In[ ]:


sns.distplot(label_dist, hist=False)


# In[ ]:


sns.distplot(label_dist,rug=True,hist=False)


# # Box Plot

# In[ ]:


tips = sns.load_dataset('tips')


# In[ ]:


sns.boxplot(x=tips['size'])


# In[ ]:


sns.boxplot(x=tips['total_bill'])


# In[ ]:


sns.boxplot(x='sex',y='total_bill',data=tips)


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='sex',palette='husl')


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker',palette='coolwarm')


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='time')


# In[ ]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='time',order=['Sun','Sat','Fri','Thur'])


# In[ ]:


sns.boxplot(x='sex', y='tip',data=tips, order=['Female','Male'])


# In[ ]:


iris = sns.load_dataset('iris')
iris.head()


# In[ ]:


sns.boxplot(data=iris, palette='coolwarm')


# In[ ]:


sns.boxplot(data=iris,orient='horizontal', palette='husl') # horizontal or h


# In[ ]:


sns.boxplot(data=iris,orient='v', palette='husl') # vertical or v


# In[ ]:


sns.boxplot(x='day',y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.boxplot(x='day',y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day',y='total_bill',data=tips,color='black')


# In[ ]:


sns.boxplot(x='day',y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day',y='total_bill',data=tips,color='0.9')


# In[ ]:


sns.boxplot(x='day',y='total_bill', data=tips, palette='husl')
sns.swarmplot(x='day',y='total_bill',data=tips,color='0.3')


# # Strip Plots

# In[ ]:


tips = sns.load_dataset('tips')
sns.stripplot(x=tips['tip'], color='green')


# In[ ]:


sns.stripplot(x=tips['total_bill'], color='red')


# In[ ]:


sns.stripplot(x='day',y='total_bill', data=tips)


# In[ ]:


sns.stripplot(x='day',y='total_bill', data=tips, jitter=False)


# In[ ]:


sns.stripplot(x='day',y='total_bill', data=tips, jitter=0.2)


# In[ ]:


sns.stripplot(y='day',x='total_bill', data=tips, jitter=1)


# In[ ]:


sns.stripplot(x='day',y='total_bill', data=tips, linewidth=0.75, hue='sex')


# In[ ]:


sns.stripplot(x='day',y='total_bill', data=tips, linewidth=0.75, hue='sex', split=True)


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips,hue='smoker',split=True, linewidth=0.75)


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips,hue='smoker',split=True, linewidth=0.75, palette='winter_r', order=['Sun','Sat','Thur','Fri'])


# In[ ]:


sns.stripplot(x='sex',y='tip', data=tips, order=['Female','Male'], palette='winter_r')


# In[ ]:


sns.stripplot(x='sex',y='tip', data=tips, order=['Female','Male'], palette='winter_r', marker='D', linewidth=0.75)


# In[ ]:


sns.stripplot(x='day',y='total_bill',data=tips,marker='D', size=4, hue='sex', edgecolor='black',split=True, linewidth=0.75, alpha=0.5)


# # PairGrid

# In[ ]:


iris = sns.load_dataset('iris')


# In[ ]:


x = sns.PairGrid(iris)
x = x.map(plt.scatter)


# In[ ]:


x = sns.PairGrid(iris)
x = x.map_diag(plt.hist)


# In[ ]:


x = sns.PairGrid(iris)
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)


# In[ ]:


x = sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)


# In[ ]:


x = sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()


# In[ ]:


x = sns.PairGrid(iris, hue='species', palette='husl') # husl,winter_r,spring_r,summer_r,RdBu
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()


# In[ ]:


x = sns.PairGrid(iris, hue='species')
x = x.map_diag(plt.hist, histtype='step', linewidth=3)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()


# In[ ]:


x = sns.PairGrid(iris, vars=['petal_length','petal_width'])
x = x.map(plt.scatter)


# In[ ]:


x = sns.PairGrid(iris,hue='species', palette='spring_r',vars=['petal_length','petal_width', 'sepal_width'])
x = x.map_diag(plt.hist, edgecolor='black')
x = x.map_offdiag(plt.scatter, edgecolor='black')
x = x.add_legend()


# In[ ]:


print(x)


# In[ ]:


x = sns.PairGrid(iris,hue='species',x_vars=['petal_length','petal_width'], y_vars=['sepal_length','sepal_width'])
x = x.map(plt.scatter)


# In[ ]:


x = sns.PairGrid(iris,hue='species')
x = x.map_diag(plt.hist)
x = x.map_upper(plt.scatter)
x = x.map_lower(sns.kdeplot)
x = x.add_legend()


# In[ ]:


x = sns.PairGrid(iris,hue='species', hue_kws={'marker':['D','s','*']})
x = x.map(plt.scatter,s=10) # s: size
x = x.add_legend()


# In[ ]:


x = sns.PairGrid(iris,hue='species', hue_kws={'marker':['D','s','*']})
x = x.map(plt.scatter, edgecolor='black')
x = x.add_legend()


# # Violin Plot

# In[ ]:


tips = sns.load_dataset('tips')
sns.violinplot(x=tips['tip'])
tips.head()


# In[ ]:


sns.violinplot(x=tips['total_bill'])


# In[ ]:


sns.violinplot(x='size',y='total_bill',data=tips)


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips)


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips, hue='sex')


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips, hue='smoker',palette='husl')


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips, hue='smoker',split=True)


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips, hue='smoker',split=True,order=['Sun','Sat','Thur','Fri'])


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',palette='husl',inner='quartile') # quartiles are visible as lines inside the violinplot


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',palette='husl',inner='quartile', split=True)


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',palette='husl',inner='quartile', split=True,scale='count') # widht of the violin is adjusted according to the number of observations


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',palette='husl',inner='quartile', scale='count')


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',palette='husl',inner='stick', scale='count')


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',scale='count',scale_hue=False) # relative widths checked


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',scale='count')


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',inner='quartile',scale='count',scale_hue=True,split=True,bw=0.7) # bw : bandwidth


# In[ ]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='smoker',inner='quartile',scale='count',scale_hue=True,split=True,bw=0.1)


# # Clustermap 

# In[ ]:


flights = sns.load_dataset('flights')
flights.head()


# In[ ]:


flights = flights.pivot('month','year','passengers')


# In[ ]:


flights


# In[ ]:


sns.heatmap(flights)


# In[ ]:


sns.clustermap(flights) # clusters together similar data


# In[ ]:


sns.clustermap(flights,col_cluster=False) #row_cluster = True


# In[ ]:


sns.heatmap(flights,cmap='coolwarm') # cmap='color-map'


# In[ ]:


sns.clustermap(flights,cmap='coolwarm', linewidth=1)


# In[ ]:


sns.clustermap(flights,cmap='Blues_r', linewidth=1.5)


# In[ ]:


sns.clustermap(flights,cmap='coolwarm', linewidths=1.5)


# In[ ]:


sns.clustermap(flights,cmap='coolwarm', linewidths=1.5, figsize=(8,8))


# Standardize across columns and rows

# In[ ]:


sns.clustermap(flights, standard_scale=1) # standard_scale:{0:rows,1:columns}


# In[ ]:


sns.clustermap(flights, standard_scale=0)


# In[ ]:


# normalising our dataset = z_score  0/1 : rows/columns
sns.clustermap(flights,z_score=0)


# In[ ]:


sns.clustermap(flights,z_score=1)


# # Heatmaps

# In[ ]:


normal = np.random.rand(12,15)
sns.heatmap(normal)


# In[ ]:


sns.heatmap(normal, annot=True) # annot : annotation


# In[ ]:


sns.heatmap(normal, vmin=0,vmax=2)


# In[ ]:


flights = sns.load_dataset('flights')


# In[ ]:


flights.head()


# In[ ]:


flights = flights.pivot('month','year','passengers')


# In[ ]:


sns.heatmap(flights)


# In[ ]:


sns.heatmap(flights, annot=True)


# In[ ]:


sns.heatmap(flights, annot=True, fmt='d', linewidths=0.9)


# In[ ]:


sns.heatmap(flights, annot=True, fmt='d', linewidths=0.9, vmin=200,vmax=650)


# In[ ]:


sns.heatmap(flights, cmap='RdBu', fmt='d', annot=True)


# In[ ]:


sns.heatmap(flights,cmap='coolwarm', center=flights.loc['June',1954], annot=True, fmt='d')


# In[ ]:


sns.heatmap(flights,cmap='coolwarm', center=flights.loc['March',1955], annot=True, fmt='d')


# In[ ]:


sns.heatmap(flights,cmap='coolwarm', center=flights.loc['June',1954], annot=True, fmt='d', cbar=False)


# # FacetGrid

# In[ ]:


tips = sns.load_dataset('tips')


# In[ ]:


x = sns.FacetGrid(tips,row='smoker',col='time')
# Initialising the FacetGrid
x = x.map(plt.hist,'total_bill')


# In[ ]:


x = sns.FacetGrid(tips,row='smoker',col='time')
x = x.map(plt.hist,'total_bill', color='red', bins=15)


# In[ ]:


x = sns.FacetGrid(tips, col='time',row='smoker')
x = x.map(plt.scatter,'total_bill','tip')


# In[ ]:


x = sns.FacetGrid(tips, col='time',row='smoker')
x = x.map(sns.regplot,'total_bill','tip')


# In[ ]:


x = sns.FacetGrid(tips,col='time',row='smoker')
x = x.map(plt.scatter,'total_bill','tip')


# In[ ]:


x = sns.FacetGrid(tips,col='time',hue='smoker')
x = x.map(plt.scatter,'total_bill','tip')
x = x.add_legend()


# In[ ]:


x = sns.FacetGrid(tips,col='time',hue='smoker')
x = x.map(plt.scatter,'total_bill','tip').add_legend()


# In[ ]:


x = sns.FacetGrid(tips,col='day')
x = x.map(sns.boxplot,'total_bill','time')


# In[ ]:


x = sns.FacetGrid(tips,col='day')
x = x.map(sns.boxplot,'time','total_bill')


# In[ ]:


x = sns.FacetGrid(tips,col='day', size=7,aspect=0.2)
x = x.map(sns.boxplot,'time','total_bill')


# In[ ]:


x = sns.FacetGrid(tips,col='day', col_order=['Sat','Sun','Fri','Thur'],size=7,aspect=0.4)
x = x.map(sns.boxplot,'time','total_bill', color='red')


# In[ ]:


x = sns.FacetGrid(tips,col='time',hue='smoker', palette='coolwarm', size=3)
x = x.map(plt.scatter,'total_bill','tip').add_legend()


# # KDE Plots (Kernel Density Estimation)

# In[ ]:


mean = [0,0]
cov = [[0.2,0],[0,3]]


# In[ ]:


x_axis,y_axis = np.random.multivariate_normal(mean,cov,size=40).T


# In[ ]:


sns.kdeplot(x_axis)


# In[ ]:


sns.kdeplot(y_axis)


# In[ ]:


sns.kdeplot(x_axis, shade=True)


# In[ ]:


sns.kdeplot(x_axis, shade=True, color='green')


# In[ ]:


sns.kdeplot(y_axis, shade=True, color='purple')


# In[ ]:


sns.kdeplot(x_axis,y_axis)


# In[ ]:


sns.kdeplot(x_axis,y_axis, shade=True)


# In[ ]:


sns.kdeplot(x_axis,y_axis, cmap='RdBu')


# In[ ]:


sns.kdeplot(x_axis,y_axis, n_levels=20, cmap='RdBu')


# In[ ]:


sns.kdeplot(y_axis,bw=1.5) # bandwidth


# In[ ]:


sns.kdeplot(y_axis,bw=0.5) # bandwidth


# In[ ]:


sns.kdeplot(y_axis,bw=0.1)


# In[ ]:


sns.kdeplot(y_axis, vertical=True)


# In[ ]:


iris = sns.load_dataset('iris')


# In[ ]:


setosa = iris.loc[iris.species == 'setosa']
versicolor = iris.loc[iris.species == 'versicolor']


# In[ ]:


sns.kdeplot(setosa.petal_length, setosa.petal_width)
sns.kdeplot(versicolor.petal_length, versicolor.petal_width)


# In[ ]:


sns.kdeplot(setosa.petal_length, setosa.petal_width, cmap='coolwarm')
sns.kdeplot(versicolor.petal_length, versicolor.petal_width, cmap='RdBu')


# In[ ]:


sns.kdeplot(setosa.petal_length, setosa.petal_width, cmap='coolwarm')
sns.kdeplot(versicolor.petal_length, versicolor.petal_width, cmap='RdBu', shade=True)


# In[ ]:


sns.kdeplot(setosa.petal_width, setosa.petal_length, cmap='coolwarm')
sns.kdeplot(versicolor.petal_width, versicolor.petal_length, cmap='RdBu')


# In[ ]:


sns.kdeplot(setosa.petal_length, setosa.petal_width, cmap='coolwarm', n_levels = 12)
sns.kdeplot(versicolor.petal_length, versicolor.petal_width, cmap='RdBu', n_levels = 14)


# # Joint Plots

# In[ ]:


tips = sns.load_dataset('tips')
tips.head()


# In[ ]:


iris = sns.load_dataset('iris')
iris.head()


# In[ ]:


sns.jointplot(x='total_bill', y='tip', data=tips)


# In[ ]:


sns.jointplot(x='sepal_length',y='sepal_width', data=iris)


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg', color='red')


# In[ ]:


sns.jointplot(x='total_bill',y='tip', data=tips, kind='hex')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')


# In[ ]:


sns.jointplot(x='sepal_length',y='sepal_width',data=iris,kind='kde')


# In[ ]:


# stat_ func


# In[ ]:


from scipy.stats import spearmanr


# In[ ]:


sns.jointplot(x='total_bill',y='size',data=tips,stat_func=spearmanr, color='blue')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips,ratio=4,size=5, color='blue')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips,ratio=6,size=5, color='blue')


# In[ ]:


sns.jointplot(x='total_bill',y='tip',data=tips,ratio=1,size=5, color='blue')


# # Reg Plots

# In[ ]:


tips = sns.load_dataset('tips')
tips.head()


# In[ ]:


sns.regplot(x='total_bill',y='tip',data=tips, color='green')


# In[ ]:


mean = [2,5]
cov = [[1.1,0.4],[2.2,3]]

x_value,y_value = np.random.multivariate_normal(mean,cov,100).T

sns.regplot(x=x_value,y=y_value,color='purple')


# In[ ]:


seriesx_value = pd.Series(x_value, name='Var-X')
seriesy_value = pd.Series(y_value, name='Var-Y')

sns.regplot(x=seriesx_value,y=seriesy_value)


# In[ ]:


sns.regplot(x=seriesx_value,y=seriesy_value, marker='D', color='grey')


# In[ ]:


sns.regplot(x='total_bill',y='tip',data=tips)


# In[ ]:


sns.regplot(x='total_bill',y='tip',data=tips,marker='D', scatter_kws={'color':'blue'},line_kws={'color':'red','linewidth':2})


# In[ ]:


sns.regplot(x=x_value,y=y_value,marker='D', scatter_kws={'color':'blue'},line_kws={'color':'red','linewidth':2})


# In[ ]:


sns.regplot(x=x_value,y=y_value,ci=0)


# In[ ]:


sns.regplot(x=x_value,y=y_value,ci=100) # ci : confidence interval


# In[ ]:


sns.regplot(x='size',y='total_bill',data=tips)


# In[ ]:


sns.regplot(x='size',y='total_bill',data=tips, x_jitter=0.18, scatter_kws={'color':'red'})


# In[ ]:


sns.regplot(x='size',y='total_bill',data=tips, x_estimator=np.mean)


# # Pair Plots

# In[ ]:


iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')


# In[ ]:


sns.pairplot(iris)


# In[ ]:


sns.pairplot(tips)


# In[ ]:


sns.pairplot(iris,hue='species')


# In[ ]:


sns.pairplot(iris,hue='species',palette='Blues_d')


# In[ ]:


# markers


# In[ ]:


sns.pairplot(iris,hue='species',markers=['o','D','s'], palette='husl')


# In[ ]:


# vars


# In[ ]:


sns.pairplot(iris,vars=['sepal_length','sepal_width'])


# In[ ]:


sns.pairplot(tips,vars=['total_bill','tip'])


# In[ ]:


sns.pairplot(iris,size=3,vars=['sepal_length','sepal_width'])


# In[ ]:


# x_vars , y_vars


# In[ ]:


sns.pairplot(iris, x_vars=['petal_length','petal_width'],y_vars=['sepal_length','sepal_width'], palette='husl',hue='species')


# In[ ]:


# diag_kind


# In[ ]:


sns.pairplot(iris, diag_kind='kde',palette='husl',hue='species')


# In[ ]:


sns.pairplot(iris,kind='reg', diag_kind='kde',palette='husl',hue='species')


# In[ ]:


sns.pairplot(tips,kind='reg')

