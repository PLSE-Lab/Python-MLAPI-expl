#!/usr/bin/env python
# coding: utf-8

# In this kernel I want to demonstrate the power of feature selection. Here I try to apply interactive graphics with plotly library. I recommend switching to it, as static graphics become outdated. Necessary imports:

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import random
import scipy.stats as stt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('pylab', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')
data.head()


# Here we can see, that our dataset is very imbalanced. It's mean, that our model can always predict first class and it will be right in 99,8% cases.

# In[ ]:


first_class = '{:.4}% - class 0'.format(pd.value_counts(data.Class)[0]/data.shape[0])
second_class = '{:.2}% - class 1'.format(pd.value_counts(data.Class)[1]/data.shape[0])
import plotly.graph_objects as go
fig = go.Figure(
    data=[go.Bar(
        y=data['Class'].value_counts().to_dense().keys(),
        x=data['Class'].value_counts(),
        orientation='h',
        text=[second_class,first_class]
    )],
    layout_title_text="Countplot for classifications"
)
fig.show()


# In[ ]:


data[data.Class==0].sample(n=492)
result = pd.concat([data[data.Class==0].sample(n=492), data[data.Class==1]], axis=0)


# In[ ]:


plt.figure(figsize=(15,15))
result = data
# sns.heatmap(result.corr(), fmt='.1f');
fig = go.Figure(
    data = [go.Heatmap( z=result.corr(), colorscale='Viridis',
                      x=list(result.columns),
                      y=list(result.columns)
                      )],
    layout_title_text="Correlation plot",
    )
fig.show()


# In[ ]:


df = result[['V1','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','Class']]
fig = go.Figure(
    data = [go.Heatmap( z=df.corr(), colorscale='Viridis',
                      x=list(df.columns),
                      y=list(df.columns)
                      )],
    layout_title_text="Countplot for classifications",
    )
fig.show()


# In[ ]:


# sns.boxplot(y=data.Amount, x=data.Class);
# 1,3,5,6,7, 9, 10, 12, 14, 16,17,18
sns.heatmap(result[['V1','V3','V5','V6','V7','V9','V10','V12','V14','V16','V17','V18']].corr(), fmt='.1f');


# In[ ]:


import plotly.express as px
tips = px.data.tips()
fig = px.histogram(tips, x="total_bill", y="tip", color="sex", marginal="rug",
                   hover_data=tips.columns)
fig.show()


# In[ ]:


data.Amount.hist(bins=100)


# In[ ]:


data[data.Class==0].shape


# In[ ]:


import plotly.figure_factory as ff
import numpy as np

# Group data together
hist_data = [data.Amount]

group_labels = ['Amount']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.show()


# In[ ]:





# In[ ]:


import plotly.express as px
data = 
fig = px.histogram(data, x="Amount", y="Time", color="Class",
                   marginal="box", # or violin, rug
                   hover_data=data.columns)
fig.show()


# In[ ]:


data.head()


# In[ ]:





# Only with visual analysis we can identify 6 features, which can easy separate 2 main classes.

# In[ ]:


f, axes = plt.subplots(1, 3,figsize=(17,5))
sns.set(font_scale=1)
sns.boxplot(y=data['V4'],x=data.Class,ax=axes[0]);
sns.boxplot(y=data['V10'], x=data.Class,ax=axes[1]);
sns.boxplot(y=data['V11'], x=data.Class,ax=axes[2]);

f, axes = plt.subplots(1, 3,figsize=(17,5))
sns.set(font_scale=1)
sns.boxplot(y=data['V12'],x=data.Class,ax=axes[0]);
sns.boxplot(y=data['V14'], x=data.Class,ax=axes[1]);
sns.boxplot(y=data['V16'], x=data.Class,ax=axes[2]);


# Now apply cluster analysis. First interesting method - T-SNE

# In[ ]:


from sklearn.manifold import TSNE
res_ = pd.concat([data[data.Class==0].sample(n=492), data[data.Class==1]], axis=0)
tsn = TSNE()
res_tsne = tsn.fit_transform(res_)
sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=res_.Class, palette='viridis');
plt.show()


# This method doesn't show any interesting. Picture has cluster structure, but not with feature - Class. Try this with our selected features: V4, 10, 11, 12, 14, 16.

# In[ ]:


data[data.Class==0].sample(n=492)
res = pd.concat([data[data.Class==0].sample(n=492), data[data.Class==1]], axis=0)
res = res[['V4','V10','V11','V12','V14','V16','Class']]


# In[ ]:


from sklearn.manifold import TSNE
tsn = TSNE()
res_tsne = tsn.fit_transform(res)
plt.figure(figsize=(7,7))
sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=res.Class, palette='copper');


# This result seems very sapid. Here doubt creeps in that the data are specially selected, because real projects rarely have so well separable data.

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
distortions = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++',
               n_init=20,max_iter=300, random_state=0)
    km.fit(res)
    distortions.append(km.inertia_)
plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[ ]:


import plotly.express as px

fig = go.Figure()
fig.add_trace(go.Scatter(x=[i for i in range(1,11)], y=distortions,
                    mode='lines',
                    name='lines'))
fig.update_layout(title='Distorsion plot',
                   xaxis_title='Number of clusters',
                   yaxis_title='Distortion')
fig.show()


# In[ ]:


[i for i in range(1,11)]


# In[ ]:





# 

# In[ ]:




