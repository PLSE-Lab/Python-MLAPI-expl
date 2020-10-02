#!/usr/bin/env python
# coding: utf-8

# ## [Benchmark] LIME Using Two Classes

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install line_profiler\n!pip install memory_profiler')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'line_profiler')
get_ipython().run_line_magic('load_ext', 'memory_profiler')


# Initial Memory

# In[ ]:


get_ipython().run_line_magic('memit', '')


# In[ ]:




from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

from sklearn.datasets import fetch_20newsgroups
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)
 
pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


c = make_pipeline(vectorizer, rf)
print(c.predict_proba([newsgroups_test.data[0]]))

explainer = LimeTextExplainer(class_names=class_names)


# In[ ]:


get_ipython().run_line_magic('memit', '')


# In[ ]:


import pandas as pd

benchmark_df = pd.DataFrame(columns=['length', 'wall_time', 'cpu_total_time', 'user_time', 'sys_time'])


# In[ ]:


benchmark_df.head()


# In[ ]:


benchmark_df.drop(benchmark_df.index, inplace=True)


# ## Explanation

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'pip install tqdm')


# In[ ]:


import cProfile, pstats, io
from tqdm.notebook import tqdm


# In[ ]:


len(newsgroups_test.data)


# In[ ]:


get_ipython().run_line_magic('memit', '')


# In[ ]:


for idx in tqdm(range(10)):
    
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())

    ps.total_tt
    benchmark_df.loc[len(benchmark_df)] = [len(newsgroups_test.data[idx]), ps.total_tt, np.nan, np.nan, np.nan]

benchmark_df.head(50)


# In[ ]:


benchmark_df.sort_values(by=['length', 'wall_time'], ascending=True, inplace= True)


# In[ ]:


benchmark_df = benchmark_df.set_index('length')


# In[ ]:


axes = benchmark_df.plot.line(lw=2, colormap='jet', marker='.', markersize=10, title="Input Length Wise BenchMarking")
axes.set_xlabel("Length (Char)")
axes.set_ylabel("Time (Sec)")


# In[ ]:


benchmark_df.drop(benchmark_df.index, inplace=True)
benchmark_df.reset_index(inplace=True)


# In[ ]:





# ## Exhaustive Run

# In[ ]:


get_ipython().run_line_magic('memit', '')


# In[ ]:



for idx in tqdm(range(len(newsgroups_test.data))):
    
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...
    exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())

    
    benchmark_df.loc[len(benchmark_df)] = [len(newsgroups_test.data[idx]), ps.total_tt, np.nan, np.nan, np.nan]
    
    if idx % 10 == 0:
        print("Pickline at {} ..".format(idx))
        benchmark_df.to_pickle("benchmark_observation.pkl")


# In[ ]:


get_ipython().run_line_magic('memit', '')


# In[ ]:


benchmark_df.head(50)
benchmark_df.sort_values(by=['length', 'wall_time'], ascending=True, inplace= True)
benchmark_df = benchmark_df.set_index('length')
axes = benchmark_df.plot.line(lw=2, colormap='jet', marker='.', markersize=10, title="Input Length Wise BenchMarking")
axes.set_xlabel("Length (Char)")
axes.set_ylabel("Time (Sec)")


# In[ ]:


n=50
benchmark_df1 = benchmark_df.reset_index()
axes = benchmark_df1.plot.bar( x='wall_time', y='length', rot=90)
ticks = axes.xaxis.get_ticklocs()
ticklabels = [int(float(l.get_text())) for l in axes.xaxis.get_ticklabels()]
axes.xaxis.set_ticks(ticks[::n])
axes.xaxis.set_ticklabels(ticklabels[::n])


# In[ ]:


ax = benchmark_df1.plot.scatter(y='length', x='wall_time', figsize=(5, 5),c='wall_time', colormap='viridis')


# ## Developing Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:





# In[ ]:


X = benchmark_df1.loc[:, 'length'].values.reshape(-1, 1)  # values converts it into a numpy array
Y = benchmark_df1.loc[:, 'wall_time'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions


# In[ ]:


import matplotlib.pyplot as plt 
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[ ]:


linear_regressor.predict(np.array(2000).reshape((-1, 1)))


# In[ ]:


linear_regressor.predict(np.array(50000).reshape((-1, 1)))


# In[ ]:




