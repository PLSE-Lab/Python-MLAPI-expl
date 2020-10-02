#!/usr/bin/env python
# coding: utf-8

# # Ensembling
#     
#     
# In this kernel we'll look at using XGBoost (Gradient Boosting) mixed in with fastai2, and you'll notice we'll be using fastai2 to prepare our data!
# 
# <font size=3 color="red">Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>

# ## Acknowledgment
# 
# I would like to thank [Muellerzr](https://github.com/muellerzr/) from where I forked this notebook from his [Fastai mega online study series](https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/blob/master/Tabular%20Notebooks/02_Ensembling.ipynb)

# In[ ]:


get_ipython().system(' pip install fastai2')


# In[ ]:


from fastai2.tabular.all import *


# 
# Let's first build our TabularPandas object:

# In[ ]:


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')


# In[ ]:


df.head()


# In[ ]:


cat_names= ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']

procs = [Categorify, FillMissing, Normalize]
y_name = 'salary'
block_y = CategoryBlock()
splits = RandomSplitter()(range_of(df))


# In[ ]:


to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=y_name,                       block_y=block_y, splits=splits)


# ## Xgboost

# In[ ]:


import xgboost as xgb
from xgboost import plot_importance


# In[ ]:


to.train.ys


# In[ ]:


X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()


# In[ ]:


model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=1e-1, )


# In[ ]:


xgb_model = model.fit(X_train, y_train)


# In[ ]:


xgb_preds = xgb_model.predict_proba(X_test)
accuracy(tensor(xgb_preds), tensor(y_test))


# In[ ]:


plot_importance(xgb_model)


# ## Bring in fastai2

# In[ ]:


dls = to.dataloaders()
learn = tabular_learner(dls, layers=[200, 100], metrics=accuracy)


# In[ ]:


learn.fit(10, 1e-2)


# In[ ]:


nn_preds = learn.get_preds()[0]
nn_preds


# In[ ]:


accuracy(tensor(nn_preds), tensor(y_test))


# In[ ]:


class PermutationImportance():
  "Calculate and plot the permutation importance"
  def __init__(self, learn:Learner, df=None, bs=None):
    "Initialize with a test dataframe, a learner, and a metric"
    self.learn = learn
    self.df = df if df is not None else None
    bs = bs if bs is not None else learn.dls.bs
    self.dl = learn.dls.test_dl(self.df, bs=bs) if self.df is not None else learn.dls[1]
    self.x_names = learn.dls.x_names.filter(lambda x: '_na' not in x)
    self.na = learn.dls.x_names.filter(lambda x: '_na' in x)
    self.y = dls.y_names
    self.results = self.calc_feat_importance()
    self.plot_importance(self.ord_dic_to_df(self.results))

  def measure_col(self, name:str):
    "Measures change after column shuffle"
    col = [name]
    if f'{name}_na' in self.na: col.append(name)
    orig = self.dl.items[col].values
    perm = np.random.permutation(len(orig))
    self.dl.items[col] = self.dl.items[col].values[perm]
    metric = learn.validate(dl=self.dl)[1]
    self.dl.items[col] = orig
    return metric

  def calc_feat_importance(self):
    "Calculates permutation importance by shuffling a column on a percentage scale"
    print('Getting base error')
    base_error = self.learn.validate(dl=self.dl)[1]
    self.importance = {}
    pbar = progress_bar(self.x_names)
    print('Calculating Permutation Importance')
    for col in pbar:
      self.importance[col] = self.measure_col(col)
    for key, value in self.importance.items():
      self.importance[key] = (base_error-value)/base_error #this can be adjusted
    return OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))

  def ord_dic_to_df(self, dict:OrderedDict):
    return pd.DataFrame([[k, v] for k, v in dict.items()], columns=['feature', 'importance'])

  def plot_importance(self, df:pd.DataFrame, limit=20, asc=False, **kwargs):
    "Plot importance with an optional limit to how many variables shown"
    df_copy = df.copy()
    df_copy['feature'] = df_copy['feature'].str.slice(0,25)
    df_copy = df_copy.sort_values(by='importance', ascending=asc)[:limit].sort_values(by='importance', ascending=not(asc))
    ax = df_copy.plot.barh(x='feature', y='importance', sort_columns=True, **kwargs)
    for p in ax.patches:
      ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y()  * 1.005))


# In[ ]:


imp = PermutationImportance(learn)


# ## Ensembling

# In[ ]:


avgs = (nn_preds + xgb_preds) /2


# In[ ]:


avgs


# In[ ]:


argmax = avgs.argmax(dim=1)
argmax


# In[ ]:


y_test


# In[ ]:


accuracy(tensor(avgs), tensor(y_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

tree = RandomForestClassifier(n_estimators=100)
tree.fit(X_train, y_train)


# In[ ]:


get_ipython().system(' pip install rfpimp')
from rfpimp import *


# In[ ]:


imp = importances(tree, X_test, to.valid.ys)


# In[ ]:


plot_importances(imp)


# In[ ]:


forest_preds = tree.predict_proba(X_test)


# In[ ]:


forest_preds


# In[ ]:


accuracy(tensor(forest_preds), tensor(y_test))


# In[ ]:


wk = (nn_preds + xgb_preds + forest_preds) / 3
accuracy(tensor(avgs), tensor(y_test))


# # Ending note <a id="3"></a>
# 
# <font size=4 color="red">This concludes my kernel. Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>
