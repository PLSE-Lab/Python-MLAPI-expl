#!/usr/bin/env python
# coding: utf-8

# # Correlates of Happiness
# **here, I use the dataset "65 world indexes" to look for predictors of the happiness score of the "world happiness report 2015"**

# ## import and preprocess data

# In[1]:


from pylab import *
import pandas as pd
import seaborn as sns
sns.set_style('white')
from scipy.cluster.hierarchy import dendrogram, fcluster
get_ipython().run_line_magic('matplotlib', 'inline')
hap = pd.read_csv("../input/world-happiness/2015.csv") #world happiness report 2015
world = pd.read_csv("../input/65-world-indexes-gathered/Kaggle.csv") #65 world indexes


# In[2]:


hap.head()


# In[3]:


world.head()


# In[4]:


world.columns = concatenate([[hap.columns[0]], world.columns[1:]]) #equalize country first column


# In[5]:


hap.shape, world.shape


# find differences in country spelling and correct for them

# In[6]:


print(set(hap['Country']).symmetric_difference(world['Country']))
#congo, dom rer. congo, hongkong, Palestine, 


# In[7]:


old_names, new_names = (['Congo (Brazzaville)', 'Congo (Kinshasa)', 'Hong Kong\xc2\xa0', 'Palestinian Territories'],
                         ['Republic of the Congo',
                          'Democratic Republic of the Congo',
                          'Hong Kong',
                          'Palestine'])


# adapt country names, 153 countries overlap in both datasets

# In[8]:


hap['Country'].replace(old_names, new_names, inplace = True)
world['Country'].replace(old_names, new_names, inplace = True)


# In[9]:


len(set(hap['Country']).intersection(world['Country'])), len(set(hap['Country']).symmetric_difference(world['Country']))


# merge data sets and remove features from happiness report

# In[10]:


merged = hap[['Country','Happiness Score']].merge(world, how = 'inner', on = 'Country')


# 'thinning' data (discard regional information, (arguably) redundant features and features from happiness report)

# In[11]:


merged.drop(['Country', u'Gross domestic product GDP 2013', 'Infant Mortality 2013 per thousands', 'Gross national income GNI per capita - 2011  Dollars', 'Birth registration funder age 5 2005-2013', 'Pre-primary 2008-2014', u'International inbound tourists thausands 2013', u'International student mobility of total tetiary enrolvemnt 2013', ], axis = 1, inplace = True)


# ## feature correlations

# In[12]:


corr_matrix = merged.corr()


# **raw correlation matrix**

# In[13]:


figure(figsize = (15, 15))
sns.heatmap(corr_matrix, xticklabels= False)


# **clusters of correlated features** (absolute value: sign does not matter)

# In[14]:


figure(figsize = (16, 16))
#sns.heatmap(merged.corr())
cg = sns.clustermap(merged.corr().applymap(abs), xticklabels = True, yticklabels = False)


# In[15]:


#figure(figsize = (9, 6))
#a = dendrogram(cg.dendrogram_col.linkage, labels = merged.columns, color_threshold = 2, leaf_font_size= 10)
fc = fcluster(cg.dendrogram_col.linkage, 2, criterion = 'distance')


# we find a cluster of highly correlated features among which is the 'Happiness Score'. we expect these features to be predictors for happiness but at the same time to be interdependent.
# these features are part of the cluster:

# In[16]:


pd.Series(data = merged.columns[fc == 1])


# ## correlates of 'Happiness Score'
# **let's extract the strongest correlations with happiness (> 0.2, irrespective of the sign)** <br>

# In[17]:


hap_corr = pd.DataFrame(merged.corr()[abs(merged.corr()['Happiness Score']) > .2]['Happiness Score'])
hap_corr.drop('Happiness Score', axis = 0, inplace = True)
hap_corr.columns = ['Corr. with happiness']
hap_corr.sort_values('Corr. with happiness', axis = 0, ascending = False)


# In[18]:


# sort by absolute value of correlation
hap_corr_best = hap_corr.apply(abs).sort_values('Corr. with happiness', axis = 0, ascending = False)


# **plot correlation matrix of 20 most correlated features with happiness**
# again, they are highly correlated among themselfes

# In[19]:


figure(figsize = (12,10))
sns.heatmap(corr_matrix.apply(abs).loc[hap_corr_best[:20].index,hap_corr_best[:20].index], annot= True)


# ## linear regression for happiness
# **how well can the most correlated features "predict" happiness (despite being highly interdependent)?**

# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score


# plot regression score against number of most important features included

# In[21]:


figure(figsize = (8,6))
scores = zeros((35, len(hap_corr_best.index)-1))
for i in range(35):
    for k in range(1,len(hap_corr_best.index)):
        X, y = array(merged[hap_corr_best.index[:k]]), array(merged['Happiness Score'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
        linReg = LinearRegression(normalize = True)
        linReg.fit(X_train,y_train)
        sc = linReg.score(X_test, y_test)
        scores[i][k-1] = sc
sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores)
xlabel("# most important features")
ylabel("$R^2$ score of lin. regression")
show()


# the $R^2$ score never beats 0.65 and is usually around 0.55-0.66 which corresponds to a coeff of corr ~0.8. this is equaivalent to the coeff of corr of the highest corelated features (see above). hence, because all predictive features are interdependendent they don't contribute additional information about happiness.

# ## going deeper

# **can we extract nonlinear dependencies or hidden composite variables?** <br>
# let's try a simple ANN with one hidden layer (may take some time depending in computing power)

# In[22]:


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

stopper = EarlyStopping(monitor='loss', min_delta=0, patience = 8, verbose= 0, mode='auto')
MinMax = MinMaxScaler()


# In[23]:


get_ipython().run_cell_magic('time', '', '\nfigure(figsize = (8,6))\nscores2 = zeros((3, len(hap_corr_best.index)-1))\nfor k in range(1,len(hap_corr_best.index)):\n    K.clear_session()\n    X, y = array(merged[hap_corr_best.index[:k]]), array(merged[\'Happiness Score\'])\n    X = MinMax.fit_transform(X)\n    for i in range(3):\n        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)\n        model = Sequential()\n        model.add(Dense(2*k, input_dim = k, activation = \'relu\'))\n        model.add(Dense(1))\n\n        model.compile(optimizer= \'rmsprop\',\n                      loss=\'mse\', metrics = [coeff_determination])\n\n        model.fit(X_train, y_train, epochs = 500, batch_size = 32, verbose = 0, callbacks= [stopper])\n        sc = model.evaluate(X_test, y_test, verbose = 0)\n        scores2[i][k-1]= sc[1]\nsns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores2)\nxlabel("# most important features")\nylabel("$R^2$ score of lin. regression")\nshow()\n#model.summary()')


# comparison of lin. regression and ANN

# In[24]:


figure(figsize = (7,5))
sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores2, err_style = "unit_traces", condition = 'ANN')
sns.tsplot(time = range(1,len(hap_corr_best.index)), data = scores, color = 'r', condition = 'linear')
ylim([0,1])
show()


# results of lin. reg. and ANN seem to be similar in terms of $R^2$ score (ANN may be slightly better but is too expensive to evaluate). no higher order dependence could be found. <br>
# results for ANN are vary variable. they did not significantly change for 2 hidden layers and different optimizers/activations. <br>
# ** of course, the result does not imply that no better prediction of happiness is possible. yet, it seem not straight forward. ** <br>
