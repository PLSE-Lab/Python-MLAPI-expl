#!/usr/bin/env python
# coding: utf-8

# Trying out some neural model and leaky relu for fun, maybe try and get it to output some probabilities of y in bins from 100-110, 110-120 etc.

# In[ ]:


from keras.layers.advanced_activations import LeakyReLU


# In[ ]:


#help(LeakyReLU)


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
print(train.shape)
test = pd.read_csv("../input/test.csv")
print(test.shape)
targets = train['y']
train = train.drop('y', axis=1)
print(train.shape)


# In[ ]:


print(test.shape)
print(train.shape)


# In[ ]:


print(targets.min())
print(targets.max())
print(len(range(70, 270, 5)))
bin1 = ((targets > 70) & (targets < 75)).astype(int)
bin2 = ((targets > 75) & (targets < 80)).astype(int)
bin3 = ((targets > 80) & (targets < 85)).astype(int)
bin4 = ((targets > 85) & (targets < 90)).astype(int)
bin5 = ((targets > 90) & (targets < 95)).astype(int)
bin6 = ((targets > 95) & (targets < 100)).astype(int)
bin7 = ((targets > 100) & (targets < 105)).astype(int)
bin8 = ((targets > 105) & (targets < 110)).astype(int)
bin9 = ((targets > 110) & (targets < 115)).astype(int)
bin10 = ((targets > 115) & (targets < 120)).astype(int)
bin11 = ((targets > 120) & (targets < 125)).astype(int)
bin12 = ((targets > 125) & (targets < 130)).astype(int)
bin13 = ((targets > 130) & (targets < 135)).astype(int)
bin14 = ((targets > 135) & (targets < 140)).astype(int)
bin15 = ((targets > 140) & (targets < 145)).astype(int)
bin16 = ((targets > 145) & (targets < 150)).astype(int)
bin17 = ((targets > 150) & (targets < 155)).astype(int)
bin18 = ((targets > 155) & (targets < 160)).astype(int)
bin19 = ((targets > 160) & (targets < 165)).astype(int)
bin20 = (targets > 165).astype(int)


# In[ ]:


bins = pd.concat([bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,
                 bin11,bin12,bin13,bin14,bin15,bin16,bin17,bin18,bin19,bin20], axis=1)
bins.shape


# In[ ]:


targets = bins


# In[ ]:


df = pd.concat([train,test], axis=0)
df.index = range(0,8418)
ids = df['ID']
df.shape


# In[ ]:


le = LabelEncoder()
enc = OneHotEncoder()
X0 = df['X0'].T
X1 = df['X1'].T
X2 = df['X2'].T
X3 = df['X3'].T
X4 = df['X4'].T
X5 = df['X5'].T
X6 = df['X6'].T
X8 = df['X8'].T

len(list(X0.unique()))


# In[ ]:


le.fit(X0)
X0 = le.transform(X0)
X0 = pd.get_dummies(X0)

le.fit(X1)
X1 = le.transform(X1)
X1 = pd.get_dummies(X1)

le.fit(X2)
X2 = le.transform(X2)
X2 = pd.get_dummies(X2)

le.fit(X3)
X3 = le.transform(X3)
X3 = pd.get_dummies(X3)

le.fit(X4)
X4 = le.transform(X4)
X4 = pd.get_dummies(X4)

le.fit(X5)
X5 = le.transform(X5)
X5 = pd.get_dummies(X5)

le.fit(X6)
X6 = le.transform(X6)
X6 = pd.get_dummies(X6)

le.fit(X8)
X8 = le.transform(X8)
X8 = pd.get_dummies(X8)


# In[ ]:


categoricals = pd.concat([X0, X1, X2, X3, X4, X5, X6, X8], axis=1)
print(categoricals.shape)


# In[ ]:


df = df.drop(['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'], axis=1)


# In[ ]:


print(list(df.dtypes.unique()))
print(df.max().max())
print(df.min().min())
print(df.shape[0]*df.shape[1])


# In[ ]:


df_ = pd.get_dummies(df)


# In[ ]:


print(df_.shape)
print(categoricals.shape)
print(targets.shape)


# In[ ]:


df = pd.concat([categoricals, df_], axis=1)
print(df.shape)
print(df.shape[0]*df.shape[1])


# In[ ]:


train = df[0:4209]
test = df[4209:]
print(train.shape)
print(test.shape)


# In[ ]:


def transforms(df):
    dfX = pd.DataFrame(df)
    m = 5
    df = np.add(dfX,abs(m))
    rt = 1/df
    rs = 1/np.sqrt(df)
    s = np.sqrt(df)
    l = np.log(df)
    d_list = [df, rt, rs, s, l]
    d = pd.concat(d_list, axis=1)
    return d


# In[ ]:


ttrain = train


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train, targets, random_state=16)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=300).fit(x_train)


# In[ ]:


train_pca = pca.transform(x_train)
test_pca = pca.transform(x_test)
print(train_pca.shape)
print(test_pca.shape)


# In[ ]:


x_train = np.concatenate((x_train, train_pca), axis=1)
x_test = np.concatenate((x_test, test_pca), axis=1)
print(x_train.shape)
print(x_test.shape)


# In[ ]:


from keras import optimizers
adam = optimizers.adam(lr=0.0001)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

def create_model():  
    model = Sequential() 
    model.add(Dense(500, input_dim=880, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(75, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(20, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


y_train, y_test = y_train.values, y_test.values


# In[ ]:


model = create_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32,
         verbose=0, epochs=100)


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print("Success rate: %.2f%%" % (score[1]*100))
print("Random guess 1-20 bins: %.2f%%" % (1/20*100))


# In[ ]:


train_pca = pca.transform(train)
train = np.concatenate((train, train_pca), axis=1)
train = scaler.transform(train)
train_data_probabilities = model.predict(train)


# In[ ]:


test_pca = pca.transform(test)
test = np.concatenate((test, test_pca), axis=1)
test = scaler.transform(test)


# In[ ]:


test_data_probabilities = model.predict(test)


# In[ ]:


import seaborn as sns
sns.set_style("dark")


# In[ ]:


bin_names = ['70-75', '75-80', '80-85', '85-90', '90-95', '95-100',
            '100-105', '105-110', '110-115', '115-120', '120-125',
            '125-130', '130-135', '135-140', '140-145', '145-150',
            '150-155', '155-160', '160-165', '165>']
averages = [72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5,
            122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 152.5, 157.5, 162.5,
            167.5]


# In[ ]:


plt.hist(np.log(train_data_probabilities+0.0000001), normed=True)
plt.title("Log distribution of predicted y location probabilities on train data")
plt.show()


# In[ ]:


plt.hist(np.log(test_data_probabilities+0.0000001), normed=True)
plt.title("Log distribution of predicted y location probabilities on test data")
plt.show()


# In[ ]:


all_probs = (np.concatenate((train_data_probabilities, test_data_probabilities), axis=0))


# In[ ]:


all_probs = pd.DataFrame(all_probs)
all_probs.columns = averages


# In[ ]:


all_probs_bins = all_probs.idxmax(axis=1)


# In[ ]:


train_data_bins = all_probs_bins[0:4209]
test_data_bins = all_probs_bins[4209:]


# In[ ]:


train_data_bins.hist()
plt.title("Train data y location frequency");


# In[ ]:


test_data_bins.hist()
plt.title("Test data y location frequency");


# In[ ]:


train_median_probabilities = pd.DataFrame(train_data_probabilities).median()
train_median_probabilities.index = bin_names


# In[ ]:


test_median_probabilities = pd.DataFrame(test_data_probabilities).median()
test_median_probabilities.index = bin_names


# In[ ]:


ax = sns.barplot(x=train_median_probabilities.index, y=train_median_probabilities)
ax.set_ylabel(ylabel='Median probability')
ax.set_title(label='Median probability of predicted y values on train data')
ax.set_xlabel(xlabel = 'Y values')
ax.set_xticklabels(labels = train_median_probabilities.index, rotation=90);


# In[ ]:


ax = sns.barplot(x=test_median_probabilities.index, y=test_median_probabilities)
ax.set_ylabel(ylabel='Median probability')
ax.set_title(label='Median probability of predicted y values on test data')
ax.set_xlabel(xlabel = 'Y values')
ax.set_xticklabels(labels = test_median_probabilities.index, rotation=90);


# In[ ]:


Y = pd.read_csv("../input/train.csv")
c = Y['y']
trueVpredict = pd.concat([train_data_bins, c], axis=1)
trueVpredict.columns = ['predicted location', 'true location']
test_data_predictions = trueVpredict['predicted location']
trueVpredict.head()


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error
r2_score = r2_score(trueVpredict['true location'],trueVpredict['predicted location'])
print("R2 all data: ", r2_score)


# In[ ]:


rmse_score = np.sqrt(mean_squared_error(trueVpredict['true location'],trueVpredict['predicted location']))
print("RMSE all data: ", rmse_score)


# In[ ]:


from sklearn.manifold import TSNE
Y = pd.read_csv("../input/train.csv")
train_c = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0)
train_c = np.clip(c, c.min(), 140)
tsne = TSNE(n_components=2, perplexity=50)
X = tsne.fit_transform(train_data_probabilities)
ax = plt.scatter(X[:, 0], X[:, 1], c=train_c, cmap="cubehelix")
plt.colorbar(ax)
plt.title("Train data probabilities in 2D");


# In[ ]:


X = tsne.fit_transform(test_data_probabilities)
test_c = test_data_predictions
test_c = np.clip(test_c, test_c.min(), 140)
ax = plt.scatter(X[:, 0], X[:, 1], c=test_c, cmap="cubehelix")
plt.colorbar(ax)
plt.title("Test data probabilities in 2D");


# In[ ]:


from mpl_toolkits import mplot3d
from matplotlib import cm
tsne = TSNE(n_components=3, perplexity=50)
X = tsne.fit_transform(train_data_probabilities)
X1 = np.meshgrid(X[:, 0])
X2 = np.meshgrid(X[:, 1])
X3 = np.meshgrid(X[:, 2])

plt.figure(figsize=(14,10))
ax = plt.axes(projection='3d')
ax.scatter3D(X1, X2, X3, c=train_c, cmap="cubehelix")
m = cm.ScalarMappable(cmap="cubehelix")
m.set_array(train_c)
cbar = plt.colorbar(m)
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
ax.set_zlabel("tsne3")
plt.title("Train data probabilities in 3D")
plt.show();


# In[ ]:


tsne = TSNE(n_components=2, perplexity=50)
X = tsne.fit_transform(X)
X1 = np.meshgrid(X[:, 0])
X2 = np.meshgrid(X[:, 1])
X3 = np.meshgrid(train_c)

plt.figure(figsize=(14,10))
ax = plt.axes(projection='3d')
ax.scatter3D(X1, X2, X3, c= train_c, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
m = cm.ScalarMappable(cmap=cm.coolwarm)
m.set_array(train_c)
cbar = plt.colorbar(m)
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
ax.set_zlabel("True Y")
plt.title("2D training probabilities versus True Y")
plt.show();


# In[ ]:


tsne = TSNE(n_components=3, perplexity=50)
X = tsne.fit_transform(test_data_probabilities)
X1 = np.meshgrid(X[:, 0])
X2 = np.meshgrid(X[:, 1])
X3 = np.meshgrid(X[:, 2])

plt.figure(figsize=(14,10))
ax = plt.axes(projection='3d')
ax.scatter3D(X1, X2, X3, c=test_c, cmap="cubehelix")
m = cm.ScalarMappable(cmap="cubehelix")
m.set_array(test_c)
cbar = plt.colorbar(m)
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
ax.set_zlabel("tsne3")
plt.title("Test data probabilities in 3D")
plt.show();


# In[ ]:


Y = pd.read_csv("../input/train.csv")
c = Y['y']
trueVpredict = pd.concat([train_data_bins, c], axis=1)
trueVpredict.columns = ['predicted location', 'true location']
trueVpredict.tail()


# In[ ]:


print("Test data predicted locations")
pd.DataFrame(test_data_bins.head(10)).T


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


X0 = test['X0']
le.fit(X0)
X0 = le.transform(X0)
X0_ = pd.get_dummies(X0)
X = np.concatenate((X0_, X), axis=1)
scaler.fit(X)
X = scaler.transform(X)

tsne = TSNE(n_components=2, perplexity=50)
X = tsne.fit_transform(X)
ax = plt.scatter(X[:, 0], X[:, 1], c=X0, cmap="cubehelix")
plt.colorbar(ax)
plt.title("X0 categorical probabilities in 2D");


# In[ ]:


X0 = test['X0']
le.fit(X0)
X0 = le.transform(X0)
X0_ = pd.get_dummies(X0)
X = np.concatenate((X0_, X), axis=1)
scaler.fit(X)
X = scaler.transform(X)

tsne = TSNE(n_components=3, perplexity=50)
X = tsne.fit_transform(X)
X1 = np.meshgrid(X[:, 0])
X2 = np.meshgrid(X[:, 1])
X3 = np.meshgrid(X[:, 2])

plt.figure(figsize=(14,10))
ax = plt.axes(projection='3d')
ax.scatter3D(X1, X2, X3, c=X0, cmap="cubehelix")
m = cm.ScalarMappable(cmap="cubehelix")
m.set_array(X0)
cbar = plt.colorbar(m)
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
ax.set_zlabel("tsne3")
plt.title("X0 categorical probabilities in 3D on Test Data")
plt.show();


# In[ ]:


X1 = np.meshgrid(X[:, 0])
X2 = np.meshgrid(X[:, 1])
X3 = np.meshgrid(test_data_bins)

plt.figure(figsize=(14,10))
ax = plt.axes(projection='3d')
ax.scatter3D(X1, X2, X3, c=X0, cmap="cubehelix")
m = cm.ScalarMappable(cmap="cubehelix")
m.set_array(X0)
cbar = plt.colorbar(m)
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
ax.set_zlabel("predicted y")
plt.title("X0 categorical probabilities in 2D versus Predicted Y on Test Data")
plt.show();


# In[ ]:


ax = plt.scatter(X1, X3, c=X0, cmap=cm.coolwarm)
plt.colorbar(ax)
plt.title("X0 categorical probabilities in 1D versus Predicted Y on Test Data");


# In[ ]:


X0s = test['X0']

df = pd.concat([X0s, pd.DataFrame(test_data_probabilities)], axis=1)
counts = df.groupby(['X0']).count().iloc[:,0]
medians = df.groupby(['X0']).max().median(axis=1)
means = df.groupby(['X0']).max().mean(axis=1)


# In[ ]:


colors = sns.color_palette("terrain", len(counts))
plt.figure(figsize=(12,8))
counts = counts.sort_values(ascending=False)
ax = sns.barplot(y = counts.index , x = counts, orient='h', palette=colors)
ax.set_xlabel(xlabel='Count of X0 entries', fontsize=16)
ax.set_ylabel(ylabel='X0 Category', fontsize=16)
ax.set_title(label='X0 category counts', fontsize=20)
plt.show();


# In[ ]:


colors = sns.color_palette("ocean", len(medians))
medians = pd.DataFrame(medians.sort_values(ascending=False))
plt.figure(figsize=(12,8))
ax = sns.barplot(y = medians.index , x = medians[0], orient='h', palette=colors)
ax.set_xlabel(xlabel='Median of maximum predicted probability based on X0 category', fontsize=16)
ax.set_ylabel(ylabel='X0 Category', fontsize=16)
ax.set_title(label='Descending median prediction certainty based on X0 category', fontsize=20)
plt.show();


# In[ ]:


colors = sns.color_palette("gist_earth", len(means))
means = pd.DataFrame(means.sort_values(ascending=False))
plt.figure(figsize=(12,8))
ax = sns.barplot(y = means.index , x = means[0], orient='h', palette=colors)
ax.set_xlabel(xlabel='Median of mean predicted probability based on X0 category', fontsize=16)
ax.set_ylabel(ylabel='X0 Category', fontsize=16)
ax.set_title(label='Descending mean prediction certainty based on X0 category', fontsize=20)
plt.show();


# In[ ]:


test = pd.read_csv("../input/test.csv")
ids = pd.DataFrame((test['ID'].values).reshape(4209,1))
sub = pd.DataFrame(((test_data_bins.values).reshape(4209,1)))
sub = pd.concat([ids, sub], axis=1)
sub.columns = ['ID', 'y']
sub.head(1)


# In[ ]:


# 0.48328 LB
sub.to_csv('./maxbin_sub.csv', index=False)


# In[ ]:


test_data_probabilities = pd.DataFrame(test_data_probabilities)
test_data_probabilities.columns = bin_names
test_data_probabilities.head()


# In[ ]:


print(test_data_probabilities.shape)
test_data_probabilities.to_csv('./test_data_probabilities.csv', index=False)


# In[ ]:


train_data = ttrain
train_data_pca = pca.transform(ttrain)
train_data = np.concatenate((train_data, train_data_pca), axis=1)
train_data = scaler.fit_transform(train_data)
train_data_probabilities = pd.DataFrame(model.predict(train_data))
train_data_probabilities.columns = bin_names
train_data_probabilities.head()


# In[ ]:


print(train_data_probabilities.shape)
train_data_probabilities.to_csv('./train_data_probabilities.csv', index=False)


# In[ ]:


weights = pd.DataFrame(test_data_probabilities)
avgs = np.ones_like(weights) * np.array(averages)
weighted_pred = weights * avgs
weighted_pred = weighted_pred.sum(axis=1)
sub = pd.DataFrame(((weighted_pred.values).reshape(4209,1)))
sub = pd.concat([ids, sub], axis=1)
sub.columns = ['ID', 'y']
sub = sub.iloc[0:4209]
print(sub.shape)
sub.head()


# In[ ]:


# 0.53952 LB
sub.to_csv('./weightedAvg_sub.csv', index=False)


# In[ ]:




