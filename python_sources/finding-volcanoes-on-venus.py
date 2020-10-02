#!/usr/bin/env python
# coding: utf-8

# <h1><i>Finding Volcanoes On Venus</i></h1>

# <h2>Introduction:</h2>
# <p>Using the dataset <a href="https://www.kaggle.com/fmena14/volcanoesvenus">Volcanoes on Venus</a>, I will try to predict whether an image from Venus has a volcano in it or not.</p>
# <sub>wish me luck.</sub>

# <h2>Overview:</h2>
# <ol>
#     <li>Importing The Libraries</li>
#     <li>Reading The Data</li>
#     <li>Handling Missing Data</li>
#     <li>Fixing The First Example Problem</li>
#     <li>Training To Classify</li>
#     <li>Training To Regress</li>
#     <li>Merging The Predictions</li>
#     <li>Visualising The Predicted Data</li>
# </ol>

# <h3>1. Importing The Libraries:</h3>
# <p>Nothing special here, just some classic old importing.</p>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso


# <h3>2. Reading The Data:</h3>
# <p>This thing took forever because, the file sizes are well over <b>200MB</b>.</p>

# In[ ]:


train_images = pd.read_csv('../input/volcanoes_train/train_images.csv')
train_labels = pd.read_csv('../input/volcanoes_train/train_labels.csv')

test_images = pd.read_csv('../input/volcanoes_test/test_images.csv')
test_labels = pd.read_csv('../input/volcanoes_test/test_labels.csv')


# <p>Now comes the interesting part, we will analyse the data and by analyse I mean just looking at it as the data is nothing but pixel values.</p>
# <p>But <i>first</i> let's look at some heads.<sub>(:P)</sub></p>

# In[ ]:


print(train_images.head())
print(train_labels.head())

print(test_images.head())
print(test_labels.head())


# <p>Those were the heads.</p>
# <p>Now enough looking at heads, let's use some matplotlib magic and look at some pictures from <b>SPACE!!!</b></p>

# In[ ]:


for x in range(10):
    row = np.array(train_images.iloc[x])
    image = row.reshape((110,110))
    plt.figure()
    plt.imshow(image,cmap="gray")

plt.show()


# <p>Doesn't it feel exciting to work on images taken far beyond from where you currently are, like millions of kilometers.</p>
# <p>Unless you are orbiting Venus, then that's another thing.</p>

# <h3>3. Handling Missing Data:</h3>
# <p>So, now it's time to clean up our data.</p>
# <p>Nobody likes Nans.</p>
# <sub>except my actual nan i like her.</sub>

# In[ ]:


print(sum(train_images.isna().sum()))
print(sum(test_images.isna().sum()))

print(train_labels.isna().sum())
print(test_labels.isna().sum())


# Whoa. Okay, So, our images don't have any missing values but our labels seem to have a lot of them.
# Let's look at the head of our labels again.

# In[ ]:


print(train_labels.head())
print(test_labels.head())


# It seems the 'Type, Radius, Number Volcanoes' for images with no volcanoes have not been provided.

# In[ ]:


plt.hist(train_labels["Volcano?"])
plt.show()


# And, there are 6000 images with no volcanoes, which explains the 6000 Nan for each label.
# 
# The best way that I think of to handle this is fill the missing data with zeros as it seems kinda logical, no volcano means zero radius, zero number of volcanoes and zero Type maybe?
# 
# Let's just go with it.

# In[ ]:


train_labels.fillna(value=0,inplace=True)
test_labels.fillna(value=0,inplace=True)


# There, fixed.

# In[ ]:


for x in train_labels.keys():
    X = train_labels[x]
    plt.figure()
    plt.hist(X)
    plt.xlabel(x)
    plt.ylabel('frequency')
plt.show()


# There you have it, some nice graphs to look at.
# 
# Now, for a really interesting problem I faced when analysing this data,  which left me scratching my head for hours.

# <h3>4. Fixing The First Example:</h3>
# Let me show you what I am talking about.

# In[ ]:


print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)


# There you see it!? we either have an extra label or a missing image.
# Let's investigate.

# In[ ]:


print(train_images.head())
print(test_images.head())


# Do you see what I'm seeing?
# The columns of the data frame, look very similar to pixel values.
# Let's check our hypothesis.
# <p>We will take each column, turn it into a floating point and round it to an integer and plot  it.</p>

# In[ ]:


columns = list(train_images.columns)
columns = [float(x) for x in columns]
first_row = np.round(columns).astype(np.int64)

image = first_row.reshape((110,110))
plt.imshow(image,cmap="gray")

columns = list(test_images.columns)
columns = [float(x) for x in columns]
first_row = np.round(columns).astype(np.int64)

image = first_row.reshape((110,110))
plt.figure()
plt.imshow(image,cmap="gray")

plt.show()


# Yup, my guess was correct, the hidden image was in the columns.
# Now, let's fix it.

# In[ ]:


columns = list(train_images.columns)
columns = [float(x) for x in columns]
first_row = np.round(columns).astype(np.int64)

train_images.loc[-1] = first_row
train_images.index += 1
train_images.sort_index(inplace=True)

columns = list(test_images.columns)
columns = [float(x) for x in columns]
first_row = np.round(columns).astype(np.int64)

test_images.loc[-1] = first_row
test_images.index += 1
test_images.sort_index(inplace=True)

print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)


# There, fixed.
# <p>Now, to my favourite part.</p>

# <h3>5. Training To Classify:</h3>
# <p>Now to train this would be tricky, because the problem here is a combination of classification(Volcano?, Type, Number Volcanoes) and regression(Radius) so we have to split the problem into <b>two</b>.</p>
# <p>First we will train the decisiontree classifier because it supports multi label classification and then produce a list of predicted classes, which we will join to our test data and train the lasso algorithm to regress the radius given pixel values, type and number of volcanoes.</p>

# In[ ]:


X_train = np.array(train_images)
y_train = np.array(train_labels.drop(['Radius'],1))

X_test = np.array(test_images)
y_test = np.array(test_labels.drop(['Radius'],1))

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)
predclf = pd.DataFrame(pred)

predclf.columns = ['Volcano?','Type','Number Volcanoes']

print(predclf.head())


# So, now we have our predicted Volcano?, Type, Number Volcanoes which we will join to the test data and perform regression.

# <h3>6. Training To Regress:</h3>
# <p>First we will join the training data with the necessary training labels and do the same with testing data and train it all on the algorithm.</p>

# In[ ]:


new_train_data = train_images[(train_labels['Volcano?'] == 1)].join(train_labels[(train_labels['Volcano?'] == 1)].drop(['Radius','Volcano?'],1))
new_train_labels = train_labels['Radius'][(train_labels['Volcano?'] == 1)]

new_test_data = test_images[(predclf['Volcano?'] == 1)].join(predclf[(predclf['Volcano?'] == 1)].drop(['Volcano?'],1))
new_test_labels = test_labels['Radius'][(predclf['Volcano?'] == 1)]

X_train = np.array(new_train_data)
y_train = np.array(new_train_labels)

X_test = np.array(new_test_data)
y_test = np.array(new_test_labels)

reg = Lasso()
reg.fit(X_train,y_train)

pred = reg.predict(X_test)
predreg = pd.DataFrame(pred)

predreg.columns = ['Radius']

print(predreg.head())


# So, now we have all of our predicted classes, all we have to do is merge them.

# <h3>7. Merging The Predictions:</h3>

# In[ ]:


pred_radius = []
y = 0
for x in range(predclf.shape[0]):
    if predclf['Volcano?'].iloc[x] == 1.0:
        pred_radius.append(predreg['Radius'].iloc[y])
        y+=1
    else:
        pred_radius.append(0.)

pred_radius = pd.DataFrame({'Radius': pred_radius})

total_pred = predclf.join(pred_radius)

print(total_pred.head())


# Everythings' looking good, except the order of the columns, so let's fix that.

# In[ ]:


cols = ['Volcano?','Type','Radius','Number Volcanoes']
total_pred = total_pred[cols]

print(total_pred)


# <b>Perfect!</b>
# <p>Let's look at what we've predicted.</p>

# <h3>8. Visualising The Predicted Data:</h3>

# In[ ]:


for x in total_pred.keys():
    X = total_pred[x]
    plt.figure()
    plt.hist(X)
    plt.xlabel(x)
plt.show()


# In[ ]:


detected = test_images[(total_pred['Volcano?'] == 1)]

for x in range(detected.shape[0]):
    image_flat = np.array(detected.iloc[x])
    image = image_flat.reshape((110,110))

    plt.figure()
    plt.imshow(image,cmap='gray')
    plt.title('Volcano '+str(x))

plt.show()


# So, those were all the images my model thinks has volcanoes.
# 
# <p>Thank you if you read all the way till here or even if you just scrolled to bottom.</p>
# <p>I apologise if my terrible humour bored you or distracted you, but anyways thank you for your time.</p>
