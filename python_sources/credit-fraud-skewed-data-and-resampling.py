#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
from sklearn import tree
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler 


# In[78]:


np.random.seed(0)
data = pd.read_csv("../input/creditcard.csv")


# ## Setup
# Importing tools, loading the data, splitting up our data into a train and test set.
# I hid this code, but take a peek if you're curious.

# In[79]:


# This is a function for calculating the F1 Score. 
# A much better way at guaging how well the algorithm 
# is doing than simply by accuracy. 
# Learn more: https://en.wikipedia.org/wiki/F1_score
# Credit to: https://stackoverflow.com/a/45305384
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[80]:


# This is a little macro to make a confusion matrix as output, nothing special.
from sklearn.metrics import confusion_matrix

def evaluate(model, X_val, Y_val, silent=False):
    predictions = model.predict(X_val)
    predictions = np.around(predictions).flatten()
    results = np.equal(Y_val, predictions)
    acc = float(np.sum(results)/len(results))
    suspected_fraud = np.nonzero(predictions)[0]
    real_fraud = np.nonzero(Y_val)[0]
    fraud_results = results[real_fraud]
    fraud_acc = float(np.sum(fraud_results)/len(fraud_results))
    correct_count = np.sum(fraud_results)
    cm = confusion_matrix(Y_val, predictions)
    if not silent:
        print("       \tPREDICTED")
        print("TRUE  |\tokay\tfraud")
        print("okay  |\t"+str(cm[0][0])+"\t"+str(cm[0][1]))
        print("fraud |\t"+str(cm[1][0])+"\t"+str(cm[1][1]))
        print("overall accuracy: "+str(acc))
        print("fraud accuracy:   "+str(fraud_acc))
    return fraud_acc


# In[81]:


#shuffling the data
data = data.sample(frac=1).reset_index(drop=True)
frauds = data[data['Class'] == 1]
validation_portion = 0.15
validation_cutoff_index = int(len(data)*validation_portion)
validation_set = data[:validation_cutoff_index]
training_set = data[validation_cutoff_index:]

training_set = training_set.sample(frac=1).reset_index(drop=True)
validation_set = validation_set.sample(frac=1).reset_index(drop=True)

X=training_set.drop(columns=['Class'])
Y=training_set['Class']
X_val=validation_set.drop(columns=['Class'])
Y_val=validation_set['Class']

print("Training size:   "+str(len(X)))
print(" fraud count: "+str(len(Y[Y==1])))
print("Validation size: "+str(len(X_val)))
print(" fraud count: "+str(len(Y_val[Y_val==1])))

labels = 'Okay', 'Fraud'
sizes = [len(Y[Y==0]), len(Y[Y==1])]
colors = ['green', 'red']
explode = (0.1,0)
# Plot
plt.pie(sizes, labels=labels,
        colors=colors,
       explode=explode)
 
plt.axis('equal')
plt.show()


# Judging from the output above, our data is clearly skewed. Skewed data is one of the worst things to deal with... so let's give it a shot and see what we can do!

# ## Neural Net
# NNs are all the rage these days, so let's throw an NN at this problem and see what happens.

# In[62]:


nn_model = Sequential()
nn_model.add(Dense(100, input_dim=30, activation='relu')) # taking in the 30 inputs
nn_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn_model.add(Dense(300, activation='relu')) # layer of 300 neurons
nn_model.add(Dense(500, activation='sigmoid')) # layer of 500 neurons w/ sigmoid activation
nn_model.add(Dense(1, activation='sigmoid')) # final layer to say if its 

num_epochs = 3
batch_size = 1024
nn_model.compile(loss='logcosh',
                 optimizer=keras.optimizers.RMSprop(lr=0.0001), 
                 metrics=[f1, 'accuracy'])
nn_model.fit(X, Y, epochs=num_epochs, batch_size=batch_size)


# In[63]:


evaluate(nn_model, X_val, Y_val)


# That was as terrible as it could be. It marked all of them as being non-fraudulent. Additionally, it took quite a while to train, and this wasn't even that much training compared to what the norm is. This would be real bad for a credit card company.

# ## Decision Tree
# Perhaps a differnet approach would work better, possibly a decision tree? We'll use the default settings so it'll be a quick estimate of how a DTree can do.

# In[64]:


tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X, Y)


# Now let's try some predictions.

# In[65]:


evaluate(tree_model, X_val, Y_val)


# Look at that, it's doing much better. Of the **74** cases of fraud, it got **59** of them correct, but missed **15**. It also incorrectly labeled **18** proper transactions as being fraudulent. However, in this situation, marking an okay transaction as fraud, is much better than letting a fraudulent transaction slip through the cracks.
# 
# I think it's worth checking another style of ML to see if that can do any better.

# In[66]:


rf_model = RandomForestClassifier(max_depth=15,
                                  warm_start=False,
                                  n_jobs=-1,
                                  random_state=0)
rf_model.fit(X,Y)


# In[67]:


evaluate(rf_model, X_val, Y_val)


# That did one better for classifying fraud, but that's negligible. What really improved though, was classifying the proper transactions, going from **18** to **8**. As mentioned earlier, this metric is less important to correctly identifying fraud, but still an improvement.
# 
# ### Trying different class weights
# Maybe we can tweak this to work a little better. Let's add in some class weights and see if we can get a better value with that.

# In[69]:


okay_val=0.5
fraud_min=0.05
fraud_max=0.951
fraud_incr=0.05
fraud_range=np.arange(fraud_min, fraud_max, fraud_incr)
results = []
for fraud_weight in tqdm(fraud_range):
    rf_model = RandomForestClassifier(max_depth=22,
    #                                 0=okay, 1=fraud
                                      class_weight={0:okay_val, 1:fraud_weight},
                                      warm_start=False,
                                      n_jobs=-1,
                                      random_state=0)
    rf_model.fit(X,Y)
    results.append(evaluate(rf_model, X_val, Y_val, silent=True))


# In[70]:


fig, ax = plt.subplots()
ax.plot(fraud_range, results)

ax.set(xlabel='fraud weight', ylabel='fraud_accuracy', title="okay weight = "+str(okay_val))
ax.grid()
plt.show()


# Seems like a 0.45 paired with a 0.5 value works out best, but there doesn't seem to be a clear answer and the variation can just be chalked up to randomness.
# 
# Perhaps we should try some sampeling techniques to get around these issues.
# 
# # Random Over-Sampling

# In[82]:


ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_sample(X, Y)

print("new size:        "+str(len(X_resampled)))
print("new fraud count: "+str(len(X_resampled[Y_resampled==1])))

labels = 'Okay', 'Fraud'
sizes = [len(Y_resampled[Y_resampled==0]), len(Y_resampled[Y_resampled==1])]
colors = ['green', 'red']
explode = (0.1,0)
# Plot
plt.pie(sizes, labels=labels,
        colors=colors,
       explode=explode)
 
plt.axis('equal')
plt.show()


# ## Neural Network (pt. 2)

# In[83]:


nn2_model = Sequential()
nn2_model.add(Dense(100, input_dim=30, activation='relu')) # taking in the 30 inputs
nn2_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn2_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn2_model.add(Dense(300, activation='relu')) # layer of 300 neurons
nn2_model.add(Dense(500, activation='sigmoid')) # layer of 500 neurons w/ sigmoid activation
nn2_model.add(Dense(1, activation='sigmoid')) # final layer to say if its 

num_epochs = 3
batch_size = 1024
nn2_model.compile(loss='logcosh',
                 optimizer=keras.optimizers.RMSprop(lr=0.0001), 
                 metrics=[f1, 'accuracy'])
nn2_model.fit(X_resampled, Y_resampled, epochs=num_epochs, batch_size=batch_size)


# In[84]:


evaluate(nn2_model, X_val, Y_val)


# Remember that neural net that got 0% for fraud detection, it just got a near perfect score! This just goes to show how awful skewed data can be.
# 
# Let's try some of the other classifiers that did okay even when the data wasn't resampled.

# In[74]:


rf2_model = RandomForestClassifier(max_depth=15,
                                  warm_start=False,
                                  n_jobs=-1,
                                  random_state=0)
rf2_model.fit(X_resampled, Y_resampled)


# In[75]:


evaluate(rf2_model, X_val, Y_val)


# Hmm, not as good as the NN, actually pretty similar to what we got last time with the DTree. This makes sense because the decision tree isn't affected by a skewed dataset as severely as NNs are.

# ## Conclusion
# This dataset was awfully skewed, so horrendously that a neural net went from 0% to 97% just by some simple random over sampling. Let's talk about that. Random over sampling takes the okay transactions and the fraudulent transactions and samples them with replacement so that the classes are balanced. In our case here, that means that the ~400 fraudulent transaction were multiplied into about 240k samples. We can naively assume that each original sample was duplicated about 600 times. This means that the neural network is very likely to be overfitted on those 400 original samples. Tread carefully when using random over sampling.
