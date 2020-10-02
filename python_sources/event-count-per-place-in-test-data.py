#!/usr/bin/env python
# coding: utf-8

# The popularity of place ids is changing with time. The script below will calculate the popularity of places in the test data. Lets start with the basic time calculations below:

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### read in train data
train = pd.read_csv('../input/train.csv')
train['hour'] = (train['time']//60)%24+1 # 1 to 24 hours
train['weekday'] = (train['time']//1440)%7+1 # 1 to 7 weekdays
train['month'] = (train['time']//43800)%12+1 # rough estimate, month = 30.4 days 
train['year'] = (train['time']//525600)+1 # year 1 and 2


# Next we will calculate median for train x, y data for each place id

# In[ ]:



places = train.groupby('place_id')['x','y'].agg(np.median) # median is better than mean for this data
places.columns = ['xc', 'yc']  # the median values xc, yc are the centers of the places
places['train_count']=0 # count of train data per place id in a small window around xc, yc
places['test_count']=0 # count of test data per place id in a small window around xc, yc

print(places.head())


# limit train data to similar timeframe as test data (from year 1). When we did the calculations for xc,yc in the previous step, we used the full data set which gave more resolution for finding xc,yc

# In[ ]:



# limit train data to similar timeframe as test data
train = train.query('month>=7 and month<=12 and year==1')


# read in the test data

# In[ ]:



### read in test data
test = pd.read_csv('../input/test.csv')
test['hour'] = (test['time']//60)%24+1 # 1 to 24 hours
test['weekday'] = (test['time']//1440)%7+1 # 1 to 7 weekdays
test['month'] = (test['time']//43800)%12+1 # rough estimate, month = 30.4 days
test['year'] = (test['time']//525600)+1 # year 2


# Calculate counts within a small window around xc,yc for both train and test data. It takes about 3 hours to go through all place ids. I have shown 5 placeids which have biggest differences between counts 

# In[ ]:



tr = np.asarray(train[['x','y']])
te = np.asarray(test[['x','y']])

print(len(tr))
print(len(te))

## ratio for scaling counts based on data sizes. It will be close to 1
train_test_ratio = len(train)/float(len(test))
print(train_test_ratio)


# In[ ]:



windowsize = 0.01
#for i,placeid in enumerate(places.index):  # use this to iterate over all placeids. Takes 3 hrs
for i,placeid in enumerate([6305130230, 3938632601, 5705309641, 5676118839, 4501921430]):  
    xmax=places.xc[placeid]+windowsize
    xmin=places.xc[placeid]-windowsize
    ymax=places.yc[placeid]+windowsize
    ymin=places.yc[placeid]-windowsize
    places = places.set_value(placeid, 'train_count', ( (xmin < tr[:,0]) & (tr[:,0] < xmax) & (ymin < tr[:,1]) & (tr[:,1] < ymax)  ).sum()  )   
    places = places.set_value(placeid, 'test_count', ( (xmin < te[:,0]) & (te[:,0] < xmax) & (ymin < te[:,1]) & (te[:,1] < ymax)  ).sum() * train_test_ratio  )   


# The count columns can be scaled and used for prediction in the model. Let's plot the results to see if the calculations are correct. The red dot in the plots is the xc,yc center for the place
# 

# In[ ]:


placeid = 6305130230
xmin= places.loc[placeid,'xc']-0.3
xmax=places.loc[placeid,'xc']+0.3
ymin=places.loc[placeid,'yc']-0.3
ymax=places.loc[placeid,'yc']+0.3
      
train_subset = train.query('( @xmin<= x <= @xmax) and ( @ymin<=y<= @ymax)')
test_subset = test.query('( @xmin<= x <= @xmax) and ( @ymin<=y<= @ymax)')
    
fig = plt.figure(1, figsize=(10,4))

ax1 = plt.subplot(1,2,1)
ax1.scatter(train_subset['x'], train_subset['y'], s=1, c='g', marker="s", label='first', edgecolors='none')
ax1.scatter(places['xc'][placeid], places['yc'][placeid], s=25, c='r', marker="s", label='first', edgecolors='none')
ax1.set(xlim=(xmin, xmax))
ax1.set(ylim=(ymin, ymax))
plt.title("train: place_id " + str(placeid)) 
plt.xlabel('x')
plt.ylabel('y')
    
ax2 = plt.subplot(1,2,2)
ax2.scatter(test_subset['x'], test_subset['y'], s=1, c='b', marker="s", label='first', edgecolors='none')
ax2.scatter(places['xc'][placeid], places['yc'][placeid], s=25, c='r', marker="s", label='first', edgecolors='none')
ax2.set(xlim=(xmin, xmax))
ax2.set(ylim=(ymin, ymax))
plt.title("test: place_id " + str(placeid)) 
plt.xlabel('x')
plt.ylabel('y')
    
plt.show()


# Let's plot a few more place ids

# In[ ]:


fig, axs = plt.subplots(4,2, figsize=(10, 20))
axs = axs.ravel()

for i,placeid in enumerate([3938632601, 5705309641, 5676118839, 4501921430]):     
    xmin= places.loc[placeid,'xc']-0.3
    xmax=places.loc[placeid,'xc']+0.3
    ymin=places.loc[placeid,'yc']-0.3
    ymax=places.loc[placeid,'yc']+0.3
     
    train_subset = train.query('( @xmin<= x <= @xmax) and ( @ymin<=y<= @ymax)')
    test_subset = test.query('( @xmin<= x <= @xmax) and ( @ymin<=y<= @ymax)')
    
    axs[i*2].scatter(train_subset['x'], train_subset['y'], s=1, c='g', marker="s", label='first', edgecolors='none')
    axs[i*2].scatter(places.loc[placeid,'xc'], places.loc[placeid,'yc'], s=25, c='r', marker="s", label='first', edgecolors='none')
    axs[i*2].set(xlim=(xmin, xmax))
    axs[i*2].set(ylim=(ymin, ymax))
    axs[i*2].set_title("train: place_id " + str(placeid)) 
    axs[i*2].set_xlabel('x')
    axs[i*2].set_ylabel('y')
    
    axs[i*2+1].scatter(test_subset['x'], test_subset['y'], s=1, c='b', marker="s", label='first', edgecolors='none')
    axs[i*2+1].scatter(places.loc[placeid,'xc'], places.loc[placeid,'yc'], s=25, c='r', marker="s", label='first', edgecolors='none')
    axs[i*2+1].set(xlim=(xmin, xmax))
    axs[i*2+1].set(ylim=(ymin, ymax))
    axs[i*2+1].set_title("test: place_id " + str(placeid)) 
    axs[i*2+1].set_xlabel('x')
    axs[i*2+1].set_ylabel('y')

plt.show()

